import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans

from models.net_inflora import Net
from models.vit_inflora import Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        # inflora
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def init_drm(self, train_loader):
        """initialzation of dimensionality reduction matrix A"""
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)  # gather the features in module.cur_matrix
            if self.debug: break
        
        if self.cur_task == 0:
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    cur_matrix = module.cur_matrix
                    U, S, V = torch.linalg.svd(cur_matrix)
                    module.lora_A_k[self.cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                    module.lora_A_v[self.cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
        else:
            kk = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    cur_matrix = module.cur_matrix
                    if self.project_type[kk] == 'remove':
                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                    else:
                        assert self.project_type[kk] == 'retain'
                        cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                    cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                    module.lora_A_k[self.cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                    module.lora_A_v[self.cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
                    kk += 1

    def psv_info(self, train_loader):
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)
        
        mat_list = []
        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
        
        self.update_DualGPM(mat_list)

        self.feature_mat = []
        for p in range(len(self.feature_list)):
            Uf = torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
            print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
            self.feature_mat.append(Uf)

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        # Design LoRA matrix through Equation (8)
        with torch.no_grad():
            self.init_drm(train_loader)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # Preserve the information about the gradient of the t-th task through DualGPM
        with torch.no_grad():
            self.psv_info(train_loader)
        
        return

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self.known_classes
                
                logits = self.network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_B_k{target_suffix}",
            f"lora_B_v{target_suffix}",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    
    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            mask = (targets >= self.known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)

            with torch.no_grad():
                if isinstance(self.network, nn.DataParallel):
                    feature = self.network.module.extract_vector(inputs)
                else:
                    feature = self.network.extract_vector(inputs)

            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)

        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def update_DualGPM (self, mat_list):
        threshold = (self.lame - self.lamb) * self.cur_task/self.sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold)
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('(DualGPM) Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} (type: {})'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

    def update_GPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self.cur_task/self.sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                self.feature_list.append(U[:,0:max(r,1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
            
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                # update GPM
                Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    self.feature_list[i]=Ui
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            logging.info('Layer {} : {}/{}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-'*40)  

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
