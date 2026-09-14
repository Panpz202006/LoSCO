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
        # 我不懂这里为啥又要再跑一遍训练集了，之前不是已经跑过一次了么？（在train函数中）
        # 并且为啥还执行了一次SVD分解。
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)  # gather the features in module.cur_matrix
            if self.debug: break
        
        if self.cur_task == 0:
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    cur_matrix = module.cur_matrix
                    U, S, V = torch.linalg.svd(cur_matrix)
                    # 用U的前r个行向量初始化A矩阵
                    module.lora_A_k[self.cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                    module.lora_A_v[self.cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            # 改进点1：替换为对W_0的权重进行
            # 阅读完之后的新感受：为啥这里初试化要用到当前（W+AB）的 activation 进行SVD分解后的结果，为啥没有用到要用AB的参数结果，
            # 这里显然是不恰当的，没有学习的AB，其实这里相当于是 (W+0,A服从高斯分布，B是全0初始化的)，那么从这角度来理解A的初始化就是用的W的U矩阵，
            # 但是没有用到特征值。 B的话还是0矩阵，因此优化空间还有。基于此，我们提出下面两种改进方案
            # 改进点2：对W进行奇异值分别，而不是对激活值，不光对A进行初始化，而且对B也进行初始化。强调关注权重W。
            # 改进点3：基于上面的结果我们还可以Sigm的影响，进而更加有针对性的初始化。强调关注
            # 改进点4：不单单针对前面第一个任务增加有效的初始化方案，除了DualGpm的初始化A和splitlora的初始化A策略，
            # 我们还可以采取一些其他的初始化B的策略，不再让B是一个零矩阵，要不然收敛的太慢了。
        else:
            kk = 0
            # 后续任务：需要投影到 N_t ∩ M_t^⊥
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    cur_matrix = module.cur_matrix
                    if self.project_type[kk] == 'remove':  #移除主成分空间的信息，保留残差空间
                        # 公式(6): Ĥ_t = H_t - M_t M_t^T H_t (移除M_t中的成分) 
                        # 通俗的理解就是，去除掉当前激活矩阵中包含的前面所有任务的激活矩阵信息。
                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                    else:
                        # 公式(7): Ĥ_t = M_t^⊥ (M_t^⊥)^T H_t (保留M_t^⊥中的成分) #保留主成分空间的信息，移除残差空间
                        assert self.project_type[kk] == 'retain'
                        # 通俗的理解就是，仅仅使用当前激活矩阵中包含的前面所有任务的激活矩阵信息。
                        cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                    cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                    # SVD分解得到B_t (公式8) 
                    module.lora_A_k[self.cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3)) 
                    module.lora_A_v[self.cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3)) 
                    module.cur_matrix.zero_() 
                    module.n_cur_matrix = 0 
                    kk += 1 

    def psv_info(self, train_loader):
        # 想要了解这个函数的功能，需要慢慢一步一步往下看。
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #这里的forward函数中get_cur_feat=True在Attention_LoRA模块中注
            # 册hook来获取当前任务的激活矩阵（即cur_matrix）。
            self.network(inputs, get_cur_feat=True)
            
        #这里的mat_list是一个列表，里面存储了每个Attention_LoRA模块的cur_matrix
        # （即当前任务的激活矩阵）。这个矩阵在前面训练过程中被更新了。
        mat_list = []
        for module in self.network.modules():
            # 判断：module 这个对象是不是 Attention_LoRA 类型的实例？
            # 返回值：True 或 False
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
        
        self.update_DualGPM(mat_list)

        self.feature_mat = []
        #feature_list是始终不变的，使用GPM和DualGPM更新得到的特征子空间矩阵列表。
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

        # 创建优化器和学习策略。
        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)
        
        # 从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A用U初试化(冻结)。
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
    
    # def clustering(self, dataloader):
    #     features = []
    #     for i, (_, inputs, targets) in enumerate(dataloader):
    #         inputs, targets = inputs.to(self.device), targets.to(self.device)
    #         mask = (targets >= self.known_classes).nonzero().view(-1)
    #         inputs = torch.index_select(inputs, 0, mask)

    #         with torch.no_grad():
    #             if isinstance(self.network, nn.DataParallel):
    #                 feature = self.network.module.extract_vector(inputs)
    #             else:
    #                 feature = self.network.extract_vector(inputs)

    #         feature = feature / feature.norm(dim=-1, keepdim=True)
    #         features.append(feature)

    #     features = torch.cat(features, 0).cpu().detach().numpy()
    #     clustering = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(features)
    #     self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

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
