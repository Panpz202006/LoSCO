import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import optimgrad
import re
from collections import defaultdict
from utils.losses import AugmentedTripletLoss
from scipy.spatial.distance import cdist
from utils.toolkit import print_trainable_params, check_params_consistency
from dataloaders.data_manager import DataManager

import optimgrad



class LoRAsub_DRS(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self.network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lambada = args["lambada"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.fc_lrate = args["fc_lrate"]
        self.margin_inter = args["margin_inter"]
        self.eval = args['eval']
        self._protos = []

        self.topk = 1  # origin is 5
        self.class_num = self.network.class_num
        self.debug = False
        self.fea_in = defaultdict(dict)

        self.data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)


        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()
    
    def freeze_network(self, train_loader):
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self.network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self.network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self.network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self.network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self.network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self.network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self.network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self.network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self.network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self.network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
        
        # Double check
        enabled = set()
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        # 和Splitlora的初试化还挺像的，从第二个任务起就需要获取表征矩阵。（由于没有梯度，那么只需要no_grad模式就行了）
        with torch.no_grad():
            if self.cur_task > 0:
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.network(inputs, get_cur_x=True)

                for module in self.network.modules():
                    if isinstance(module, Attention_LoRA):
                        self.fea_in[module.lora_A_k[self.cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self.device)
                        self.fea_in[module.lora_A_v[self.cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self.device)
                        self.fea_in[module.lora_B_k[self.cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self.device)
                        self.fea_in[module.lora_B_v[self.cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self.device)
                        module.cur_matrix.zero_()
                        module.matrix_kv = 0
                        module.n_cur_matrix = 0


    def init_model_optimizer(self):
        if self.cur_task == 0:
            lr = self.init_lr
        else:
            lr = self.lrate

        #这个是单独拿出来中间可学习参数，是不包括classifier_pool的。
        fea_params = [p for n, p in self.network.named_parameters() if
                      not bool(re.search('classifier_pool', n)) and p.requires_grad == True]
        #这个是单独拿出来分类器可学习参数，只有classifier_pool会参与训练。
        cls_params = [p for n, p in self.network.named_parameters() if bool(re.search('classifier_pool', n))]
        
        # 这个是LoRA的优化器，包含了可学习参数和一些超参数，主要是针对可学习参数的学习率和权重衰减。
        model_optimizer_arg = {'params': 
            [{'params': fea_params, 'svd': True, 'lr': lr,'thres': 0.99},
            {'params': cls_params, 'weight_decay': self.weight_decay,'lr': self.fc_lrate}],
                               'weight_decay': self.weight_decay,
                               'betas': (0.9, 0.999)
                               }
        # self.args['model_optimizer'] = 'Adam'
        ## 获取类，不是实例
        self.model_optimizer = getattr(optimgrad, self.args['optim'])(**model_optimizer_arg)
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.epochs)

    # 这里主要是Adam的初试化思想
    def build_optimizer(self): 
        self.init_model_optimizer() 
        if self.cur_task == 0: 
            self.run_epoch = self.init_epoch 
        else: 
            self.update_optim_transforms() 
            self.run_epoch = self.epochs 
        # return optimizer, scheduler

    def _train_function(self, train_loader):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                ret = self.network(inputs)
                logits = ret['logits']
                features = ret['features']
                feature = features / features.norm(dim=-1, keepdim=True)

                logits = self.network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                #这里应该加上ATL loss函数进行下一步计算

                criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self.device)

                ATL = criterion(feature, labels, self._protos, self.device)
                loss += self.lambada * ATL

                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            self.model_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    #重写这一部分，主要lora_sub_drs的训练过程还需要test_loader.
    def incremental_train(self, data_manager):
        self.build_train_loader(data_manager)
        self.build_test_loader(data_manager)

        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader, self.test_loader)

    def _extract_vectors(self, loader):
        self.network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self.network, nn.DataParallel):
                _vectors = tensor2numpy(self.network.module.extract_vector(_inputs.to(self.device)))
            else:
                _vectors = tensor2numpy(self.network.extract_vector(_inputs.to(self.device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _build_protos(self):
        self.network.to(self.device)
        with torch.no_grad():
            for class_idx in range(self.known_classes, self.total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)

    #这个是核心部分。
    def _train(self, train_loader, test):
        self.network.to(self.device)
        self.freeze_network(train_loader) #这里需要注意，冻结A和B。
        print_trainable_params(self.network) #在这里打印可学习的参数以验证是否正确冻结。



        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        # 设计需要更新的表征矩阵在建立优化器中完成。
        # with torch.no_grad():
        #     self.init_drm(train_loader) 

        # 创建优化器和学习策略。这里自己搭建adam-ncsl优化器
        self.build_optimizer()
        check_params_consistency(self.network, self.model_optimizer)
        
        self._build_protos()
        


        #这里是正式的训练函数，从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A。
        # 这里的lora sub drs 需要同时学习A和B，导致可学习的参数比常规 
        self._train_function(train_loader)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # # Preserve the information about the gradient of the t-th task through DualGPM
        # with torch.no_grad():
        #     self.psv_info(train_loader)
        
        return

    def update_optim_transforms(self):
        # 基于输入特征计算协方差矩阵的特征分解 
        self.model_optimizer.get_eigens(self.fea_in)

        # 构建变换矩阵用于参数更新 
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)







