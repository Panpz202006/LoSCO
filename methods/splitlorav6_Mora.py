import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans

from models.net_splitlora_R import Net
from models.vit_splitlora_R import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class SplitLoRAV6(BaseLearner):

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
        self.in_dim = args['in_dim']
        self.alpha = args['alpha']

        self.rank = args['rank']
        # self.atl = agrs['atl']
        # self.random_coeffs_K = nn.ParameterDict()
        # self.random_coeffs_V = nn.ParameterDict()
    
    @staticmethod
    def init_lora_with_mora_style(feature_matrix, rank, target_norm, scale_factor, group_size=16, target_dim=None):
        """
        MoRA 风格的分组压缩降维初始化
        """
        dim, feat_dim = feature_matrix.shape
        
        # 如果指定了 target_dim，确保 feat_dim 匹配
        if target_dim is not None and feat_dim != target_dim:
            if feat_dim < target_dim:
                # 补零到 target_dim
                feature_matrix = torch.nn.functional.pad(feature_matrix, (0, target_dim - feat_dim))
            else:
                # 截断到 target_dim
                feature_matrix = feature_matrix[:, :target_dim]
            feat_dim = target_dim
        
        # 确保 dim 能被 group_size 整除
        if dim % group_size != 0:
            pad_size = group_size - (dim % group_size)
            feature_matrix = torch.nn.functional.pad(feature_matrix, (0, 0, 0, pad_size))
            dim = feature_matrix.shape[0]
        
        new_dim = dim // group_size
        
        # 分组压缩
        compressed = feature_matrix.view(new_dim, group_size, feat_dim).sum(dim=1)
        
        # 随机投影
        random_proj = torch.randn(new_dim, rank).to(feature_matrix.device)
        A_init = compressed.T @ random_proj  # [feat_dim, rank]
        
        # 归一化 + 缩放
        A_init = A_init / A_init.norm() * target_norm * scale_factor
        
        return A_init.T  # [rank, feat_dim]
        
    def init_drm(self, train_loader):
        if self.cur_task == 0:
            print("Using Original LoRA Initialization for Task 0")
            return
        else:
            print("Using SplitLora Initialization for Task t(t>=1)")
            i = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    feature_matrix = self.feature_list[i].to(self.device)
                    
                    target_norm_k = module.lora_A_k[self.cur_task].weight.data.norm()
                    target_norm_v = module.lora_A_v[self.cur_task].weight.data.norm()
                    scale_factor = 1 / (self.cur_task + 1)
                    
                    # 获取期望的维度
                    target_dim = module.lora_A_k[self.cur_task].weight.shape[1]
                    
                    A_init_k = self.init_lora_with_mora_style(
                        feature_matrix, self.rank, target_norm_k, scale_factor, 
                        group_size=16, target_dim=target_dim
                    )
                    A_init_v = self.init_lora_with_mora_style(
                        feature_matrix, self.rank, target_norm_v, scale_factor,
                        group_size=16, target_dim=target_dim
                    )
                    
                    module.lora_A_k[self.cur_task].weight.data.copy_(A_init_k)
                    module.lora_A_v[self.cur_task].weight.data.copy_(A_init_v)
                    
                    i += 1


                    # # print(f"*************{i}**********")
                    # #
                    # # random_coeffs_A = module.xxx
                    # # random_coeffs_V = 
                    
                    # # random_coeffs_K = (torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device))
                    # print("feature_list_dim", self.feature_list[i].shape[1])
                    # random_coeffs_V = (torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device))
                    # random_coeffs_K = random_coeffs_V #共享的
                    # # 第2行：用特征矩阵乘以随机系数
                    # self.feature_list[i] = self.feature_list[i].to(self.device)
                    # initialized_matrix_K = self.feature_list[i] @ random_coeffs_K
                    # initialized_matrix_V = self.feature_list[i] @ random_coeffs_V

                    # # 第3行：归一化 + 缩放
                    # initialized_matrix_K = initialized_matrix_K / initialized_matrix_K.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    # initialized_matrix_V = initialized_matrix_V / initialized_matrix_V.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    # # 将UR矩阵的结果赋值给A（应该指的是 Y=X(W+AB中的A，即A<-UR)
                    # module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)   
                    # module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)   
                    # i += 1


                
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
        # 用Splitlora代替update_DualGPM(mat_list)
        for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = torch.linalg.svd(activation, full_matrices=False) #在这里接受输入激活值
                sval_total = (S).sum() #在这里计算总的能量
                sval_ratio = (S)/sval_total #在这里计算各个能量占得总的能量的比率
                k = torch.arange(1, self.in_dim+1).to(self.device)-1
                k = (self.in_dim-k.float()) / self.in_dim
                sval_ratio = sval_ratio.to(self.device)
                result = (self.cur_task+1)*(1-torch.cumsum(sval_ratio,dim=-1)) - self.alpha*k
                r = torch.argmin(result).item()

                # 这一步很多余，我感觉没必要！
                # temp = U[:,max(r,1)+1:]
                # Uf=torch.matmul(temp,temp.T)
                # self.feature_list[i] = Uf
                if i >= len(self.feature_list):
                    self.feature_list.append(U[:, max(r, 1) + 1:])
                else:
                    self.feature_list[i] = U[:, max(r, 1) + 1:]


                
    #这个是整个Splitlora算法的核心，其他都是参数传入、数据集载入、
    def _train(self, train_loader):
        self.network.to(self.device)
        # 保证B和分类池中的分类头是可学习的
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

        # if self.atl:
        #     self._build_protos()

        
        # 从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A用U初试化(冻结)。
        self._train_function(train_loader, optimizer, scheduler)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # Preserve the information about the gradient of the t-th task through Splitlora
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

                # if self.atl:
                #     #这里应该加上ATL loss函数进行下一步计算
                #     criterion = AugmentedTripletLoss(margin=self.margin_inter)

                #     ATL = criterion(feature, labels, self._protos, self.device)
                #     loss += self.lambada * ATL

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
            f"scaling_factor",
        #     "random_coeffs_K",  # 如果所有任务共享
        # "random_coeffs_V",  # 如果所有任务共享

        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    
    

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
