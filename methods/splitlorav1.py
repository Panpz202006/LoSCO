import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans

from models.net_splitlora import Net
from models.vit_splitlora import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency

# 为了满足ATL loss推出的几个包
from utils.losses import AugmentedTripletLoss
# from dataloaders.data_manager import DataManager
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

class SplitLoRAV1(BaseLearner):

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

        # ATL loss from Lora_Sub_DRS
        self.atl = args['atl']
        self._protos = []
        self.lambada = args["lambada"]
        self.margin_inter = args["margin_inter"]

        # #datasets args
        # self.dataset_name, self.shuffle, self.seed = args['dataset'], args['shuffle'], args['seed']
        # self.init_cls, self.increment =  args['init_cls'], args['increment']

        # self.data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)

        self.feature_list_remove = []  # 用于 remove 策略的残差空间
        self.feature_list_retain = []  # 用于 retain 策略的主成分空间
        self.project_type = []  # 记录每层的策略类型 
        self.use_reatain = args["use_reatain"]
        self.pca = args["pca"] # 使用pca降维获取参与空间进而进行充分学习
        self.svd = args["svd"]
        # 维护一个全局残差空间，因为上一个任务学习的残差空间可以与接下来任务的残差空间重合，
        # 因此，需要将当前任务的残差空间投影在global 残差空间中，然后执行：当前任务的残差空间 - 投影残差 
        self.global_res = args["global_res"]
        self.global_res_list = []
        self.global_res_compress_threshold = args.get("global_res_compress_threshold", 0.99)  # 压缩阈值
        self.global_res_allow_overlap = args.get("global_res_allow_overlap", 0.0)  # 允许重叠比例


        # Space 方法
        self.space = args.get("space", False)  # SPACE 方法开关
        self.space_variance_threshold = args.get("space_variance_threshold", 0.90)  # 方差保留阈值
        
        # SPACE 专用：核心空间（只读知识库）和残差空间（草稿纸）
        self.core_space_list = []      # 永久冻结的核心基 U_core
        self.residual_dim_list = []    # 每层当前可用的残差维度数       
        self.tsne = args["tsne"] 
        self.mds = args["mds"] 
        self.umap = args["umap"] 

        #创建A=UR中的R为可学习的系数
        self.learnable_coeffs = nn.ParameterList()  # 存储每层每个任务的可学习系数
        self.learnable_R = args['learnable_R']

    def init_drm(self, train_loader):
        """initialzation of dimensionality reduction matrix A"""
        # 这里很显然是后面的任务在前一个任务的残余空间中学习。
        # 然而，Inforlora是在前一个

        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)  # gather the features in module.cur_matrix
            if self.debug: break
        
        if self.cur_task == 0:
            #对于Splitlora，他的第一个任务的初始化为lora的原始方式。
            print("Using Original LoRA Initialization for Task 0")
            return
        else:
            # 后续任务：使用上一个任务保存到的self.feature_list
            print("Using SplitLora Initialization for Task t(t>=1)")
            i = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    if self.pca:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        pca = PCA(n_components=10)
                        features_reduced = pca.fit_transform(features)  # shape: (m, k)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    elif self.tsne:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        pca = TSNE(n_components=10)
                        features_reduced = pca.fit_transform(features)  # shape: (m, k)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    elif self.mds:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        pca = MDS(n_components=10)
                        features_reduced = pca.fit_transform(features)  # shape: (m, k)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))

                    elif self.svd:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        # features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        U, S, Vt = torch.linalg.svd(self.feature_list[i], full_matrices=False)
                        k = 10
                        U_k = U[:, :k]
                        S_k = S[:k]
                        # 降维后的数据
                        initialized_matrix = U_k * S_k  # shape: (m, k)
                        initialized_matrix = initialized_matrix.to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    elif self.umap:  # 新增 UMAP 分支
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        features = self.feature_list[i].cpu().numpy()
                        
                        # 导入 UMAP
                        import umap
                        # 创建 UMAP 对象，降到 10 维
                        # 关键参数设置：
                        # n_neighbors: 控制局部与全局结构平衡，通常 15-30 适合特征降维
                        # min_dist: 控制点之间的最小距离，设为 0.1 比较通用
                        # random_state: 保证结果可复现
                        reducer = umap.UMAP(
                            n_components=10, 
                            n_neighbors=15, 
                            min_dist=0.1, 
                            metric='euclidean',
                            random_state=42
                        )
                        features_reduced = reducer.fit_transform(features)
                        
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))                    
                                # elif self.global_res:
                    #     pass
                    #     #将收集得到的 
                    #     self.feature_list[i] = self.feature_list[i].to(self.device)
                    #                         self.
                    elif self.global_res:
                        # 全局残差空间：避免不同任务间的方向重合
                        # 核心思想：新方向 = 当前残差 - proj_{全局占用}(当前残差)
                        
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        
                        # 初始化或获取当前层的全局占用空间
                        if len(self.global_res_list) <= i:
                            # 第一个任务，初始化全局空间
                            self.global_res_list.append({
                                'occupied': None,      # 已占用的方向空间
                                'available': None,     # 可用方向空间
                                'history_residuals': [] # 历史残差列表
                            })
                        
                        global_space = self.global_res_list[i]
                        
                        # 获取当前任务的残差空间（存储在 feature_list[i] 中）
                        current_residual = self.feature_list[i]  # [d, r_current]
                        
                        # 检查是否有历史占用的空间
                        if global_space['occupied'] is not None:
                            occupied = global_space['occupied'].to(self.device)  # [d, r_occupied]
                            
                            # === 步骤1: 计算当前残差在已占用空间上的投影（重合部分）===
                            # projection = occupied @ (occupied^T @ current_residual)
                            # 这计算的是当前残差中与历史任务重合的部分
                            UtH = occupied.T @ current_residual  # [r_occupied, r_current]
                            projection = occupied @ UtH          # [d, r_current]
                            
                            # === 步骤2: 减去投影，得到真正可用的新方向 ===
                            # new_direction = current_residual - projection
                            new_direction = current_residual - projection
                            
                            # === 步骤3: 正交化并过滤数值误差 ===
                            Q_new, R_new = torch.linalg.qr(new_direction, mode='reduced')
                            # 过滤掉接近零的列（数值误差导致的）
                            col_norms = torch.norm(Q_new, dim=0)
                            valid_cols = col_norms > 1e-6
                            Q_new = Q_new[:, valid_cols]
                            
                            # 计算重合率（用于监控）
                            overlap_ratio = 1 - Q_new.shape[1] / max(current_residual.shape[1], 1)
                            print(f"  Layer {i}: Overlap ratio={overlap_ratio:.2%}, "
                                f"Original={current_residual.shape[1]}, "
                                f"New={Q_new.shape[1]}, "
                                f"Occupied={occupied.shape[1]}")
                            
                            # === 步骤4: 更新全局占用空间 ===
                            # 将当前任务的完整残差空间加入历史占用
                            combined_occupied = torch.cat([occupied, current_residual], dim=1)
                            # 重新正交化
                            Q_occ, R_occ = torch.linalg.qr(combined_occupied, mode='reduced')
                            # 可选：压缩占用空间（保留主要方向，防止维度爆炸）
                            if Q_occ.shape[1] > self.in_dim * 2:  # 超过阈值时压缩
                                U_occ, S_occ, V_occ = torch.linalg.svd(Q_occ, full_matrices=False)
                                # 保留 99% 能量
                                cumsum_energy = torch.cumsum(S_occ**2, dim=0) / (S_occ**2).sum()
                                k = torch.searchsorted(cumsum_energy, 0.99).item() + 1
                                Q_occ = U_occ[:, :k]
                                print(f"  Layer {i}: Compressed occupied space from {combined_occupied.shape[1]} to {k}")
                            
                            global_space['occupied'] = Q_occ
                            
                            # === 步骤5: 更新全局可用空间 ===
                            if global_space['available'] is not None:
                                # 合并新的可用方向到全局可用空间
                                combined_available = torch.cat([global_space['available'], Q_new], dim=1)
                                Q_avail, R_avail = torch.linalg.qr(combined_available, mode='reduced')
                                # 过滤零列
                                col_norms_avail = torch.norm(Q_avail, dim=0)
                                valid_avail = col_norms_avail > 1e-6
                                global_space['available'] = Q_avail[:, valid_avail]
                            else:
                                global_space['available'] = Q_new
                            
                            # 保存历史残差
                            global_space['history_residuals'].append(current_residual)
                            
                            # 使用新方向作为当前任务的初始化基
                            if Q_new.shape[1] >= self.rank:
                                # 有足够的新方向
                                self.feature_list[i] = Q_new
                            else:
                                # 新方向不足，使用可用空间补充
                                print(f"  Layer {i}: WARNING - new directions ({Q_new.shape[1]}) < rank ({self.rank})")
                                if global_space['available'] is not None and global_space['available'].shape[1] > 0:
                                    # 从全局可用空间中补充
                                    supplement = global_space['available'][:, :min(self.rank - Q_new.shape[1], 
                                                                                    global_space['available'].shape[1])]
                                    combined = torch.cat([Q_new, supplement], dim=1)
                                    self.feature_list[i] = combined
                                else:
                                    # 完全没有可用空间，使用当前残差（允许重叠）
                                    print(f"  Layer {i}: No available space, using original residual (may overlap)")
                                    self.feature_list[i] = current_residual
                            
                        else:
                            # 第一个任务：直接使用残差空间初始化全局空间
                            print(f"  Layer {i}: First task, initializing global space with residual shape={current_residual.shape}")
                            global_space['occupied'] = current_residual
                            global_space['available'] = current_residual  # 初始时所有方向都可用
                            global_space['history_residuals'] = [current_residual]
                            # feature_list[i] 保持不变（已经是残差空间）
                        
                        # 生成随机投影系数（与原版相同）
                        random_coeffs = torch.randn(self.feature_list[i].shape[1], self.rank).to(
                            module.lora_A_k[self.cur_task].weight.device)
                        
                        # 用特征矩阵乘以随机系数
                        initialized_matrix = self.feature_list[i] @ random_coeffs
                        
                        # 归一化 + 缩放
                        scale = 1.0 / (self.cur_task + 1)
                        initialized_matrix_K = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                            module.lora_A_k[self.cur_task].weight.data.norm() * scale
                        initialized_matrix_V = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                            module.lora_A_v[self.cur_task].weight.data.norm() * scale
                    # elif self.learnable_R:  # 新增分支
                    #     # 确保可学习系数容器有足够的空间
                    #     while len(self.learnable_coeffs) <= i:
                    #         self.learnable_coeffs.append(nn.ModuleDict())
                        
                    #     # 检查当前任务的系数是否已存在
                    #     if str(self.cur_task) not in self.learnable_coeffs[i]:
                    #         # 创建可学习的随机系数
                    #         coeffs = torch.randn(self.feature_list[i].shape[1], self.rank)
                    #         learnable_coeff = nn.Parameter(coeffs, requires_grad=True)
                    #         self.learnable_coeffs[i][str(self.cur_task)] = learnable_coeff
                            
                    #         # 注册到网络（确保优化器能捕捉到）
                    #         self.network.register_parameter(
                    #             f'learnable_coeff_layer{i}_task{self.cur_task}', 
                    #             learnable_coeff
                    #         )
                    #     else:
                    #         learnable_coeff = self.learnable_coeffs[i][str(self.cur_task)]
                        
                    #     # 使用可学习系数计算 A
                    #     self.feature_list[i] = self.feature_list[i].to(self.device)
                    #     learnable_coeff = learnable_coeff.to(self.device)
                    #     initialized_matrix = self.feature_list[i] @ learnable_coeff
                        
                    #     # 归一化和缩放（注意：这里会破坏梯度，建议去掉或在反向传播后做）
                    #     scale = 1.0 / (self.cur_task + 1)
                    #     initialized_matrix_K = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                    #                         module.lora_A_k[self.cur_task].weight.data.norm() * scale
                    #     initialized_matrix_V = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                    #                         module.lora_A_v[self.cur_task].weight.data.norm() * scale
                    elif self.learnable_R:
                        # 确保可学习系数容器有足够的空间
                        while len(self.learnable_coeffs) <= i:
                            self.learnable_coeffs.append(nn.ParameterDict())
                        
                        key = str(self.cur_task)
                        if key not in self.learnable_coeffs[i]:
                            # 创建可学习的随机系数
                            coeffs = torch.randn(self.feature_list[i].shape[1], self.rank)
                            learnable_coeff = nn.Parameter(coeffs, requires_grad=True)
                            self.learnable_coeffs[i][key] = learnable_coeff
                        else:
                            learnable_coeff = self.learnable_coeffs[i][key]
                        
                        # 使用可学习系数计算 A
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        learnable_coeff = learnable_coeff.to(self.device)
                        initialized_matrix = self.feature_list[i] @ learnable_coeff
                        
                        # 归一化和缩放
                        scale = 1.0 / (self.cur_task + 1)
                        initialized_matrix_K = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                            module.lora_A_k[self.cur_task].weight.data.norm() * scale
                        initialized_matrix_V = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                            module.lora_A_v[self.cur_task].weight.data.norm() * scale
                                            

                    # elif self.learnable_R:  # 新增分支
                    #     # 确保可学习系数容器有足够的空间
                    #     while len(self.learnable_coeffs) <= i:
                    #         self.learnable_coeffs.append(nn.ModuleDict())
                        
                    #     # 检查当前任务的系数是否已存在
                    #     if str(self.cur_task) not in self.learnable_coeffs[i]:
                    #         # 创建可学习的随机系数
                    #         coeffs = torch.randn(self.feature_list[i].shape[1], self.rank)
                    #         learnable_coeff = nn.Parameter(coeffs, requires_grad=True)
                    #         self.learnable_coeffs[i][str(self.cur_task)] = learnable_coeff
                            
                    #         # 注册到网络（确保优化器能捕捉到）
                    #         self.network.register_parameter(
                    #             f'learnable_coeff_layer{i}_task{self.cur_task}', 
                    #             learnable_coeff
                    #         )
                    #     else:
                    #         learnable_coeff = self.learnable_coeffs[i][str(self.cur_task)]
                        
                    #     # 使用可学习系数计算 A
                    #     self.feature_list[i] = self.feature_list[i].to(self.device)
                    #     learnable_coeff = learnable_coeff.to(self.device)
                    #     initialized_matrix = self.feature_list[i] @ learnable_coeff
                        
                    #     # 归一化和缩放（注意：这里会破坏梯度，建议去掉或在反向传播后做）
                    #     scale = 1.0 / (self.cur_task + 1)
                    #     initialized_matrix_K = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                    #                         module.lora_A_k[self.cur_task].weight.data.norm() * scale
                    #     initialized_matrix_V = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                    #                         module.lora_A_v[self.cur_task].weight.data.norm() * scale

                    else:
                        # print(f"*************{i}**********")
                        random_coeffs = torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        # 第2行：用特征矩阵乘以随机系数
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        initialized_matrix = self.feature_list[i] @ random_coeffs
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        # 将UR矩阵的结果赋值给A（应该指的是 Y=X(W+AB中的A，即A<-UR)

                    module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)   
                    module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)   
                    i += 1

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


    def incremental_train(self, data_manager):
        self.data_manager = data_manager  #用于下面创建原型，用于ATL loss计算。
        self.build_train_loader(data_manager)
        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader)

    def _build_protos(self):
        print("collecting protos!")
        self.network.to(self.device)
        with torch.no_grad():
            for class_idx in range(self.known_classes, self.total_classes):
                # dataset_name, shuffle, seed, init_cls, increment,
                # data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
                # dataset_name=self.dataset_name, shuffle=self.shuffle, seed=self.seed, init_cls=self.init_cls, increment=self.increment = args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment']
                # data_manager = DataManager(dataset_name=self.dataset_name, shuffle=self.shuffle, seed=self.seed, init_cls=self.init_cls, increment=self.increment)
                idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=False)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)
                # del vectors, class_mean, data, targets, idx_dataset, idx_loader
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

    #这个是整个Splitlora算法的核心，其他都是参数传入、数据集载入、
    def _train(self, train_loader):
        self.network.to(self.device)
        # 保证B和分类池中的分类头是可学习的
        self.freeze_network()
        print_trainable_params(self.network)

        # Design LoRA matrix through Equation (8)
        if self.space:
            # 训练前：在残差空间中初始化 A
            self.space_init_drm(train_loader)
        elif self.use_reatain:
            self.init_drm_improvedv1(train_loader)
        else:
            self.init_drm(train_loader)


        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        # 创建优化器和学习策略。
        if self.args['optimizer'] == 'Muon':
            optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw = self.build_optimizer(self.network.parameters())
        else :
            optimizer, scheduler = self.build_optimizer(self.network.parameters())

        # check_params_consistency(self.network, optimizer)
        
        if self.atl:
            print("using ATL loss !")
            self._build_protos()

        # 从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A用U初试化(冻结)。
        if self.args['optimizer'] == 'Muon':
            self._train_function_muon(train_loader, optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw)
        else:
            self._train_function(train_loader, optimizer, scheduler)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # Preserve the information about the gradient of the t-th task through Splitlora
        # with torch.no_grad():
        #     if self.use_reatain:
        #         self.psv_info_improvedv1(train_loader)
        #     else:
        #         self.psv_info(train_loader)  #
        with torch.no_grad():
            if self.space:
                # 训练后：投影减法 + PCA 提纯，核心空间扩展，残差空间缩小
                self.space_psv_info(train_loader)
            else:
                if self.use_reatain:
                    self.psv_info_improvedv1(train_loader)
                else:
                    self.psv_info(train_loader)
        return
    def build_optimizer(self, parameters):
        if isinstance(parameters, list) and isinstance(parameters[0], dict):
            filtered_groups = []
            for group in parameters:
                params = [p for p in group['params'] if p.requires_grad]
                if len(params) > 0:
                    new_group = group.copy()
                    new_group['params'] = params
                    filtered_groups.append(new_group)

            if len(filtered_groups) == 0:
                raise ValueError("No trainable parameters found!")
            trainable_params = filtered_groups
        else:
            trainable_params = [p for p in parameters if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError("No trainable parameters found!")
        # optimizer
        import torch.optim as optim

        if self.method == "splitlorav1":
            if self.args['optimizer'] == 'adamw':
                optimizer = optim.AdamW([
        {'params': trainable_params[-2:], 'lr': self.lrate_head, 'weight_decay': self.weight_decay},
        {'params': trainable_params[:-2], 'lr': self.lrate,  'weight_decay': self.weight_decay}
                ])
            
            elif self.args['optimizer'] == 'Muon':
                print("Using Moun and adamw optimizer！")
                # 从主体参数中分离出二维权重矩阵
                main_params = trainable_params[:-2]
                head_params = trainable_params[-2:]
                
                muon_params = [p for p in main_params if p.ndim == 2]      # 线性层权重
                adamw_params = [p for p in main_params if p.ndim != 2]     # bias、LayerNorm等
                
                # 创建两个优化器
                optimizer_muon = optim.Muon(muon_params, lr=self.lrate, momentum=0.95)
                optimizer_adamw = optim.AdamW([
                    # {'params': adamw_params, 'lr': self.lrate, 'weight_decay': self.weight_decay},
                    {'params': head_params, 'lr': self.lrate_head, 'weight_decay': self.weight_decay}
                ])
                
        if self.args['optimizer'] == 'Muon':
            # scheduler
            if self.scheduler == 'constant':
                scheduler = None
            elif self.scheduler == 'cosine':
                scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=self.epochs, eta_min=0)
                scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=self.epochs, eta_min=0)

            elif self.scheduler == 'steplr':
                # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestone, gamma=self.lrate_decay)
                scheduler_muon = optim.lr_scheduler.MultiStepLR(optimizer_muon, milestones=self.milestone, gamma=self.lrate_decay)
                scheduler_adamw = optim.lr_scheduler.MultiStepLR(optimizer_adamw, milestones=self.milestone, gamma=self.lrate_decay) 
            else:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

            return optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw
        else:
            # scheduler
            if self.scheduler == 'constant':
                scheduler = None
            elif self.scheduler == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
            elif self.scheduler == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestone, gamma=self.lrate_decay)
            else:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

            return optimizer, scheduler

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                # mask = (targets >= self.known_classes).nonzero().view(-1)
                # inputs = torch.index_select(inputs, 0, mask)
                # targets = torch.index_select(targets, 0, mask)-self.known_classes
                
                # logits = self.network(inputs)['logits']
                # loss = F.cross_entropy(logits, targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                ret = self.network(inputs)
                logits = ret['logits']


                loss = F.cross_entropy(logits, targets)
                if self.atl:
                    #这里应该加上ATL loss函数进行下一步计算
                    criterion = AugmentedTripletLoss(margin=self.margin_inter)
                    features = ret['features']
                    feature = features / features.norm(dim=-1, keepdim=True)
                    ATL = criterion(feature, labels, self._protos, self.device)
                    loss += self.lambada * ATL


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # scheduler_muon.step()
            # scheduler_adamw.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def _train_function_muon(self, train_loader, optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

        
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                # mask = (targets >= self.known_classes).nonzero().view(-1)
                # inputs = torch.index_select(inputs, 0, mask)
                # targets = torch.index_select(targets, 0, mask)-self.known_classes
                
                # logits = self.network(inputs)['logits']
                # loss = F.cross_entropy(logits, targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                ret = self.network(inputs)
                logits = ret['logits']


                loss = F.cross_entropy(logits, targets)
                if self.atl:
                    #这里应该加上ATL loss函数进行下一步计算
                    criterion = AugmentedTripletLoss(margin=self.margin_inter)
                    features = ret['features']
                    feature = features / features.norm(dim=-1, keepdim=True)
                    ATL = criterion(feature, labels, self._protos, self.device)
                    loss += self.lambada * ATL


                # optimizer.zero_grad()                 optimizer_muon, optimizer_adamw
                optimizer_muon.zero_grad()
                optimizer_adamw.zero_grad()
                loss.backward()
                # optimizer.step()
                optimizer_muon.step()
                optimizer_adamw.step()  #返回的是loss


                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()
            scheduler_muon.step()
            scheduler_adamw.step()

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
    
    def init_drm_improvedv1(self, train_loader):
        """初始化 A 矩阵，支持 remove 和 retain 两种策略"""
        # 收集当前任务的激活
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)
            if self.debug:
                break
        
        if self.cur_task == 0:
            #对于Splitlora，他的第一个任务的初始化为lora的原始方式。
            print("Using Original LoRA Initialization for Task 0")
            return
        else:
            # 后续任务：根据策略初始化
            print(f"Using SplitLoRA Initialization for Task {self.cur_task}")
            layer_idx = 0
            
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    cur_matrix = module.cur_matrix
                    cur_matrix = cur_matrix.to(self.device)
                    
                    if cur_matrix.shape[0] == 0:
                        layer_idx += 1
                        continue
                    
                    # 根据策略应用不同的投影
                    if layer_idx < len(self.project_type):
                        strategy = self.project_type[layer_idx]
                        
                        if strategy == 'remove':
                            # remove 策略：投影到残差空间
                            if layer_idx < len(self.feature_list_remove):
                                U_main = self.feature_list_remove[layer_idx].to(self.device)
                                # H' = H - U_main @ U_main^T @ H
                                if U_main.shape[1] > 0:
                                    UtH = U_main.T @ cur_matrix
                                    cur_matrix = cur_matrix - U_main @ UtH
                                    
                        elif strategy == 'retain':
                            # retain 策略：投影到主成分空间
                            if layer_idx < len(self.feature_list_retain):
                                U_main = self.feature_list_retain[layer_idx].to(self.device)
                                # H' = U_main @ U_main^T @ H
                                if U_main.shape[1] > 0:
                                    UtH = U_main.T @ cur_matrix
                                    cur_matrix = U_main @ UtH
                    
                    # 对投影后的矩阵进行 SVD 分解，初始化 A
                    if cur_matrix.shape[0] > 0 and cur_matrix.shape[1] > 0:
                        try:
                            cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                            if cU.shape[1] >= self.rank:
                                # 使用投影后的主方向初始化 A
                                init_values = cU[:, :self.rank].T / math.sqrt(3)
                                module.lora_A_k[self.cur_task].weight.data.copy_(init_values)
                                module.lora_A_v[self.cur_task].weight.data.copy_(init_values)
                        except Exception as e:
                            print(f"Warning: SVD failed for layer {layer_idx}: {e}")
                            # 降级到随机初始化
                            module.lora_A_k[self.cur_task].weight.data.normal_(0, 0.01)
                            module.lora_A_v[self.cur_task].weight.data.normal_(0, 0.01)
                    
                    # 清理
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
                    layer_idx += 1



    def psv_info_improvedv1(self, train_loader):
        """保存投影矩阵，支持 remove 和 retain 两种策略"""
        # 收集激活矩阵
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)
            
        # 收集所有 Attention_LoRA 模块的 cur_matrix
        mat_list = []
        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
                
        # 为每一层决定使用 remove 还是 retain 策略
        num_layers = len(mat_list)
        
        for i in range(num_layers):
            activation = mat_list[i]
            U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            
            # 计算能量分布
            sval_total = (S).sum()
            sval_ratio = (S) / sval_total
            cumulative_energy = torch.cumsum(sval_ratio, dim=-1)
            cumulative_energy = cumulative_energy.to(self.device)
            
            # 动态阈值（可以根据需要调整）
            # 方法1：基于层索引（浅层 retain，深层 remove）
            use_retain = (i < num_layers // 2)  # 前半层用 retain，后半层用 remove
            
            # 方法2：基于能量分布自动判断（类似 DualGPM）
            # threshold = 0.8  # 能量阈值
            # r = torch.sum(cumulative_energy < threshold).item()
            # use_retain = (r >= activation.shape[0] // 2)
            
            if use_retain:
                # retain 策略：保留主成分空间
                # 选择保留前 k 个主成分（覆盖主要能量）
                k = min(self.rank * 2, activation.shape[0] // 2)  # 保留更多成分
                projection_basis = U[:, :k]  # 主成分空间
                
                # 保存投影矩阵的基（用于后续投影）
                if i >= len(self.feature_list_retain):
                    self.feature_list_retain.append(projection_basis)
                else:
                    self.feature_list_retain[i] = projection_basis
                
                # 记录策略类型
                if i >= len(self.project_type):
                    self.project_type.append('retain')
                else:
                    self.project_type[i] = 'retain'
                    
                # 为了兼容性，也保存到 feature_list（原版 SplitLoRA 使用）
                if i >= len(self.feature_list):
                    self.feature_list.append(projection_basis)
                else:
                    self.feature_list[i] = projection_basis
                    
                print(f"Layer {i}: Using RETAIN strategy, kept {k} principal components")
                
            else:
                # remove 策略：保留残差空间（原版 SplitLoRA 的行为）
                # 计算 r 值（与原版相同）
                k = torch.arange(1, self.in_dim + 1).to(self.device) - 1
                k = (self.in_dim - k.float()) / self.in_dim
                sval_ratio = sval_ratio.to(self.device)
                k = k.to(self.device)
                result = (self.cur_task + 1) * (1 - cumulative_energy) - self.alpha * k
                r = torch.argmin(result).item()
                
                # 保存残差空间
                residual_basis = U[:, max(r, 1) + 1:]
                
                if i >= len(self.feature_list_remove):
                    self.feature_list_remove.append(residual_basis)
                else:
                    self.feature_list_remove[i] = residual_basis
                    
                # 记录策略类型
                if i >= len(self.project_type):
                    self.project_type.append('remove')
                else:
                    self.project_type[i] = 'remove'
                    
                # 为了兼容性，也保存到 feature_list
                if i >= len(self.feature_list):
                    self.feature_list.append(residual_basis)
                else:
                    self.feature_list[i] = residual_basis
                    
                print(f"Layer {i}: Using REMOVE strategy (residual space), r={r}")

    # ===================== SPACE 核心函数 =====================
    def space_psv_info(self, train_loader):
        """
        SPACE 风格的任务结束处理：投影减法 + PCA 提纯 + 核心空间扩展
        这一步在任务训练完成后执行，对应论文 Algorithm 3
        """
        # 1. 收集当前任务的激活矩阵
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)
            if self.debug:
                break
        
        mat_list = []
        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
        
        layer_idx = 0
        for activation in mat_list:
            activation = activation.to(self.device)  # [n_samples, in_dim]
            
            # 2. 如果是第一个任务：纯 PCA 初始化核心空间
            if self.cur_task == 0:
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                
                # 按方差阈值保留主成分
                total_var = (S ** 2).sum()
                cumsum_var = torch.cumsum(S ** 2, dim=0) / total_var
                k = torch.searchsorted(cumsum_var, self.space_variance_threshold).item() + 1
                k = min(k, activation.shape[0])  # 不超过样本数
                
                U_core = U[:, :k]  # 核心空间基
                self.core_space_list.append(U_core)
                
                # 残差空间 = 全空间减去核心空间（这里简化为剩余的奇异向量）
                # 实际残差维度 = in_dim - k
                residual_dim = self.in_dim - k
                self.residual_dim_list.append(residual_dim)
                
                print(f"  Layer {layer_idx}: Task 0 -> Core dim={k}, Residual dim={residual_dim}")
                
            else:
                # 3. 后续任务：投影减法 + PCA 提纯
                U_core_old = self.core_space_list[layer_idx].to(self.device)  # [n_samples, r_core]
                
                # === 步骤 1: 投影减法 (Projection-Subtraction) ===
                # 将当前激活投影到核心空间上
                # A_proj = U_core @ (U_core^T @ A)
                UtA = U_core_old.T @ activation  # [r_core, in_dim]
                A_proj = U_core_old @ UtA          # [n_samples, in_dim]
                
                # 残差部分 = 原始激活 - 能被核心解释的部分
                A_residual = activation - A_proj   # [n_samples, in_dim]
                
                # === 步骤 2: 对残差做 PCA，找出真正的新知识 ===
                U_res, S_res, Vh_res = torch.linalg.svd(A_residual, full_matrices=False)
                
                # 按方差阈值保留残差中的主成分
                if S_res.sum() > 1e-10:
                    total_var_res = (S_res ** 2).sum()
                    cumsum_var_res = torch.cumsum(S_res ** 2, dim=0) / total_var_res
                    k_new = torch.searchsorted(cumsum_var_res, self.space_variance_threshold).item() + 1
                else:
                    k_new = 0  # 残差中没有有效信息
                
                k_new = min(k_new, self.residual_dim_list[layer_idx])  # 不能超过可用残差维度
                
                if k_new > 0:
                    U_new = U_res[:, :k_new]  # 新增的核心基
                else:
                    U_new = torch.zeros((activation.shape[0], 0), device=self.device)
                
                # === 步骤 3: 扩展核心空间（精华毕业进入百科全书）===
                if U_new.shape[1] > 0:
                    # 将新基与旧核心基合并，并正交化
                    U_core_new = torch.cat([U_core_old, U_new], dim=1)
                    Q_core, R_core = torch.linalg.qr(U_core_new, mode='reduced')
                    
                    # 过滤接近零的列
                    col_norms = torch.norm(Q_core, dim=0)
                    valid_cols = col_norms > 1e-6
                    U_core_new = Q_core[:, valid_cols]
                else:
                    U_core_new = U_core_old
                
                # 更新核心空间
                self.core_space_list[layer_idx] = U_core_new
                
                # 更新残差维度（草稿纸变小了）
                new_residual_dim = self.in_dim - U_core_new.shape[1]
                self.residual_dim_list[layer_idx] = max(0, new_residual_dim)
                
                # === 步骤 4: 保存当前任务可用的残差空间（用于下一个任务的初始化）===
                # 这就是越来越小的 U_residual
                U_residual = U_res[:, k_new:] if k_new < U_res.shape[1] else torch.zeros((activation.shape[0], 0), device=self.device)
                
                # 存储到 feature_list，供下一个任务使用
                if layer_idx >= len(self.feature_list):
                    self.feature_list.append(U_residual)
                else:
                    self.feature_list[layer_idx] = U_residual
                
                print(f"  Layer {layer_idx}: Task {self.cur_task} -> "
                      f"Old Core={U_core_old.shape[1]}, New Added={k_new}, "
                      f"Total Core={U_core_new.shape[1]}, Residual={U_residual.shape[1]}")
            
            layer_idx += 1
    
    def space_init_drm(self, train_loader):
        """
        SPACE 风格的任务开始初始化：在上一轮留下的残差空间中学习
        核心空间冻结，残差空间可训练
        """
        # 收集激活（用于确定当前残差空间的分布）
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True)
            if self.debug:
                break
        
        if self.cur_task == 0:
            print("SPACE: Task 0 - Original LoRA initialization")
            return
        
        print(f"SPACE: Task {self.cur_task} - Learning in residual space")
        
        layer_idx = 0
        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                cur_matrix = module.cur_matrix.to(self.device)
                
                # 获取当前层的信息
                U_core = self.core_space_list[layer_idx].to(self.device)  # 核心空间（只读）
                residual_dim_available = self.residual_dim_list[layer_idx]
                
                if residual_dim_available <= 0:
                    print(f"  Layer {layer_idx}: WARNING - No residual space left!")
                    # 没有残差空间了，只能用随机初始化（理论上应该扩展网络）
                    module.lora_A_k[self.cur_task].weight.data.normal_(0, 0.01)
                    module.lora_A_v[self.cur_task].weight.data.normal_(0, 0.01)
                    layer_idx += 1
                    continue
                
                # === 关键步骤：在残差空间中生成初始化矩阵 ===
                # 残差空间 = 全空间正交补于核心空间
                # 我们用一个随机矩阵投影到残差空间中来生成 A
                
                # 方法1：如果 feature_list 中保存了上一轮的残差基 U_residual
                if layer_idx < len(self.feature_list) and self.feature_list[layer_idx].shape[1] > 0:
                    U_residual = self.feature_list[layer_idx].to(self.device)  # [n_samples, r_residual]
                    
                    # 生成随机系数 R，尺寸适配 U_residual
                    # R 的维度: [r_residual, rank]
                    random_coeffs = torch.randn(U_residual.shape[1], self.rank).to(self.device)
                    
                    # A = U_residual @ R
                    initialized_matrix = U_residual @ random_coeffs
                    
                else:
                    # 方法2：如果没有保存残差基，直接从激活中提取
                    # 先将激活投影到核心空间的正交补上
                    if cur_matrix.shape[0] > 0:
                        # 计算核心空间的正交补
                        # 使用 Householder 变换或简单 QR
                        I = torch.eye(cur_matrix.shape[0], device=self.device)
                        proj_core = U_core @ U_core.T
                        proj_residual = I - proj_core
                        
                        # 在残差空间中生成随机方向
                        random_dirs = torch.randn(cur_matrix.shape[0], self.rank).to(self.device)
                        initialized_matrix = proj_residual @ random_dirs
                    else:
                        initialized_matrix = torch.randn(self.in_dim, self.rank).to(self.device)
                
                # 归一化 + 缩放（适配 LoRA 的 scale）
                scale = 1.0 / (self.cur_task + 1)
                initialized_matrix_K = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                    module.lora_A_k[self.cur_task].weight.data.norm() * scale
                initialized_matrix_V = initialized_matrix / (initialized_matrix.norm() + 1e-8) * \
                                    module.lora_A_v[self.cur_task].weight.data.norm() * scale
                
                # 赋值给 lora_A
                module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)
                module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)
                
                print(f"  Layer {layer_idx}: A initialized in residual space "
                      f"(core dim={U_core.shape[1]}, residual available={residual_dim_available})")
                
                # 清理
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
                layer_idx += 1