import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils.losses import AugmentedTripletLoss

from models.net_splitlora_R_only_use_shared_adapter import Net
from models.vit_splitlora_R_only_use_shared_adapter import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency
from utils.toolkit import print_args, format_elapsed_time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

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
        self._protos = []
        self.rank = args['rank']
        self.atl = "true"
        # self.random_coeffs_K = nn.ParameterDict()
        # self.random_coeffs_V = nn.ParameterDict()
        self.margin_inter = 1.0
        # self.n_cur_matrix = 0
        # self.cur_matrix = torch.zeros(dim ,dim)
        # self.mds = True
        self.pca = False
        self.mds = False
        # self.pca = True
        self.ema_decay = args['ema_decay']
        self.ema_decay2 = args['ema_decay2']
        self.ema_type = args["ema_type"]
        # 在 __init__ 中添加
        self.grad_ema_decay = args.get('grad_ema_decay', 0.99)    # 梯度EMA衰减系数
        self.grad_direction_eta = args.get('grad_direction_eta', 0.5)  # α调整幅度控制


        # ========== adaptEMAv2 需要的全局统计量 ==========
        self.global_loss_sum = 0.0          # 所有任务所有batch的loss累加
        self.global_loss_count = 0          # 所有任务所有batch的样本数
        self.global_param_sum = {}          # 所有任务所有步的参数累加（逐参数）
        self.global_param_count = 0         # 所有任务所有步的参数更新次数
        
        # 可选的平滑版本（跨任务EMA）
        self.global_loss_ema = None         # 跨任务损失EMA
        self.global_loss_ema_decay = 0.95   # EMA衰减系数


        #         # ========== EMA衰减调度参数 ==========
        # self.ema_decay_start = args.get('ema_decay_start', 0.9)      # 起始decay
        # self.ema_decay_end = args.get('ema_decay_end', 0.9999)      # 终止decay

        self.ema_decay_start = args.get('ema_decay_start', 0.9999)      # 起始decay
        self.ema_decay_end = args.get('ema_decay_end', 0.9)      # 终止decay
        self.ema_decay_schedule = args.get('ema_decay_schedule', 'cosine')  # 'linear' 或 'cosine'
        


        ######################
        self.off_merge_knob = args.get('off_merge_knob', "True") 

    def init_drm(self, train_loader):

        """initialzation of dimensionality reduction matrix A"""
        # 我不懂这里为啥又要再跑一遍训练集了，之前不是已经跑过一次了么？（在train函数中）
        # 并且为啥还执行了一次SVD分解。
        # for i, (_, inputs, targets) in enumerate(train_loader):
        #     inputs, targets = inputs.to(self.device), targets.to(self.device)
        #     self.network(inputs, get_cur_feat=True)  # gather the features in module.cur_matrix
        #     if self.debug: break
        
        if self.cur_task == 0:
                   
            return
        else:
            # 后续任务：使用上一个任务保存到的self.feature_list
            print("Using SplitLora Initialization for Task t(t>=1)")
            i = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    if self.mds:
                        # ########################   对initialized_matrix_K和initialized_matrix_V 进行重排列，避免特征冲突   #######################
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
                    elif self.pca:
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
                    
                    else:
                        random_coeffs_K = torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        random_coeffs_V = torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        # 第2行：用特征矩阵乘以随机系数
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        initialized_matrix_K = self.feature_list[i] @ random_coeffs_K
                        initialized_matrix_V = self.feature_list[i] @ random_coeffs_V

                        initialized_matrix_K = initialized_matrix_K / initialized_matrix_K.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix_V / initialized_matrix_V.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        # 将UR矩阵的结果赋值给A（应该指的是 Y=X(W+AB中的A，即A<-UR)
                    
                    module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)   
                    module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)   


                    i += 1


        ########################计算权重的svd分解###########################
    def incremental_train(self, data_manager):
        self.data_manager = data_manager  #用于下面创建原型，用于ATL loss计算。
        self.build_train_loader(data_manager)
        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader)

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
    def _build_protos(self):
        self._protos = []
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
                
    def psv_info(self, train_loader):
        # 想要了解这个函数的功能，需要慢慢一步一步往下看。
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #这里的forward函数中get_cur_feat=True在Attention_LoRA模块中注
            # 册hook来获取当前任务的激活矩阵（即cur_matrix）。
            self.network(inputs, get_cur_feat=True, get_weight=False)
            
        #这里的mat_list是一个列表，里面存储了每个Attention_LoRA模块的cur_matrix
        # （即当前任务的激活矩阵）。这个矩阵在前面训练过程中被更新了。
        mat_list = []
        for module in self.network.modules():
            # 判断：module 这个对象是不是 Attention_LoRA 类型的实例？
            # 返回值：True 或 False
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                n_cur_matrix = module.n_cur_matrix 
                print("self.n_cur_matrix:", n_cur_matrix)
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
        get_weight = False
        if get_weight:
            ########################计算权重的svd分解###########################
            for module in self.network.modules():
                # 判断：module 这个对象是不是 Attention_LoRA 类型的实例？
                # 返回值：True 或 False
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    self.weight_k = module.weight_k 
                    self.weight_v = module.weight_v

                    module.weight_k.zero_()
                    module.weight_v.zero_()


            ########################计算权重的svd分解###########################
        

    def save_task(self, cur_task):
        """
        保存当前任务训练完成后的模型权重（仅保存共享适配器 + 分类器）
        用于后续的离线合并（WSM风格）
        """
        import os
        
        # 创建保存目录
        save_dir = os.path.join(self.args.get('logdir', './checkpoints'), 'task_checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        # 只保存共享适配器和分类器参数（减少存储开销）
        checkpoint = {}
        for name, param in self.network.named_parameters():
            # 只保存共享适配器和分类器，不保存 backbone 和 lora_A/lora_B
            if 'Lora_shared' in name or 'classifier_pool' in name:
                checkpoint[name] = param.data.clone().detach().cpu()
        
        # 保存额外信息
        checkpoint['task_id'] = cur_task
        checkpoint['known_classes'] = self.known_classes
        checkpoint['total_classes'] = self.total_classes
        
        # 如果有EMA参数，也一并保存（用于对比实验）
        if hasattr(self, 'ema_params') and self.ema_params is not None:
            checkpoint['ema_params'] = {}
            for name, param in self.ema_params.items():
                checkpoint['ema_params'][name] = param.clone().detach().cpu()
        
        # 保存到文件
        save_path = os.path.join(save_dir, f'task_{cur_task}_checkpoint.pt')
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved for task {cur_task} at {save_path}")
        
        # 更新checkpoint列表（用于后续合并）
        if not hasattr(self, 'checkpoint_list'):
            self.checkpoint_list = []
        self.checkpoint_list.append(save_path)
        

    # def load_all_task_weight_merge(self, cur_task):
    #     pass



    #这个是整个Splitlora算法的核心，其他都是参数传入、数据集载入、
    def _train(self, train_loader):
        self.network.to(self.device)
        # 保证B和分类池中的分类头是可学习的
        self.freeze_network()
        print_trainable_params(self.network)

        # 仅仅使用共享适配器就不需要Splitlora的初始化步骤
        # # Design LoRA matrix through Equation (8)
        # with torch.no_grad():
        #     self.init_drm(train_loader)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        # 创建优化器和学习策略。
        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)


        # 从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A用U初试化(冻结)。
        self._train_function(train_loader, optimizer, scheduler)

        # 这里应该是每次训练一个任务之后进行的权重保存策略（只保存科学系的参数权重）。
        self.save_task(self.cur_task)

        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # 仅仅使用共享适配器就不需要Splitlora的保存前向初试化的结果
        # # Preserve the information about the gradient of the t-th task through Splitlora
        # with torch.no_grad():
        #     self.psv_info(train_loader)
        # return

    def Weight_pri(weight, alpha=0.99, device=None):
        original_device = weight.device
        original_dtype = weight.dtype
        
        if device is not None:
            compute_device = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            weight = weight.to(device=compute_device, dtype=torch.float32)
        else:
            weight = weight.to(dtype=torch.float32)

        # 对权重矩阵做 SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        total_sum = S.sum()
        cumsum = S.cumsum(dim=0)
        r = (cumsum / total_sum >= alpha).nonzero(as_tuple=True)[0][0].item() + 1

        # print(r, end="")

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_r = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))
        
        return weight_r.to(device=original_device, dtype=original_dtype)
    def cosine_similarity(self, W_prev, delta):
        """计算余弦相似度，范围 [-1, 1]"""
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        return cos_sim
    
    def cosine_similarity_loss(self, W_prev, delta):
        # 展平
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        # 计算余弦相似度
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        # 我们希望余弦相似度接近0（正交），所以 loss = |cos_sim|
        return torch.abs(cos_sim)
    
    def _train_function(self, train_loader, optimizer, scheduler):

        
        #初始化EMA等
        if self.cur_task >= 1:
            ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
            if self.ema_type == "adaptEMA":
                # use_adaptive_ema = getattr(self, 'use_adaptive_ema', False)  # 默认False，使用固定系数                
                ema_decay = None  # 自适应计算，不固定
                # 用于存储历史EMA参数和当前参数，计算损失
                ema_params = {}
                prev_ema_params = {}  # 存储上一步的EMA参数用于计算梯度/损失                
                # 初始化EMA参数副本
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        print(f"Adaptive EMA tracking: {name}")        
            
            elif self.ema_type == "grad_direction":  # <--- 新增：梯度方向一致性EMA
                ema_decay = None
                ema_params = {}           # EMA参数
                prev_ema_params = {}      # 上一步EMA参数（用于计算delta）
                ema_grad_params = {}      # 历史梯度的EMA（用于方向一致性）
                grad_ema_decay = 0.99     # 梯度EMA的衰减系数
                
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        ema_grad_params[name] = torch.zeros_like(param.data)
                        print(f"Grad-Direction EMA tracking: {name}")
            elif self.ema_type == "EMA_scheduled":  # 新增一个类型
                    ema_params = {}
                    for name, param in self.network.named_parameters():
                        if any(key in name for key in ema_keys):
                            ema_params[name] = param.data.clone().detach()
                            print(f"Scheduled EMA tracking: {name}")
    
            elif self.ema_type == "AdEMAMix": #
                # AdEMAMix 参数跟踪（慢速和快速EMA）
                # 快速EMA（类似标准EMA）
                ema_params = {} # 这个代表快速响应部分
                # 慢速EMA（AdEMAMix的慢速平均）
                ema_slow_params = {}
                # AdEMAMix超参数
                beta1 = 0.9      # 快速EMA衰减率
                beta3 = 0.9999   # 慢速EMA衰减率
                alpha_adamemix = 5.0  # 慢速EMA的混合权重
                
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach() # 这个代表快速响应部分
                        ema_slow_params[name] = param.data.clone().detach()
                        print(f"AdEMAMix tracking (fast & slow): {name}")
                
                # 用于计算beta3_t的动态调整（可选）
                T_alpha_beta3 = self.epochs * len(train_loader)  # 总迭代次数
            elif self.ema_type == "adaptEMAv2":  # <--- 新增：为 v2 单独初始化
                # adaptEMAv2 使用全局统计量（在 __init__ 中已初始化）
                # 但仍然需要 ema_params 和 prev_ema_params
                ema_params = {}
                prev_ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        print(f"Adaptive EMA v2 tracking: {name}")
            #只需要在第二个任务开始定义这个全局变量
            elif self.ema_type == "Cum_avg":
                self.ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
                if self.cur_task==1:
                    self.ema_params = {}
                    self.cnt = 1
                    # 初始化EMA参数副本
                    for name, param in self.network.named_parameters():
                        if any(key in name for key in self.ema_keys):
                            self.ema_params[name] = param.data.clone().detach()

            else:
                ema_decay = self.ema_decay
                ema_params = {}
                # 初始化EMA参数副本
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        print(f"EMA tracking: {name}")
        # training phase
        prog_bar = tqdm(range(self.epochs))
        global_step = 0  # 全局迭代计数器，用于AdEMAMix的beta3调度
        
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0
            print("\n")
            print(f"cur_task {self.cur_task}, Epoch {epoch}: "
                f"Head LR={optimizer.param_groups[0]['lr']:.6f}, "
                f"Backbone LR={optimizer.param_groups[1]['lr']:.6f}")
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes
                ret_minus = self.network(inputs)
                logits = (ret_minus['logits'])
                loss_minus = F.cross_entropy(logits, targets)
                
                loss = loss_minus

                optimizer.zero_grad()
                loss.backward()
                
                # AdEMAMix 更新需要在 optimizer.step() 之后手动更新EMA
                optimizer.step()
                global_step += 1

                # 选择不同的EMA series更新
                if self.cur_task >= 1:

                    if self.ema_type == "EMA2":
                        # 标准EMA更新
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = 0.5 * (ema_decay * ema_params[name] + (1 - ema_decay) * param.data \
                                            + ema_decay2 * ema_params[name] + (1 - ema_decay2) * param.data)

                    elif self.ema_type == "EMA_scheduled":
                        # ---------- 计算当前步的 decay ----------
                        global_step = epoch * len(train_loader) + i

                        # 计算总的 步数
                        total_steps = self.epochs * len(train_loader)

                        progress = global_step / total_steps  # 0 → 1
                        
                        if self.ema_decay_schedule == "linear":
                            # 线性衰减
                            ema_decay_current = self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * progress
                        elif self.ema_decay_schedule == "cosine":
                            # 余弦衰减
                            cos_val = (1 - math.cos(math.pi * progress)) / 2
                            ema_decay_current = self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * cos_val
                        else:
                            # 默认使用固定值
                            ema_decay_current = self.ema_decay
                        
                        # ---------- 更新EMA ----------
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = ema_decay_current * ema_params[name] + (1 - ema_decay_current) * param.data
                    elif self.ema_type == "grad_direction":  # <--- 新增：梯度方向一致性EMA
                        for name, param in self.network.named_parameters():
                            if name not in ema_params:
                                continue
                            if param.grad is None:
                                continue
                            
                            theta_t = param.data
                            bar_theta_prev = prev_ema_params[name]
                            grad_flat = param.grad.view(-1)
                            ema_grad_flat = ema_grad_params[name].view(-1)
                            
                            # 计算梯度范数
                            grad_norm = torch.norm(grad_flat, p=2)
                            ema_grad_norm = torch.norm(ema_grad_flat, p=2)
                            
                            # ---------- 核心：梯度方向一致性（余弦相似度） ----------
                            if grad_norm > 1e-8 and ema_grad_norm > 1e-8:
                                cos_sim = torch.dot(grad_flat, ema_grad_flat) / (grad_norm * ema_grad_norm + 1e-8)
                            else:
                                cos_sim = torch.tensor(0.0, device=param.device)
                            
                            # 更新历史梯度EMA（用于下一步的方向比较）
                            grad_ema_decay = getattr(self, 'grad_ema_decay', 0.99)
                            ema_grad_params[name] = grad_ema_decay * ema_grad_params[name] + (1 - grad_ema_decay) * param.grad
                            
                            # ---------- 将cos_sim映射到alpha ----------
                            # 方向一致（cos_sim → 1）→ α大 → 稳定性增强
                            # 方向冲突（cos_sim → -1）→ α小 → 可塑性增强
                            # 添加eta控制调整幅度，防止α变化过激
                            eta = getattr(self, 'grad_direction_eta', 0.5)
                            alpha_star = 0.5 + eta * 0.5 * cos_sim
                            
                            # 裁剪到合理范围
                            alpha_star = torch.clamp(alpha_star, 0.99, 0.99999)
                            
                            # 可选：打印调试信息
                            if i % 100 == 0 and epoch == 0:
                                print(f"  {name}: cos_sim={cos_sim.item():.4f}, alpha_star={alpha_star.item():.8f}")
                            
                            # ---------- 参数EMA更新 ----------
                            ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                            prev_ema_params[name] = ema_params[name].clone().detach()
                    elif self.ema_type == "adaptEMA":
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                theta_t = param.data
                                bar_theta_prev = prev_ema_params[name]
                                delta = bar_theta_prev - theta_t
                                delta_norm_sq = torch.norm(delta, p=2).pow(2)

                                # 直接使用当前 loss 作为 L(theta_t)
                                loss_theta_t = loss.item()

                                # 用当前梯度近似计算 L(bar_theta_prev) 的近似值
                                # 使用泰勒展开: L(bar_theta_prev) ≈ L(theta_t) + <∇L(theta_t), bar_theta_prev - theta_t>
                                if param.grad is not None and delta_norm_sq > 1e-8:
                                    grad_flat = param.grad.view(-1)
                                    delta_flat = delta.view(-1)
                                    loss_bar_prev_approx = loss.item() + torch.dot(grad_flat, delta_flat).item()
                                else:
                                    loss_bar_prev_approx = loss.item()

                                # 计算 alpha*
                                # L_avg 近似：由于是单batch，直接用 loss 作为平均损失
                                if param.grad is not None and delta_norm_sq > 1e-8:
                                    # L = torch.norm(param.grad, p=2) / (torch.sqrt(delta_norm_sq) + 1e-8)
                                    # L = torch.clamp(L, min=1e-6, max=1e3)
                                    L = 15.0
                                    
                                    # 直接使用公式：alpha = 0.5 + (loss_bar_prev_approx - loss_theta_t) / (L * delta_norm_sq + 1e-8)
                                    alpha_star = 0.5 + (loss_bar_prev_approx - loss_theta_t) / (L * delta_norm_sq + 1e-8)
                                else:
                                    alpha_star = torch.tensor(0.5, device=param.device)

                                alpha_star = torch.clamp(alpha_star, 0.9, 0.99999)
                                
                                # EMA 更新
                                ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                                prev_ema_params[name] = ema_params[name].clone().detach() 

                    elif self.ema_type == "adaptEMAv2":
                        # ---------- 更新全局损失累计平均（所有任务共享） ----------
                        batch_size = inputs.shape[0]
                        self.global_loss_sum += loss.item() * batch_size
                        self.global_loss_count += batch_size
                        loss_global_avg = self.global_loss_sum / self.global_loss_count
                        
                        # 更新跨任务损失EMA（平滑版本）
                        if self.global_loss_ema is None:
                            self.global_loss_ema = loss_global_avg
                        else:
                            self.global_loss_ema = (
                                self.global_loss_ema_decay * self.global_loss_ema +
                                (1 - self.global_loss_ema_decay) * loss_global_avg
                            )
                        
                        # ---------- 更新全局参数累计平均（逐参数，所有任务共享） ----------
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                theta_t = param.data
                                if name not in self.global_param_sum:
                                    self.global_param_sum[name] = torch.zeros_like(theta_t)
                                
                                # 累积平均公式
                                self.global_param_sum[name] = (
                                    self.global_param_sum[name] * self.global_param_count + theta_t
                                ) / (self.global_param_count + 1)
                        self.global_param_count += 1
                        
                        # ---------- 计算 alpha_star ----------
                        for name, param in self.network.named_parameters():
                            if name not in ema_params:
                                continue
                            
                            theta_t = param.data
                            bar_theta_prev = prev_ema_params[name]
                            delta = bar_theta_prev - theta_t
                            delta_norm_sq = torch.norm(delta, p=2).pow(2)
                            
                            if param.grad is not None and delta_norm_sq > 1e-8:
                                grad_norm = torch.norm(param.grad, p=2)
                                delta_norm = torch.sqrt(delta_norm_sq)
                                
                                # 使用全局损失EMA作为稳定的损失基准
                                loss_theta_t = self.global_loss_ema if self.global_loss_ema is not None else loss_global_avg
                                
                                # 使用全局损失EMA与当前batch的差异
                                loss_diff = loss_theta_t - loss.item()
                                
                                # 估计 Lipschitz 常数
                                # L = grad_norm / (delta_norm + 1e-8)
                                # L = torch.clamp(L, min=1e-6, max=1e3)
                                L = 5.0
                                
                                # 计算 alpha_star
                                if abs(loss_diff) > 1e-6:
                                    alpha_star = 0.5 + loss_diff / (L * delta_norm_sq + 1e-8)
                                else:
                                    # 备用：梯度方向一致性
                                    grad_flat = param.grad.view(-1)
                                    delta_flat = delta.view(-1)
                                    grad_delta = torch.dot(grad_flat, delta_flat).item()
                                    alpha_star = 0.5 + grad_delta / (grad_norm * delta_norm + 1e-8)
                            else:
                                alpha_star = torch.tensor(0.5, device=param.device)
                            print(f"cur_task: {self.cur_task}, epoch: {epoch}, alpha_star: {alpha_star}")
                            alpha_star = torch.clamp(alpha_star, 0.9, 0.99999)
                            print(f"cliped : cur_task: {self.cur_task}, epoch: {epoch}, alpha_star: {alpha_star}")
                            # EMA更新
                            ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                            prev_ema_params[name] = ema_params[name].clone().detach()
                    elif self.ema_type == "AdEMAMix":
                        # AdEMAMix 更新策略
                        # 使用AdEMAMix的更新公式：
                        # fast_ema = beta1 * fast_ema + (1 - beta1) * param
                        # slow_ema = beta3_t * slow_ema + (1 - beta3_t) * param
                        # 最终参数 = fast_ema + alpha_t * slow_ema
                        
                        # 动态调整 beta3_t（如果使用T_alpha_beta3）
                        if T_alpha_beta3 is not None:
                            # 使用与AdEMAMix论文相同的调度策略
                            # 注意：这里使用global_step而不是epoch
                            step = global_step
                            T = T_alpha_beta3
                            # 计算当前的 alpha_t 和 beta3_t
                            alpha_t = min(step * alpha_adamemix / T, alpha_adamemix)
                            # 使用对数插值计算beta3_t
                            # 从beta1逐渐过渡到beta3
                            progress = min(step / T, 1.0)
                            # 对数空间插值
                            log_beta1 = math.log(beta1)
                            log_beta3 = math.log(beta3)
                            log_beta3_t = (1 - progress) * log_beta3 + progress * log_beta1
                            beta3_t = math.exp(log_beta3_t)
                        else:
                            alpha_t = alpha_adamemix
                            beta3_t = beta3
                        
                        for name, param in self.network.named_parameters():
                            if name in ema_params: #ema_params代表快速反应部分
                                # 快速EMA更新
                                ema_params[name] = beta1 * ema_params[name] + (1 - beta1) * param.data
                                # 慢速EMA更新
                                ema_slow_params[name] = beta3_t * ema_slow_params[name] + (1 - beta3_t) * param.data
                                # 注意：AdEMAMix的最终参数是 fast_ema + alpha_t * slow_ema
                                # 但这里我们只存储两个EMA，稍后在训练结束后合并
                                final_alpha_t = min(global_step * alpha_adamemix / T_alpha_beta3, alpha_adamemix) if T_alpha_beta3 is not None else alpha_adamemix
                                # 混合参数
                                ema_params[name] = ema_params[name] + final_alpha_t * ema_slow_params[name]
                    elif self.ema_type == "Cum_avg":
                        # 累计平均EMA更新 (default)
                        for name, param in self.network.named_parameters():
                            if name in self.ema_params:
                                self.ema_params[name] = (self.cnt * self.ema_params[name] + param.data )/(self.cnt+1)
                                self.cnt += 1

                    else:
                        # 标准EMA更新 (default)
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = (ema_decay * ema_params[name] + (1 - ema_decay) * param.data )
                                # 这个是偏差纠正项目
                                # ema_params[name] = ema_params[name] / (1 - ema_decay)
                        # pass

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
        

        if self.cur_task >=1:

            if self.ema_type == "Cum_avg":
                #将上面EMA学习到的参数应用到shared parameters
                with torch.no_grad():
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            param.data.copy_(self.ema_params[name])            
            else:
                #将上面EMA学习到的参数应用到shared parameters
                with torch.no_grad():
                    for name, param in self.network.named_parameters():
                        if name in ema_params:
                            param.data.copy_(ema_params[name])
                # pass

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        if self.cur_task==0:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "Lora_shared",
        ]
        else:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "Lora_shared"

        ]
        for name, param in self.network.named_parameters():
            # print(name) 
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




