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

from models.net_splitlora_R import Net
from models.vit_splitlora_R import VisionTransformer, PatchEmbed, Block, Attention_LoRA
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
        self.margin_inter = 1.0
        self.pca = False
        self.mds = False
        self.ema_decay = args['ema_decay']
        self.ema_decay2 = args['ema_decay2']
        self.ema_type = args["ema_type"]
        self.grad_ema_decay = args.get('grad_ema_decay', 0.99)
        self.grad_direction_eta = args.get('grad_direction_eta', 0.5)

        self.R = args['R']

        # ========== adaptEMAv2 需要的全局统计量 ==========
        self.global_loss_sum = 0.0
        self.global_loss_count = 0
        self.global_param_sum = {}
        self.global_param_count = 0
        self.global_loss_ema = None
        self.global_loss_ema_decay = 0.95

        # ========== EMA衰减调度参数 ==========
        self.ema_decay_start = args.get('ema_decay_start', 0.9999)
        self.ema_decay_end = args.get('ema_decay_end', 0.9)
        self.ema_decay_schedule = args.get('ema_decay_schedule', 'cosine')

        # ========== grad_direction 需要的平滑状态 ==========
        self.alpha_smooth = {}
        self.alpha_smooth_decay = 0.9

        self.flag = 0

        # ============ Flat-LoRA 相关配置 ============
        # 扰动强度，建议值: 0.05 ~ 0.15
        self.flat_sigma = args.get('flat_sigma', 0.05)
        # 是否使用cosine递增策略
        self.ortho_loss_type = args.get('ortho_loss_type', False)
        self.cos_loss_type = args.get('cos_loss_type', False)

        # 当前训练步数（用于cosine递增）
        self.flat_current_step = 0
        # 总训练步数（在_train中设置）
        self.flat_total_steps = 0
        # 存储每个Linear层的filter norms，用于复现扰动（内存高效）
        self.flat_filter_norms = {}
        # 随机种子，用于复现扰动
        self.flat_seed = None
        # 存储被扰动的模块的原始B值，用于恢复
        self.flat_original_B = {}
        # ============================================
        self.ortho_loss_weight = args.get("ortho_loss_weight", 0.08)

        self.flat_cosine_anneal = args.get('flat_cosine_anneal', True)

    def _generate_flat_perturbation(self, W, module_name, sigma):
        """
        Flat-LoRA的filter-wise随机扰动生成 (论文 Eq. 7)
        
        Args:
            W: 权重矩阵 [m, n]
            module_name: 模块名称，用于缓存filter norms
            sigma: 当前扰动强度
        
        Returns:
            epsilon: 与W同形状的扰动矩阵
        """
        m, n = W.shape
        
        # 计算每个filter的L2范数 (按输出维度)
        filter_norms = torch.norm(W, dim=1)  # [m]
        
        # 缓存filter norms用于内存高效的扰动复现
        self.flat_filter_norms[module_name] = filter_norms.clone()
        
        # Flat-LoRA Eq. 7: N(0, sigma^2/n * ||W_i,:||^2)
        std = sigma / math.sqrt(n) * filter_norms.unsqueeze(1)  # [m, n]
        
        # 使用固定种子生成扰动（可复现）
        if self.flat_seed is not None:
            torch.manual_seed(self.flat_seed + hash(module_name) % 10000)
        
        epsilon = torch.randn_like(W) * std
        
        return epsilon

    def _apply_flat_perturbation_to_lora_modules(self, sigma):
        """
        对所有LoRA模块的B矩阵施加Flat-LoRA扰动
        在计算梯度前对merged权重加扰动，等效于对B矩阵加扰动
        """
        # 重置扰动存储
        self.flat_filter_norms = {}
        
        for name, module in self.network.named_modules():
            if not isinstance(module, Attention_LoRA):
                continue
            
            # 对K和V的LoRA模块分别处理
            for proj_type in ['k', 'v']:
                # 获取当前任务的B矩阵
                if proj_type == 'k':
                    B = getattr(module, f'lora_B_k_{self.cur_task}', None)
                else:
                    B = getattr(module, f'lora_B_v_{self.cur_task}', None)
                
                if B is None:
                    continue
                
                B_weight = B.weight
                m, r = B_weight.shape
                
                # 按filter-wise生成扰动
                # 对于B矩阵，按输出维度计算filter norms
                filter_norms = torch.norm(B_weight, dim=0)  # [r]
                std = sigma / math.sqrt(m) * filter_norms.unsqueeze(0)  # [m, r]
                
                # 保存当前B的原始值用于恢复
                key = f"{name}_{proj_type}"
                if key not in self.flat_original_B:
                    self.flat_original_B[key] = B_weight.clone()
                
                # 生成扰动并加到B上（用于前向传播的梯度计算）
                epsilon_B = torch.randn_like(B_weight) * std
                B_weight.add_(epsilon_B)

    def _restore_lora_modules(self):
        """恢复被扰动的LoRA参数"""
        for key, original_B in self.flat_original_B.items():
            # 解析key获取模块和投影类型
            # key格式: "Attention_LoRA_name_proj_type"
            parts = key.rsplit('_', 1)
            if len(parts) != 2:
                continue
            module_name, proj_type = parts[0], parts[1]
            
            # 找到对应的模块和B矩阵
            for name, module in self.network.named_modules():
                if name == module_name and isinstance(module, Attention_LoRA):
                    if proj_type == 'k':
                        B = getattr(module, f'lora_B_k_{self.cur_task}', None)
                    else:
                        B = getattr(module, f'lora_B_v_{self.cur_task}', None)
                    
                    if B is not None:
                        B.weight.data.copy_(original_B)
                    break
        
        self.flat_original_B = {}
        self.flat_filter_norms = {}

    def _get_flat_sigma(self, step, total_steps):
        """
        获取当前步的扰动强度
        使用cosine递增策略（论文推荐）
        """
        if not self.flat_cosine_anneal:
            return self.flat_sigma
        
        # Cosine递增: 从0逐渐增长到sigma_max
        if total_steps <= 0:
            return self.flat_sigma
        
        progress = step / total_steps
        sigma_t = self.flat_sigma * (1 + math.cos(math.pi * progress)) / 2
        return max(sigma_t, 0.001)  # 避免完全为0

    def init_drm(self, train_loader):
        """initialzation of dimensionality reduction matrix A"""
        if self.cur_task == 0:
            return
        else:
            print("Using SplitLora Initialization for Task t(t>=1)")
            i = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    if self.mds:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        features = self.feature_list[i].cpu().numpy()
                        pca = MDS(n_components=10)
                        features_reduced = pca.fit_transform(features)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    elif self.pca:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        features = self.feature_list[i].cpu().numpy()
                        pca = PCA(n_components=10)
                        features_reduced = pca.fit_transform(features)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    else:
                        random_coeffs_K = torch.randn(self.feature_list[i].shape[1], self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        random_coeffs_V = torch.randn(self.feature_list[i].shape[1], self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        initialized_matrix_K = self.feature_list[i] @ random_coeffs_K
                        initialized_matrix_V = self.feature_list[i] @ random_coeffs_V
                        initialized_matrix_K = initialized_matrix_K / initialized_matrix_K.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix_V / initialized_matrix_V.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    
                    module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)   
                    module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)   
                    i += 1

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
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
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.network(inputs, get_cur_feat=True, get_weight=False)
            
        mat_list = []
        for module in self.network.modules():
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                n_cur_matrix = module.n_cur_matrix 
                print("self.n_cur_matrix:", n_cur_matrix)
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
                
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            sval_total = (S).sum()
            sval_ratio = (S) / sval_total
            k = torch.arange(1, self.in_dim + 1).to(self.device) - 1
            k = (self.in_dim - k.float()) / self.in_dim
            sval_ratio = sval_ratio.to(self.device)
            result = (self.cur_task + 1) * (1 - torch.cumsum(sval_ratio, dim=-1)) - self.alpha * k
            r = torch.argmin(result).item()

            if i >= len(self.feature_list):
                self.feature_list.append(U[:, max(r, 1) + 1:])
            else:
                self.feature_list[i] = U[:, max(r, 1) + 1:]
                
    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        # ============ Flat-LoRA: 计算总步数用于cosine递增 ============
        self.flat_total_steps = self.epochs * len(train_loader)
        self.flat_current_step = 0
        self.flat_original_B = {}
        # ============================================================

        with torch.no_grad():
            self.init_drm(train_loader)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        with torch.no_grad():
            self.psv_info(train_loader)
        return

    def Weight_pri(weight, alpha=0.99, device=None):
        original_device = weight.device
        original_dtype = weight.dtype
        
        if device is not None:
            compute_device = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            weight = weight.to(device=compute_device, dtype=torch.float32)
        else:
            weight = weight.to(dtype=torch.float32)

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        total_sum = S.sum()
        cumsum = S.cumsum(dim=0)
        r = (cumsum / total_sum >= alpha).nonzero(as_tuple=True)[0][0].item() + 1

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_r = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))
        
        return weight_r.to(device=original_device, dtype=original_dtype)
    
    def cosine_similarity(self, W_prev, delta):
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        return cos_sim
    
    def cosine_similarity_loss(self, W_prev, delta):
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        return torch.abs(cos_sim)
    
    def _train_function(self, train_loader, optimizer, scheduler):


        # ========== EMA初始化（原有逻辑，保持不变） ==========
        if self.cur_task >= 1:
            ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
            if self.ema_type == "adaptEMA":
                ema_decay = None
                ema_params = {}
                prev_ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        print(f"Adaptive EMA tracking: {name}")        
            elif self.ema_type == "grad_direction":
                ema_decay = None
                ema_params = {}
                prev_ema_params = {}
                ema_grad_params = {}
                grad_ema_decay = 0.99
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        ema_grad_params[name] = torch.zeros_like(param.data)
                        print(f"Grad-Direction EMA tracking: {name}")
                if not hasattr(self, 'alpha_smooth'):
                    self.alpha_smooth = {}
            elif self.ema_type == "EMA_scheduled":
                ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        print(f"Scheduled EMA tracking: {name}")
            elif self.ema_type == "AdEMAMix":
                ema_params = {}
                ema_slow_params = {}
                beta1 = 0.9
                beta3 = 0.9999
                alpha_adamemix = 5.0
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        ema_slow_params[name] = param.data.clone().detach()
                        print(f"AdEMAMix tracking (fast & slow): {name}")
                T_alpha_beta3 = self.epochs * len(train_loader)
            elif self.ema_type == "adaptEMAv2":
                ema_params = {}
                prev_ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        prev_ema_params[name] = param.data.clone().detach()
                        print(f"Adaptive EMA v2 tracking: {name}")
            elif self.ema_type == "Cum_avg":
                if self.flag ==1:
                    pass
                else:
                    self.ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
                    if self.cur_task == 1:
                        self.ema_params = {}
                        self.cnt = 1
                        for name, param in self.network.named_parameters():
                            if any(key in name for key in self.ema_keys):
                                self.ema_params[name] = param.data.clone().detach()
                    # self.flag = 1

            elif self.ema_type == "randomEMA":
                import random
                ema_decay = random.uniform(0.9, 1.0)
                print(f"cur_task: {self.cur_task}, ema_decay:{ema_decay}")
                ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        print(f"EMA tracking: {name}")                
            elif self.ema_type == "kda_delta":  # <--- 新增
                kda_beta = getattr(self, 'kda_beta', 0.01)
                ema_params = {}
                for name, param in self.network.named_parameters():
                    if any(key in name for key in ema_keys):
                        ema_params[name] = param.data.clone().detach()
                        print(f"KDA-Delta EMA tracking: {name}")

            elif self.ema_decay == "global_ema":
                ema_decay = math.pow(0.70, 1/ (len(train_loader)*10))

                self.ema_params = {}

                for name, module in self.network.named_modules():
                    if not isinstance(module, Attention_LoRA):
                        continue
                    
                    module_path = name
                    
                    # ========== K 的 W ==========
                    A_k_shared = module.Lora_shared_A_k.weight
                    B_k_shared = module.Lora_shared_B_k.weight
                    W_k_shared = B_k_shared @ A_k_shared
                    
                    # ========== V 的 W ==========
                    A_v_shared = module.Lora_shared_A_v.weight
                    B_v_shared = module.Lora_shared_B_v.weight
                    W_v_shared = B_v_shared @ A_v_shared
                    
                    # 存储
                    self.ema_params[f"{module_path}.W_k_shared"] = W_k_shared.clone().detach()
                    self.ema_params[f"{module_path}.W_v_shared"] = W_v_shared.clone().detach()
                    
                    # print(f"   Tracking W_k_shared, W_v_shared for: {module_path}")

            else:
                import math

                ema_decay = math.pow(self.R, 1/ (len(train_loader)*10))

                if self.flag ==1:
                    print(f"curent task {self.cur_task}, ema decay {ema_decay}")
                    pass
                else:          
                    print(f"curent task {self.cur_task}, ema decay {ema_decay}")
                    self.ema_params = {}
                    for name, param in self.network.named_parameters():
                        if any(key in name for key in ema_keys):
                            self.ema_params[name] = param.data.clone().detach()
                            print(f"EMA tracking: {name}")
                    # self.flag = 1
        
        # training phase
        prog_bar = tqdm(range(self.epochs))
        global_step = 0
        
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0
            ortho_loss_sum = 0.0
            ortho_batches = 0
            print("\n")
            print(f"cur_task {self.cur_task}, Epoch {epoch}: "
                  f"Head LR={optimizer.param_groups[0]['lr']:.6f}, "
                  f"Backbone LR={optimizer.param_groups[1]['lr']:.6f}")
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes
                
                # ============================================================
                # Flat-LoRA: 在梯度计算前对LoRA的B矩阵施加扰动
                # 使用cosine递增的扰动强度
                # ============================================================
                # current_sigma = self._get_flat_sigma(self.flat_current_step, self.flat_total_steps)
                # self._apply_flat_perturbation_to_lora_modules(current_sigma)
                
                # 前向传播（在扰动后的权重上）
                ret_minus = self.network(inputs)
                logits = (ret_minus['logits'])
                loss_minus = F.cross_entropy(logits, targets)
                loss = loss_minus
                
                # ============================================================
                # 恢复扰动（确保参数在优化步骤前恢复）
                # ============================================================
                # self._restore_lora_modules()
                self.flat_current_step += 1

                if self.ortho_loss_type and self.cur_task >= 1:
                    #############################################################################
                    ortho_loss = 0.0
                    layer_count = 0

                    for module in self.network.modules():
                        if not isinstance(module, Attention_LoRA):
                            continue

                        for proj in ('k', 'v'):
                            if proj == 'k':
                                B_tasks = module.lora_B_k
                                B_shared = module.Lora_shared_B_k
                            else:
                                B_tasks = module.lora_B_v
                                B_shared = module.Lora_shared_B_v

                            # 每个任务单独约束，再按任务数平均，避免 loss 随任务数增长
                            proj_loss = 0.0
                            for t in range(self.cur_task + 1):
                                cross = B_shared.weight.T @ B_tasks[t].weight  # [r, r]
                                proj_loss += torch.norm(cross, p='fro') ** 2
                            ortho_loss += proj_loss / (self.cur_task + 1)
                            layer_count += 1

                    # 按层平均，避免损失随层数/投影数线性增长
                    if layer_count > 0:
                        ortho_loss = ortho_loss / layer_count
                    #############################################################################
                    loss += self.ortho_loss_weight * ortho_loss
                    ortho_loss_sum += ortho_loss.item()
                    ortho_batches += 1
                    
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                global_step += 1

                # ========== EMA系列更新（原有逻辑，保持不变） ==========
                if self.cur_task >= 1:
                    if self.ema_type == "EMA2":
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = 0.5 * (ema_decay * ema_params[name] + (1 - ema_decay) * param.data \
                                            + ema_decay2 * ema_params[name] + (1 - ema_decay2) * param.data)
                    elif self.ema_type == "kda_delta":
                        # KDA风格的选择性EMA更新
                        # 使用当前参数与EMA参数的差异方向作为k_t
                        
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                # 当前参数 θ_t
                                theta_t = param.data
                                
                                # EMA参数 S_{t-1}
                                S_prev = ema_params[name]
                                
                                # 计算方向 k_t = (θ_t - S_prev) / ||θ_t - S_prev||  (归一化的变化方向)
                                delta = theta_t - S_prev
                                delta_norm = torch.norm(delta)
                                
                                if delta_norm > 1e-8:
                                    k_t = delta / delta_norm  # 单位方向向量
                                    
                                    # KDA Delta规则: S_t = S_{t-1}(I - β·k·k^T) + β·θ_t·k^T
                                    # 简化版: 只擦除与变化方向平行的部分
                                    S_flat = S_prev.view(-1)
                                    k_flat = k_t.view(-1)
                                    
                                    # S · k (标量)
                                    S_dot_k = torch.dot(S_flat, k_flat)
                                    
                                    # θ · k (标量)
                                    theta_dot_k = torch.dot(theta_t.view(-1), k_flat)
                                    
                                    # 更新: 擦除旧方向 + 写入新方向
                                    # KDA Delta规则: S_t = S_{t-1}(I - β·k·k^T) + β·θ_t·k^T
                                    S_new_flat = S_flat - kda_beta * S_dot_k * k_flat + kda_beta * theta_dot_k * k_flat
                                    
                                    ema_params[name] = S_new_flat.view_as(S_prev)
                                else:
                                    # 变化太小，保持不动
                                    ema_params[name] = S_prev
                    elif self.ema_type == "EMA_scheduled":
                        global_step = epoch * len(train_loader) + i
                        total_steps = self.epochs * len(train_loader)
                        progress = global_step / total_steps
                        if self.ema_decay_schedule == "linear":
                            ema_decay_current = self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * progress
                        elif self.ema_decay_schedule == "cosine":
                            cos_val = (1 - math.cos(math.pi * progress)) / 2
                            ema_decay_current = self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * cos_val
                        else:
                            ema_decay_current = self.ema_decay
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = ema_decay_current * ema_params[name] + (1 - ema_decay_current) * param.data
                    elif self.ema_type == "grad_direction":
                        for name, param in self.network.named_parameters():
                            if name not in ema_params:
                                continue
                            if param.grad is None:
                                continue
                            theta_t = param.data
                            bar_theta_prev = prev_ema_params[name]
                            grad_flat = param.grad.view(-1)
                            ema_grad_flat = ema_grad_params[name].view(-1)
                            grad_norm = torch.norm(grad_flat, p=2)
                            ema_grad_norm = torch.norm(ema_grad_flat, p=2)
                            if grad_norm > 1e-8 and ema_grad_norm > 1e-8:
                                cos_sim = torch.dot(grad_flat, ema_grad_flat) / (grad_norm * ema_grad_norm + 1e-8)
                            else:
                                cos_sim = torch.tensor(0.0, device=param.device)
                            grad_ema_decay = getattr(self, 'grad_ema_decay', 0.99)
                            ema_grad_params[name] = grad_ema_decay * ema_grad_params[name] + (1 - grad_ema_decay) * param.grad
                            normalized = (cos_sim + 1) * 0.5
                            alpha_raw = 0.9 + 0.09999 * torch.sigmoid(1.0 * (normalized - 0.5))
                            if name not in self.alpha_smooth:
                                self.alpha_smooth[name] = alpha_raw.clone().detach()
                            else:
                                self.alpha_smooth[name] = 0.9 * self.alpha_smooth[name] + 0.1 * alpha_raw
                            alpha_star = torch.clamp(self.alpha_smooth[name], 0.9, 0.99999)
                            ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                            prev_ema_params[name] = ema_params[name].clone().detach()
                    elif self.ema_type == "adaptEMA":
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                theta_t = param.data
                                bar_theta_prev = prev_ema_params[name]
                                delta = bar_theta_prev - theta_t
                                delta_norm_sq = torch.norm(delta, p=2).pow(2)
                                loss_theta_t = loss.item()
                                if param.grad is not None and delta_norm_sq > 1e-8:
                                    grad_flat = param.grad.view(-1)
                                    delta_flat = delta.view(-1)
                                    loss_bar_prev_approx = loss.item() + torch.dot(grad_flat, delta_flat).item()
                                else:
                                    loss_bar_prev_approx = loss.item()
                                if param.grad is not None and delta_norm_sq > 1e-8:
                                    L = 15.0
                                    alpha_star = 0.5 + (loss_bar_prev_approx - loss_theta_t) / (L * delta_norm_sq + 1e-8)
                                else:
                                    alpha_star = torch.tensor(0.5, device=param.device)
                                alpha_star = torch.clamp(alpha_star, 0.9, 0.99999)
                                ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                                prev_ema_params[name] = ema_params[name].clone().detach()
                    elif self.ema_type == "adaptEMAv2":
                        batch_size = inputs.shape[0]
                        self.global_loss_sum += loss.item() * batch_size
                        self.global_loss_count += batch_size
                        loss_global_avg = self.global_loss_sum / self.global_loss_count
                        if self.global_loss_ema is None:
                            self.global_loss_ema = loss_global_avg
                        else:
                            self.global_loss_ema = (
                                self.global_loss_ema_decay * self.global_loss_ema +
                                (1 - self.global_loss_ema_decay) * loss_global_avg
                            )
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                theta_t = param.data
                                if name not in self.global_param_sum:
                                    self.global_param_sum[name] = torch.zeros_like(theta_t)
                                self.global_param_sum[name] = (
                                    self.global_param_sum[name] * self.global_param_count + theta_t
                                ) / (self.global_param_count + 1)
                        self.global_param_count += 1
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
                                loss_theta_t = self.global_loss_ema if self.global_loss_ema is not None else loss_global_avg
                                loss_diff = loss_theta_t - loss.item()
                                L = 5.0
                                if abs(loss_diff) > 1e-6:
                                    alpha_star = 0.5 + loss_diff / (L * delta_norm_sq + 1e-8)
                                else:
                                    grad_flat = param.grad.view(-1)
                                    delta_flat = delta.view(-1)
                                    grad_delta = torch.dot(grad_flat, delta_flat).item()
                                    alpha_star = 0.5 + grad_delta / (grad_norm * delta_norm + 1e-8)
                            else:
                                alpha_star = torch.tensor(0.5, device=param.device)
                            alpha_star = torch.clamp(alpha_star, 0.9, 0.99999)
                            ema_params[name] = alpha_star * ema_params[name] + (1 - alpha_star) * theta_t
                            prev_ema_params[name] = ema_params[name].clone().detach()
                    elif self.ema_type == "AdEMAMix":
                        if T_alpha_beta3 is not None:
                            step = global_step
                            T = T_alpha_beta3
                            alpha_t = min(step * alpha_adamemix / T, alpha_adamemix)
                            progress = min(step / T, 1.0)
                            log_beta1 = math.log(beta1)
                            log_beta3 = math.log(beta3)
                            log_beta3_t = (1 - progress) * log_beta3 + progress * log_beta1
                            beta3_t = math.exp(log_beta3_t)
                        else:
                            alpha_t = alpha_adamemix
                            beta3_t = beta3
                        for name, param in self.network.named_parameters():
                            if name in ema_params:
                                ema_params[name] = beta1 * ema_params[name] + (1 - beta1) * param.data
                                ema_slow_params[name] = beta3_t * ema_slow_params[name] + (1 - beta3_t) * param.data
                                final_alpha_t = min(global_step * alpha_adamemix / T_alpha_beta3, alpha_adamemix) if T_alpha_beta3 is not None else alpha_adamemix
                                ema_params[name] = ema_params[name] + final_alpha_t * ema_slow_params[name]
                    elif self.ema_type == "Cum_avg":
                        for name, param in self.network.named_parameters():
                            if name in self.ema_params:
                                self.ema_params[name] = (self.cnt * self.ema_params[name] + param.data) / (self.cnt + 1)
                                self.cnt += 1

                    elif self.ema_decay == "global_ema":
                        # 更新 W_k_shared 和 W_v_shared 的 EMA
                        for name, module in self.network.named_modules():
                            if not isinstance(module, Attention_LoRA):
                                continue
                            
                            module_path = name
                            
                            # 计算当前的 W_k_shared
                            A_k_shared = module.Lora_shared_A_k.weight
                            B_k_shared = module.Lora_shared_B_k.weight
                            W_k_shared_current = B_k_shared @ A_k_shared
                            
                            # 计算当前的 W_v_shared
                            A_v_shared = module.Lora_shared_A_v.weight
                            B_v_shared = module.Lora_shared_B_v.weight
                            W_v_shared_current = B_v_shared @ A_v_shared
                            
                            # 更新 EMA
                            key_k = f"{module_path}.W_k_shared"
                            key_v = f"{module_path}.W_v_shared"
                            
                            if key_k in self.ema_params:
                                self.ema_params[key_k] = (
                                    ema_decay * self.ema_params[key_k] + 
                                    (1 - ema_decay) * W_k_shared_current
                                )
                            
                            if key_v in self.ema_params:
                                self.ema_params[key_v] = (
                                    ema_decay * self.ema_params[key_v] + 
                                    (1 - ema_decay) * W_v_shared_current
                                )
                    else:
                        for name, param in self.network.named_parameters():
                            # ema_params = ["Lora_shared_A_k", "Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
                            ema_keys_k = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
                            if name in self.ema_params:
                                self.ema_params[name] = (ema_decay * self.ema_params[name] + (1 - ema_decay) * param.data)
                                # param.data.copy_(self.ema_params[name])     
                                       

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            #     self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            # prog_bar.set_description(info)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if ortho_batches > 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Ortho {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc,
                    ortho_loss_sum / ortho_batches)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

        # ========== EMA参数应用到共享参数（原有逻辑） ==========
        if self.cur_task >= 1:
            if self.ema_type == "Cum_avg":
                with torch.no_grad():
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            param.data.copy_(self.ema_params[name])  

            elif self.ema_decay == "global_ema":
                with torch.no_grad():
                    for name, module in self.network.named_modules():
                        if not isinstance(module, Attention_LoRA):
                            continue
                        
                        module_path = name
                        key_k = f"{module_path}.W_k_shared"
                        key_v = f"{module_path}.W_v_shared"
                        
                        # 如果模块有 W_k_shared 和 W_v_shared 参数，直接赋值
                        if hasattr(module, 'W_k_shared') and key_k in self.ema_params:
                            module.W_k_shared.data.copy_(self.ema_params[key_k])
                        
                        if hasattr(module, 'W_v_shared') and key_v in self.ema_params:
                            module.W_v_shared.data.copy_(self.ema_params[key_v])

                      
            else:
                with torch.no_grad():
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            param.data.copy_(self.ema_params[name])

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        if self.cur_task == 0:
            unfrozen_keys = [
                f"classifier_pool{target_suffix}",
                f"lora_B_k{target_suffix}",
                f"lora_B_v{target_suffix}",
                "Lora_shared",
            ]
        else:
            unfrozen_keys = [
                f"classifier_pool{target_suffix}",
                f"lora_B_k{target_suffix}",
                f"lora_B_v{target_suffix}",
                "Lora_shared"
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