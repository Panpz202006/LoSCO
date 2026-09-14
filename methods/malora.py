"""
MALoRA: Manifold-Aligned Low-Rank Adaptation for Continual Learning
基于 InfLoRA 结构，使用共享的 ΔW 和黎曼优化
"""

import math
import logging
import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from models.net_malora import Net_MALoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params

# Try to import Riemannian optimizer
try:
    from methods.RiemannianLoRA import RiemannianSGD
    RIEMANNIAN_AVAILABLE = True
except ImportError:
    RIEMANNIAN_AVAILABLE = False
    print("Warning: RiemannianSGD not found, using standard optimizer")


class MALoRA(BaseLearner):
    """
    MALoRA: Manifold-Aligned Low-Rank Adaptation
    使用共享的 delta_W 在固定秩流形上优化
    """
    def __init__(self, args):
        super().__init__(args)
        
        # MALoRA 特有参数
        self.rank = args.get("rank", 8)
        self.lame = args.get("lame", 0.95)
        self.lamb = args.get("lamb", 0.85)
        self.use_riemannian = args.get("use_riemannian", False)
        self.riemannian_lr = args.get("riemannian_lr", 0.01)
        
        # 历史子空间（用于梯度投影，防止遗忘）
        self.feature_list = []      # 存储子空间基向量
        self.feature_mat_list = []  # 存储投影矩阵
        self.project_type = []      # 'remove' or 'retain'
        
        # 当前任务状态
        self._cur_task = -1
        self._known_classes = 0
        
        # 存储每个任务的类别索引范围
        self._task_indices = []  # 每个任务包含的类别索引列表
        
        # 初始化网络
        self.network = Net_MALoRA(args)
        self.network.to(self.device)
        
        # 优化器（延迟创建）
        self._optimizer = None
        
        # 训练参数
        self.epochs = args.get("epochs", 20)
        self.lrate = args.get("lrate", 0.0005)
        self.weight_decay = args.get("weight_decay", 0.0)
        self.batch_size = args.get("batch_size", 128)
        self.num_workers = args.get("num_workers", 16)
        
        logging.info(f"MALoRA initialized: rank={self.rank}, lame={self.lame}, lamb={self.lamb}")

    # ==================== 核心训练方法 ====================
    
    def before_task(self, data_manager):
        """每个任务开始前的准备"""
        self._cur_task += 1
        
        # 计算当前任务的类别索引范围
        start_cls = sum(data_manager._increments[:self._cur_task])
        end_cls = start_cls + data_manager.get_task_size(self._cur_task)
        current_indices = list(range(start_cls, end_cls))
        self._task_indices.append(current_indices)
        
        self._known_classes = start_cls
        
        # 更新分类头（添加新任务的分类器）
        self.network.update_fc()
        
        # 初始化 delta_W（使用随机初始化，跳过耗时的特征收集）
        if self._cur_task == 0:
            with torch.no_grad():
                self._init_delta_w_random()
        
        # 冻结网络（只训练 delta_W 和当前分类头）
        self._freeze_network()
        
        logging.info(f'Task {self._cur_task}: classes {start_cls}-{end_cls-1}, known_classes={self._known_classes}')

    def incremental_train(self, data_manager):
        """训练当前任务"""
        # 获取当前任务的训练数据加载器
        train_loader = self._create_train_loader(data_manager, self._cur_task)
        
        # 创建优化器
        self._optimizer = self._create_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self.epochs)
        
        # 训练循环
        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            self.network.train()
            losses = 0.
            correct, total = 0, 0
            
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                logits = self.network(inputs, inference=False)
                loss = F.cross_entropy(logits, targets)
                
                # 反向传播
                self._optimizer.zero_grad()
                loss.backward()
                
                # 梯度投影到历史子空间（防止遗忘）
                for name, param in self.network.named_parameters():
                    if param.grad is not None and 'delta_W' in name:
                        param.grad = self._project_gradient_to_history(param.grad)
                
                self._optimizer.step()
                
                # 保持 delta_W 的低秩性
                for module in self.network.get_attention_modules():
                    module._truncate_rank()
                    module._update_factors()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f'Task {self._cur_task}, Epoch {epoch+1}/{self.epochs} => Loss {losses/len(train_loader):.3f}, Acc {train_acc:.2f}'
            prog_bar.set_description(info)
            logging.info(info)
        
        # 训练完成后，更新历史子空间（用于后续任务）
        with torch.no_grad():
            self._update_history_subspaces(train_loader)
    
    def incremental_test(self, data_manager):
        """测试所有已学任务"""
        self.network.eval()
        
        # 计算每个任务的准确率
        accy_task = []
        for task_idx in range(self._cur_task + 1):
            test_loader = self._create_test_loader(data_manager, task_idx)
            correct, total = 0, 0
            
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.no_grad():
                    logits = self.network(inputs, inference=True)
                    preds = logits.max(1)[1]
                
                correct += (preds == targets).cpu().sum()
                total += len(targets)
            
            acc_task = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            accy_task.append(acc_task)
        
        # 计算平均准确率
        avg_acc = np.mean(accy_task)
        
        # 格式化成 BaseLearner 期望的返回格式
        accy = {'top1': avg_acc, 'grouped': accy_task}
        accy_with_task = {'top1': avg_acc}
        
        logging.info(f'Task {self._cur_task} evaluation: avg_acc={avg_acc:.2f}')
        
        return accy, accy_with_task, accy_task
    
    def after_task(self):
        """每个任务结束后的清理"""
        pass

    # ==================== 辅助方法 ====================
    
    def _freeze_network(self):
        """冻结网络参数，只训练 delta_W 和当前分类头"""
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
            if 'delta_W' in name:
                param.requires_grad_(True)
            elif f'classifier_pool.{self._cur_task}' in name:
                param.requires_grad_(True)
        
        print_trainable_params(self.network)
    
    def _get_task_indices(self, data_manager, task_idx):
        """获取任务包含的类别索引"""
        start_cls = sum(data_manager._increments[:task_idx])
        end_cls = start_cls + data_manager.get_task_size(task_idx)
        return list(range(start_cls, end_cls))
    
    def _create_train_loader(self, data_manager, task_idx):
        """创建训练数据加载器"""
        # 获取当前任务的类别索引
        indices = self._task_indices[task_idx]
        
        # 获取数据集
        dataset = data_manager.get_dataset(
            indices=indices,
            source='train',
            mode='train',
            appendent=None,
            ret_data=False
        )
        
        # 创建 DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader
    
    def _create_test_loader(self, data_manager, task_idx):
        """创建测试数据加载器"""
        # 获取当前任务的类别索引
        indices = self._task_indices[task_idx]
        
        # 获取数据集
        dataset = data_manager.get_dataset(
            indices=indices,
            source='test',
            mode='test',
            appendent=None,
            ret_data=False
        )
        
        # 创建 DataLoader
        test_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return test_loader
    
    def _init_delta_w_random(self):
        """随机初始化 delta_W（快速，跳过特征收集）"""
        logging.info("Initializing delta_W with random values...")
        
        for module in self.network.get_attention_modules():
            # 使用小随机值初始化
            nn.init.normal_(module.delta_W, std=0.02)
            # 强制保持低秩
            module._truncate_rank()
            module._update_factors()
            # 重置特征收集矩阵
            module.reset_input_matrix()
        
        logging.info("delta_W initialization complete.")
    
    def _init_delta_w(self, data_manager):
        """初始化 delta_W（使用第一个任务的数据）- 备用方法，默认不使用"""
        # 这个方法保留但默认不使用，因为特征收集太慢
        # 如需使用，请在 before_task 中调用此方法而不是 _init_delta_w_random
        logging.info("Initializing delta_W with data-driven method (may be slow)...")
        
        # 获取第一个任务的训练数据加载器
        train_loader = self._create_train_loader(data_manager, 0)
        
        # 限制收集的样本数量
        max_samples = 500
        collected_samples = 0
        
        for batch_idx, (_, inputs, _) in enumerate(train_loader):
            if collected_samples >= max_samples:
                break
            inputs = inputs.to(self.device)
            self.network(inputs, get_cur_feat=True)
            collected_samples += inputs.size(0)
        
        # 初始化每个注意力层的 delta_W
        for module in self.network.get_attention_modules():
            if module.cur_matrix is not None and module.n_cur_matrix > 0:
                features = module.cur_matrix
                
                # 采样
                if features.shape[0] > 500:
                    idx = torch.randperm(features.shape[0])[:500]
                    features = features[idx]
                
                mean = features.mean(dim=0, keepdim=True)
                features_centered = features - mean
                
                U, S, V = torch.linalg.svd(features_centered, full_matrices=False)
                r = min(self.rank, len(S))
                
                delta_init = U[:, :r] @ torch.diag(S[:r]) @ V[:r, :]
                module.delta_W.data = delta_init.to(self.device)
                module._truncate_rank()
                module._update_factors()
                module.reset_input_matrix()
        
        logging.info("delta_W initialization complete.")
    
    def _create_optimizer(self):
        """创建优化器"""
        delta_w_params = []
        classifier_params = []
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if 'delta_W' in name:
                    delta_w_params.append(param)
                elif 'classifier_pool' in name:
                    classifier_params.append(param)
        
        # 使用 Adam 优化器
        optimizer = torch.optim.Adam([
            {'params': delta_w_params, 'lr': self.lrate},
            {'params': classifier_params, 'lr': self.lrate}
        ], weight_decay=self.weight_decay)
        
        return optimizer
    
    def _project_gradient_to_history(self, grad):
        """将梯度投影到历史子空间的正交补（防止遗忘）"""
        if len(self.feature_list) == 0:
            return grad
        
        grad_flat = grad.view(-1, 1)
        grad_device = grad.device
        
        for i, feature in enumerate(self.feature_list):
            if i < len(self.project_type) and self.project_type[i] == 'remove':
                feature_tensor = torch.from_numpy(feature).float().to(grad_device)
                proj = feature_tensor @ (feature_tensor.T @ grad_flat)
                grad_flat = grad_flat - proj
        
        return grad_flat.view(grad.shape)
    
    def _update_history_subspaces(self, train_loader):
        """更新历史子空间（使用 DualGPM）"""
        # 收集特征
        for _, inputs, _ in train_loader:
            inputs = inputs.to(self.device)
            self.network(inputs, get_cur_feat=True)
        
        mat_list = []
        for module in self.network.get_attention_modules():
            if module.cur_matrix is not None and module.n_cur_matrix > 0:
                mat_list.append(deepcopy(module.cur_matrix.numpy()))
            module.reset_input_matrix()
        
        if mat_list:
            self._update_dual_gpm(mat_list)
    
    def _update_dual_gpm(self, mat_list):
        """Dual GPM 子空间更新"""
        threshold = (self.lame - self.lamb) * self._cur_task / self.args.get("sessions", 10) + self.lamb
        logging.info(f'Threshold: {threshold}')
        
        if len(self.feature_list) == 0:
            # 第一个任务：初始化子空间
            for i, activation in enumerate(mat_list):
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S ** 2).sum()
                if sval_total > 0:
                    sval_ratio = (S ** 2) / sval_total
                    r = np.sum(np.cumsum(sval_ratio) < threshold)
                    r = max(r, 1)
                else:
                    r = 1
                self.feature_list.append(U[:, :r])
                self.project_type.append('remove' if r < activation.shape[0] / 2 else 'retain')
        else:
            # 后续任务：更新子空间
            for i, activation in enumerate(mat_list):
                if i >= len(self.feature_list):
                    continue
                    
                if self.project_type[i] == 'remove':
                    if i < len(self.feature_mat_list):
                        proj_mat = self.feature_mat_list[i].numpy()
                        act_hat = activation - proj_mat @ activation
                    else:
                        act_hat = activation
                    
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    sval_total = (S ** 2).sum()
                    if sval_total > 0:
                        sval_ratio = (S ** 2) / sval_total
                        accumulated_sval = 0
                        r = 0
                        for ii in range(len(sval_ratio)):
                            if accumulated_sval < threshold:
                                accumulated_sval += sval_ratio[ii]
                                r += 1
                            else:
                                break
                        if r > 0:
                            new_U = np.hstack((self.feature_list[i], U[:, :r]))
                            if new_U.shape[1] > new_U.shape[0]:
                                self.feature_list[i] = new_U[:, :new_U.shape[0]]
                            else:
                                self.feature_list[i] = new_U
        
        # 更新投影矩阵
        self.feature_mat_list = []
        for feature in self.feature_list:
            proj_mat = feature @ feature.T
            self.feature_mat_list.append(torch.from_numpy(proj_mat).float())
        
        # 打印子空间信息
        logging.info('-' * 40)
        logging.info('DualGPM Subspace Summary')
        for i in range(len(self.feature_list)):
            logging.info(f'Layer {i+1}: {self.feature_list[i].shape[1]}/{self.feature_list[i].shape[0]} (type: {self.project_type[i]})')
        logging.info('-' * 40)

    def get_parameters(self, config):
        """返回可训练参数"""
        return self.network.parameters()