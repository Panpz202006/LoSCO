# scheduler_utils.py - 完全兼容 timm 的 create_scheduler_v2

import math
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, CosineAnnealingLR

def create_scheduler_v2(
    optimizer,
    sched='cosine',
    num_epochs=100,
    decay_epochs=1000,
    decay_milestones=None,
    decay_rate=0.1,
    min_lr=1e-6,
    warmup_epochs=0,
    warmup_lr=1e-6,
    warmup_prefix=False,
    cooldown_epochs=0,
    **kwargs
):
    """
    完全兼容 timm.scheduler.create_scheduler_v2 的自定义实现
    
    Args:
        optimizer: 优化器
        sched: 调度器类型 ('cosine', 'multistep', 'steplr', 'step')
        num_epochs: 总训练轮数
        decay_epochs: 延迟衰减起始轮数
        decay_milestones: 多阶梯衰减轮数列表
        decay_rate: 衰减率
        min_lr: 最小学习率
        warmup_epochs: 预热轮数
        warmup_lr: 预热初始学习率
        warmup_prefix: 是否将预热轮数从总轮数中扣除
        cooldown_epochs: 冷却轮数
    """
    
    if sched == 'cosine':
        scheduler = create_cosine_with_warmup(
            optimizer, num_epochs, decay_epochs, 
            min_lr, warmup_epochs, warmup_lr, 
            warmup_prefix, cooldown_epochs
        )
    elif sched in ['multistep', 'steplr']:
        # multistep 和 steplr 使用同样的逻辑
        if sched == 'steplr' and decay_milestones is None:
            # 如果是 steplr 但没有指定 milestones，使用 decay_epochs 作为步长
            milestones = [decay_epochs] if decay_epochs else []
        else:
            milestones = decay_milestones or []
        
        scheduler = create_multistep_with_warmup(
            optimizer, milestones, decay_rate,
            warmup_epochs, warmup_lr
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched}")
    
    # 返回 scheduler 和 num_epochs（保持接口一致）
    return scheduler, num_epochs


def create_multistep_with_warmup(
    optimizer,
    milestones,
    decay_rate=0.1,
    warmup_epochs=0,
    warmup_lr=1e-6
):
    """创建 MultiStepLR 调度器（带预热）"""
    
    if not milestones:
        # 如果没有 milestones，返回恒等调度器
        return LambdaLR(optimizer, lambda epoch: 1.0)
    
    def lr_lambda(epoch):
        # 预热阶段
        if epoch < warmup_epochs:
            if warmup_epochs > 0:
                return float(epoch) / float(warmup_epochs)
            return 0.0
        
        # 计算衰减（milestones 相对于 warmup 之后）
        effective_epoch = epoch - warmup_epochs
        decay_steps = sum([1 for m in milestones if effective_epoch >= m])
        return decay_rate ** decay_steps
    
    return LambdaLR(optimizer, lr_lambda)


def create_cosine_with_warmup(
    optimizer,
    num_epochs,
    decay_epochs,
    min_lr=1e-6,
    warmup_epochs=0,
    warmup_lr=1e-6,
    warmup_prefix=False,
    cooldown_epochs=0
):
    """创建 Cosine 调度器（带预热和延迟衰减）"""
    
    def lr_lambda(epoch):
        # 预热阶段
        if epoch < warmup_epochs:
            if warmup_epochs > 0:
                return float(epoch) / float(warmup_epochs)
            return 0.0
        
        # 处理 warmup_prefix
        effective_epoch = epoch
        if warmup_prefix:
            effective_epoch = epoch - warmup_epochs
        
        # 延迟衰减阶段（在 decay_epochs 之前保持 lr 不变）
        if effective_epoch < decay_epochs:
            return 1.0
        
        # 冷却阶段（最后 cooldown_epochs 轮保持最小学习率）
        if cooldown_epochs > 0 and effective_epoch > num_epochs - cooldown_epochs:
            return 0.0  # 对应 min_lr / base_lr
        
        # 余弦衰减
        progress = (effective_epoch - decay_epochs) / max(1, num_epochs - decay_epochs - cooldown_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)
    
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from abc import abstractmethod
from utils.toolkit import accuracy
from utils.toolkit import print_trainable_params, check_params_consistency


class BaseLearner(object):
    def __init__(self, args):

        self.args = args

        self.dataset = args['dataset']
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self.sessions = args['sessions']

        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lrate = args['lrate']
        self.weight_decay = args['weight_decay']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.num_workers = args['num_workers']

        self.milestone = args.get('milestone', None)
        self.lrate_decay = args.get('lrate_decay', None)
        self.fc_lrate = args.get('fc_lrate', None)
        
        self.topk = 5
        self.cur_task = -1
        self.known_classes = 0  # Number of classes seen so far
        self.total_classes = 0  # Total number of classes in the current task

        self.debug = False
        self.device = args['device'][0]
        self.multiple_gpus = args['device']
        self.method = args['method']
        self.lrate_head = args['lrate_head']
        self.moe_list = args['moe_list']


    @property
    def feature_dim(self):
        if isinstance(self.network, nn.DataParallel):
            return self.network.module.feature_dim
        else:
            return self.network.feature_dim

    def build_train_loader(self, data_manager):
        train_dataset = data_manager.get_dataset(
            np.arange(self.known_classes, self.total_classes),
            source='train', mode='train')
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)

    def build_test_loader(self, data_manager):
        test_dataset = data_manager.get_dataset(
            np.arange(0, self.total_classes),
            source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers)
    
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
        if self.method == "splitlora" or self.method == "splitlorav6" or self.method == "splitlorav16":
            if self.optimizer == 'sgd':
                optimizer = optim.SGD(
                    trainable_params,
                    momentum=0.9,
                    lr=self.lrate,
                    weight_decay=self.weight_decay
                )
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.Adam([
                    # trainable_params,
                    # lr=self.lrate,
                    # weight_decay=self.weight_decay,
                    # betas=(0.9,0.999)
        {'params': trainable_params[-2:], 'lr': self.lrate_head, 'weight_decay': self.weight_decay, 'betas': (0.9, 0.999)},
        {'params': trainable_params[:-2], 'lr': self.lrate,  'weight_decay': self.weight_decay, 'betas': (0.9, 0.999)}
                ])
            elif self.args['optimizer'] == 'adamw':
                # 获取所有可训练参数的名称和形状
                trainable_params = []
                trainable_names = []
                
                for name, param in self.network.named_parameters():
                    if param.requires_grad:
                        trainable_names.append(name)
                        trainable_params.append(param)
                from timm.optim import create_optimizer_v2

                param_groups = [
                    {'params': trainable_params[-2:], 'lr': self.lrate_head, 'weight_decay': self.weight_decay},
                    {'params': trainable_params[:-2], 'lr': self.lrate, 'weight_decay': self.weight_decay}
                ]

                optimizer = create_optimizer_v2(
                    param_groups,
                    opt='adamW',
                    weight_decay=self.weight_decay,
                    foreach=True
                )
            elif self.args['optimizer'] == 'AdEMAMix':
                from methods.AdEMAMix import AdEMAMix
                optimizer = AdEMAMix([
        {'params': trainable_params[-2:], 'lr': self.lrate_head, 'weight_decay': self.weight_decay},
        {'params': trainable_params[:-2], 'lr': self.lrate,  'weight_decay': self.weight_decay}
                ])    
                
                #             
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
                    {'params': adamw_params, 'lr': self.lrate, 'weight_decay': self.weight_decay},
                    {'params': head_params, 'lr': self.lrate_head, 'weight_decay': self.weight_decay}
                ])
                
                # 返回两个优化器，训练循环中需要分别调用
                optimizer = (optimizer_muon, optimizer_adamw) 
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")
        else:
            if self.optimizer == 'sgd':
                optimizer = optim.SGD(
                    trainable_params,
                    momentum=0.9,
                    lr=self.lrate,
                    weight_decay=self.weight_decay
                )
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.Adam(
                    trainable_params,
                    lr=self.lrate,
                    weight_decay=self.weight_decay,
                    betas=(0.9,0.999)
                )
            elif self.args['optimizer'] == 'adamw':
                optimizer = optim.AdamW(
                    trainable_params,
                    lr=self.lrate,
                    weight_decay=self.weight_decay
                )
            elif self.args['optimizer'] == 'Muon':
                # from muon.muon import Muon
                optimizer = optim.Muon(trainable_params, lr=0.001, momentum=0.95)   
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")
        # else:
        #     for name, param in self.network.named_parameters():
        #         if 'classifier_pool' in name:
        #             classifier_params.append(param)
        #         elif 'lora_B' in name:
        #             # 根据层数区分
        #             if 'blocks.0' in name or 'blocks.1' in name or 'blocks.2' in name or \
        #             'blocks.3' in name or 'blocks.4' in name or 'blocks.5' in name:
        #                 lora_b_early.append(param)
        #             else:
        #                 lora_b_deep.append(param)
        
        if self.method == "splitlora" or self.method == "splitlorav6" or self.method == "splitlorav16":
            # scheduler
            if self.scheduler == 'constant':
                scheduler = None
            elif self.scheduler == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
            elif self.scheduler == 'steplr':
                # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestone, gamma=self.lrate_decay)
                scheduler, num_epochs = create_scheduler_v2(optimizer, sched="multistep", num_epochs=10, decay_epochs=1000, decay_milestones=[5, 8],
                                                decay_rate=0.1, min_lr=1e-5, warmup_epochs=0, warmup_lr=1e-5)
            else:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

            return optimizer, scheduler
        else:
                        # scheduler
            if self.scheduler == 'constant':
                scheduler = None
            elif self.scheduler == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
            elif self.scheduler == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestone, gamma=self.lrate_decay)

                assert num_epochs == 10

            else:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

            return optimizer, scheduler
    def get_trainable_params(self, parameters):
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
        return trainable_params
            
    def before_task(self, data_manager):
        #在任务新的任务训练之前需要初始化参数：包括任务数量+1，总的任务数量+加上当前任务的总数，fc的任务数更新。
        self.cur_task += 1
        self.total_classes = self.known_classes + data_manager.get_task_size(self.cur_task)
        self.network.update_fc(self.total_classes)

    def after_task(self):
        self.known_classes = self.total_classes

    def incremental_train(self, data_manager):
        self.build_train_loader(data_manager)
        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader)

    #这个是整个算法的核心
    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
        
        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)

        # to be implemented
        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module
        return
    
    def incremental_test(self, data_manager):
        self.build_test_loader(data_manager)
        y_pred, y_pred_with_task, y_true, y_task_pred, y_task_true = self._test(self.test_loader)
        accy = self._evaluate(y_pred, y_true)
        accy_with_task = self._evaluate(y_pred_with_task, y_true)
        accy_task = round((y_task_pred == y_task_true).sum().item() * 100 / len(y_task_pred), 2)

        return accy, accy_with_task, accy_task
    
    def _test(self, test_loader):
        self.network.eval()
           
        self.training = False        
        
        y_pred, y_true = [], []  # task-agnostic prediction
        y_pred_with_task = []    # task-aware prediction
        y_task_pred, y_task_true = [], []  # prediction of task id

        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                if isinstance(self.network, nn.DataParallel):
                    # print("非直接推理")
                    outputs = self.network.module.interface(inputs)
                    # outputs = 10*torch.sigmoid(outputs)

                else:
                    outputs = self.network.interface(inputs)  # logits
                    # outputs = 10*torch.sigmoid(outputs)
                    # print("直接推理")

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            # task-agnostic prediction
            y_pred.append(predicts.view(-1).cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # prediction of task id
            if self.init_cls == self.increment:
                self.class_num = self.increment
            
            # 预测样本属于哪个任务组
            y_task_pred.append((torch.div(predicts.view(-1), self.class_num, rounding_mode='trunc')).cpu())
            y_task_true.append((torch.div(targets, self.class_num, rounding_mode='trunc')).cpu())
            
            # task-aware prediction
            outputs_with_task = torch.zeros_like(outputs)[:, :self.class_num]
            for idx, i in enumerate(torch.div(targets, self.class_num, rounding_mode='trunc')):
                en, be = self.class_num*i, self.class_num*(i+1)
                # 0,10; 10,20; 20,30

                # print("idx:{}, i:{}, targets shape:{}, self.class_num:{}, outputs[idx, en:be] shape:{}, outputs[idx, en:be]:{}".format(
                #     idx, 
                #     i, 
                #     targets.shape, 
                #     self.class_num, 
                #     outputs[idx, en:be].shape, 
                #     outputs[idx, en:be]
                # ))

                outputs_with_task[idx] = outputs[idx, en:be]

            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (torch.div(targets, self.class_num, rounding_mode='trunc'))*self.class_num
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            
        return (
            np.concatenate(y_pred),
            np.concatenate(y_pred_with_task),
            np.concatenate(y_true),
            torch.cat(y_task_pred),
            torch.cat(y_task_true)
        )

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred, y_true, self.known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret
    
    def freeze_network(self):
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
        
    @abstractmethod
    def _train_function(self, train_loader, optimizer, scheduler):
        pass

