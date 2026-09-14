import logging
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from models.net_cllora_e import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.function import KD_loss, Orthogonality_loss


class CLLoRAe(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        self.network = Net(args)

        # cllora
        self.msa = args['msa']
        self.shared_pos = args['shared_pos']
        self.kd_ratio = args['kd_ratio']
        self.temperature = args['temperature']
        self.R = args.get('R', 0.5)  # EMA decay base (per total number of steps)

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self.build_train_loader_for_protonet(data_manager)
        self.network.replace_fc(self.train_loader_for_protonet)

    def build_train_loader_for_protonet(self, data_manager):
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self.known_classes, self.total_classes),
            source='train', mode='test')
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)
        
    def after_task(self):
        super().after_task()
        self.network.save_old_shared_lora()

    # def _train_function(self, train_loader, optimizer, scheduler):
    #     prog_bar = tqdm(range(self.epochs))
    #     for _, epoch in enumerate(prog_bar):
    #         self.network.train()
    #         losses = 0.
    #         correct, total = 0, 0
            
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             mask = (targets >= self.known_classes).nonzero().view(-1)
    #             inputs = torch.index_select(inputs, 0, mask)
    #             targets = torch.index_select(targets, 0, mask)-self.known_classes
                
    #             logits = self.network(inputs, self.cur_task)['logits']
    #             loss = F.cross_entropy(logits, targets)

    #             if self.cur_task > 0:
    #                 # Knowledge Distillation
    #                 logits_kd, logits_teacher = self.network.forward_kd(inputs)
    #                 loss_kd = self.kd_ratio * KD_loss(logits_kd, logits_teacher, T=self.temperature)

    #                 optimizer.zero_grad()
    #                 loss_kd.backward()

    #                 # Gradient Reassignment
    #                 with torch.no_grad():
    #                     for pos in self.shared_pos:
    #                         for k, use_msa in enumerate(self.msa):
    #                             if not use_msa:
    #                                 continue
    #                             proj = ['q', 'k', 'v'][k]
    #                             old_B = next(iter_attn_lora_B(self.network, pos, proj, use_new=False)).detach()
    #                             scale = torch.norm(old_B, dim=1)
    #                             scale = len(scale) * scale / torch.sum(scale)
    #                             new_B = next(iter_attn_lora_B(self.network, pos, proj, use_new=True))
    #                             if new_B.grad is not None:
    #                                 new_B.grad.mul_(scale.unsqueeze(1))   
    #                 optimizer.step()

    #                 # Block-wise Orthogonality
    #                 blk_weights = self.network.image_encoder.block_weights
    #                 loss_orth = Orthogonality_loss(blk_weights[:self.cur_task], blk_weights[self.cur_task])
    #                 loss += 0.0001 * loss_orth
                
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             losses += loss.item()
    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #             total += len(targets)

    #         scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    #         info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
    #             self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
    #         prog_bar.set_description(info)

    #     logging.info(info)
    def _train_function(self, train_loader, optimizer, scheduler):
        # EMA for task-shared LoRA (Lora_shared), only from the 2nd task on
        if self.cur_task > 0:
            ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
            import math
            ema_decay = math.pow(self.R, 1 / (len(train_loader) * self.epochs))

            self.ema_params = {}
            for name, param in self.network.named_parameters():
                if any(key in name for key in ema_keys):
                    self.ema_params[name] = param.data.clone().detach()
                    print(f"EMA tracking: {name}")

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes
                
                logits = self.network(inputs, self.cur_task)['logits']
                loss = F.cross_entropy(logits, targets)

                # ========== 累积所有损失 ==========
                total_loss = loss
                
                if self.cur_task > 0:
                    # Knowledge Distillation
                    logits_kd, logits_teacher = self.network.forward_kd(inputs)
                    loss_kd = self.kd_ratio * KD_loss(logits_kd, logits_teacher, T=self.temperature)
                    total_loss += loss_kd
                    
                    # Block-wise Orthogonality
                    blk_weights = self.network.image_encoder.block_weights
                    loss_orth = Orthogonality_loss(blk_weights[:self.cur_task], blk_weights[self.cur_task])
                    total_loss += 0.0001 * loss_orth
                
                # ========== 只调用一次 backward 和 step ==========
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient Reassignment（在 backward 之后，step 之前修改梯度）
                if self.cur_task > 0:
                    with torch.no_grad():
                        for pos in self.shared_pos:
                            for k, use_msa in enumerate(self.msa):
                                if not use_msa:
                                    continue
                                proj = ['q', 'k', 'v'][k]
                                old_B = next(iter_attn_lora_B(self.network, pos, proj, use_new=False)).detach()
                                scale = torch.norm(old_B, dim=1)
                                scale = len(scale) * scale / torch.sum(scale)
                                new_B = next(iter_attn_lora_B(self.network, pos, proj, use_new=True))
                                if new_B.grad is not None:
                                    new_B.grad.mul_(scale.unsqueeze(1))
                
                optimizer.step()

                # ========== EMA update (after each step) ==========
                if self.cur_task > 0:
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            self.ema_params[name] = (ema_decay * self.ema_params[name] + (1 - ema_decay) * param.data)

                # ========== 日志记录 ==========
                losses += total_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)
        if self.cur_task > 0:
            # 将 EMA 累积的共享知识写回 Lora_shared
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if name in self.ema_params:
                        param.data.copy_(self.ema_params[name])

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"lora_q.lora_B",
            f"lora_v.lora_B",
            f"lora_q{target_suffix}.lora",
            f"lora_v{target_suffix}.lora",
            f"proxy_fc",
            "Lora_shared",
        ]
        for name, param in self.network.named_parameters():
            trainable = any(key in name for key in unfrozen_keys)
            if name.endswith(f"block_weights{target_suffix}"):
                trainable = True
            param.requires_grad_(trainable)
    

def iter_attn_lora_B(model, pos, proj, use_new):
    block_prefix = f"image_encoder.blocks.{pos}.attn"
    proj_key = f"lora_{proj}_old" if not use_new else f"lora_{proj}"

    for name, param in model.named_parameters():
        if (
            name.startswith(block_prefix)
            and proj_key in name
            and name.endswith("lora_B.weight")
        ):
            yield param