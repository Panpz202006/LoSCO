import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from models.net_sdlora_e import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency
from models.vit_sdlora_e import Attention_LoRA


class SDLoRAo(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        self.network = Net(args)
        self.ortho_loss_type = args.get('ortho_loss_type', False)
        self.ortho_loss_weight = args.get("ortho_loss_weight", 0.1)
        self.R = args['R']
        self.lambda_o = args.get("lambda_o", 0.01)   # 核范数权重
        self.lambda_s = args.get("lambda_s", 0.001)  # L1范数权重
        self.mu = args.get("mu", 0.1)                # 保真项权重
        self.ortho_loss_typev2 = args.get('ortho_loss_typev2', True)

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        encoder_params = self.network.image_encoder.parameters()
        cls_params = [p for p in self.network.classifier_pool.parameters() if p.requires_grad==True]

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
        
        encoder_params = {'params': encoder_params, 'lr': self.lrate, 'weight_decay': self.weight_decay}
        cls_params = {'params': cls_params, 'lr': self.fc_lrate, 'weight_decay': self.weight_decay}

        network_params = [encoder_params, cls_params]
        optimizer, scheduler = self.build_optimizer(network_params)
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module
        return
    
    def _train_function(self, train_loader, optimizer, scheduler):
        print("lambda_o, lambda_s, mu",self.lambda_o, self.lambda_s, self.mu)
        if self.cur_task >= 1:
            ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
            import math
            ema_decay = math.pow(self.R, 1/ (len(train_loader)*self.epochs))

            self.ema_params = {}
            for name, param in self.network.named_parameters():
                if any(key in name for key in ema_keys):
                    self.ema_params[name] = param.data.clone().detach()
                    print(f"EMA tracking: {name}")
                    
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            self.training = True
            losses = 0.
            correct, total = 0, 0
            ortho_loss_sum = 0.0
            ortho_batches = 0

            # v2 三个正则分量(加权后)的逐batch累加, 用于观察是否下降
            v2_nuc_sum = 0.0
            v2_l1_sum = 0.0
            v2_fid_sum = 0.0
            v2_batches = 0


            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self.known_classes
                
                logits = self.network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

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

                            proj_loss = 0.0
                            for t in range(self.cur_task + 1):
                                cross = B_shared.weight.T @ B_tasks[t].weight  # [r, r]
                                proj_loss += torch.norm(cross, p='fro') ** 2
                            ortho_loss += proj_loss / (self.cur_task + 1)
                            layer_count += 1

                    if layer_count > 0:
                        ortho_loss = ortho_loss / layer_count
                    loss += self.ortho_loss_weight * ortho_loss
                    ortho_loss_sum += ortho_loss.item()

                    ortho_batches += 1

                # if self.ortho_loss_typev2 and self.cur_task >= 1:
                #     nuclear_loss = 0.0
                #     l1_loss = 0.0
                #     fidelity_loss = 0.0
                    
                #     for name, param in self.network.named_parameters():
                #         if name in self.ema_params:
                #             # 核范数近似
                #             nuclear_loss += torch.norm(param.data, p='fro')
                #             # L1稀疏
                #             l1_loss += torch.norm(param.data, p=1)
                #             # 保真项
                #             Z_o = self.ema_params[name]
                #             diff = param.data - Z_o
                #             fidelity_loss += torch.norm(diff, p='fro') ** 2
                    
                #     loss += self.lambda_o * nuclear_loss + self.lambda_s * l1_loss + (self.mu / 2) * fidelity_loss

                if self.ortho_loss_typev2 and self.cur_task >= 1:

                    nuclear_loss = 0.0
                    l1_loss = 0.0
                    fidelity_loss = 0.0

                    shared_count = 0
                    task_count = 0


                    for name, param in self.network.named_parameters():

                        #这里计算共享适配器
                        if name in self.ema_params:

                            # Current shared adapter
                            Z_o = param

                            # Low-rank regularization
                            nuclear_loss += torch.norm(Z_o, p='fro')

                            # Historical shared knowledge
                            Z_o_ema = self.ema_params[name].detach()

                            # Fidelity / stability constraint
                            fidelity_loss += torch.norm(
                                Z_o - Z_o_ema,
                                p='fro'
                            ) ** 2
                            shared_count += 1

                        #这里计算 任务特定适配器
                        # 注意: named_parameters() 的 name 是完整路径(如
                        #   ...attn.lora_B_k.3.weight), 直接与短串 "lora_B_k" 判等永远为 False,
                        #   之前导致 L1 项恒为 0 (lambda_s 无效)。改为子串匹配, 且只对当前任务的
                        #   lora_B_k / lora_B_v (该任务训练时被解冻、有梯度的部分) 施加 L1 稀疏。
                        cur_task_sfx = f".{self.cur_task}.weight"
                        for _tag in ("lora_B_k", "lora_B_v"):
                            if _tag + cur_task_sfx in name:
                                l1_loss += torch.norm(param, p=1)
                                task_count += 1
                                break

                    # ============================================================
                    # 3. Normalize by parameter groups
                    # ============================================================
                    if shared_count > 0:
                        nuclear_loss = nuclear_loss / shared_count
                        fidelity_loss = fidelity_loss / shared_count

                    if task_count > 0:
                        l1_loss = l1_loss / task_count

                    # ============================================================
                    # 4. Unified objective
                    # ============================================================
                    loss += (
                        self.lambda_o * nuclear_loss
                        + self.lambda_s * l1_loss
                        + (self.mu) * fidelity_loss
                    )

                    v2_nuc_sum += self.lambda_o * nuclear_loss
                    v2_l1_sum += self.lambda_s * l1_loss
                    v2_fid_sum += self.mu * 0.5* fidelity_loss
                    v2_batches += 1


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.cur_task >= 1:      
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            self.ema_params[name] = (ema_decay * self.ema_params[name] + (1 - ema_decay) * param.data)

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if v2_batches > 0:
                print('[v2loss] Task {}, Epoch {}/{} => Loss {:.3f} | lo*nuc {:.4f}  ls*l1 {:.4f}  mu*fid {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader),
                    v2_nuc_sum / v2_batches, v2_l1_sum / v2_batches, v2_fid_sum / v2_batches))

            # info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            #     self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)

            if ortho_batches > 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Ortho {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc,
                    ortho_loss_sum / ortho_batches)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)

            prog_bar.set_description(info)

        logging.info(info)
        if self.cur_task >= 1:
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if name in self.ema_params:
                        param.data.copy_(self.ema_params[name])

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_A_k{target_suffix}",
            f"lora_A_v{target_suffix}",
            f"lora_B_k{target_suffix}",
            f"lora_B_v{target_suffix}",
            f"scaling_factor",
            "Lora_shared",

        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    