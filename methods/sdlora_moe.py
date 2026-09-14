import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from models.net_sdlora_moe import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class SDLoRA_MOE(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        self.network = Net(args)

    

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()   #不执行冻结，一次可以选择10个expert
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
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            self.training = True
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self.known_classes
                
                # 只调用一次
                outputs = self.network(inputs)
                logits = outputs['logits']
                total_load_balance_loss = outputs['total_load_balance_loss']
                total_etf_loss = outputs['total_etf_loss']

                loss = F.cross_entropy(logits, targets)
                loss += total_load_balance_loss*0.01
                loss += total_etf_loss*0.01
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def freeze_network(self):
        unfrozen_keys = []
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_A_k{target_suffix}",
            f"lora_A_v{target_suffix}",
            f"lora_B_k{target_suffix}",
            f"lora_B_v{target_suffix}",
            f"scaling_factor",
        ]

        unfrozen_keys.append(f"classifier_pool{target_suffix}")
        # self.moe_list = [8, 9, 10, 11] , 从输入端获取
        for j in self.moe_list:
            target_suffix_bloock = f".{j}"
            for i in range (10): #这里相当于有10个专家， 这十个专家一起参与训练。
                # if 
                target_suffix = f".{i}"
                unfrozen_keys.append(f"{target_suffix_bloock}.attn.lora_A_k{target_suffix}")
                unfrozen_keys.append(f"{target_suffix_bloock}.attn.lora_A_v{target_suffix}")
                unfrozen_keys.append(f"{target_suffix_bloock}.attn.lora_B_k{target_suffix}")
                unfrozen_keys.append(f"{target_suffix_bloock}.attn.lora_B_v{target_suffix}")
                unfrozen_keys.append(f"scaling_factor")
            unfrozen_keys.append(f"{target_suffix_bloock}.attn.gate")

        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    