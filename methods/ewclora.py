import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from models.net_ewclora import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class EWCLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        self.network = Net(args)

        # ewclora
        self.gamma = args["gamma"]
        self.ewc_weight = args["lambda"]
        self.omega_W = []  # Importance matrix
        self.count_updates = 0
        
    def after_task(self):
        super().after_task()

        # Compute Fisher Information Matrix
        print("=== Update Importance Matrix ===")
        self.count_updates += 1
        fisher = FisherComputer(self.cur_task, self.network, self.train_loader, 
                                self.increment, F.cross_entropy, self.device)
        fisher_W = fisher.compute(max_batches=None)

        omega_W_bk = self.omega_W[:]
        self.omega_W = []

        new_a_params = filter(lambda p: getattr(p, '_is_new_a', False), self.network.parameters())
        new_b_params = filter(lambda p: getattr(p, '_is_new_b', False), self.network.parameters())
        for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
            if len(omega_W_bk) != 0:
                self.omega_W.append(self.gamma * omega_W_bk[idx] + fisher_W[idx])
            else:
                self.omega_W.append(fisher_W[idx])

        self.network.accumulate_and_reset_lora()

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
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                # 这个是获取数据集，包括样本和标签。
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 1. 找出新类别的索引
                mask = (targets >= self.known_classes).nonzero().view(-1)
                # 2. 根据索引筛选输入样本
                inputs = torch.index_select(inputs, 0, mask)
                # 3. 根据索引筛选标签，并重新映射为从0开始的标签
                targets = torch.index_select(targets, 0, mask)-self.known_classes

                logits = self.network(inputs, use_new=True)['logits']
                loss = F.cross_entropy(logits, targets)

                # regularization loss
                if self.count_updates != 0:
                    new_a_params = filter(lambda p: getattr(p, '_is_new_a', False), self.network.parameters())
                    new_b_params = filter(lambda p: getattr(p, '_is_new_b', False), self.network.parameters())
                    ewc_loss = 0.
                    for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
                        delta_W = p_b @ p_a
                        ewc_term = self.omega_W[idx].type(torch.float32).to(self.device) * (delta_W ** 2)
                        ewc_loss += torch.sum(ewc_term)
                        
                    weighted_ewc_loss = self.ewc_weight/2. * ewc_loss
                    loss += weighted_ewc_loss
                    
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
            f"lora_new_A_k",
            f"lora_new_A_v",
            f"lora_new_B_k",
            f"lora_new_B_v",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))


class FisherComputer:
    def __init__(self, task_id, network, dataloader, increment, criterion, device=torch.device('cpu')):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.increment = increment
        self.criterion = criterion
        self.device = device

        self.task_id = task_id
        self.fisher_W = []
        self._init_fisher_storage()

    def compute(self, max_batches=None):
        self.model.eval()
        num_samples = 0
        
        for i, (_, inputs, targets) in enumerate(tqdm(self.dataloader, desc="Computing Fisher")):
            if max_batches and i >= max_batches:
                break
            # Empirical Fisher
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            logits = self.model.forward(inputs, use_new=True, register_hook=True)['logits']
            targets = targets - self.task_id * self.increment
            loss = self.criterion(logits, targets)
            loss.backward()

            batch_size = inputs.size(0)
            num_samples += batch_size

            idx = 0
            for module in self.model.modules():
                if hasattr(module, 'delta_w_k_new_grad'):
                    grad_k = module.delta_w_k_new_grad
                    if grad_k is not None:
                        self.fisher_W[idx] += (grad_k.detach() ** 2) * batch_size
                    idx += 1
                if hasattr(module, 'delta_w_v_new_grad'):
                    grad_v = module.delta_w_v_new_grad
                    if grad_v is not None:
                        self.fisher_W[idx] += (grad_v.detach() ** 2) * batch_size
                    idx += 1                    
        self.fisher_W = [fw / num_samples for fw in self.fisher_W]

        return self.fisher_W
    
    def _init_fisher_storage(self):
        for module in self.model.modules():
            if hasattr(module, 'lora_new_B_k') and hasattr(module, 'lora_new_A_k'):
                delta_w_k_new = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
                self.fisher_W.append(torch.zeros_like(delta_w_k_new))
            if hasattr(module, 'lora_new_B_v') and hasattr(module, 'lora_new_A_v'):
                delta_w_v_new = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
                self.fisher_W.append(torch.zeros_like(delta_w_v_new))


def _solve_sylvester_cg(B, A, GB, GA, eps=1e-6, tol=1e-6, maxiter=200, verbose=False):
    """
    (B B^T) G + G (A^T A) = GB A + B GA
    B: (m, r)
    A: (r, n)
    GB: (m, r)
    GA: (r, n)
    """
    m, n = B.shape[0], A.shape[1]
    R = GB @ A + B @ GA
    mn = m * n

    def matvec(vec):
        G = vec.view(m, n)
        MG = B @ (B.T @ G)
        GN = (G @ A.T) @ A
        out = MG + GN
        if eps != 0.0:
            out = out + eps * G
        return out.reshape(mn)

    b = R.reshape(mn)

    x_vec = torch.zeros_like(b)
    r_vec = b - matvec(x_vec)
    p = r_vec.clone()
    rsold = torch.dot(r_vec, r_vec)

    for k in range(maxiter):
        Ap = matvec(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-30)
        x_vec = x_vec + alpha * p
        r_vec = r_vec - alpha * Ap
        rsnew = torch.dot(r_vec, r_vec)
        if verbose:
            print(f"iter={k}, residual={rsnew.sqrt().item():.3e}")
        if torch.sqrt(rsnew) <= tol * torch.sqrt(torch.dot(b, b)):
            break
        beta = rsnew / (rsold + 1e-30)
        p = r_vec + beta * p
        rsold = rsnew

    return x_vec.view(m, n)