import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from models.net_ewclora_e import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency

from models.vit_ewclora_e import Attention_LoRA

class EWCLoRAo(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        self.network = Net(args)

        # ewclora
        self.gamma = args["gamma"]
        self.ewc_weight = args["lambda"]
        self.omega_W = []  # Importance matrix
        self.count_updates = 0

        self.R = args['R']

        self.ortho_loss_type = args.get('ortho_loss_type', False)
        self.ortho_loss_weight = args.get("ortho_loss_weight", 0.1)

               # v2 正则化参数（从 SDLoRAo 移植）
        self.ortho_loss_typev2 = args.get('ortho_loss_typev2', True)
        self.lambda_o = args.get("lambda_o", 0.01)   # 核范数权重
        self.lambda_s = args.get("lambda_s", 0.001)  # L1范数权重
        self.mu = args.get("mu", 0.1)                # 保真项权重

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
            losses = 0.
            correct, total = 0, 0

            ortho_loss_sum = 0.0
            ortho_batches = 0

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

                    if self.ortho_loss_type and self.cur_task >= 1:
                        ortho_loss = 0.0
                        layer_count = 0

                        for module in self.network.modules():
                            if not isinstance(module, Attention_LoRA):
                                continue

                            for proj in ('k', 'v'):
                                if proj == 'k':
                                    B_accum = module.lora_B_k        # 之前所有任务累加后的B
                                    B_new = module.lora_new_B_k      # 当前任务的B
                                    B_shared = module.Lora_shared_B_k
                                else:
                                    B_accum = module.lora_B_v
                                    B_new = module.lora_new_B_v
                                    B_shared = module.Lora_shared_B_v

                                # 对应 splitlorav6 中 for t in range(self.cur_task + 1)
                                # 每个任务单独约束，再按任务数平均，避免 loss 随任务数增长
                                proj_loss = 0.0
                                n_terms = 0
                                for Bt in (B_accum, B_new):
                                    cross = B_shared.weight.T @ Bt.weight  # [r, r]
                                    proj_loss += torch.norm(cross, p='fro') ** 2
                                    n_terms += 1

                                ortho_loss += proj_loss / n_terms
                                layer_count += 1

                        # 按层平均，避免损失随层数/投影数线性增长
                        if self.cur_task >=1 and layer_count > 0:
                            ortho_loss = ortho_loss / layer_count
                        loss += self.ortho_loss_weight * ortho_loss

                        ortho_loss_sum += ortho_loss.item()

                        ortho_batches += 1

                    if self.ortho_loss_typev2 and self.cur_task >= 1:
                        nuclear_loss = 0.0
                        l1_loss = 0.0
                        fidelity_loss = 0.0

                        # v2 三个正则分量(加权后)的逐batch累加
                        v2_nuc_sum = 0.0
                        v2_l1_sum = 0.0
                        v2_fid_sum = 0.0
                        v2_batches = 0
                        shared_count = 0
                        task_count = 0
                        
                        for name, param in self.network.named_parameters():
                            # 计算共享适配器 (核范数 + 保真项)
                            if name in self.ema_params:
                                Z_o = param
                                nuclear_loss += torch.norm(Z_o, p='fro')
                                Z_o_ema = self.ema_params[name].detach()
                                fidelity_loss += torch.norm(Z_o - Z_o_ema, p='fro') ** 2
                                shared_count += 1
                            
                            # 计算任务特定适配器 (L1稀疏)
                            cur_task_sfx = f".{self.cur_task}.weight"
                            for _tag in ("lora_new_B_k", "lora_new_B_v"):
                                if _tag + cur_task_sfx in name:
                                    l1_loss += torch.norm(param, p=1)
                                    task_count += 1
                                    break
                        
                        # Normalize by parameter groups
                        if shared_count > 0:
                            nuclear_loss = nuclear_loss / shared_count
                            fidelity_loss = fidelity_loss / shared_count
                        if task_count > 0:
                            l1_loss = l1_loss / task_count
                        
                        # Unified objective
                        loss += (
                            self.lambda_o * nuclear_loss
                            + self.lambda_s * l1_loss
                            + self.mu *0.5 *fidelity_loss
                        )
                        
                        v2_nuc_sum += self.lambda_o * nuclear_loss
                        v2_l1_sum += self.lambda_s * l1_loss
                        v2_fid_sum += self.mu * fidelity_loss
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

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            #     self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)

            if ortho_batches > 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Ortho {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc,
                    ortho_loss_sum / ortho_batches)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            if self.cur_task>=1 and v2_batches > 0:
                print('[v2loss] Task {}, Epoch {}/{} => Loss {:.3f} | lo*nuc {:.4f}  ls*l1 {:.4f}  mu*fid {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader),
                    v2_nuc_sum / v2_batches, v2_l1_sum / v2_batches, v2_fid_sum / v2_batches))
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
            f"lora_new_A_k",
            f"lora_new_A_v",
            f"lora_new_B_k",
            f"lora_new_B_v",
                        "Lora_shared",


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