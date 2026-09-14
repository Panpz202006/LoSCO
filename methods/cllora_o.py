import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from models.net_cllora_e import Net
from models.vit_cllora_e import Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.function import KD_loss, Orthogonality_loss


class CLLoRAo(BaseLearner):

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

        # lora-level orthogonality between task-shared Lora_shared and task-specific LoRA
        self.ortho_loss_type = args.get('ortho_loss_type', True)
        self.ortho_loss_weight = args.get('ortho_loss_weight', 0.1)

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
            ortho_loss_sum = 0.0
            ortho_batches = 0

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

                    # LoRA-level Orthogonality: task-shared Lora_shared vs. task-specific LoRA
                    if self.ortho_loss_type:
                        loss_lora_ortho = lora_orthogonality_loss(self.network, self.cur_task)
                        total_loss += self.ortho_loss_weight * loss_lora_ortho
                        ortho_loss_sum += loss_lora_ortho.item()
                        ortho_batches += 1

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

            if ortho_batches > 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Ortho {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc,
                    ortho_loss_sum / ortho_batches)
            else:
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


def lora_orthogonality_loss(network, cur_task):
    """
    Push the task-shared Lora_shared (EMA-updated) to be orthogonal to the
    task-specific LoRA of each trained task, following sdlora_o/ewclora_o.

    - specific blocks: per-task ModuleList -> constrain vs. lora_{k/v}[t], t in [0, cur_task]
    - shared blocks:   single continuously-updated LoRA -> constrain vs. lora_{k/v}
      (current) and lora_{k/v}_old (snapshot before the current task)
    """
    ortho_loss = 0.0
    layer_count = 0

    for module in network.modules():
        if not isinstance(module, Attention_LoRA):
            continue

        for proj in ('k', 'v'):
            shared_B = getattr(module, f'Lora_shared_B_{proj}')
            targets = []
            if module.is_specific:
                lora_list = getattr(module, f'lora_{proj}', None)
                if isinstance(lora_list, nn.ModuleList):
                    for t in range(cur_task + 1):
                        lora_t = lora_list[t]
                        if lora_t is not None:
                            targets.append(lora_t.lora_B)
            elif module.is_shared:
                lora_cur = getattr(module, f'lora_{proj}', None)
                if lora_cur is not None:
                    targets.append(lora_cur.lora_B)
                lora_old = getattr(module, f'lora_{proj}_old', None)
                if lora_old is not None:
                    targets.append(lora_old.lora_B)

            proj_loss = 0.0
            n_terms = 0
            for tb in targets:
                cross = shared_B.weight.T @ tb.weight  # [r, r]
                proj_loss += torch.norm(cross, p='fro') ** 2
                n_terms += 1

            if n_terms > 0:
                ortho_loss += proj_loss / n_terms
                layer_count += 1

    if layer_count > 0:
        ortho_loss = ortho_loss / layer_count
    else:
        ref = next((p for p in network.parameters() if 'Lora_shared_B' in p), None)
        ortho_loss = torch.zeros((), device=ref.device) if ref is not None else torch.tensor(0.0)
    return ortho_loss
