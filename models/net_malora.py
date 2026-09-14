"""
Network wrapper for MALoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_malora import create_vit_malora


class Net_MALoRA(nn.Module):
    """
    Network with shared delta_W for all tasks.
    """
    def __init__(self, args):
        super().__init__()

        # ViT backbone with MALoRA attention
        self.image_encoder = create_vit_malora(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            r=args.get("rank", 8),
            pretrained=args.get("pretrained", True)
        )

        # Classifier pool for incremental learning
        self.classifier_pool = nn.ModuleList()
        self.init_cls_num = args.get("init_cls", 10)
        self.inc_cls_num = args.get("increment", 10)
        self.task_num = args.get("sessions", 10)
        self._cur_task = -1

        # Initialize first classifier
        self._add_classifier()

    def _add_classifier(self):
        """Add a new classifier for the next task."""
        if self._cur_task == -1:
            num_classes = self.init_cls_num
        else:
            num_classes = self.inc_cls_num
        
        new_classifier = nn.Linear(768, num_classes, bias=True)
        self.classifier_pool.append(new_classifier)

    def update_fc(self):
        """Update classifier for new task."""
        self._cur_task += 1
        if self._cur_task >= len(self.classifier_pool):
            self._add_classifier()

    def forward(self, x, get_cur_feat=False, inference=False):
        """
        Forward pass.
        Args:
            x: input images [B, 3, H, W]
            get_cur_feat: whether to collect features for manifold estimation
            inference: if True, use all classifiers; if False, only current task
        Returns:
            logits: [B, total_classes] or [B, inc_cls_num]
        """
        features = self.image_encoder(x, get_cur_feat=get_cur_feat)

        if inference:
            # Use all learned classifiers
            logits_list = []
            for classifier in self.classifier_pool:
                logits_list.append(classifier(features))
            logits = torch.cat(logits_list, dim=1)
        else:
            # Use only current task classifier
            logits = self.classifier_pool[self._cur_task](features)

        return logits

    def extract_features(self, x):
        """Extract features without classification."""
        return self.image_encoder(x)

    def get_attention_modules(self):
        """Get all MALoRA attention modules."""
        modules = []
        for block in self.image_encoder.blocks:
            if hasattr(block.attn, 'delta_W'):
                modules.append(block.attn)
        return modules

    def get_manifold_info(self):
        """Get factor matrices for Riemannian optimization."""
        info = []
        for module in self.get_attention_modules():
            A, B, S = module.get_manifold_info()
            info.append((A, B, S))
        return info