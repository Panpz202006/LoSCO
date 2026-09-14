import torch
import torch.nn as nn
import numpy as np

from models.vit_cllora import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from models.vit_cllora import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.modules.linears import CosineLinear


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant)
    pretrained_cfg['num_classes'] = 0
    
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


class ViT(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64, 
            msa=None, shared_pos=None, specific_pos=None):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank, 
            msa=msa, shared_pos=shared_pos, specific_pos=specific_pos)
        
    def forward(self, x, task_id=None, shared=False, use_new=False):
        x = self.patch_embed(x)  
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        
        if shared:
            x = self.forward_shared(x, use_new)
        else:
            x= self.forward_feature(x, task_id, use_new)

        x = self.norm(x)
            
        return x
    
    def forward_shared(self, x, use_new=False):
        # forward only with shared-blocks
        for pos in self.shared_pos:
            x = self.blocks[pos](x, use_new=use_new)
        return x
    
    def forward_feature(self, x, task_id, use_new=False):
        block_weight = self.block_weights[task_id]
        for i, blk in enumerate(self.blocks):
            if i in self.specific_pos:
                blk_weight = block_weight[self.specific_pos.index(i)]
            else:
                blk_weight = None
            x = blk(x, task_id, blk_weight, use_new)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self.fc = None

        model_kwargs = dict(patch_size=16, embed_dim=768, 
                            depth=12, num_heads=12,
                            n_tasks=args["sessions"], rank=args["rank"],
                            msa=args["msa"], shared_pos=args["shared_pos"], specific_pos=args["specific_pos"])
        
        self.image_encoder = _create_vision_transformer(args["load"], pretrained=True, **model_kwargs)

        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()
        
        self._cur_task = -1
        self._device = args['device'][0]

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim
    
    def forward(self, image, task_id):
        image_features = self.image_encoder(image, task_id, use_new=True)[:,0,:]
        logits = self.proxy_fc(image_features)

        return {
            'logits': logits,
            'features': image_features
        }
    
    def forward_kd(self, image):
        image_features = self.image_encoder(image, shared=True, use_new=True)[:,0,:]
        image_features_teacher = self.image_encoder(image, shared=True, use_new=False)[:,0,:]
        logits = self.proxy_fc(image_features)
        logits_teacher = self.proxy_fc(image_features_teacher)

        return logits, logits_teacher

    def interface(self, image):
        logits = []
        for task_id in range(self._cur_task + 1):
            image_features = self.image_encoder(image, task_id, use_new=True)[:,0,:]
            logits.append(self.fc.forward_all(image_features, task_id, inc=self.increment, feature_dim=self.feature_dim))
        logits = torch.cat(logits, dim=1)
        return logits

        # features = []
        # for task_id in range(self._cur_task + 1):
        #     image_feature = self.image_encoder(image, task_id, use_new=True)  # [batch, 197, 768]
        #     features.append(image_feature)
        # output = torch.Tensor().to(features[0].device)
        # for x in features:
        #     cls = x[:, 0, :]  # [batch, 768]
        #     output = torch.cat((output, cls), dim=1)  # [batch, 768*num_task]
        # logits = self.fc.forward_diagonal(output, self._cur_task, inc=self.increment, feature_dim=self.feature_dim)
        # return logits
    

    def update_fc(self, nb_classes):
        """
        Expand the classifier to accommodate new classes.
        Inputs:
            - nb_classes: total number of classes seen so far
        Updates:
            - proxy_fc: classifier for current task
            - fc: classifier for all seen tasks
        """
        self._cur_task += 1
        
        # create proxy classifier (for current task)
        cur_classes = self.init_cls if self._cur_task == 0 else self.increment
        self.proxy_fc = self.generate_fc(self.feature_dim, cur_classes).to(self._device)
        
        # create final classifier (for all tasks)
        new_fc = self.generate_fc(self.feature_dim * (self._cur_task+1), nb_classes).to(self._device)
        new_fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_classes = self.fc.out_features
            new_fc.sigma.data.copy_(self.fc.sigma.data)
            new_fc.weight.data[:old_classes, :-self.feature_dim].copy_(self.fc.weight.data)

        self.fc = new_fc

    def replace_fc(self, train_loader):
        """
        Replace the weights of the final classifier with prototypes computed from the training data.
        """
        self.image_encoder.eval()

        with torch.no_grad():
            # compute prototype for current dat via each task branch
            for task_id in range(self._cur_task + 1):

                feature_list, target_list = [], []
                for i, (_, inputs, targets) in enumerate(train_loader): 
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    feature = self.image_encoder(inputs, task_id, use_new=True)[:,0,:]
                    feature_list.append(feature.cpu())
                    target_list.append(targets.cpu())

                feature_list = torch.cat(feature_list, dim=0)
                target_list = torch.cat(target_list, dim=0)

                class_list = np.unique(target_list)
                for cls in class_list:
                    data_idx = (target_list == cls).nonzero().squeeze(-1)
                    feature = feature_list[data_idx]
                    proto = feature.mean(0)
                    self.fc.weight.data[cls, task_id*self.feature_dim : (task_id+1)*self.feature_dim] = proto
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def save_old_shared_lora(self):
        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.save_old_shared_lora()

