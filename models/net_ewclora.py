import torch
import torch.nn as nn

from models.vit_ewclora import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from models.vit_ewclora import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant)
    
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
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, use_new, register_hook=False):
        x = self.patch_embed(x)  
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, use_new, register_hook=register_hook)

        x = self.norm(x)
        
        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, 
                            depth=12, num_heads=12, 
                            n_tasks=args["sessions"], rank=args["rank"])

        self.image_encoder = _create_vision_transformer(args["load"], pretrained=True, **model_kwargs)
        self.class_num = args["init_cls"]

        # Linear classifier for each task
        self.classifier_pool = nn.ModuleList([
            nn.Linear(768, self.class_num, bias=True)
            for i in range(args["sessions"])
        ])

        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self._cur_task = -1

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_features(self, image, task=None):
        if task == None:
            image_features = self.image_encoder(image, self._cur_task)
        else:
            image_features = self.image_encoder(image, task)
        
        image_features = image_features[:,0,:]  # [128,768]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def accumulate_and_reset_lora(self):
        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.accumulate_and_reset_lora()

    def forward(self, image, use_new, fc_only=False, register_hook=False):
        if fc_only:
            fc_outs = []
            for ti in range(self._cur_task + 1):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features = self.image_encoder(image, use_new=use_new, register_hook=register_hook)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in [self.classifier_pool[self._cur_task]]:
            logits.append(classifier(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, use_new=True):
        logits = []
        image_features = self.image_encoder(image, use_new=use_new)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in self.classifier_pool[: self._cur_task+1]:
            logits.append(classifier(image_features))

        logits = torch.cat(logits, dim=1)
        return logits
    
    def update_fc(self, nb_classes):
        self._cur_task +=1
