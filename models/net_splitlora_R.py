import torch
import torch.nn as nn

#splitlora也可以直接使用vitlora定义好的进行就OK了
from models.vit_splitlora_R import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from models.vit_splitlora_R import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn


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


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False, get_weight=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat, get_weight=get_weight)

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

        # ============ 控制变量：对齐随机种子 ============
        # 用途：比较「用 gate / 不用 gate」时，保证 LoRA 和分类头初值完全一致，
        #       避免 gate 构造时消耗随机数导致的初始化漂移。
        # 用法：config 里加 "align_rng_seed": 1993，两个版本都设成同一个值。
        #       默认 None（关闭）＝ 完全保持原有初始化行为。
        align_seed = args.get('align_rng_seed', None)
        if align_seed is not None:
            torch.manual_seed(align_seed)
        # =================================================

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

    def extract_vector(self, image, task=None):
        if task == None:
            image_features = self.image_encoder(image, self._cur_task)
        else:
            image_features = self.image_encoder(image, task)
        
        image_features = image_features[:,0,:]  # [128,768]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False, get_weight=False):
        if fc_only:
            fc_outs = []
            for ti in range(self._cur_task+1):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features = self.image_encoder(image, task_id=self._cur_task, get_feat=get_feat, get_cur_feat=get_cur_feat, get_weight=get_weight)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in [self.classifier_pool[self._cur_task]]:
            logits.append(classifier(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    # 这个是推理函数
    def interface(self, image, task_id=None):
        logits = []
        image_features = self.image_encoder(image, task_id=self._cur_task if task_id is None else task_id)
        image_features = image_features[:,0,:]
        #在 ViT 中，第0个位置是 CLS token，用于分类任务
        # 输出：[batch_size, hidden_dim]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in self.classifier_pool[: self._cur_task+1]:
            logits.append(classifier(image_features))

        #共享一个图像编码器，每个任务有独立的分类头，
        # 推理时把所有已学任务的分类结果拼接起来，得到对所有已见类别的预测。
        logits = torch.cat(logits, dim=1)
        return logits
    
    def update_fc(self, nb_classes):
        self._cur_task +=1
