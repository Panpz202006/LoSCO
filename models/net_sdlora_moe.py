import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_sdlora_moe import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from models.vit_sdlora_moe import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn


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

def etf_loss(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    num_tokens, num_experts, _ = expert_outputs.shape

    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_outputs.device)


    norm_outputs = F.normalize(expert_outputs, p=2, dim=-1, eps=eps)
    
    gram_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

    target_val = -1.0 / (num_experts - 1)
    target_matrix = torch.full(
        (num_experts, num_experts), 
        target_val, 
        device=expert_outputs.device, 
        dtype=expert_outputs.dtype
    )

    target_matrix.fill_diagonal_(1.0)

    loss = F.mse_loss(gram_matrix, target_matrix.unsqueeze(0).expand_as(gram_matrix))
    
    return loss

class ViT(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64, moe_list=None):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank, moe_list=moe_list)


    def forward(self, x, task_id, register_blk=-1):
        x = self.patch_embed(x)  
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        self.i_w = 0.1

        total_load_balance_loss = torch.tensor(0.0, device=x.device)
        total_etf_loss = torch.tensor(0.0, device=x.device)

        for i, blk in enumerate(self.blocks):
            if i in self.moe_list:
                # MoE 层：训练和推理都需要使用 MoE
                if self.training:
                    # 训练：解包所有返回值
                    moe_output, moe_lb_loss, all_expert_outputs_k, selected_experts_k, all_expert_outputs_v, selected_experts_v = blk(x, task_id, register_blk==i)
                    
                    # 计算 ETF loss（修正：使用 K 和 V）
                    if all_expert_outputs_k is not None and all_expert_outputs_v is not None:
                        moe_etf_l_k = etf_loss(all_expert_outputs_k)
                        moe_etf_l_v = etf_loss(all_expert_outputs_v)  # 修正！
                        total_etf_loss += 0.5*(moe_etf_l_k + moe_etf_l_v)
                        total_load_balance_loss += moe_lb_loss
                    
                    # 训练时的归一化融合
                    moe_output_normalized = (
                        moe_output * x.norm(dim=-1, keepdim=True)
                        / (moe_output.norm(dim=-1, keepdim=True) + 1e-6)
                    )
                    x = self.i_w * moe_output_normalized + (1 - self.i_w) * x
                else:
                    # 推理：直接获取输出（BlocK_MOE 返回 tensor）
                    # 在sd-lora定义的就是这么直接进行。
                    # x = blk(x, task_id, register_blk==i)
                    moe_output = blk(x, task_id, register_blk==i)
                    moe_output_normalized = (
                        moe_output * x.norm(dim=-1, keepdim=True)
                        / (moe_output.norm(dim=-1, keepdim=True) + 1e-6)
                    )                    
                    x = self.i_w * moe_output_normalized + (1 - self.i_w) * x
                    
            else:
                # 普通 Block
                x = blk(x, task_id, register_blk==i)

        x = self.norm(x)
        
        return x, total_load_balance_loss, total_etf_loss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, 
                            depth=12, num_heads=12, 
                            n_tasks=args["sessions"], rank=args["rank"], moe_list=args["moe_list"])

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

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _, _ = self.image_encoder(image, self._cur_task)
        else:
            image_features, _, _ = self.image_encoder(image, task)
        
        image_features = image_features[:,0,:]  # [128,768]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self._cur_task + 1):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, total_load_balance_loss, total_etf_loss = self.image_encoder(image, task_id=self._cur_task)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in [self.classifier_pool[self._cur_task]]:
            logits.append(classifier(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            "total_load_balance_loss":total_load_balance_loss,
            "total_etf_loss": total_etf_loss
        }

    def interface(self, image, task_id=None):
        logits = []
        image_features,_,_ = self.image_encoder(image, task_id=self._cur_task if task_id is None else task_id)
        image_features = image_features[:,0,:]

        image_features = image_features.view(image_features.size(0),-1)

        for classifier in self.classifier_pool[: self._cur_task+1]:
            logits.append(classifier(image_features))

        logits = torch.cat(logits, dim=1)
        return logits



    def update_fc(self, nb_classes):
        self._cur_task +=1
