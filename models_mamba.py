# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
    'test',
]

def compute_alignment_loss(mask_img: torch.Tensor, mask_lidar: torch.Tensor, loss_type='kl'):
    """
    mask_img: 图像引导 mask，形状为 [B, 1, H, W]
    mask_lidar: lidar 投影生成的原始掩码 [B, 1, H, W]
    """
    if loss_type == 'kl':
        img_soft = F.log_softmax(mask_img, dim=1)
        lidar_soft = F.softmax(mask_lidar, dim=1)
        loss = F.kl_div(img_soft, lidar_soft, reduction='batchmean')
    elif loss_type == 'cosine':
        loss = 1 - F.cosine_similarity(mask_img.flatten(1), mask_lidar.flatten(1)).mean()
    else:
        raise NotImplementedError("Unknown alignment loss")


    return loss

class SoftAlign(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=16):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),  # 输出位移场 dx, dy
        )

    def forward(self, mask):
        B, _, H, W = mask.shape
        flow = self.offset_conv(mask)  # (B, 2, H, W)

        # 构建标准网格
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=mask.device),
            torch.linspace(-1, 1, W, device=mask.device),
            indexing="ij"
        )
        grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # 偏移加入网格，归一化后用于 grid_sample
        flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        warped_grid = (grid + flow).clamp(-1, 1)
        aligned_mask = F.grid_sample(mask, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return aligned_mask


class TransformerDetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_queries=1, num_heads=8, num_decoder_layers=6):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.class_embed = nn.Linear(embed_dim, num_classes + 1)  # +1 for background class
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)  # bbox coordinates: cx, cy, w, h

    def forward(self, fused_features):
        # fused_features: [B, sequence_len, embed_dim]
        B = fused_features.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, embed_dim]

        # Transformer decoder
        hs = self.transformer_decoder(queries, fused_features)  # [B, num_queries, embed_dim]

        # Predict class and bbox
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_bbox = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_bbox}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    output_dim if i == num_layers - 1 else hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.0, drop_path=0.0):
        super(CrossAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attn_weights_img_to_lidar = None  # 保存权重
        self.attn_weights_lidar_to_img = None

        # Layer norms
        self.norm_img = nn.LayerNorm(dim)
        self.norm_lidar = nn.LayerNorm(dim)

        # Multi-head attentions
        self.attn_img_to_lidar = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout,
                                                       batch_first=False)
        self.attn_lidar_to_img = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout,
                                                       batch_first=False)

        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)

    def forward(self, x_img, x_lidar):
        img_tokens = self.norm_img(x_img)
        lidar_tokens = self.norm_lidar(x_lidar)

        img_tokens_t = img_tokens.transpose(0, 1)  # [N_img, B, C]
        lidar_tokens_t = lidar_tokens.transpose(0, 1)  # [N_lidar, B, C]

        attn_img, attn_weights_img = self.attn_img_to_lidar(
            img_tokens_t, lidar_tokens_t, lidar_tokens_t,
            need_weights=True, average_attn_weights=False
        )
        attn_lidar, attn_weights_lidar = self.attn_lidar_to_img(
            lidar_tokens_t, img_tokens_t, img_tokens_t,
            need_weights=True, average_attn_weights=False
        )

        self.attn_weights_img_to_lidar = attn_weights_img  # shape: (B, num_heads, N_img, N_lidar)
        self.attn_weights_lidar_to_img = attn_weights_lidar

        x_img = x_img + self.drop_path(attn_img.transpose(0, 1))
        x_lidar = x_lidar + self.drop_path(attn_lidar.transpose(0, 1))

        return x_img, x_lidar


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding (适配大尺寸图像) """

    def __init__(
            self,
            img_size=(1280, 960),  # 修改1: 直接指定目标尺寸
            patch_size=(64, 64),  # 修改2: 增大patch尺寸减少计算量
            stride=64,  # 修改3: stride与patch_size一致
            in_chans=1,
            embed_dim=768,
            norm_layer=None,
            flatten=True
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride  # 新增参数记录stride

        # 计算grid_size (确保整除)
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # 使用Conv2d进行分块投影
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride  # stride与kernel_size一致
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # 允许动态尺寸输入 (删除固定尺寸断言)
        # 修改4: 支持动态输入尺寸（若需要固定尺寸可保留断言）
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        d_state=16,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type,
                        if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=(1280, 960),
                 lidar_size=(1280, 960),
                 patch_size=64,
                 stride=64,
                 depth=24,
                 embed_dim=768,
                 d_state=16,
                 # 图片
                 in_chans=3,
                 # 激光雷达
                 lidar_chans=1,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 flip_lidar_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 num_queries=1,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.flip_lidar_sequences_ratio = flip_lidar_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        self.depth = depth
        self.num_queries = num_queries

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 感受野扩大
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),  # 输出3通道用于引导图像
            nn.Sigmoid()
        )

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1)  # 输出两个通道作为偏移向量 dx, dy
        )

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.soft_align = SoftAlign()

        # 图像分支
        self.patch_embed_img = PatchEmbed(
            img_size=img_size,  # 1280*960
            patch_size=patch_size,  # 64 （64*64）
            stride=patch_size,  # stride与patch_size一致 64
            in_chans=in_chans,  # 3 dimensiion
            embed_dim=embed_dim  # 768
        )
        self.num_patches_img = self.patch_embed_img.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1

        if if_abs_pos_embed:
            num_patches = 300  # 计算得到 (1280//64) * (960//64) = 20*15=300
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers_img = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.layers_lidar = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,  # 默认的16
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed_img.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # 在 VisionMamba 的 __init__ 最后添加:
        self.detection_head = TransformerDetectionHead(
            embed_dim=self.embed_dim,  # 需与主干输出特征维度一致
            num_classes=self.num_classes,
            # num_queries=1,  # 可自定义
            num_queries=self.num_queries,
            num_heads=8,
            num_decoder_layers=6
        )

        # 融合后的单流Transformer模块（从第16层开始，共 depth-16 层）
        self.fused_blocks = nn.ModuleList([
            create_block(
                embed_dim,
                d_state=d_state,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=layer_idx,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[layer_idx],  # 使用与层索引对应的drop_path概率
                if_divide_out=if_divide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs
            )
            for layer_idx in range(16, depth)
        ])

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, img, lidar_mask, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        # 使用 mask encoder 引导图像
        offset = self.offset_predictor(lidar_mask)  # [B, 2, H, W]

        # 创建像素坐标网格并添加 offset
        B, _, H, W = offset.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=lidar_mask.device),
                                        torch.linspace(-1, 1, W, device=lidar_mask.device))
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        grid = grid + offset.permute(0, 2, 3, 1)  # 加上偏移

        # 插值变换 lidar_mask 到引导图像对齐位置
        lidar_mask_aligned = F.grid_sample(lidar_mask, grid, align_corners=True)

        # 再送入小CNN
        guided_mask = self.mask_encoder(lidar_mask_aligned)  # [B, 3, H, W]
        img = img * guided_mask

        x = self.patch_embed_img(img)  # [B, N, C]
        B, M, _ = x.shape

        # 插入 cls token
        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                if self.use_middle_cls_token:
                    token_position = M // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                else:
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
            M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # Token打乱（用于训练鲁棒性）
        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)
            x = x[:, shuffle_indices, :]
            if isinstance(token_position, list):
                token_position = [torch.where(shuffle_indices == pos)[0].item() for pos in token_position]
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

        if_flip = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip = True

        residual = None
        hidden_states = x

        # 主干处理（Mamba或双向Mamba）
        if not self.if_bidirectional:
            for layer in self.layers_img:
                if if_flip and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)
                if if_flip and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        else:
            for i in range(len(self.layers_img) // 2):
                hidden_states_f, residual_f = self.layers_img[i * 2](hidden_states, residual)
                hidden_states_b, residual_b = self.layers_img[i * 2 + 1](hidden_states.flip([1]),
                                                                         residual.flip(
                                                                             [1]) if residual is not None else None)
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        # 最后归一化
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.if_cls_token:
            return hidden_states, {
                "mask_img": guided_mask,
                "mask_lidar": lidar_mask_aligned
            }

        if self.final_pool_type == 'none':
            return hidden_states, {
                "mask_img": guided_mask,
                "mask_lidar": lidar_mask_aligned
            }
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1), {
                "mask_img": guided_mask,
                "mask_lidar": lidar_mask_aligned
            }
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1).values, {
                "mask_img": guided_mask,
                "mask_lidar": lidar_mask_aligned
            }
        elif self.final_pool_type == 'all':
            return hidden_states, {
                "mask_img": guided_mask,
                "mask_lidar": lidar_mask_aligned
            }
        else:
            raise NotImplementedError

    def forward(self, img, lidar, return_features=False, inference_params=None,
                if_random_cls_token_position=False, if_random_token_rank=False,
                return_attn=False):

        fused_features, aux_info = self.forward_features(
            img, lidar,
            inference_params=inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank
        )

        predictions = self.detection_head(fused_features)

        # 只需要返回预测 + 特征
        if return_attn and return_features:
            return predictions, {}, fused_features  # attention dict 已无意义，返回空

        elif return_attn:
            return predictions, {}  # 同上

        elif return_features:
            return predictions, fused_features

        else:
            return predictions


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                             **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                              **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def test(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=64,
        stride=64,
        embed_dim=768,
        depth=24,
        in_chans=3,
        lidar_chans=1,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        # num_classes=1,  # 你可以根据自己的任务改成目标类别数
        if_abs_pos_embed=True,
        if_cls_token=True,
        use_middle_cls_token=False,
        bimamba_type="v2",
        final_pool_type='mean',
        # final_pool_type='none',
        num_queries=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def test_5(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=64,
        stride=64,
        embed_dim=768,
        depth=24,
        in_chans=3,
        lidar_chans=1,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        num_classes=1,  # 你可以根据自己的任务改成目标类别数
        if_abs_pos_embed=True,
        if_cls_token=False,
        use_middle_cls_token=False,
        bimamba_type="v2",
        # final_pool_type='mean',
        final_pool_type='none',
        num_queries=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
