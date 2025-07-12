# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

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


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.0, drop_path=0.0):
#         super(CrossAttentionBlock, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         # Layer norms for each modality's input
#         self.norm_img = nn.LayerNorm(dim)
#         self.norm_lidar = nn.LayerNorm(dim)
#         # Multi-head attention modules for each direction
#         self.attn_img_to_lidar = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout)
#         self.attn_lidar_to_img = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout)
#         # DropPath for residual (stochastic depth, if applicable)
#         self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)
#         # You may include an MLP (feed-forward) part here if needed, with its own norm and drop_path,
#         # to mirror a full Transformer block structure. For simplicity, only cross-attention is shown.
#
#     def forward(self, x_img, x_lidar):
#         """Perform cross-attention between image and LiDAR token sequences.
#         Args:
#             x_img: Tensor of shape [B, N_img, C] (image tokens)
#             x_lidar: Tensor of shape [B, N_lidar, C] (LiDAR tokens)
#         Returns:
#             Updated (x_img, x_lidar) after cross-attending to each other.
#         """
#         # Normalize inputs
#         img_tokens = self.norm_img(x_img)  # shape [B, N_img, C]
#         lidar_tokens = self.norm_lidar(x_lidar)  # shape [B, N_lidar, C]
#
#         # Prepare for PyTorch MultiheadAttention: needs shape [N, B, C]
#         # Transpose to [sequence_length, batch, embed_dim]
#         img_tokens_t = img_tokens.transpose(0, 1)  # [N_img, B, C]
#         lidar_tokens_t = lidar_tokens.transpose(0, 1)  # [N_lidar, B, C]
#
#         # Cross attention: Image queries attend to LiDAR keys/values
#         attn_img, _ = self.attn_img_to_lidar(query=img_tokens_t, key=lidar_tokens_t, value=lidar_tokens_t)
#         attn_img = attn_img.transpose(0, 1)  # back to [B, N_img, C]
#
#         # Cross attention: LiDAR queries attend to Image keys/values
#         attn_lidar, _ = self.attn_lidar_to_img(query=lidar_tokens_t, key=img_tokens_t, value=img_tokens_t)
#         attn_lidar = attn_lidar.transpose(0, 1)  # back to [B, N_lidar, C]
#
#         # Residual connection with optional DropPath
#         # Add the cross-attention outputs back to the original inputs
#         x_img = x_img + self.drop_path(attn_img)
#         x_lidar = x_lidar + self.drop_path(attn_lidar)
#
#         return x_img, x_lidar

# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.0, drop_path=0.0):
#         super(CrossAttentionBlock, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#
#         self.attn_weights_img_to_lidar = None  # 保存权重
#         self.attn_weights_lidar_to_img = None
#
#         # Layer norms
#         self.norm_img = nn.LayerNorm(dim)
#         self.norm_lidar = nn.LayerNorm(dim)
#
#         # Multi-head attentions
#         self.attn_img_to_lidar = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout, batch_first=False)
#         self.attn_lidar_to_img = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout, batch_first=False)
#
#         self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)
#
#     def forward(self, x_img, x_lidar):
#         img_tokens = self.norm_img(x_img)
#         lidar_tokens = self.norm_lidar(x_lidar)
#
#         img_tokens_t = img_tokens.transpose(0, 1)  # [N_img, B, C]
#         lidar_tokens_t = lidar_tokens.transpose(0, 1)  # [N_lidar, B, C]
#
#         attn_img, attn_weights_img = self.attn_img_to_lidar(img_tokens_t, lidar_tokens_t, lidar_tokens_t, need_weights=True)
#         attn_lidar, attn_weights_lidar = self.attn_lidar_to_img(lidar_tokens_t, img_tokens_t, img_tokens_t, need_weights=True)
#
#         self.attn_weights_img_to_lidar = attn_weights_img  # 保存下来
#         self.attn_weights_lidar_to_img = attn_weights_lidar
#
#         x_img = x_img + self.drop_path(attn_img.transpose(0, 1))
#         x_lidar = x_lidar + self.drop_path(attn_lidar.transpose(0, 1))
#
#         return x_img, x_lidar

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
        self.attn_img_to_lidar = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout, batch_first=False)
        self.attn_lidar_to_img = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=dropout, batch_first=False)

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

    def __init__(
            self,
            img_size=(1280, 960),
            patch_size=(64, 64),
            stride=64,
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
        self.stride = stride

        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
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
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
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

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        cross_num_heads = embed_dim // 64
        cross_dropout = 0.0  # 交叉注意力的dropout概率
        cross_drop_path = 0.0  # 交叉注意力的DropPath
        self.cross_attn_block_8 = CrossAttentionBlock(dim=embed_dim,
                                                          num_heads=cross_num_heads,
                                                          qkv_bias=True,
                                                          dropout=cross_dropout,
                                                          drop_path=cross_drop_path)
        # self.cross_attn_block_8_lidar = CrossAttentionBlock(dim=embed_dim,
        #                                                     num_heads=cross_num_heads,
        #                                                     qkv_bias=True,
        #                                                     dropout=cross_dropout,
        #                                                     drop_path=cross_drop_path)
        # 第16层后的交叉注意力模块（图像分支和激光雷达分支）
        self.cross_attn_block_16 = CrossAttentionBlock(dim=embed_dim,
                                                           num_heads=cross_num_heads,
                                                           qkv_bias=True,
                                                           dropout=cross_dropout,
                                                           drop_path=cross_drop_path)
        # self.cross_attn_block_16_lidar = CrossAttentionBlock(dim=embed_dim,
        #                                                      num_heads=cross_num_heads,
        #                                                      qkv_bias=True,
        #                                                      dropout=cross_dropout,
        #                                                      drop_path=cross_drop_path)

        mlp_hidden_dim = embed_dim * 4
        self.cross_attn_8_img_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.cross_attn_8_lidar_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.cross_attn_16_img_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.cross_attn_16_lidar_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

        # 图像分支
        self.patch_embed_img = PatchEmbed(
            img_size=img_size, # 1280*960
            patch_size=patch_size, # 64 （64*64）
            stride=patch_size,  # stride与patch_size一致 64
            in_chans=in_chans, # 3 dimensiion
            embed_dim=embed_dim # 768
        )
        self.num_patches_img = self.patch_embed_img.num_patches

        # 激光雷达分支
        self.patch_embed_lidar = PatchEmbed(
            img_size=lidar_size,
            patch_size=patch_size,
            stride=patch_size,
            in_chans=lidar_chans,
            embed_dim=embed_dim
        )
        self.num_patches_lidar = self.patch_embed_lidar.num_patches

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
                    d_state=d_state,  #  默认的16
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
        self.patch_embed_lidar.apply(segm_init_weights)
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

    def forward_features(self, img, lidar, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x_img = self.patch_embed_img(img)
        B_img, M_img, _ = x_img.shape # B

        x_lidar = self.patch_embed_lidar(lidar)
        B_lidar, M_lidar, _ = x_lidar.shape

        assert B_img == B_lidar and M_img == M_lidar, "Batch and sequence length must match"

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head_img = self.cls_token_head.expand(B_img, -1, -1)
                cls_token_tail_img = self.cls_token_tail.expand(B_img, -1, -1)
                token_position_img = [0, M_img + 1]
                x_img = torch.cat((cls_token_head_img, x_img, cls_token_tail_img), dim=1)
                M_img = x_img.shape[1]

                # lidar
                cls_token_head_lidar = self.cls_token_head.expand(B_lidar, -1, -1)
                cls_token_tail_lidar = self.cls_token_tail.expand(B_lidar, -1, -1)
                token_position_lidar = [0, M_lidar + 1]
                x_lidar = torch.cat((cls_token_head_lidar, x_lidar, cls_token_tail_lidar), dim=1)
                M_lidar = x_lidar.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token_img = self.cls_token.expand(B_img, -1, -1)
                    token_position_img = M_img // 2
                    # add cls token in the middle
                    x_img = torch.cat((x_img[:, :token_position_img, :], cls_token_img, x_img[:, token_position_img:, :]), dim=1)

                    # lidar
                    cls_token_lidar = self.cls_token.expand(B_lidar, -1, -1)
                    token_position_lidar = M_lidar // 2
                    # add cls token in the middle
                    x_lidar = torch.cat((x_lidar[:, :token_position_lidar, :], cls_token_lidar, x_lidar[:, token_position_lidar:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token_img = self.cls_token.expand(B_img, -1, -1)
                    token_position_img = random.randint(0, M_img)
                    x_img = torch.cat((x_img[:, :token_position_img, :], cls_token_img, x_img[:, token_position_img:, :]), dim=1)
                    print("token_position: ", token_position_img)

                    cls_token_lidar = self.cls_token.expand(B_lidar, -1, -1)
                    token_position_lidar = random.randint(0, M_lidar)
                    x_lidar = torch.cat((x_lidar[:, :token_position_lidar, :], cls_token_lidar, x_lidar[:, token_position_lidar:, :]), dim=1)
                    print("token_position: ", token_position_lidar)
                else:
                    cls_token_img = self.cls_token.expand(B_img, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position_img = 0
                    x_img = torch.cat((cls_token_img, x_img), dim=1)

                    cls_token_lidar = self.cls_token.expand(B_lidar, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position_lidar = 0
                    x_lidar = torch.cat((cls_token_lidar, x_lidar), dim=1)

                M_img = x_img.shape[1]
                M_lidar = x_lidar.shape[1]

        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x_img = x_img + self.pos_embed
            x_img = self.pos_drop(x_img)

            x_lidar = x_lidar + self.pos_embed
            x_lidar = self.pos_drop(x_lidar)

        if if_random_token_rank:
            # img
            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M_img)

            if isinstance(token_position_img, list):
                print("original value: ", x_img[0, token_position_img[0], 0], x_img[0, token_position_img[1], 0])
            else:
                print("original value: ", x_img[0, token_position_img, 0])
            print("original token_position: ", token_position_img)

            # 执行 shuffle
            x_img = x_img[:, shuffle_indices, :]

            if isinstance(token_position_img, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position_img = [torch.where(shuffle_indices == token_position_img[i])[0].item() for i in range(len(token_position_img))]
                token_position_img = new_token_position_img
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position_img = torch.where(shuffle_indices == token_position_img)[0].item()

            if isinstance(token_position_img, list):
                print("new value: ", x_img[0, token_position_img[0], 0], x_img[0, token_position_img[1], 0])
            else:
                print("new value: ", x_img[0, token_position_img, 0])
            print("new token_position: ", token_position_img)

            # lidar
            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M_lidar)

            if isinstance(token_position_lidar, list):
                print("original value: ", x_lidar[0, token_position_lidar[0], 0], x_lidar[0, token_position_lidar[1], 0])
            else:
                print("original value: ", x_lidar[0, token_position_lidar, 0])
            print("original token_position: ", token_position_lidar)

            # 执行 shuffle
            x_lidar = x_lidar[:, shuffle_indices, :]

            if isinstance(token_position_lidar, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position_lidar = [torch.where(shuffle_indices == token_position_lidar[i])[0].item() for i in range(len(token_position_lidar))]
                token_position_lidar = new_token_position_lidar
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position_lidar = torch.where(shuffle_indices == token_position_lidar)[0].item()

            if isinstance(token_position_lidar, list):
                print("new value: ", x_lidar[0, token_position_lidar[0], 0], x_lidar[0, token_position_lidar[1], 0])
            else:
                print("new value: ", x_lidar[0, token_position_lidar, 0])
            print("new token_position: ", token_position_lidar)


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x_img = x_img.flip([1])
            if_flip_img_sequences = True

        if_flip_lidar_sequences = False
        if self.flip_lidar_sequences_ratio > 0 and (self.flip_lidar_sequences_ratio - random.random()) > 1e-5:
            x_lidar = x_lidar.flip([1])
            if_flip_lidar_sequences = True

        # Early dual-stream layers (0-7)
        if not self.if_bidirectional:
            residual_img, residual_lidar = None, None

            for layer_idx in range(self.depth):

                if if_flip_img_sequences and self.if_rope:
                    x_img = x_img.flip([1])
                    if residual_img is not None:
                        residual_img = residual_img.flip([1])

                if self.if_rope:
                    x_img = self.rope(x_img)
                    if residual_img is not None and self.if_rope_residual:
                        residual_img = self.rope(residual_img)

                if if_flip_img_sequences and self.if_rope:
                    x_img = x_img.flip([1])
                    if residual_img is not None:
                        residual_img = residual_img.flip([1])

                x_img, residual_img = self.layers_img[layer_idx](
                    x_img, residual_img, inference_params=inference_params
                )

                if if_flip_lidar_sequences and self.if_rope:
                    x_lidar = x_lidar.flip([1])
                    if residual_lidar is not None:
                        residual_lidar = residual_lidar.flip([1])

                if self.if_rope:
                    x_lidar = self.rope(x_lidar)
                    if residual_lidar is not None and self.if_rope_residual:
                        residual_lidar = self.rope(residual_lidar)

                if if_flip_lidar_sequences and self.if_rope:
                    x_lidar = x_lidar.flip([1])
                    if residual_lidar is not None:
                        residual_lidar = residual_lidar.flip([1])

                x_lidar, residual_lidar = self.layers_lidar[layer_idx](
                    x_lidar, residual_lidar, inference_params=inference_params
                )

                if layer_idx == 7:
                    x_img, x_lidar = self.cross_attn_block_8(x_img, x_lidar)
                    x_img = self.cross_attn_8_img_ffn(x_img)  # 图像分支交叉注意力 + 前馈
                    x_lidar = self.cross_attn_8_lidar_ffn(x_lidar)  # 激光雷达分支交叉注意力 + 前馈

                if layer_idx == 15:
                    x_img, x_lidar = self.cross_attn_block_16(x_img, x_lidar)
                    x_img = self.cross_attn_16_img_ffn(x_img)
                    x_lidar = self.cross_attn_16_lidar_ffn(x_lidar)

                    # 融合后进入单流处理
                    x_fused = torch.cat([x_img, x_lidar], dim=1)
                    residual_fused = None

                    for fused_layer_idx in range(16, self.depth):
                        layer = self.fused_blocks[fused_layer_idx - 16]

                        if if_flip_img_sequences and self.if_rope:
                            x_fused = x_fused.flip([1])
                            if residual_fused is not None:
                                residual_fused = residual_fused.flip([1])

                        if self.if_rope:
                            x_fused = self.rope(x_fused)
                            if residual_fused is not None and self.if_rope_residual:
                                residual_fused = self.rope(residual_fused)

                        if if_flip_img_sequences and self.if_rope:
                            x_fused = x_fused.flip([1])
                            if residual_fused is not None:
                                residual_fused = residual_fused.flip([1])

                        x_fused, residual_fused = layer(
                            x_fused, residual_fused, inference_params=inference_params
                        )

                    break

        else:
            # 如果是双向bidirectional，完全类似地扩展img和lidar分支即可
            residual_img, residual_lidar = None, None
            for i in range(self.depth // 2):

                if self.if_rope:
                    x_img = self.rope(x_img)
                    if residual_img is not None and self.if_rope_residual:
                        residual_img = self.rope(residual_img)

                x_img_f, residual_img_f = self.layers_img[i * 2](x_img, residual_img, inference_params=inference_params)
                x_img_b, residual_img_b = self.layers_img[i * 2 + 1](x_img.flip([1]),
                                                                     None if residual_img is None else residual_img.flip(
                                                                         [1]), inference_params=inference_params)
                x_img = x_img_f + x_img_b.flip([1])
                residual_img = residual_img_f + residual_img_b.flip([1])

                if self.if_rope:
                    x_lidar = self.rope(x_lidar)
                    if residual_lidar is not None and self.if_rope_residual:
                        residual_lidar = self.rope(residual_lidar)

                x_lidar_f, residual_lidar_f = self.layers_lidar[i * 2](x_lidar, residual_lidar,
                                                                       inference_params=inference_params)
                x_lidar_b, residual_lidar_b = self.layers_lidar[i * 2 + 1](x_lidar.flip([1]),
                                                                           None if residual_lidar is None else residual_lidar.flip(
                                                                               [1]), inference_params=inference_params)
                x_lidar = x_lidar_f + x_lidar_b.flip([1])
                residual_lidar = residual_lidar_f + residual_lidar_b.flip([1])

                if i * 2 + 1 == 7:
                    x_img, x_lidar = self.cross_attn_block_8(x_img, x_lidar)

                if i * 2 + 1 == 15:
                    x_img, x_lidar = self.cross_attn_block_16(x_img, x_lidar)

                    x_fused = torch.cat([x_img, x_lidar], dim=1)
                    residual_fused = None
                    for fused_layer_idx in range(16, self.depth):
                        layer = self.fused_blocks[fused_layer_idx - 16]

                        if self.if_rope:
                            x_fused = self.rope(x_fused)
                            if residual_fused is not None and self.if_rope_residual:
                                residual_fused = self.rope(residual_fused)

                        x_fused_f, residual_fused_f = layer(x_fused, residual_fused, inference_params=inference_params)
                        x_fused_b, residual_fused_b = layer(x_fused.flip([1]),
                                                            None if residual_fused is None else residual_fused.flip(
                                                                [1]), inference_params=inference_params)
                        x_fused = x_fused_f + x_fused_b.flip([1])
                        residual_fused = residual_fused_f + residual_fused_b.flip([1])

                    break  # 单流处理完毕，跳出循环

        # Final normalization
        if not self.fused_add_norm:
            if residual_fused is None:
                residual_fused = x_fused
            else:
                residual_fused = residual_fused + self.drop_path(x_fused)
            hidden_states = self.norm_f(residual_fused.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(x_fused),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual_fused,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # if self.if_cls_token:
        #     if self.use_double_cls_token:
        #         return (hidden_states[:, token_position_img[0], :] + hidden_states[:, token_position_img[1], :]) / 2
        #     else:
        #         return hidden_states[:, token_position_img, :]
        #
        # # pooling
        # if self.final_pool_type == 'mean':
        #     return hidden_states.mean(dim=1)
        # elif self.final_pool_type == 'max':
        #     return hidden_states.max(dim=1).values
        # else:
        #     return hidden_states
        if self.if_cls_token:
            hidden_states = torch.cat(
                [x for i, x in enumerate(hidden_states.split(1, dim=1)) if i != token_position_img], dim=1)

        return hidden_states  # [B, seq_len, embed_dim]

    def forward(self, img, lidar, return_features=False, inference_params=None,
                if_random_cls_token_position=False, if_random_token_rank=False,
                return_attn=False):

        # x = self.forward_features(img, lidar, inference_params,
        #                           if_random_cls_token_position=if_random_cls_token_position,
        #                           if_random_token_rank=if_random_token_rank)
        # if return_features:
        #     return x
        # x = self.head(x)
        # if self.final_pool_type == 'max':
        #     x = x.max(dim=1)[0]
        # return x

        fused_features = self.forward_features(
            img, lidar,
            inference_params=inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank
        )

        predictions = self.detection_head(fused_features)

        if return_attn and return_features:
            attn_dict = {
                "cross_attn_8_img2lidar": self.cross_attn_block_8.attn_weights_img_to_lidar,
                "cross_attn_8_lidar2img": self.cross_attn_block_8.attn_weights_lidar_to_img,
                "cross_attn_16_img2lidar": self.cross_attn_block_16.attn_weights_img_to_lidar,
                "cross_attn_16_lidar2img": self.cross_attn_block_16.attn_weights_lidar_to_img,
            }
            return predictions, attn_dict, fused_features

        elif return_attn:
            attn_dict = {
                "cross_attn_8_img2lidar": self.cross_attn_block_8.attn_weights_img_to_lidar,
                "cross_attn_8_lidar2img": self.cross_attn_block_8.attn_weights_lidar_to_img,
                "cross_attn_16_img2lidar": self.cross_attn_block_16.attn_weights_img_to_lidar,
                "cross_attn_16_lidar2img": self.cross_attn_block_16.attn_weights_lidar_to_img,
            }
            return predictions, attn_dict

        elif return_features:
            return predictions, fused_features

        else:
            return predictions

@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
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
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
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
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
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
        # num_classes=0,
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
        # num_classes=0,
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