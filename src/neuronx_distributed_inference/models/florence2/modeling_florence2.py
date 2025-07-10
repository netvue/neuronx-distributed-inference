# coding=utf-8
# Copyright 2024 The NeuronX Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple, List
from collections import OrderedDict

import torch
from torch import nn
from transformers.activations import ACT2FN
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.utils.distributed import get_tp_group
from transformers import AutoModelForCausalLM


class Florence2InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> list[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "vision_config",
        ]

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
        
        # Explicitly set output_attentions, output_hidden_states, use_return_dict
        # These are usually present in PretrainedConfig, but if not, set defaults
        if not hasattr(hf_config, 'output_attentions'):
            hf_config.output_attentions = False
        if not hasattr(hf_config, 'output_hidden_states'):
            hf_config.output_hidden_states = False
        if not hasattr(hf_config, 'use_return_dict'):
            hf_config.use_return_dict = True # Default for HF models

        # Extract relevant parameters from HuggingFace config
        config_params = {
            "hidden_size": hf_config.text_config.d_model,
            "num_attention_heads": hf_config.text_config.decoder_attention_heads,
            "num_hidden_layers": hf_config.text_config.decoder_layers,
            "num_key_value_heads": hf_config.text_config.decoder_attention_heads, # Florence-2 uses MHA, so num_key_value_heads = num_attention_heads
            "pad_token_id": hf_config.pad_token_id,
            "vocab_size": hf_config.text_config.vocab_size,
            "max_position_embeddings": hf_config.text_config.max_position_embeddings,
            "rms_norm_eps": 1e-5,
            "hidden_act": hf_config.text_config.activation_function,
            "intermediate_size": hf_config.text_config.decoder_ffn_dim,
            "vision_config": hf_config.vision_config.to_dict(),
        }

        # Create an instance of Florence2InferenceConfig
        # Pass hf_config as the first argument, and other params as kwargs
        instance = cls(hf_config=hf_config, neuron_config=cls.get_neuron_config_cls()(), **config_params)
        instance.add_derived_config()
        return instance

    @classmethod
    def get_neuron_config_cls(cls) -> type[NeuronConfig]:
        return NeuronConfig

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        config=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=True,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.act = act_layer()
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            bias=True,
            input_is_parallel=True,
            tensor_model_parallel_group=get_tp_group(config),
        )

    def forward(self, x, size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x, size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None
        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm and not self.pre_norm:
            x = self.norm(x)
        return x, (H, W)


class ChannelAttention(nn.Module):
    def __init__(self, dim, groups=8, qkv_bias=True, config=None):
        super().__init__()
        self.groups = groups
        self.qkv = ColumnParallelLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.proj = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            tensor_model_parallel_group=get_tp_group(config),
        )

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (float(N) ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, config=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = ColumnParallelLinear(
            dim, dim * 3, bias=qkv_bias, tensor_model_parallel_group=get_tp_group(config)
        )
        self.proj = RowParallelLinear(
            dim, dim, bias=True, input_is_parallel=True, tensor_model_parallel_group=get_tp_group(config)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
        config=None,
    ):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias, config=config),
            drop_path,
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, config=config),
            drop_path,
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)
        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class ChannelBlock(nn.Module):
    def __init__(
        self,
        dim,
        groups,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
        config=None,
    ):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias, config=config),
            drop_path,
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, config=config),
            drop_path,
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)
        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class NeuronFlorence2VisionEncoder(nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        super().__init__()
        vision_config = config.vision_config
        self.config = config
        self.num_classes = vision_config.get("num_classes", 1000)
        self.embed_dims = vision_config.get("dim_embed", [64, 128, 192, 256])
        self.num_heads = vision_config.get("num_heads", [3, 6, 12, 24])
        self.num_groups = vision_config.get("num_groups", [3, 6, 12, 24])
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = False

        depths = vision_config.get("depths", [1, 1, 3, 1])
        patch_size = vision_config.get("patch_size", [7, 2, 2, 2])
        patch_stride = vision_config.get("patch_stride", [4, 2, 2, 2])
        patch_padding = vision_config.get("patch_padding", [3, 0, 0, 0])
        patch_prenorm = vision_config.get("patch_prenorm", [False, False, False, False])
        window_size = vision_config.get("window_size", 7)
        mlp_ratio = vision_config.get("mlp_ratio", 4.0)
        qkv_bias = vision_config.get("qkv_bias", True)
        drop_path_rate = vision_config.get("drop_path_rate", 0.1)
        norm_layer = nn.LayerNorm
        in_chans = 3

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(self.num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i],
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(
                        OrderedDict([
                            (
                                "spatial_block",
                                SpatialBlock(
                                    self.embed_dims[i],
                                    self.num_heads[i],
                                    window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop_path_rate=dpr[depth_offset + j * 2],
                                    config=config,
                                ),
                            ),
                            (
                                "channel_block",
                                ChannelBlock(
                                    self.embed_dims[i],
                                    self.num_groups[i],
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop_path_rate=dpr[depth_offset + j * 2 + 1],
                                    config=config,
                                ),
                            ),
                        ])
                    )
                    for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            ColumnParallelLinear(self.embed_dims[-1], self.num_classes, bias=True, tensor_model_parallel_group=get_tp_group(config))
            if self.num_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if x.ndim == 2 and x.shape[1] == 3:
                # [B, 3] -> [B, 3, H, W] using projection_dim as default resolution
                dim = self.config.vision_config.get("projection_dim", 768)
                x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dim, dim)
            elif x.ndim == 3 and x.shape[0] == 3:
                # [3, H, W] -> [1, 3, H, W]
                x = x.unsqueeze(0)

        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor for vision encoder, but got shape {x.shape}")

        # Ensure input dtype matches model weights before conv
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)

        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            x, input_size = block(x, input_size)

        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.norms(x)
        x = self.head(x)
        return x


class NeuronFlorence2Attention(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(tensor_model_parallel_group=tensor_model_parallel_group)
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.neuron_config.tp_degree
        self.torch_dtype = config.hf_config.torch_dtype
        self.fused_qkv = False
        self.clip_qkv = False # Add this line # Add this line # Add this line

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            tensor_model_parallel_group=get_tp_group(config),
        )

        self.init_gqa_properties()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads_per_partition, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class NeuronFlorence2MLP(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            gather_output=False,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            tensor_model_parallel_group=get_tp_group(config),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class NeuronFlorence2DecoderLayer(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronFlorence2Attention(config)
        self.cross_attn = NeuronFlorence2Attention(config)
        self.mlp = NeuronFlorence2MLP(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.encoder_attn_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states_norm = self.encoder_attn_layernorm(hidden_states)
            hidden_states, _, _ = self.cross_attn(
                hidden_states=hidden_states_norm,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states_norm = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_norm)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, past_key_value)

        return outputs


class NeuronFlorence2Model(NeuronBaseModel):

    def setup_attr_for_model(self, config: InferenceConfig):
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.on_device_sampling = False
        self.torch_dtype = config.hf_config.torch_dtype # Add this line

    def init_model(self, config: InferenceConfig):
        self.vision_encoder = NeuronFlorence2VisionEncoder(config)
        self.decoder_layers = nn.ModuleList([NeuronFlorence2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, tensor_model_parallel_group=get_tp_group(config))
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.layernorm_embedding = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,  # This is the 5th positional argument
        # The rest are kwargs or optional positional args that might be passed by ModelWrapper
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        llava_args: Optional[List] = []  # llava_args remains, but pixel_values is not extracted from it
    ):
        # --- DEBUGGING PRINTS START ---
        print(f"DEBUG: Inside NeuronFlorence2Model.forward")
        print(f"DEBUG: pixel_values type: {type(pixel_values)}")
        if isinstance(pixel_values, torch.Tensor):
            print(f"DEBUG: pixel_values shape: {pixel_values.shape}")
        else:
            print(f"DEBUG: pixel_values is not a tensor.")
        # --- DEBUGGING PRINTS END ---

        encoder_hidden_states = self.vision_encoder(pixel_values)

        hidden_states = self.embed_tokens(input_ids)


        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=hidden_states.device
        ).expand(bsz, -1)
        hidden_states = hidden_states + self.embed_positions(positions + 2)
        hidden_states = self.layernorm_embedding(hidden_states)

        presents = []
        for i, layer in enumerate(self.decoder_layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, past_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            presents.append(past_key_value)

        logits = self.lm_head(hidden_states)
        return logits, presents

    def input_generator(self):
        # These are dummy inputs for tracing
        batch_size = self.neuron_config.batch_size
        # Use max_context_length for context encoding model tracing
        seq_len = self.neuron_config.max_context_length

        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        seq_ids = torch.arange(0, batch_size, dtype=torch.long)

        # Dummy pixel_values (assuming 3 channels, 768x768 image as per Florence-2)
        pixel_values = torch.randn(batch_size, 3, 768, 768, dtype=self.torch_dtype)

        # --- DEBUGGING PRINTS START ---
        print(f"DEBUG: Inside NeuronFlorence2Model.input_generator")
        print(f"DEBUG: pixel_values type (in input_generator): {type(pixel_values)}")
        print(f"DEBUG: pixel_values shape (in input_generator): {pixel_values.shape}")
        # --- DEBUGGING PRINTS END ---

        # The forward method expects: input_ids, attention_mask, position_ids, seq_ids, pixel_values, *args, **kwargs
        return (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            pixel_values,  # This is the 5th positional argument
            # Remaining arguments for forward (past_key_values, inputs_embeds, etc.) can be passed as kwargs
        )


class NeuronFlorence2ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronFlorence2Model

    def __init__(self, model_path, config):
        super().__init__(model_path, config=config)
        self.text_config = config.hf_config.text_config

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_state_dict = {}
        for name, tensor in state_dict.items():
            # Vision Encoder (DaViT)
            if name.startswith("vision_tower."):
                new_name = name.replace("vision_tower.", "vision_encoder.")
                # Handle convs (ConvEmbed)
                if "convs." in new_name:
                    # convs.X.proj.weight/bias -> vision_encoder.convs.X.proj.weight/bias
                    # convs.X.norm.weight/bias -> vision_encoder.convs.X.norm.weight/bias
                    pass # No further renaming needed for convs
                # Handle blocks (SpatialBlock and ChannelBlock)
                elif "blocks." in new_name:
                    # Remove .fn. from PreNorm (e.g., .window_attn.fn.qkv -> .window_attn.qkv)
                    new_name = new_name.replace(".fn.", ".")
                    # Handle Mlp layers within spatial_block and channel_block (e.g., .ffn.net.fc1 -> .ffn.fc1)
                    if "ffn.net." in new_name:
                        new_name = new_name.replace("ffn.net.", "ffn.")
                # Handle norms and head
                elif new_name.startswith("vision_encoder.norms."):
                    pass # No change
                elif new_name.startswith("vision_encoder.head."):
                    pass # No change
                neuron_state_dict[new_name] = tensor

            # Language Model (Decoder)
            elif name.startswith("language_model.model.decoder."):
                new_name = name.replace("language_model.model.decoder.", "")
                if new_name.startswith("embed_tokens."):
                    neuron_state_dict["embed_tokens." + new_name.split("embed_tokens.")[1]] = tensor
                elif new_name.startswith("embed_positions."):
                    neuron_state_dict["embed_positions." + new_name.split("embed_positions.")[1]] = tensor
                elif new_name.startswith("layernorm_embedding."):
                    neuron_state_dict["layernorm_embedding." + new_name.split("layernorm_embedding.")[1]] = tensor
                elif new_name.startswith("layers."):
                    new_name = new_name.replace("layers.", "decoder_layers.")
                    if "self_attn_layer_norm." in new_name:
                        new_name = new_name.replace("self_attn_layer_norm.", "input_layernorm.")
                    elif "encoder_attn_layer_norm." in new_name:
                        new_name = new_name.replace("encoder_attn_layer_norm.", "encoder_attn_layernorm.")
                    elif "final_layer_norm." in new_name:
                        new_name = new_name.replace("final_layer_norm.", "final_layernorm.")
                    elif "fc1." in new_name:
                        new_name = new_name.replace("fc1.", "mlp.fc1.")
                    elif "fc2." in new_name:
                        new_name = new_name.replace("fc2.", "mlp.fc2.")
                    elif "encoder_attn." in new_name:
                        new_name = new_name.replace("encoder_attn.", "cross_attn.")
                    neuron_state_dict[new_name] = tensor
                else:
                    print(f"Unhandled language_model.model.decoder key: {name}")

            # LM Head
            elif name.startswith("language_model.lm_head."):
                neuron_state_dict[name.replace("language_model.", "")] = tensor

            # Shared Embeddings (from language_model.model.shared)
            elif name.startswith("language_model.model.shared."):
                neuron_state_dict[name.replace("language_model.model.shared.", "embed_tokens.")] = tensor

            # Explicitly ignore keys that are not mapped to our NeuronX model
            elif (name.startswith("image_projection.") or 
                  name.startswith("image_proj_norm.") or 
                  name.startswith("image_pos_embed.") or 
                  name.startswith("visual_temporal_embed.") or 
                  name == "language_model.final_logits_bias" or 
                  name.startswith("language_model.model.encoder.")): # Explicitly ignore the HF text encoder
                print(f"Ignoring key: {name}")
            else:
                print(f"Unhandled key: {name}")
        return neuron_state_dict

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Initialize past_key_values
        past_key_values = None

        # Greedy decoding loop
        for _ in range(max_new_tokens):
            # Calculate position_ids for the current step
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            bsz, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

            # Forward pass
            logits, past_key_values = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                llava_args=[pixel_values],
                **kwargs,
            )

            # Get the next token (greedy approach)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Append the next token to input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # Update attention_mask for the next iteration (if needed)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1
                )
        return input_ids

    @classmethod
    def get_config_cls(cls):
        return Florence2InferenceConfig

    
