# coding=utf-8
# Copyright 2024 Microsoft, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Florence2 model for NxD Inference."""

import collections.abc
import logging
import math
from typing import List, Optional, Tuple, Union, Type

import torch
import torch.utils.checkpoint
import torch.distributed as dist
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding, OutputChannelParallelConv2d
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.utils import torch_int

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper, DecoderModelInstance, EncoderModelInstance
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.florence2.configuration_florence2_neuron import Florence2VisionInferenceConfig, Florence2LanguageInferenceConfig, Florence2InferenceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





class NeuronFlorence2Embeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: Florence2VisionInferenceConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        logger.info(f"use_mask_token {use_mask_token}")
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        )
        logger.info(f"self.mask_token {self.mask_token}")
        self.patch_embeddings = NeuronFlorence2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[torch.BoolTensor] = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class NeuronFlorence2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: Florence2InferenceConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # self.projection = nn.Conv2d(
        #     num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        # )
        self.projection = OutputChannelParallelConv2d( # FIXME: in checkpoint bias is not sharded: Incorrect tensor shape at checkpoint keyprojection.bias: received 768, expected 24.
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True # Assuming bias is always true for Florence2
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: Optional[
            torch.BoolTensor
        ] = False,  # swap bool to torch.BoolTensor for Neuron
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class NeuronFlorence2Attention(NeuronAttentionBase):
    def __init__(self, config: Florence2InferenceConfig, attention_type: str):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.neuron_config.tp_degree
        self.torch_dtype = config.neuron_config.torch_dtype
        self.fused_qkv = False
        self.clip_qkv = None
        self.bias = True
        self.attention_type = attention_type
        self.channel_group_size = config.channel_group_size

        self.o_proj_layer_name = "o_proj"

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            gather_output=False,
            dtype=self.torch_dtype,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            gather_output=False,
            dtype=self.torch_dtype,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            gather_output=False,
            dtype=self.torch_dtype,
        )
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=self.torch_dtype,
        )

        self.init_gqa_properties()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.attention_type == "spatial":
            return super().forward(hidden_states)
        elif self.attention_type == "channel":
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1).contiguous()
            hidden_states = hidden_states.view(batch_size, hidden_size, int(seq_len**0.5), int(seq_len**0.5))

            batch_size, num_channels, height, width = hidden_states.shape
            num_groups = num_channels // self.channel_group_size

            hidden_states = hidden_states.view(
                batch_size, num_groups, self.channel_group_size, height, width
            ).permute(0, 2, 1, 3, 4).contiguous()
            hidden_states = hidden_states.view(batch_size * self.channel_group_size, num_groups, -1)

            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            q = q.view(batch_size * self.channel_group_size, num_groups, self.num_attention_heads, -1).permute(0, 2, 1, 3)
            k = k.view(batch_size * self.channel_group_size, num_groups, self.num_attention_heads, -1).permute(0, 2, 3, 1)
            v = v.view(batch_size * self.channel_group_size, num_groups, self.num_attention_heads, -1).permute(0, 2, 1, 3)

            attn_weights = torch.matmul(q, k)
            attn_weights = attn_weights / (self.head_dim ** 0.5)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(batch_size * self.channel_group_size, num_groups, -1)

            attn_output = self.o_proj(attn_output)

            attn_output = attn_output.view(batch_size, self.channel_group_size, num_groups, height, width).permute(0, 2, 1, 3, 4).contiguous()
            attn_output = attn_output.view(batch_size, num_channels, height, width)
            attn_output = attn_output.flatten(2).transpose(1, 2)

            return (attn_output,)



class NeuronFlorence2Intermediate(nn.Module):
    def __init__(self, config: Florence2InferenceConfig) -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            raise ValueError(
                f"{config.hidden_act} is not supported. Choose from {list(ACT2FN.keys())}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class NeuronFlorence2Output(nn.Module):
    def __init__(self, config: Florence2InferenceConfig) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class NeuronFlorence2Layer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: Florence2InferenceConfig, attention_type: str) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # NeuronAttentionBase includes qkv project layers (CPL) and output project (RPL) layers
        # but HF separates into ViTSelfAttention which only has qkv, then an another ViTSelfOutput that has the output project layer
        self.attention = NeuronFlorence2Attention(config, attention_type)
        self.intermediate = NeuronFlorence2Intermediate(config)
        self.output = NeuronFlorence2Output(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
        )

        # NeuronAttentionBases output tuple (attn_output, past_key_value, cos_cache, sin_cache)
        attention_output = self_attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class NeuronFlorence2Encoder(nn.Module):
    def __init__(self, config: Florence2VisionInferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attention_type = "channel" if (i % 2 == 1) else "spatial"
            self.layer.append(NeuronFlorence2Layer(config, attention_type))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[tuple, torch.Tensor]:
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs

        return hidden_states


class NeuronFlorence2Pooler(nn.Module):
    def __init__(self, config: Florence2VisionInferenceConfig):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NeuronFlorence2LanguageModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Florence2LanguageInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.d_model
        self.num_attention_heads = config.encoder_attention_heads
        self.num_key_value_heads = config.encoder_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Florence2LanguageInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.d_model,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=not config.neuron_config.vocab_parallel,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            sequence_dimension=1,
            pad=True,
        )

        self.embed_positions = LlamaRotaryEmbedding(
            config.d_model,
            max_position_embeddings=config.max_position_embeddings,
        )

        self.encoder_layers = nn.ModuleList(
            [NeuronFlorence2Layer(config, "spatial") for _ in range(config.encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [NeuronFlorence2Layer(config, "spatial") for _ in range(config.decoder_layers)]
        )

        self.lm_head = ColumnParallelLinear(
            config.d_model,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
            pad=True,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Encoder
        encoder_hidden_states = inputs_embeds
        for encoder_layer in self.encoder_layers:
            encoder_hidden_states = encoder_layer(encoder_hidden_states)

        # Decoder
        decoder_hidden_states = decoder_inputs_embeds
        for decoder_layer in self.decoder_layers:
            decoder_hidden_states = decoder_layer(decoder_hidden_states)

        lm_logits = self.lm_head(decoder_hidden_states)

        return (lm_logits,)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (NeuronFlorence2Encoder, NeuronFlorence2Decoder)):
            module.gradient_checkpointing = value


class NeuronFlorence2VisionModel(NeuronEncoderBase):
    """
    Neuron version of HF ViTModel

    Difference from original: Move/remove input arguments that result in dynamic graph at runtime.

    - Move init argument `use_mask_token`, `add_pooling_layer`, and `interpolate_pos_encoding` to
     `config:Florence2InferenceConfig`. All Default to `False`.

    - No `_prune_heads()` method. This is a feature in HF PretrainedModel that prune attention heads
     by `self.encoder.layer[layer].attention.prune_heads(heads)`. However, it is not used in all HF
     ViT models and not supported by `NeuronAttentionBase`.

    - The forward pass does not take `head_mask` input. This is the mask to nullify selected heads of
     the HF self-attention modules `ViTSelfAttention`. It is default to `None` in all HF ViT model.
     And `NeuronAttentionBase` does not support this.

    - The forward pass does not take `output_attentions` input. If set `True`, HF ViTModel outputs
     all raw attention weights after softmax. It is default to `None` in all HF ViT model and not
     supported by `NeuronAttentionBase`. And will increase the output tensor size and increase latency
     due to data transfer between devices.

    - The forward pass does not take `output_hidden_states` input. If set `True`, HF ViTModel outputs
     all hidden states of every ViT layer. It is default to `None` in all HF ViT model. And will increase
     the output tensor size and increase latency due to data transfer between devices.
    """

    def __init__(self, config: Florence2VisionInferenceConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = NeuronFlorence2Embeddings(config, use_mask_token=self.config.use_mask_token)
        self.encoder = NeuronFlorence2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = NeuronFlorence2Pooler(config) if self.config.add_pooling_layer else None

    def get_input_embeddings(self) -> NeuronFlorence2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[
            torch.BoolTensor
        ] = None,  # only used in ViTForMaskedImageModeling
    ) -> Tuple:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=self.config.interpolate_pos_encoding,
        )

        sequence_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        head_outputs = (
            (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        )
        return head_outputs


class NeuronFlorence2ForConditionalGeneration(NeuronBaseForCausalLM):
    _model_cls = NeuronFlorence2VisionModel

    def __init__(self, config: Florence2InferenceConfig, **kwargs):
        super().__init__("dummy_model_path", config=config, **kwargs)

        self.vision_model = NeuronFlorence2VisionModel(config.vision_config)
        self.language_model = NeuronFlorence2LanguageModel(config.text_config)

        self.image_projection = nn.Parameter(
            torch.empty(config.vision_config.dim_embed[-1], config.projection_dim)
        )
        self.image_proj_norm = nn.LayerNorm(config.projection_dim)

        # TODO: Add image_pos_embed and visual_temporal_embedding if needed

    def _encode_image(self, pixel_values):
        # This part is adapted from Florence2ForConditionalGeneration._encode_image
        # in Florence-2-base-ft/modeling_florence2.py
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_model(pixel_values)[0] # Assuming vision_model returns a tuple
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')

        # TODO: Add image_pos_embed and visual_temporal_embedding logic here

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.config.vision_config.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds
    ):
        batch_size, image_token_length = image_features.size()[:-1]
        device = image_features.device
        image_attention_mask = torch.ones(batch_size, image_token_length, device=device)

        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = torch.ones(batch_size, task_prefix_embeds.size(1), device=device)

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # concat [image embeds, task prefix embeds]
        inputs_embeds = torch.cat([image_features, task_prefix_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, task_prefix_attention_mask], dim=1)

        return inputs_embeds, attention_mask

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        image_features = None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                # (batch_size, num_image_tokens, hidden_size)
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0] # Assuming language_model returns a tuple with logits as the first element
        return (logits,)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        return self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def generate(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs
        ):

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

        return self.language_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            **kwargs
        )



class ModelWrapperFlorence2(ModelWrapper):
    """
    Neuron ModelWrapper class for NeuronFlorence2Model.
    Generates input shapes for trace and compilation. Disables bucketing.
    """

    def __init__(
        self,
        config: Florence2InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.tag = tag
        self.bucket_config = None  # Set to None because we don't have bucketing
        

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        # pixel_values
        pixel_values = torch.ones(
            [
                self.neuron_config.batch_size,
                3,
                self.config.vision_config.image_size,
                self.config.vision_config.image_size,
            ],
            dtype=self.neuron_config.torch_dtype
        )
        # input_ids
        input_ids = torch.ones(
            [self.neuron_config.batch_size, 1], dtype=torch.int64
        )
        # attention_mask
        attention_mask = torch.ones(
            [self.neuron_config.batch_size, 1], dtype=torch.int64
        )
        # decoder_input_ids
        decoder_input_ids = torch.ones(
            [self.neuron_config.batch_size, 1], dtype=torch.int64
        )
        # decoder_attention_mask
        decoder_attention_mask = torch.ones(
            [self.neuron_config.batch_size, 1], dtype=torch.int64
        )

        inputs = [
            (
                pixel_values,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
            )
        ]
        return inputs

    def get_model_instance(self):
        return DecoderModelInstance(model_cls=self.model_cls, config=self.config, model_init_kwargs=self.model_init_kwargs)

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        output = self._forward(*args)

        return output


class NeuronFlorence2ForConditionalGenerationApp(NeuronApplicationBase):
    """
    Neuron Application class for Florence2 conditional generation case.
    Wraps NeuronFlorence2ForConditionalGeneration with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronFlorence2ForConditionalGeneration

    def __init__(self, model_path: str, config: InferenceConfig, **kwargs):
        super().__init__(model_path=model_path, config=config, neuron_config=config.neuron_config)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return ModelWrapperFlorence2

    def forward(self, pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        return self.models[0](
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    def get_compiler_args(self):
        # Flag for model type
        compiler_args = "-O1 --model-type=transformer"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        # Prevent auto-down casting when running with fp32
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        logger.info(f"{self._model_cls.__name__} compiler_args: {compiler_args}")
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_state_dict = {}
        for key, value in state_dict.items():
            # Vision model mapping
            if "vision_tower" in key:
                new_key = key.replace("vision_tower.vision_tower", "vision_model")
                if "patch_embedding.projection.bias" in new_key:
                    # Shard the bias for OutputChannelParallelConv2d
                    tp_degree = config.neuron_config.tp_degree
                    if tp_degree > 1:
                        rank = torch.distributed.get_rank()
                        bias_per_rank = value.shape[0] // tp_degree
                        neuron_state_dict[new_key] = value[rank * bias_per_rank : (rank + 1) * bias_per_rank]
                    else:
                        neuron_state_dict[new_key] = value
                    continue
                if "patch_embedding" in new_key:
                    new_key = new_key.replace("patch_embedding", "embeddings.patch_embeddings.projection")
                if "attention.output" in new_key:
                    new_key = new_key.replace("attention.output.dense", "attention.o_proj")
                if "attention.self" in new_key:
                    new_key = new_key.replace("attention.self", "attention")
                    # Handle QKV split for vision attention
                    if "query.weight" in new_key:
                        qkv_weight = state_dict[key]
                        hidden_size = qkv_weight.shape[0] // 3
                        q_weight, k_weight, v_weight = torch.split(qkv_weight, hidden_size)
                        neuron_state_dict[new_key.replace("query", "q_proj")] = q_weight
                        neuron_state_dict[new_key.replace("query", "k_proj")] = k_weight
                        neuron_state_dict[new_key.replace("query", "v_proj")] = v_weight
                        if key.replace(".weight", ".bias") in state_dict:
                            qkv_bias = state_dict[key.replace(".weight", ".bias")]
                            q_bias, k_bias, v_bias = torch.split(qkv_bias, hidden_size)
                            neuron_state_dict[new_key.replace("query.weight", "q_proj.bias")] = q_bias
                            neuron_state_dict[new_key.replace("query.weight", "k_proj.bias")] = k_bias
                            neuron_state_dict[new_key.replace("query.weight", "v_proj.bias")] = v_bias
                        continue
                if "intermediate.dense" in new_key:
                    new_key = new_key.replace("intermediate.dense", "intermediate.dense")
                if "output.dense" in new_key:
                    new_key = new_key.replace("output.dense", "output.dense")
                if "layernorm_before" in new_key:
                    new_key = new_key.replace("layernorm_before", "layernorm_before")
                if "layernorm_after" in new_key:
                    new_key = new_key.replace("layernorm_after", "layernorm_after")
                neuron_state_dict[new_key] = value
            # Language model mapping
            elif "language_model" in key:
                new_key = key.replace("language_model.model.", "language_model.")
                if "shared.weight" in new_key:
                    neuron_state_dict[new_key.replace("shared.weight", "embed_tokens.weight")] = value
                    neuron_state_dict[new_key.replace("shared.weight", "lm_head.weight")] = value
                    continue
                if "encoder.layers" in new_key:
                    new_key = new_key.replace("encoder.layers", "encoder_layers")
                    # These are already separate linear layers in HF, no need to split
                    if "self_attn.q_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.q_proj", "attention.q_proj")] = value
                        continue
                    if "self_attn.k_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.k_proj", "attention.k_proj")] = value
                        continue
                    if "self_attn.v_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.v_proj", "attention.v_proj")] = value
                        continue
                    if "self_attn.out_proj" in new_key:
                        new_key = new_key.replace("self_attn.out_proj", "attention.o_proj")
                    if "self_attn_layer_norm" in new_key:
                        new_key = new_key.replace("self_attn_layer_norm", "layernorm_before")
                    if "final_layer_norm" in new_key:
                        new_key = new_key.replace("final_layer_norm", "layernorm_after")
                    if "fc1" in new_key:
                        new_key = new_key.replace("fc1", "intermediate.dense")
                    if "fc2" in new_key:
                        new_key = new_key.replace("fc2", "output.dense")
                    neuron_state_dict[new_key] = value
                elif "decoder.layers" in new_key:
                    new_key = new_key.replace("decoder.layers", "decoder_layers")
                    # These are already separate linear layers in HF, no need to split
                    if "self_attn.q_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.q_proj", "attention.q_proj")] = value
                        continue
                    if "self_attn.k_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.k_proj", "attention.k_proj")] = value
                        continue
                    if "self_attn.v_proj" in new_key:
                        neuron_state_dict[new_key.replace("self_attn.v_proj", "attention.v_proj")] = value
                        continue
                    if "self_attn.out_proj" in new_key:
                        new_key = new_key.replace("self_attn.out_proj", "attention.o_proj")
                    if "self_attn_layer_norm" in new_key:
                        new_key = new_key.replace("self_attn_layer_norm", "layernorm_before")
                    # Cross-attention layers
                    if "encoder_attn.q_proj" in new_key:
                        neuron_state_dict[new_key.replace("encoder_attn.q_proj", "cross_attention.q_proj")] = value
                        continue
                    if "encoder_attn.k_proj" in new_key:
                        neuron_state_dict[new_key.replace("encoder_attn.k_proj", "cross_attention.k_proj")] = value
                        continue
                    if "encoder_attn.v_proj" in new_key:
                        neuron_state_dict[new_key.replace("encoder_attn.v_proj", "cross_attention.v_proj")] = value
                        continue
                    if "encoder_attn.out_proj" in new_key:
                        new_key = new_key.replace("encoder_attn.out_proj", "cross_attention.o_proj")
                    if "encoder_attn_layer_norm" in new_key:
                        new_key = new_key.replace("encoder_attn_layer_norm", "cross_attention_layernorm")
                    if "final_layer_norm" in new_key:
                        new_key = new_key.replace("final_layer_norm", "layernorm_after")
                    if "fc1" in new_key:
                        new_key = new_key.replace("fc1", "intermediate.dense")
                    if "fc2" in new_key:
                        new_key = new_key.replace("fc2", "output.dense")
                    neuron_state_dict[new_key] = value
                elif "embed_positions.weight" in new_key:
                    neuron_state_dict[new_key] = value
                elif "layernorm_embedding" in new_key:
                    neuron_state_dict[new_key] = value
                else:
                    neuron_state_dict[new_key] = value
            # Image projection mapping
            elif "image_projection" in key:
                neuron_state_dict[key] = value
            elif "image_proj_norm" in key:
                neuron_state_dict[key] = value
            else:
                neuron_state_dict[key] = value
        return neuron_state_dict
