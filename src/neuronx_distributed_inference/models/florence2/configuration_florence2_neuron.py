from typing import List, Type
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

class Florence2NeuronConfig(NeuronConfig):
    # Add Florence-2 specific Neuron configurations here
    # For example, if there are specific quantization or attention implementations
    # that are unique to Florence-2's architecture on Neuron.
    pass

class Florence2VisionInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "drop_path_rate",
            "patch_size",
            "patch_stride",
            "patch_padding",
            "patch_prenorm",
            "enable_checkpoint",
            "dim_embed",
            "num_heads",
            "num_groups",
            "depths",
            "window_size",
            "projection_dim",
            "visual_temporal_embedding",
            "image_pos_embed",
            "image_feature_source",
        ]
    @classmethod
    def from_hf_config(cls, hf_config, neuron_config: Florence2NeuronConfig, **kwargs):
        config = cls(
            neuron_config=neuron_config,
            drop_path_rate=hf_config.vision_config.drop_path_rate,
            patch_size=hf_config.vision_config.patch_size,
            patch_stride=hf_config.vision_config.patch_stride,
            patch_padding=hf_config.vision_config.patch_padding,
            patch_prenorm=hf_config.vision_config.patch_prenorm,
            enable_checkpoint=hf_config.vision_config.enable_checkpoint,
            dim_embed=hf_config.vision_config.dim_embed,
            num_heads=hf_config.vision_config.num_heads,
            num_groups=hf_config.vision_config.num_groups,
            depths=hf_config.vision_config.depths,
            window_size=hf_config.vision_config.window_size,
            projection_dim=hf_config.vision_config.projection_dim,
            visual_temporal_embedding=hf_config.vision_config.visual_temporal_embedding,
            image_pos_embed=hf_config.vision_config.image_pos_embed,
            image_feature_source=hf_config.vision_config.image_feature_source,
            **kwargs,
        )
        return config


class Florence2LanguageInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "vocab_size",
            "max_position_embeddings",
            "encoder_layers",
            "encoder_ffn_dim",
            "encoder_attention_heads",
            "decoder_layers",
            "decoder_ffn_dim",
            "decoder_attention_heads",
            "encoder_layerdrop",
            "decoder_layerdrop",
            "activation_function",
            "d_model",
            "dropout",
            "attention_dropout",
            "activation_dropout",
            "init_std",
            "classifier_dropout",
            "scale_embedding",
            "use_cache",
            "num_labels",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "is_encoder_decoder",
            "decoder_start_token_id",
            "forced_eos_token_id",
        ]
    @classmethod
    def from_hf_config(cls, hf_config, neuron_config: Florence2NeuronConfig, **kwargs):
        config = cls(
            neuron_config=neuron_config,
            vocab_size=hf_config.text_config.vocab_size,
            max_position_embeddings=hf_config.text_config.max_position_embeddings,
            encoder_layers=hf_config.text_config.encoder_layers,
            encoder_ffn_dim=hf_config.text_config.encoder_ffn_dim,
            encoder_attention_heads=hf_config.text_config.encoder_attention_heads,
            decoder_layers=hf_config.text_config.decoder_layers,
            decoder_ffn_dim=hf_config.text_config.decoder_ffn_dim,
            decoder_attention_heads=hf_config.text_config.decoder_attention_heads,
            encoder_layerdrop=hf_config.text_config.encoder_layerdrop,
            decoder_layerdrop=hf_config.text_config.decoder_layerdrop,
            activation_function=hf_config.text_config.activation_function,
            d_model=hf_config.text_config.d_model,
            dropout=hf_config.text_config.dropout,
            attention_dropout=hf_config.text_config.attention_dropout,
            activation_dropout=hf_config.text_config.activation_dropout,
            init_std=hf_config.text_config.init_std,
            classifier_dropout=hf_config.text_config.classifier_dropout,
            scale_embedding=hf_config.text_config.scale_embedding,
            use_cache=hf_config.text_config.use_cache,
            num_labels=hf_config.text_config.num_labels,
            pad_token_id=hf_config.text_config.pad_token_id,
            bos_token_id=hf_config.text_config.bos_token_id,
            eos_token_id=hf_config.text_config.eos_token_id,
            is_encoder_decoder=hf_config.text_config.is_encoder_decoder,
            decoder_start_token_id=hf_config.text_config.decoder_start_token_id,
            forced_eos_token_id=hf_config.text_config.forced_eos_token_id,
            **kwargs,
        )
        return config

class Florence2InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        super().add_derived_config()
        # Propagate neuron_config to nested configs
        self.vision_config.neuron_config = self.neuron_config
        self.text_config.neuron_config = self.neuron_config

    def get_required_attributes(self) -> List[str]:
        return [
            "vocab_size",
            "projection_dim",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "is_encoder_decoder",
            "decoder_start_token_id",
            "forced_eos_token_id",
            "vision_config",
            "text_config",
        ]

    @classmethod
    def get_nested_configs(cls) -> List[str]:
        return ["vision_config", "text_config"]

    @classmethod
    def get_config_cls_for_nested_config(cls, nested_config_name: str) -> Type[InferenceConfig]:
        if nested_config_name == "vision_config":
            return Florence2VisionInferenceConfig
        elif nested_config_name == "text_config":
            return Florence2LanguageInferenceConfig
        else:
            raise ValueError(f"Unknown nested config name: {nested_config_name}")
    @classmethod
    def from_hf_config(cls, hf_config, neuron_config: Florence2NeuronConfig, **kwargs):
        vision_config = Florence2VisionInferenceConfig.from_hf_config(hf_config, neuron_config, **kwargs)
        text_config = Florence2LanguageInferenceConfig.from_hf_config(hf_config, neuron_config, **kwargs)
        config = cls(
            neuron_config=neuron_config,
            vision_config=vision_config,
            text_config=text_config,
            vocab_size=hf_config.text_config.vocab_size,
            projection_dim=hf_config.vision_config.projection_dim,
            pad_token_id=hf_config.text_config.pad_token_id,
            bos_token_id=hf_config.text_config.bos_token_id,
            eos_token_id=hf_config.text_config.eos_token_id,
            is_encoder_decoder=hf_config.text_config.is_encoder_decoder,
            decoder_start_token_id=hf_config.text_config.decoder_start_token_id,
            forced_eos_token_id=hf_config.text_config.forced_eos_token_id,
            **kwargs
        )
        config.add_derived_config()
        return config
