import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.nnx.nn.linear import default_embed_init
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from src.common.model import ZooModel, stack_layers
from src.common.modules import (
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    SwigluMLP,
    TransformerDecoderLayer,
)
from src.common.sharding import BaseModelShardingConfig
from src.common.types import ShardingRule
from src.inference import GenerationMixin


# adapted from https://github.com/google/tunix/blob/main/tunix/models/llama3/model.py
@dataclass(slots=True, frozen=True)
class Qwen3ShardingConfig(BaseModelShardingConfig):
    act_btd: ShardingRule
    act_btf: ShardingRule
    act_btnh: ShardingRule

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None

        return Qwen3ShardingConfig(
            embedding=("tp", fsdp),
            lm_head=(fsdp, "tp"),
            attn_q_weight=("tp", fsdp, None),
            attn_kv_weight=("tp", fsdp, None),
            attn_o_weight=("tp", None, fsdp),
            ffn_up_proj=(fsdp, "tp"),
            ffn_down_proj=("tp", fsdp),
            rms_norm_scale=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
        )


class Qwen3RMSNorm(RMSNorm):
    def __init__(
        self,
        num_features,
        *,
        shardings,
        rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        epsilon=0.000001,
    ):
        super().__init__(
            num_features,
            shardings=shardings,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=epsilon,
        )


class Qwen3MLP(SwigluMLP):
    def __init__(
        self, config, *, shardings, rngs, dtype=jnp.bfloat16, param_dtype=jnp.float32
    ):
        super().__init__(
            config, shardings=shardings, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )


class Qwen3MultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        config,
        layer_idx,
        *,
        shardings,
        rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        super().__init__(
            config,
            layer_idx,
            shardings=shardings,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )


class Qwen3DecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx,
        *,
        shardings,
        rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        super().__init__(
            config,
            layer_idx,
            shardings=shardings,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )


class Qwen3RotaryEmbedding(RotaryEmbedding):
    def __init__(self, config, *, dtype=jnp.bfloat16, param_dtype=jnp.float32):
        super().__init__(config, dtype=dtype, param_dtype=param_dtype)


class Qwen3Model(nnx.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        shardings=Qwen3ShardingConfig.get_default_sharding(),
    ):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nnx.with_partitioning(
                default_embed_init,
                sharding=shardings.embedding,
            ),
        )
        layers = [
            Qwen3DecoderLayer(
                config,
                layer_idx,
                rngs=rngs,
                dtype=dtype,
                shardings=shardings,
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.layers: Qwen3DecoderLayer = stack_layers(layers)

        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.rotary_emb = Qwen3RotaryEmbedding(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.has_sliding_layers = "sliding_attention" in config.layer_types

    def __call__(
        self,
        input_ids: tp.Optional[jax.Array] = None,
        position_ids: tp.Optional[jax.Array] = None,
        past_key_values: tp.Optional[nnx.Cache] = None,
        use_cache: tp.Optional[bool] = None,
        cache_position: tp.Optional[jax.Array] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        # for now do the computation without caching, we will handle that
        # later on, for now, we need the computation to be correct

        position_ids = jnp.arange(0, hidden_states.shape[1])
        position_ids = jnp.expand_dims(position_ids, 0)
        position_embeddings = self.rotary_emb(position_ids)

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def _layer_scan(layer: Qwen3DecoderLayer, carry: jax.Array):
            state = layer(
                hidden_states=carry,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )

            return state

        hidden_states = _layer_scan(self.layers, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


def create_causal_lm_params_mapping(config: Qwen3Config):
    attention_mapping = {
        "model.layers.{idx}.input_layernorm.scale": "model.layers.{idx}.input_layernorm.weight",
        "model.layers.{idx}.mlp.down_proj.kernel": "model.layers.{idx}.mlp.down_proj.weight",
        "model.layers.{idx}.mlp.gate_proj.kernel": "model.layers.{idx}.mlp.gate_proj.weight",
        "model.layers.{idx}.mlp.up_proj.kernel": "model.layers.{idx}.mlp.up_proj.weight",
        "model.layers.{idx}.post_attention_layernorm.scale": "model.layers.{idx}.post_attention_layernorm.weight",
        "model.layers.{idx}.self_attn.k_norm.scale": "model.layers.{idx}.self_attn.k_norm.weight",
        "model.layers.{idx}.self_attn.k_proj.kernel": "model.layers.{idx}.self_attn.k_proj.weight",
        "model.layers.{idx}.self_attn.o_proj.kernel": "model.layers.{idx}.self_attn.o_proj.weight",
        "model.layers.{idx}.self_attn.q_norm.scale": "model.layers.{idx}.self_attn.q_norm.weight",
        "model.layers.{idx}.self_attn.q_proj.kernel": "model.layers.{idx}.self_attn.q_proj.weight",
        "model.layers.{idx}.self_attn.v_proj.kernel": "model.layers.{idx}.self_attn.v_proj.weight",
    }

    layers_mappings = []

    for idx in range(config.num_hidden_layers):
        mapping = {
            k.format(idx=idx): v.format(idx=idx) for k, v in attention_mapping.items()
        }

        layers_mappings.append(mapping)

    skeleton = {
        "lm_head.kernel": "lm_head.weight",
        "model.embed_tokens.embedding": "model.embed_tokens.weight",
        "model.norm.scale": "model.norm.weight",
    }

    for mapping in layers_mappings:
        skeleton.update(**mapping)

    return skeleton


class Qwen3ForCausalLM(ZooModel, GenerationMixin):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        shardings=Qwen3ShardingConfig.get_default_sharding(),
    ):
        self.vocab_size = config.vocab_size
        self.PARAMS_MAPPING = create_causal_lm_params_mapping(config)

        self.model = Qwen3Model(
            config,
            rngs=rngs,
            shardings=shardings,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.lm_head,
            ),
        )

    def __call__(
        self,
        input_ids: tp.Optional[jax.Array] = None,
        position_ids: tp.Optional[jax.Array] = None,
        past_key_values: tp.Optional[nnx.Cache] = None,
        use_cache: tp.Optional[bool] = None,
        cache_position: tp.Optional[jax.Array] = None,
        logits_to_keep: tp.Union[int, jax.Array] = 0,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits
