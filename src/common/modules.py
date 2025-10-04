import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from flax.nnx import initializers
from flax.nnx.nn import dtypes
from flax.typing import Shape
from jax import lax
from transformers import PretrainedConfig

from src.common.sharding import BaseModelShardingConfig
from src.common.types import PositionEmbeddings, ShardingRule


# adapted from https://github.com/google/tunix/blob/main/tunix/models/qwen3/model.py
class Einsum(nnx.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(
        self,
        einsum_str: str,
        shape: Shape,
        *,
        rngs: nnx.Rngs,
        sharding: ShardingRule,
    ):
        self.einsum_str = einsum_str
        self.shape = shape
        self.kernel = nnx.Param(
            nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
        )

    @jax.named_scope("Einsum")
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.einsum(self.einsum_str, x, self.kernel.value)


class RMSNorm(nnx.Module):
    def __init__(
        self,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        epsilon: float = 1e-6,
        shardings=BaseModelShardingConfig.get_default_sharding(),
    ):
        scale_init = nnx.with_partitioning(
            initializers.ones_init(),
            sharding=shardings.rms_norm_scale,
        )

        self.scale = nnx.Param(scale_init(rngs(), (num_features,), param_dtype))

        self.epsilon = epsilon
        self.dtype = dtype
        self.promote_dtype = dtypes.promote_dtype

    @jax.named_scope("RMSNorm")
    def __call__(self, h_t: jax.Array):
        dtype = h_t.dtype
        h_t = h_t.astype(jnp.float32)
        variance = jnp.pow(h_t, 2).mean(-1, keepdims=True)
        h_t = h_t * lax.rsqrt(variance + self.epsilon)

        scale, h_t = self.promote_dtype((self.scale.value, h_t), dtype=self.dtype)
        h_t = scale * h_t
        return h_t.astype(dtype)


class SwigluMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        shardings=BaseModelShardingConfig.get_default_sharding(),
    ):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        Linear = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.ffn_up_proj,
            ),
        )

        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.ffn_up_proj,
            ),
        )

        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.ffn_down_proj,
            ),
        )

        self.act_fn = jax.nn.silu

    @jax.named_scope("MLP")
    def __call__(self, x: jax.Array):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x: jax.Array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int = 1,
):
    # The unsqueeze dim is used to broadcast along the heads dimension.
    cos = jnp.expand_dims(cos, unsqueeze_dim)
    sin = jnp.expand_dims(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.expand_dims(hidden_states, axis=2)
    hidden_states = jnp.broadcast_to(
        hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim)
    )

    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        shardings=BaseModelShardingConfig.get_default_sharding(),
    ):
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        Linear = partial(
            nnx.Linear,
            use_bias=config.attention_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.q_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_attention_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_q_weight,
            ),
        )

        self.k_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_kv_weight,
            ),
        )

        self.v_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_kv_weight,
            ),
        )

        self.o_proj = Linear(
            in_features=config.num_attention_heads * self.head_dim,
            out_features=config.hidden_size,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_o_weight,
            ),
        )

        Norm = partial(
            RMSNorm,
            num_features=self.head_dim,
            epsilon=config.rms_norm_eps,
            shardings=shardings,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.q_norm = Norm()
        self.k_norm = Norm()

        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = dtypes.promote_dtype

    @jax.named_scope("MultiHeadAttention")
    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        use_cache: bool = False,
        past_key_values: tp.Optional[nnx.Cache] = None,
        cache_position: tp.Optional[jax.Array] = None,
    ):
        B, S, _ = hidden_states.shape

        def _apply_qkv(x: jax.Array, proj: nnx.Linear, norm=jax.nn.identity):
            out = rearrange(proj(x), "b s (nh hd) -> b s nh hd", hd=self.head_dim)
            # out = rearrange(norm(out), "b s nh hd -> b nh s hd")
            out = norm(out)
            return out

        query_states = _apply_qkv(hidden_states, self.q_proj, self.q_norm)
        key_states = _apply_qkv(hidden_states, self.k_proj, self.k_norm)
        value_states = _apply_qkv(hidden_states, self.v_proj)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos=cos,
            sin=sin,
            unsqueeze_dim=2,
        )

        q, k, v = self.promote_dtype(
            (query_states, key_states, value_states),
            dtype=self.dtype,
        )

        attn_output = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scaling,
            is_causal=True,
            local_window_size=self.sliding_window,
            implementation="xla",
        )

        attn_output = attn_output.reshape(B, S, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        shardings=BaseModelShardingConfig.get_default_sharding(),
    ):
        self.hidden_size = config.hidden_size

        self.self_attn = MultiHeadAttention(
            config=config,
            layer_idx=layer_idx,
            rngs=rngs,
            shardings=shardings,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.mlp = SwigluMLP(
            config,
            rngs=rngs,
            shardings=shardings,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        Norm = partial(
            RMSNorm,
            num_features=config.hidden_size,
            epsilon=config.rms_norm_eps,
            rngs=rngs,
            shardings=shardings,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.attention_type = config.layer_types[layer_idx]

    @jax.named_scope("TransformerDecoderLayer")
    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: PositionEmbeddings = None,
        past_key_values: tp.Optional[nnx.Cache] = None,
        use_cache: tp.Optional[bool] = False,
        cache_position: tp.Optional[jax.Array] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.dtype = dtype
        self.param_dtype = param_dtype

        inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )

        # Store as a non-trainable buffer
        self.inv_freq = inv_freq

    @jax.named_scope("RoPE")
    def __call__(self, position_ids: jax.Array):
        # position_ids: [batch_size, seq_len]
        t = position_ids.astype(self.inv_freq.dtype)
        # freqs: [batch_size, seq_len, dim // 2]
        freqs = jnp.einsum("...i,j->...ij", t, self.inv_freq)
        # emb: [batch_size, seq_len, dim]
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        # cos/sin: [batch_size, seq_len, dim]
        cos = jnp.cos(emb).astype(self.dtype)
        sin = jnp.sin(emb).astype(self.dtype)

        return cos, sin
