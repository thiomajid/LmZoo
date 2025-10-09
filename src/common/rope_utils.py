"""
JAX-compatible RoPE utilities adapted from transformers.modeling_rope_utils.

This module provides various RoPE (Rotary Position Embedding) implementations
including default, linear scaling, dynamic NTK, YaRN, LongRoPE, and Llama3 variants.
"""

import math
import typing as tp
from functools import partial, wraps

import jax
import jax.numpy as jnp
from transformers import PretrainedConfig

RopeInitFn = tp.Callable[
    [tp.Optional[PretrainedConfig], tp.Optional[int]], tuple[jax.Array, float]
]


def _compute_default_rope_parameters(
    config: tp.Optional[PretrainedConfig] = None,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: tp.Optional[PretrainedConfig] = None,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies with linear scaling.

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    factor = config.rope_scaling["factor"]

    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config)

    # Then applies linear scaling to the frequencies.
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: tp.Optional[PretrainedConfig] = None,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies with NTK scaling.

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length, used to update the dynamic RoPE at inference time.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, jax.Array):
        seq_len = jnp.maximum(seq_len, max_position_embeddings).astype(seq_len.dtype)
    else:
        seq_len = max(seq_len, max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (
        dim / (dim - 2)
    )
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies with YaRN scaling.

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    factor = config.rope_scaling["factor"]
    attention_factor = config.rope_scaling.get("attention_factor")
    mscale = config.rope_scaling.get("mscale")
    mscale_all_dim = config.rope_scaling.get("mscale_all_dim")
    original_max_position_embeddings = (
        config.rope_scaling.get("original_max_position_embeddings")
        or config.max_position_embeddings
    )

    def get_mscale(scale, mscale=1):
        if mscale <= 0:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )

        else:
            attention_factor = get_mscale(factor, mscale)

    # Optional config options
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (
            dim
            * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot,
        high_rot,
        dim,
        base,
        max_position_embeddings,
        truncate,
    ):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001

        linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (
            max_val - min_val
        )
        ramp_func = jnp.clip(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_scaling.get("truncate", True)
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate
    )

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: PretrainedConfig,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies with LongRoPE scaling.

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    long_factor = jnp.array(config.rope_scaling["long_factor"], dtype=jnp.float32)
    short_factor = jnp.array(config.rope_scaling["short_factor"], dtype=jnp.float32)
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")

    # NOTE: Handle models that modify `max_position_embeddings`
    if hasattr(config, "original_max_position_embeddings"):
        original_max_position_embeddings = config.original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = long_factor
    else:
        ext_factors = short_factor.astype(jnp.float32)
    inv_freq_shape = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: PretrainedConfig,
    seq_len: tp.Optional[int] = None,
) -> tuple[jax.Array, float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config: The model configuration.
        device: Unused in JAX (kept for compatibility).
        seq_len: The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (inv_freq, attention_factor), containing the inverse frequencies
        for the RoPE embeddings and the post-processing scaling factor.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling[
        "low_freq_factor"
    ]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling[
        "high_freq_factor"
    ]  # `4` in the original implementation
    old_context_len = config.rope_scaling[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS: dict[str, RopeInitFn] = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}


# dynamic rope update decorator definition
def _longrope_frequency_update(rope_module, position_ids):
    seq_len = jnp.max(position_ids) + 1
    if seq_len > rope_module.max_seq_len_cached:
        # LongRoPE updates the frequencies based on the sequence length
        inv_freq, attention_scaling = rope_module.rope_init_fn(
            rope_module.config, seq_len=seq_len.item()
        )

        rope_module.attention_scaling = attention_scaling
        rope_module.inv_freq = inv_freq
        rope_module.max_seq_len_cached = seq_len


def _dynamic_frequency_update(rope_module, position_ids):
    seq_len = jnp.max(position_ids) + 1
    if seq_len > rope_module.max_seq_len_cached:
        # Dynamic NTK updates the frequencies based on the sequence length
        inv_freq, attention_scaling = rope_module.rope_init_fn(
            rope_module.config, seq_len=seq_len.item()
        )
        rope_module.attention_scaling = attention_scaling
        rope_module.inv_freq = inv_freq
        rope_module.max_seq_len_cached = seq_len


@partial(jax.profiler.annotate_function, name="Dynamic RoPE Update")
def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable): The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
    """

    @wraps(rope_forward)
    def wrapper(self, position_ids):
        if self.rope_type == "longrope":
            _longrope_frequency_update(self, position_ids)
        elif self.rope_type == "dynamic":
            _dynamic_frequency_update(self, position_ids)

        return rope_forward(self, position_ids)

    return wrapper
