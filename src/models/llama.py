from dataclasses import dataclass

import jax.numpy as jnp

from src.common.base_modules import RMSNorm, RotaryEmbedding
from src.common.sharding import BaseModelShardingConfig
from src.common.types import ShardingRule


@dataclass(slots=True, frozen=True)
class LlamaShardingConfig(BaseModelShardingConfig):
    act_btd: ShardingRule
    act_btf: ShardingRule
    act_btnh: ShardingRule

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None

        return LlamaShardingConfig(
            embedding=(fsdp, "tp"),
            lm_head=("tp", fsdp),
            attn_q_weight=(None, "tp"),
            attn_kv_weight=(None, "tp"),
            attn_o_weight=("tp", None),
            ffn_up_proj=(None, "tp"),
            ffn_down_proj=("tp", None),
            rms_norm_scale=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
        )


class LlamaRMSNorm(RMSNorm):
    def __init__(
        self,
        num_features,
        *,
        rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        epsilon=0.000001,
        shardings=LlamaShardingConfig.get_default_sharding(),
    ):
        super().__init__(
            num_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=epsilon,
            shardings=shardings,
        )


class LlamaRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config, *, dtype=jnp.bfloat16, param_dtype=jnp.float32):
        super().__init__(config, dtype=dtype, param_dtype=param_dtype)
