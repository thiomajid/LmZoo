from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from src.common.types import ShardingRule


# copied from https://github.com/google/tunix/blob/main/tunix/models/llama3/model.py
def shard(x: jnp.ndarray, s: tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == "cpu":
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*s)))


@dataclass(slots=True, frozen=True)
class BaseModelShardingConfig:
    embedding: ShardingRule
    lm_head: ShardingRule
    attn_q_weight: ShardingRule
    attn_kv_weight: ShardingRule
    attn_o_weight: ShardingRule
    ffn_up_proj: ShardingRule
    ffn_down_proj: ShardingRule
    rms_norm_scale: ShardingRule

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None

        return BaseModelShardingConfig(
            embedding=(fsdp, "tp"),
            lm_head=("tp", fsdp),
            attn_q_weight=(None, "tp"),
            attn_kv_weight=(None, "tp"),
            attn_o_weight=("tp", None),
            ffn_up_proj=(None, "tp"),
            ffn_down_proj=("tp", None),
            rms_norm_scale=("tp",),
        )
