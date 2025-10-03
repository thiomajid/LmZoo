import typing as tp

import jax

PositionEmbeddings = tuple[jax.Array, jax.Array]
"""Cosine and sinus arrays for RoPE"""

ShardingRule = tuple[tp.Optional[str], ...]
