import jax
import jax.numpy as jnp


def adjust_position_ids(position_ids: jax.Array, pad_token_id: int):
    attention_mask = (position_ids != pad_token_id).astype(jnp.int32)
    cumsum = jnp.cumsum(attention_mask, axis=-1)
    cumsum = cumsum * attention_mask
    return cumsum


if __name__ == "__main__":
    padding_idx = 0

    left_ids = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 0, 0, 1, 2],
            [1, 2, 3, 4, 5],
        ]
    )

    right_ids = jnp.array(
        [
            [1, 2, 3, 4, 0],
            [1, 2, 0, 0, 0],
            [1, 2, 3, 4, 5],
        ]
    )

    shifted_left = adjust_position_ids(left_ids, padding_idx)
    shifted_right = adjust_position_ids(right_ids, padding_idx)

    print("fds")
