import jax

if __name__ == "__main__":
    with jax.profiler.trace("./logs"):
        key = jax.random.key(0)
        x = jax.random.normal(key, (1024, 1024))
        y = x @ x
        _ = jax.block_until_ready(y)
