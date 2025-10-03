import jax
from huggingface_hub import upload_file

upload_file(repo_id="", path_or_fileobj="", path_in_repo="", token="")

if __name__ == "__main__":
    with jax.profiler.trace("./logs"):
        key = jax.random.key(0)
        x = jax.random.normal(key, (1024, 1024))
        y = x @ x
        _ = jax.block_until_ready(y)
