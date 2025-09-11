# HF Zoo

A collection of popular Hugging Face models ported to JAX/Flax NNX.

## Overview

This repository contains implementations of popular language models from Hugging Face's model hub, reimplemented using JAX and Flax NNX. The goal is to provide high-performance, scalable versions of these models that can take advantage of JAX's compilation and automatic differentiation capabilities.

## Features

- **JAX/Flax NNX Implementation**: All models are built using Flax's new NNX API for better modularity and ease of use
- **Sharded Model Loading**: Support for loading large models with parameter sharding across devices
- **Efficient Inference**: Optimized inference routines with JAX compilation
- **HuggingFace Compatibility**: Easy loading from HuggingFace model repositories

## Installation

This project uses `uv` for dependency management. To install the dependencies:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --group dev
```

## Requirements

- Python >= 3.12
- JAX >= 0.7.1
- Flax >= 0.11.2
- Transformers >= 4.56.1


## Usage

### Loading a Model

```python
import jax
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
from hf_zoo.src.models.qwen3 import Qwen3ForCausalLM

# Create mesh for sharding
mesh = jax.make_mesh((1, 8), ("dp", "tp")) # On a TPUv2-8 for instance
rngs = nnx.Rngs(456)

# Load model from HuggingFace
model = Qwen3ForCausalLM.from_pretrained(
    "REPO_ID",  # Replace with actual repo ID
    local_dir=Path("./models/qwen3-0.6B"),
    rngs=rngs,
    mesh=mesh,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
)
```

### Generation

```python
# Generate text
input_ids = jnp.array([[1, 2, 3, 4]])  # Your tokenized input
key = rngs()

generated = model.generate(
    key=key,
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.7
)
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License.