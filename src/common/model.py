import typing as tp
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from src.common.sharding import BaseModelShardingConfig


def stack_layers(layers: tp.Sequence[nnx.Module]) -> nnx.Module:
    """Stack the modules into a single module with batched parameters"""

    # Split each layer into graphdef and state
    graphdefs, states, remainder_state = zip(
        *(nnx.split(layer, nnx.Param, ...) for layer in layers)
    )

    # All layers should have the same structure => use the first graphdef as template
    template_graphdef = graphdefs[0]

    # Stack the states (parameters) along a new axis
    stacked_state = jax.tree.map(lambda *args: jnp.stack(args, axis=0), *states)
    remainder_state = jax.tree.map(
        lambda *args: jnp.stack(args, axis=0), *remainder_state
    )

    merged_state = nnx.merge_state(stacked_state, remainder_state)

    # Create the stacked layers by merging template graphdef with stacked state
    stacked_layers = nnx.merge(template_graphdef, merged_state)
    return stacked_layers


def unstack_layers(layers: nnx.Module, num_hidden_layers: int) -> nnx.List[nnx.Module]:
    """
    Reverses the stacking operation to retrieve a list of individual layer modules.

    This function takes a single module where parameters and other state variables
    have been stacked along a new leading axis and converts it back into a list
    of separate, independent modules.

    Returns:
        A list of individual nnx.Module layers.
    """

    # The graph definition is the shared structure for all layers
    graphdef, stacked_param_state = nnx.split(layers, ...)

    # unstack the parameters, we create a list of parameter states,
    # where each element corresponds to one layer
    unstacked_params = [
        jax.tree.map(lambda leaf: leaf[i], stacked_param_state)
        for i in range(num_hidden_layers)
    ]

    # reconstruct each layer module.
    reconstructed_layers = []
    for i in range(num_hidden_layers):
        # For each layer, get its state
        param_state_i = unstacked_params[i]
        layer_i = nnx.merge(graphdef, param_state_i)
        reconstructed_layers.append(layer_i)

    return nnx.List(reconstructed_layers)


class VmappableLayersMixin:
    num_hidden_layers: int
    layers: tp.Union[nnx.Module, nnx.List]

    def stack_layers(self):
        """
        Batches params of the model's layers property along a leading axis
        to allow inference with a function wrapped with nnx.scan
        """

        layers_sequence = self.layers
        assert len(layers_sequence) == self.num_hidden_layers, (
            f"The SequentialWrapper has {len(layers_sequence)} modules while the model expects {self.num_hidden_layers} modules"
        )

        #! set sequential layers to None to avoid state conflicts or mixed values in model state
        self.layers = None
        self.layers = stack_layers(layers_sequence)

    def unstack_layers(self):
        """
        Wraps the backbone batched layers around a SequentialWrapper module to
        mimick the structure of torch.nn.Sequential, where each passed module is
        an attribute with an integer-based index.

        Don't use a module with SequentialWrapper for inference, this transformation
        is mainly used for loading weights from HF models.
        """
        layers_seq = unstack_layers(self.layers, self.num_hidden_layers)

        #! need to set it to None, otherwise Flax will merge the states of both batched and sequential layers
        self.layers = None
        self.layers = layers_seq


class ZooModel(nnx.Module):
    PARAMS_MAPPING: dict

    def create_state_from_numpy_weights(
        self,
        state_dict: dict[str, np.ndarray],
    ) -> nnx.State:
        def _load_weights(path, leaf):
            # stop at path[:-1] to not include the ".value" of each nnx.Param object
            full_path = ".".join(map(lambda p: str(p.key), path[:-1]))
            torch_key = self.PARAMS_MAPPING.get(full_path, None)
            if torch_key is None:
                return leaf

            weight = state_dict[torch_key]
            if full_path.endswith("kernel"):
                if weight.ndim == 2:  # standard linear layer
                    return jnp.array(weight.T)
                else:
                    # something like a convolution kernel
                    # torch has the in_features as last dimension while Flax
                    # always has the out_features on the trailing axis
                    return jnp.array(weight.swapaxes(-1, -2))

            return jnp.array(weight)

        updated_state = jax.tree.map_with_path(_load_weights, nnx.state(self))
        return updated_state

    @classmethod
    def from_config(
        cls,
        config: AutoConfig,
        *,
        seed: int,
        mesh: jax.sharding.Mesh,
        shardings=BaseModelShardingConfig.get_default_sharding(),
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        @partial(
            nnx.jit,
            static_argnames=("seed", "config_fn", "shardings", "dtype", "param_dtype"),
        )
        def create_sharded_model(
            seed: int,
            shardings: BaseModelShardingConfig,
            config_fn: tp.Callable[[], AutoConfig],
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        ):
            rngs = nnx.Rngs(seed)
            model = cls(
                config_fn(),
                rngs=rngs,
                shardings=shardings,
                dtype=dtype,
                param_dtype=param_dtype,
            )

            return model, rngs

        model: tp.Self
        with jax.set_mesh(mesh):
            model, rngs = create_sharded_model(
                seed,
                shardings,
                lambda: config,
                dtype,
                param_dtype,
            )

        return model, rngs

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        local_dir: Path,
        seed: int,
        shardings: tp.Any,
        mesh: jax.sharding.Mesh,
        checkpoint_step: int = 0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        token: tp.Optional[str] = None,
        revision: tp.Optional[str] = None,
    ):
        if not local_dir.exists():
            local_dir.mkdir(parents=True)

        snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=local_dir,
            revision=revision,
        )

        config = AutoConfig.from_pretrained(local_dir)

        @partial(
            nnx.jit,
            static_argnames=("seed", "config_fn", "shardings", "dtype", "param_dtype"),
        )
        def create_sharded_model(
            seed: int,
            shardings: BaseModelShardingConfig,
            config_fn: tp.Callable[[], AutoConfig],
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        ):
            rngs = nnx.Rngs(seed)
            model = cls(
                config_fn(),
                rngs=rngs,
                shardings=shardings,
                dtype=dtype,
                param_dtype=param_dtype,
            )

            return model, rngs

        model: tp.Self
        with jax.set_mesh(mesh):
            jax.debug.print("=" * 30)
            jax.debug.print("Creating {name} model", name=cls.__name__)

            model, rngs = create_sharded_model(
                seed,
                shardings,
                lambda: config,
                dtype,
                param_dtype,
            )

            with ocp.CheckpointManager(local_dir) as mngr:
                jax.debug.print("=" * 30)
                jax.debug.print(
                    "Ensuring that the sharding matches the current device topology"
                )

                # `get_abstract_model` handles device topology changes
                graphdef, abstract_state = nnx.get_abstract_model(lambda: model, mesh)

                # def set_sharding(x: jax.ShapeDtypeStruct):
                #     spec = x.sharding.spec
                #     return x.update(sharding=NamedSharding(mesh, spec))

                # new_topology_state = jax.tree.map(set_sharding, abstract_state)

                jax.debug.print("=" * 30)
                jax.debug.print("Restoring checkpoint state")
                restored = mngr.restore(
                    checkpoint_step, args=ocp.args.StandardRestore(abstract_state)
                )

            nnx.update(model, restored)

        return model, rngs
