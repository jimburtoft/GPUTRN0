from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from tpugpu.config import ExpertTrainConfig
from tpugpu.experts.train import TrainState, create_train_state, latest_checkpoint_path


def load_expert_state(
    expert_name: str,
    checkpoint_dir: str,
    *,
    batch_size: int = 1,
    image_size: int = 32,
    num_channels: int = 1,
    num_classes: int = 10,
    hidden_channels: int = 64,
    seed: int = 0,
) -> TrainState:
    config = ExpertTrainConfig(
        expert_name=expert_name,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_channels=num_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        seed=seed,
    )
    init_state = create_train_state(config, jax.random.PRNGKey(seed))
    checkpoint_path = latest_checkpoint_path(f"{checkpoint_dir}/{expert_name}")
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for expert {expert_name!r} in {checkpoint_dir!r}")

    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    restored_state = restored["state"]
    if isinstance(restored_state, dict):
        return replace(
            init_state,
            step=restored_state["step"],
            params=restored_state["params"],
            opt_state=restored_state["opt_state"],
        )
    return replace(
        init_state,
        step=restored_state.step,
        params=restored_state.params,
        opt_state=restored_state.opt_state,
    )


@jax.jit
def predict_velocity(state: TrainState, x_t: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
    return state.apply_fn({"params": state.params}, x_t, t, y)


def predict_velocity_numpy(state: TrainState, x_t: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_t_jax = jnp.asarray(x_t, dtype=jnp.float32)
    t_jax = jnp.asarray(t, dtype=jnp.float32)
    y_jax = jnp.asarray(y, dtype=jnp.int32)
    velocity = predict_velocity(state, x_t_jax, t_jax, y_jax)
    return np.asarray(velocity, dtype=np.float32)
