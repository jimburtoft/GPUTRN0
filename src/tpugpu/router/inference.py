from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from tpugpu.experts.train import latest_checkpoint_path
from tpugpu.router.model import RouterMLP
from tpugpu.router.train import RouterTrainConfig, RouterState, _create_router_state


def load_router_state(router_name: str, checkpoint_dir: str) -> tuple[RouterState, dict]:
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve() / router_name
    checkpoint_path = latest_checkpoint_path(str(checkpoint_root))
    if checkpoint_path is None:
        raise FileNotFoundError(f"No router checkpoint found for {router_name} under {checkpoint_root}")

    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    metadata = restored.get("metadata", {})
    config_dict = metadata.get("config", {})
    config = RouterTrainConfig(**config_dict) if config_dict else RouterTrainConfig(expert_names=())

    state = _create_router_state(config, jax.random.PRNGKey(config.seed))
    restored_state = restored["state"]
    if isinstance(restored_state, dict):
        restored_params = restored_state["params"]
        state = RouterState(
            step=restored_state["step"],
            apply_fn=state.apply_fn,
            params=restored_params,
            tx=state.tx,
            opt_state=state.tx.init(restored_params),
        )
    else:
        state = restored_state
    return state, {"checkpoint_path": checkpoint_path, "config": asdict(config)}


def predict_expert_id(state: RouterState, x_t, t, y) -> int:
    logits = state.apply_fn(
        {"params": state.params},
        jnp.asarray(x_t, dtype=jnp.float32),
        jnp.asarray(t, dtype=jnp.float32),
        jnp.asarray(y, dtype=jnp.int32),
    )
    return int(jnp.argmax(logits, axis=-1)[0])
