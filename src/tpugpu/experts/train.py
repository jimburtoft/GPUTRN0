from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import flax
from flax.training import train_state
import jax
import jax.image
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from tpugpu.config import ExpertTrainConfig
from tpugpu.data.mnist import (
    NumpyDataset,
    batch_iterator,
    filter_by_class_ids,
    filter_by_cluster_id,
    load_mnist_numpy,
    split_cluster_assignments,
)
from tpugpu.eval.reporting import (
    compute_pca_fid,
    ensure_dir,
    save_image_grid,
    save_json,
    save_label_histogram,
    save_training_curves,
    save_tsne_plot,
)
from tpugpu.experts.model import SmallConditionalUNet


class TrainState(train_state.TrainState):
    pass


def create_train_state(config: ExpertTrainConfig, rng: jax.Array) -> TrainState:
    model = SmallConditionalUNet(
        hidden_channels=config.hidden_channels,
        num_classes=config.num_classes,
        out_channels=config.num_channels,
    )
    dummy_x = jnp.zeros((config.batch_size, config.image_size, config.image_size, config.num_channels))
    dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.float32)
    dummy_y = jnp.zeros((config.batch_size,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t, dummy_y)["params"]
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def flow_matching_loss(
    params: flax.core.FrozenDict,
    state: TrainState,
    batch: dict[str, jax.Array],
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    images = batch["images"]
    labels = batch["labels"]
    noise_rng, time_rng = jax.random.split(rng)

    x1 = jax.random.normal(noise_rng, images.shape)
    t = jax.random.uniform(time_rng, (images.shape[0],), minval=0.0, maxval=1.0)
    t_broadcast = t[:, None, None, None]
    x_t = (1.0 - t_broadcast) * x1 + t_broadcast * images
    target_velocity = images - x1

    pred_velocity = state.apply_fn({"params": params}, x_t, t, labels)
    loss = jnp.mean((pred_velocity - target_velocity) ** 2)
    metrics = {"loss": loss, "velocity_mse": loss}
    return loss, metrics


@jax.jit
def train_step(
    state: TrainState,
    batch: dict[str, jax.Array],
    rng: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    grad_fn = jax.value_and_grad(flow_matching_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state, batch, rng)
    state = state.apply_gradients(grads=grads)
    metrics["loss"] = loss
    return state, metrics


def save_checkpoint(state: TrainState, checkpoint_dir: str, step: int, metadata: dict) -> None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = {"state": state, "metadata": metadata}
    path = os.path.join(checkpoint_dir, f"step_{step}")
    checkpointer.save(path, ckpt, force=True)


def latest_checkpoint_path(checkpoint_dir: str) -> str | None:
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_root.exists():
        return None
    candidates = []
    for child in checkpoint_root.iterdir():
        if child.is_dir() and child.name.startswith("step_"):
            try:
                step = int(child.name.split("_", 1)[1])
            except ValueError:
                continue
            candidates.append((step, child))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def load_json(path: str | Path) -> dict | None:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None
    import json

    with path.open() as f:
        return json.load(f)


def restore_training_state(
    config: ExpertTrainConfig,
    state: TrainState,
    artifact_root: Path,
) -> tuple[TrainState, int, int, list[dict[str, float]]]:
    checkpoint_dir = os.path.join(config.checkpoint_dir, config.expert_name)
    checkpoint_path = latest_checkpoint_path(checkpoint_dir)
    history_payload = load_json(artifact_root / "metrics" / "history.json")
    metrics_history = history_payload["history"] if history_payload is not None else []

    if checkpoint_path is None:
        return state, 0, 0, metrics_history

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)
    restored_state = restored["state"]
    if isinstance(restored_state, dict):
        restored_params = restored_state["params"]
        restored_state = TrainState(
            step=restored_state["step"],
            apply_fn=state.apply_fn,
            params=restored_params,
            tx=state.tx,
            opt_state=state.tx.init(restored_params),
        )
    restored_metadata = restored.get("metadata", {})
    global_step = int(restored_metadata.get("global_step", 0))
    start_epoch = int(metrics_history[-1]["epoch"]) if metrics_history else 0
    print(f"resumed from checkpoint: {checkpoint_path}")
    print(f"resume global_step={global_step} start_epoch={start_epoch}")
    return restored_state, global_step, start_epoch, metrics_history


@jax.jit
def sample_step(
    state: TrainState,
    x_t: jax.Array,
    t_scalar: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    t_vec = jnp.full((x_t.shape[0],), t_scalar, dtype=jnp.float32)
    return state.apply_fn({"params": state.params}, x_t, t_vec, labels)


def sample_images(
    state: TrainState,
    labels: np.ndarray,
    image_shape: tuple[int, int, int],
    seed: int,
    num_steps: int,
) -> np.ndarray:
    rng = jax.random.PRNGKey(seed)
    x_t = jax.random.normal(rng, (labels.shape[0], *image_shape))
    labels_jax = jnp.asarray(labels, dtype=jnp.int32)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_scalar = jnp.asarray(step * dt, dtype=jnp.float32)
        velocity = sample_step(state, x_t, t_scalar, labels_jax)
        x_t = x_t + dt * velocity
    return np.asarray(x_t)


def build_eval_subset(dataset: NumpyDataset, num_examples: int, seed: int) -> NumpyDataset:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset.labels.shape[0])
    if num_examples < len(indices):
        indices = rng.choice(indices, size=num_examples, replace=False)
    return NumpyDataset(dataset.images[indices], dataset.labels[indices])


def run_epoch_eval(
    state: TrainState,
    test_ds: NumpyDataset,
    config: ExpertTrainConfig,
    epoch: int,
    artifact_root: Path,
) -> dict[str, float]:
    eval_ds = build_eval_subset(test_ds, config.eval_num_real, seed=config.seed + epoch)
    label_rng = np.random.default_rng(config.seed + 10_000 + epoch)
    generated_labels = label_rng.choice(eval_ds.labels, size=config.eval_num_generated, replace=True)
    generated_images = sample_images(
        state,
        generated_labels,
        (config.image_size, config.image_size, config.num_channels),
        seed=config.seed + 20_000 + epoch,
        num_steps=config.num_diffusion_steps,
    )

    real_for_fid = eval_ds.images[: min(len(eval_ds.images), len(generated_images))]
    real_labels_for_fid = eval_ds.labels[: len(real_for_fid)]
    gen_for_fid = generated_images[: len(real_for_fid)]
    gen_labels_for_fid = generated_labels[: len(gen_for_fid)]
    labels_for_grid = generated_labels[: min(64, len(generated_labels))]

    epoch_dir = ensure_dir(artifact_root / f"epoch_{epoch:03d}")
    save_image_grid(generated_images[:64], labels_for_grid, epoch_dir / "generated_grid.png", f"Generated samples epoch {epoch}")
    save_label_histogram(generated_labels, epoch_dir / "generated_label_histogram.png", f"Generated label mix epoch {epoch}")
    save_tsne_plot(
        real_for_fid[:200],
        gen_for_fid[:200],
        real_labels_for_fid[:200],
        gen_labels_for_fid[:200],
        epoch_dir / "generated_tsne.png",
        f"Real vs generated t-SNE epoch {epoch}",
        seed=config.seed + 30_000 + epoch,
    )

    pca_fid = compute_pca_fid(real_for_fid, gen_for_fid)
    metrics = {
        "epoch": epoch,
        "num_real_eval": int(real_for_fid.shape[0]),
        "num_generated_eval": int(gen_for_fid.shape[0]),
        "pca_fid": float(pca_fid),
    }
    save_json(metrics, epoch_dir / "metrics.json")
    return metrics


def resolve_datasets(config: ExpertTrainConfig) -> tuple[NumpyDataset, NumpyDataset]:
    train_ds, test_ds = load_mnist_numpy(image_size=config.image_size)
    if config.cluster_assignments_path is not None:
        if config.cluster_id is None:
            raise ValueError("cluster_id must be set when cluster_assignments_path is provided")
        train_cluster_ids, test_cluster_ids = split_cluster_assignments(config.cluster_assignments_path)
        train_ds = filter_by_cluster_id(train_ds, train_cluster_ids, config.cluster_id)
        test_ds = filter_by_cluster_id(test_ds, test_cluster_ids, config.cluster_id)
        return train_ds, test_ds

    train_ds = filter_by_class_ids(train_ds, config.class_ids)
    test_ds = filter_by_class_ids(test_ds, config.class_ids)
    return train_ds, test_ds


def train_expert(config: ExpertTrainConfig) -> None:
    train_ds, test_ds = resolve_datasets(config)

    rng = jax.random.PRNGKey(config.seed)
    init_rng, loop_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng)
    artifact_root = ensure_dir(Path(config.artifact_dir) / config.expert_name)
    metrics_dir = ensure_dir(artifact_root / "metrics")

    print("config:", asdict(config))
    print("train samples:", len(train_ds.labels))
    print("eval samples:", len(test_ds.labels))
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    global_step = 0
    start_epoch = 0
    metrics_history: list[dict[str, float]] = []
    if config.resume:
        state, global_step, start_epoch, metrics_history = restore_training_state(config, state, artifact_root)

    checkpoint_dir = os.path.join(config.checkpoint_dir, config.expert_name)

    for epoch in range(start_epoch, config.num_epochs):
        losses = []
        for batch in batch_iterator(train_ds, config.batch_size, seed=config.seed + epoch):
            loop_rng, step_rng = jax.random.split(loop_rng)
            state, metrics = train_step(state, batch, step_rng)
            loss = float(metrics["loss"])
            losses.append(loss)
            global_step += 1
            if global_step % config.log_every_steps == 0:
                print(f"epoch={epoch+1} step={global_step} loss={loss:.6f}")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch+1} train_loss={mean_loss:.6f}")
        epoch_metrics: dict[str, float] = {"epoch": epoch + 1, "train_loss": mean_loss}
        if config.checkpoint_every_epochs > 0 and (epoch + 1) % config.checkpoint_every_epochs == 0:
            metadata = {"config": asdict(config), "global_step": global_step, "epoch": epoch + 1}
            save_checkpoint(state, checkpoint_dir, global_step, metadata)
            print(f"epoch={epoch+1} saved checkpoint to {checkpoint_dir}")
        if (epoch + 1) % config.sample_every_epochs == 0:
            eval_metrics = run_epoch_eval(state, test_ds, config, epoch + 1, artifact_root)
            epoch_metrics.update(eval_metrics)
            print(f"epoch={epoch+1} pca_fid={eval_metrics['pca_fid']:.6f}")
        metrics_history.append(epoch_metrics)
        save_json({"history": metrics_history, "config": asdict(config)}, metrics_dir / "history.json")
        save_training_curves(metrics_history, metrics_dir / "training_curves.png")

    metadata = {"config": asdict(config), "global_step": global_step}
    save_checkpoint(state, checkpoint_dir, global_step, metadata)
    print(f"saved checkpoint to {checkpoint_dir}")
