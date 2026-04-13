from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import flax
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from tpugpu.data.mnist import NumpyDataset, batch_iterator, load_mnist_numpy
from tpugpu.eval.reporting import (
    ensure_dir,
    save_class_accuracy_bar,
    save_confusion_matrix,
    save_expert_histogram,
    save_json,
    save_router_training_curves,
)
from tpugpu.experts.train import create_train_state, latest_checkpoint_path
from tpugpu.router.model import RouterMLP


@dataclass
class RouterTrainConfig:
    expert_names: tuple[str, ...]
    checkpoint_dir: str = "./outputs/checkpoints"
    router_checkpoint_dir: str = "./outputs/router_checkpoints"
    artifact_dir: str = "./outputs/router"
    expert_label_splits: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
    image_size: int = 32
    num_channels: int = 1
    num_classes: int = 10
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    seed: int = 0
    router_name: str = "router_mnist"
    label_mode: str = "oracle"


class RouterState(train_state.TrainState):
    pass


def _restore_expert_state(expert_name: str, checkpoint_dir: str, batch_size: int, seed: int):
    ckpt_root = Path(checkpoint_dir).expanduser().resolve() / expert_name
    checkpoint_path = latest_checkpoint_path(str(ckpt_root))
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for expert {expert_name} under {ckpt_root}")
    state = create_train_state(
        config=type(
            "ExpertCfg",
            (),
            {
                "hidden_channels": 64,
                "num_classes": 10,
                "num_channels": 1,
                "batch_size": batch_size,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "image_size": 32,
                "seed": seed,
            },
        )(),
        rng=jax.random.PRNGKey(seed),
    )
    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    restored_state = restored["state"]
    if isinstance(restored_state, dict):
        state = state.replace(params=restored_state["params"])
    else:
        state = restored_state
    return state, checkpoint_path


def _create_router_state(config: RouterTrainConfig, rng: jax.Array) -> RouterState:
    model = RouterMLP(
        num_experts=len(config.expert_label_splits),
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
    )
    dummy_x = jnp.zeros((config.batch_size, config.image_size, config.image_size, config.num_channels))
    dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.float32)
    dummy_y = jnp.zeros((config.batch_size,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t, dummy_y)["params"]
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    return RouterState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def _router_train_step(state: RouterState, x_t: jax.Array, t: jax.Array, y: jax.Array, target: jax.Array):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x_t, t, y)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == target)
        return loss, {"loss": loss, "acc": acc}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics["loss"] = loss
    return state, metrics


def _oracle_targets(labels: np.ndarray, expert_label_splits: tuple[tuple[int, ...], ...]) -> np.ndarray:
    targets = np.full(labels.shape[0], -1, dtype=np.int32)
    for expert_idx, class_ids in enumerate(expert_label_splits):
        mask = np.isin(labels, np.asarray(class_ids, dtype=np.int32))
        targets[mask] = expert_idx
    if np.any(targets < 0):
        raise ValueError("Some labels were not covered by the provided expert_label_splits")
    return targets


def _interpolate_noisy_batch(images: jax.Array, seed: int) -> tuple[jax.Array, jax.Array]:
    rng = jax.random.PRNGKey(seed)
    noise_rng, time_rng = jax.random.split(rng)
    x1 = jax.random.normal(noise_rng, images.shape)
    t = jax.random.uniform(time_rng, (images.shape[0],), minval=0.0, maxval=1.0)
    t_broadcast = t[:, None, None, None]
    x_t = (1.0 - t_broadcast) * x1 + t_broadcast * images
    return x_t, t


def _compute_confusion_matrix(target: np.ndarray, pred: np.ndarray, num_experts: int) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=np.int32)
    for true_idx, pred_idx in zip(target, pred):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def _compute_class_accuracy(labels: np.ndarray, target: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full((num_classes,), np.nan, dtype=np.float32)
    for class_id in range(num_classes):
        mask = labels == class_id
        if np.any(mask):
            acc[class_id] = float(np.mean(pred[mask] == target[mask]))
    return acc


def train_router(config: RouterTrainConfig) -> None:
    if config.label_mode != "oracle":
        raise NotImplementedError("Only oracle label_mode is implemented right now.")

    train_ds, test_ds = load_mnist_numpy(image_size=config.image_size)
    test_targets = _oracle_targets(test_ds.labels, config.expert_label_splits)

    expert_checkpoint_paths: list[str | None] = []
    if config.expert_names:
        expert_restore = [
            _restore_expert_state(name, config.checkpoint_dir, config.batch_size, config.seed + idx)
            for idx, name in enumerate(config.expert_names)
        ]
        expert_checkpoint_paths = [path for _, path in expert_restore]

    rng = jax.random.PRNGKey(config.seed)
    init_rng, _ = jax.random.split(rng)
    state = _create_router_state(config, init_rng)

    artifact_root = ensure_dir(Path(config.artifact_dir) / config.router_name)
    metrics_dir = ensure_dir(artifact_root / "metrics")
    checkpoint_dir = Path(config.router_checkpoint_dir).expanduser().resolve() / config.router_name
    metrics_history: list[dict[str, float]] = []

    print("router_config:", asdict(config))
    print("expert_checkpoints:", expert_checkpoint_paths)
    print("train samples:", len(train_ds.labels))
    print("eval samples:", len(test_ds.labels))
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    for epoch in range(config.num_epochs):
        losses = []
        accs = []
        for batch_idx, batch in enumerate(batch_iterator(train_ds, config.batch_size, seed=config.seed + epoch)):
            np_labels = np.asarray(batch["labels"], dtype=np.int32)
            targets = _oracle_targets(np_labels, config.expert_label_splits)
            x_t, t = _interpolate_noisy_batch(batch["images"], seed=config.seed + epoch * 100_000 + batch_idx)
            state, metrics = _router_train_step(
                state,
                x_t,
                t,
                batch["labels"],
                jnp.asarray(targets, dtype=jnp.int32),
            )
            losses.append(float(metrics["loss"]))
            accs.append(float(metrics["acc"]))

        eval_labels = np.asarray(test_ds.labels[: config.batch_size], dtype=np.int32)
        eval_x_t, eval_t = _interpolate_noisy_batch(jnp.asarray(test_ds.images[: config.batch_size]), seed=config.seed + 999_999 + epoch)
        eval_logits = state.apply_fn(
            {"params": state.params},
            eval_x_t,
            eval_t,
            jnp.asarray(eval_labels, dtype=jnp.int32),
        )
        eval_pred = np.asarray(jnp.argmax(eval_logits, axis=-1), dtype=np.int32)
        eval_target = test_targets[: config.batch_size]
        eval_acc = float(np.mean(eval_pred == eval_target))
        num_experts = len(config.expert_label_splits)
        confusion = _compute_confusion_matrix(eval_target, eval_pred, num_experts)
        class_accuracy = _compute_class_accuracy(eval_labels, eval_target, eval_pred, config.num_classes)
        eval_pred_hist = np.bincount(eval_pred, minlength=num_experts)

        entry = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_acc": float(np.mean(accs)),
            "eval_acc": eval_acc,
            "eval_pred_histogram": eval_pred_hist.tolist(),
        }
        metrics_history.append(entry)
        print(
            f"epoch={epoch+1} router_train_loss={entry['train_loss']:.6f} "
            f"router_train_acc={entry['train_acc']:.6f} router_eval_acc={entry['eval_acc']:.6f}"
        )
        epoch_dir = ensure_dir(artifact_root / f"epoch_{epoch + 1:03d}")
        save_confusion_matrix(
            confusion.astype(np.float32),
            epoch_dir / "router_confusion_matrix.png",
            f"Router confusion matrix epoch {epoch + 1}",
            x_label="Predicted expert",
            y_label="Target expert",
            tick_labels=[f"E{i}" for i in range(num_experts)],
        )
        save_class_accuracy_bar(
            np.nan_to_num(class_accuracy, nan=0.0),
            epoch_dir / "router_class_accuracy.png",
            f"Router class accuracy epoch {epoch + 1}",
        )
        save_expert_histogram(
            eval_pred,
            num_experts,
            epoch_dir / "router_predicted_expert_histogram.png",
            f"Predicted expert histogram epoch {epoch + 1}",
        )
        save_json(
            {
                "epoch": epoch + 1,
                "eval_target": eval_target.tolist(),
                "eval_pred": eval_pred.tolist(),
                "eval_labels": eval_labels.tolist(),
                "confusion_matrix": confusion.tolist(),
                "class_accuracy": np.nan_to_num(class_accuracy, nan=-1.0).tolist(),
                "eval_pred_histogram": eval_pred_hist.tolist(),
            },
            epoch_dir / "router_eval_details.json",
        )
        save_json(
            {
                "config": asdict(config),
                "expert_checkpoints": expert_checkpoint_paths,
                "history": metrics_history,
            },
            metrics_dir / "history.json",
        )
        save_router_training_curves(metrics_history, metrics_dir / "router_training_curves.png")
        metadata = {
            "config": asdict(config),
            "epoch": epoch + 1,
            "history_length": len(metrics_history),
        }
        ocp.PyTreeCheckpointer().save(
            str(checkpoint_dir / f"step_{epoch + 1}"),
            {"state": state, "metadata": metadata},
            force=True,
        )

    summary = {
        "config": asdict(config),
        "expert_checkpoints": expert_checkpoint_paths,
        "history": metrics_history,
    }
    with (artifact_root / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
