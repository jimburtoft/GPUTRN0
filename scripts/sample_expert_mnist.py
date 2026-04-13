from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp

from tpugpu.config import ExpertTrainConfig
from tpugpu.eval.reporting import ensure_dir, save_image_grid, save_label_histogram
from tpugpu.experts.train import create_train_state, latest_checkpoint_path, sample_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a trained expert checkpoint.")
    parser.add_argument("--expert-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--output-dir", type=str, default="./outputs/inference")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated labels, e.g. 0,2,3,5,6")
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-diffusion-steps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=64)
    return parser.parse_args()


def restore_params(config: ExpertTrainConfig) -> tuple[object, str]:
    checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve() / config.expert_name
    checkpoint_path = latest_checkpoint_path(str(checkpoint_root))
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_root}")

    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(config, rng)
    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    restored_state = restored["state"]
    if isinstance(restored_state, dict):
        params = restored_state["params"]
        state = state.replace(params=params)
    else:
        state = restored_state
    return state, checkpoint_path


def main() -> None:
    args = parse_args()
    requested_labels = [int(part) for part in args.labels.split(",") if part.strip()]
    if not requested_labels:
        raise ValueError("Expected at least one label in --labels")

    labels = np.repeat(np.asarray(requested_labels, dtype=np.int32), args.samples_per_label)
    config = ExpertTrainConfig(
        expert_name=args.expert_name,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=max(args.batch_size, labels.shape[0]),
        num_diffusion_steps=args.num_diffusion_steps,
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
        seed=args.seed,
    )
    state, checkpoint_path = restore_params(config)

    images = sample_images(
        state,
        labels=labels,
        image_shape=(config.image_size, config.image_size, config.num_channels),
        seed=config.seed,
        num_steps=config.num_diffusion_steps,
    )

    output_dir = ensure_dir(Path(args.output_dir) / args.expert_name)
    labels_slug = "-".join(str(label) for label in requested_labels)
    save_image_grid(
        images,
        labels,
        output_dir / f"forced_labels_{labels_slug}_grid.png",
        f"{args.expert_name} forced labels {requested_labels}",
    )
    save_label_histogram(
        labels,
        output_dir / f"forced_labels_{labels_slug}_histogram.png",
        f"{args.expert_name} forced label histogram",
    )

    summary = {
        "expert_name": args.expert_name,
        "checkpoint_path": checkpoint_path,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "labels": requested_labels,
        "samples_per_label": args.samples_per_label,
        "num_generated": int(labels.shape[0]),
        "seed": args.seed,
        "num_diffusion_steps": args.num_diffusion_steps,
    }
    with (output_dir / f"forced_labels_{labels_slug}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved artifacts under {output_dir}")


if __name__ == "__main__":
    main()
