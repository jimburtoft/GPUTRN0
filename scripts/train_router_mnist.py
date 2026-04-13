#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tpugpu.router.train import RouterTrainConfig, train_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST DDM router.")
    parser.add_argument("--router-name", type=str, default="router_mnist_oracle")
    parser.add_argument("--expert-names", type=str, default="")
    parser.add_argument(
        "--expert-label-splits",
        type=str,
        default="0,1,2,3,4|5,6,7,8,9",
        help="Pipe-separated class groups aligned with --expert-names",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--router-checkpoint-dir", type=str, default="./outputs/router_checkpoints")
    parser.add_argument("--artifact-dir", type=str, default="./outputs/router")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-mode", type=str, default="oracle")
    return parser.parse_args()


def parse_label_splits(spec: str) -> tuple[tuple[int, ...], ...]:
    groups = []
    for group in spec.split("|"):
        classes = tuple(int(token) for token in group.split(",") if token)
        if not classes:
            raise ValueError(f"Empty class group in --expert-label-splits: {spec}")
        groups.append(classes)
    return tuple(groups)


def main() -> None:
    args = parse_args()
    expert_names = tuple(name.strip() for name in args.expert_names.split(",") if name.strip())
    label_splits = parse_label_splits(args.expert_label_splits)
    if expert_names and len(expert_names) != len(label_splits):
        raise ValueError("--expert-names and --expert-label-splits must have the same number of groups")

    config = RouterTrainConfig(
        expert_names=expert_names,
        checkpoint_dir=args.checkpoint_dir,
        router_checkpoint_dir=args.router_checkpoint_dir,
        artifact_dir=args.artifact_dir,
        expert_label_splits=label_splits,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        router_name=args.router_name,
        label_mode=args.label_mode,
    )
    train_router(config)


if __name__ == "__main__":
    main()
