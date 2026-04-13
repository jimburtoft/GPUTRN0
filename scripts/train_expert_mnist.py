#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tpugpu.config import ExpertTrainConfig
from tpugpu.experts.train import train_expert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single MNIST DDM expert in JAX.")
    parser.add_argument("--expert-name", type=str, default="expert_a")
    parser.add_argument("--class-ids", type=str, default="0,1,2,3,4")
    parser.add_argument("--cluster-assignments-path", type=str, default=None)
    parser.add_argument("--cluster-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--artifact-dir", type=str, default="./outputs/experiments")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--sample-every-epochs", type=int, default=1)
    parser.add_argument("--eval-num-real", type=int, default=512)
    parser.add_argument("--eval-num-generated", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_ids = tuple(int(x) for x in args.class_ids.split(",") if x)
    config = ExpertTrainConfig(
        expert_name=args.expert_name,
        class_ids=class_ids,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        artifact_dir=args.artifact_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        sample_every_epochs=args.sample_every_epochs,
        eval_num_real=args.eval_num_real,
        eval_num_generated=args.eval_num_generated,
        eval_batch_size=args.eval_batch_size,
        resume=args.resume,
        cluster_assignments_path=args.cluster_assignments_path,
        cluster_id=args.cluster_id,
        seed=args.seed,
    )
    train_expert(config)


if __name__ == "__main__":
    main()
