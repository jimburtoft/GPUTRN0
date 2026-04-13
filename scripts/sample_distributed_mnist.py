#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tpugpu.eval.reporting import ensure_dir, save_image_grid
from tpugpu.router.expert_client import ExpertClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST by calling a remote expert service.")
    parser.add_argument("--expert-url", type=str, required=True)
    parser.add_argument("--labels", type=str, default="0,1,2,3,4")
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_values = [int(value) for value in args.labels.split(",") if value]
    labels = np.repeat(np.asarray(label_values, dtype=np.int32), args.samples_per_label)
    rng = np.random.default_rng(args.seed)
    x_t = rng.standard_normal((labels.shape[0], 32, 32, 1), dtype=np.float32)
    client = ExpertClient(args.expert_url)
    dt = 1.0 / args.num_steps
    for step in range(args.num_steps):
        t = np.full((labels.shape[0],), step * dt, dtype=np.float32)
        velocity = client.predict_velocity(x_t, t, labels)
        x_t = x_t + dt * velocity

    output_path = Path(args.output_path).expanduser().resolve()
    ensure_dir(output_path.parent)
    save_image_grid(x_t, labels, output_path, "Distributed remote expert samples")
    print(f"saved samples to {output_path}")


if __name__ == "__main__":
    main()
