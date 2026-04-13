#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from tpugpu.data.mnist import load_mnist_numpy


def main() -> None:
    train_ds, test_ds = load_mnist_numpy(image_size=32)
    output_dir = Path("./outputs/router_data").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "train_samples": int(train_ds.labels.shape[0]),
        "test_samples": int(test_ds.labels.shape[0]),
        "train_label_histogram": {str(i): int((train_ds.labels == i).sum()) for i in range(10)},
        "test_label_histogram": {str(i): int((test_ds.labels == i).sum()) for i in range(10)},
    }
    with (output_dir / "mnist_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved summary to {output_dir / 'mnist_summary.json'}")


if __name__ == "__main__":
    main()
