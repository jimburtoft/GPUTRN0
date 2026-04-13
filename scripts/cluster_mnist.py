#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster MNIST into two visually distinct groups.")
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./outputs/mnist_clusters")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    train_ds, test_ds = tfds.as_numpy(
        tfds.load("mnist", split=["train", "test"], batch_size=-1, as_supervised=True)
    )
    train_images, train_labels = train_ds
    test_images, test_labels = test_ds
    images = np.concatenate([train_images, test_images], axis=0).astype(np.float32)
    labels = np.concatenate([train_labels, test_labels], axis=0).astype(np.int32)
    if images.ndim == 4:
        images = images[..., 0]
    images = images / 255.0
    return images, labels


def build_summary(labels: np.ndarray, cluster_ids: np.ndarray, num_clusters: int) -> dict:
    summary: dict[str, object] = {
        "num_examples": int(labels.shape[0]),
        "num_clusters": int(num_clusters),
        "clusters": [],
    }
    for cluster_id in range(num_clusters):
        mask = cluster_ids == cluster_id
        cluster_labels = labels[mask]
        counts = np.bincount(cluster_labels, minlength=10)
        label_hist = {str(i): int(counts[i]) for i in range(10)}
        sorted_labels = sorted(label_hist.items(), key=lambda item: item[1], reverse=True)
        summary["clusters"].append(
            {
                "cluster_id": cluster_id,
                "size": int(mask.sum()),
                "digit_histogram": label_hist,
                "top_digits": [int(k) for k, v in sorted_labels if v > 0][:5],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_mnist()
    flat_images = images.reshape(images.shape[0], -1)

    pca = PCA(n_components=args.num_components, random_state=args.seed)
    embedding = pca.fit_transform(flat_images)
    plot_embedding = embedding[:, :2]

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=20)
    cluster_ids = kmeans.fit_predict(embedding)

    summary = build_summary(labels, cluster_ids, args.num_clusters)
    summary["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))

    np.save(output_dir / "cluster_ids.npy", cluster_ids)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "embedding_2d.npy", plot_embedding)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
