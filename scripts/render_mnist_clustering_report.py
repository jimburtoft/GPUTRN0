#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-ready figures for MNIST clustering.")
    parser.add_argument("--input-dir", type=str, default="./outputs/mnist_clusters")
    parser.add_argument("--samples-per-cluster", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_artifacts(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    images = np.load(input_dir / "images.npy")
    labels = np.load(input_dir / "labels.npy")
    cluster_ids = np.load(input_dir / "cluster_ids.npy")
    embedding_2d = np.load(input_dir / "embedding_2d.npy")
    with (input_dir / "summary.json").open() as f:
        summary = json.load(f)
    return images, labels, cluster_ids, embedding_2d, summary


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_cluster_scatter(
    out_dir: Path,
    embedding_2d: np.ndarray,
    cluster_ids: np.ndarray,
    labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

    ax = axes[0]
    ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=np.where(cluster_ids == 0, "#0f766e", "#b91c1c"),
        s=3,
        alpha=0.45,
        linewidths=0,
    )
    ax.set_title("MNIST clustered into two groups")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")

    ax = axes[1]
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=colors[digit],
            s=3,
            alpha=0.35,
            linewidths=0,
            label=str(digit),
        )
    ax.set_title("Same embedding, colored by digit label")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.savefig(out_dir / "cluster_embedding.png", bbox_inches="tight")
    plt.close(fig)


def save_digit_histogram(out_dir: Path, summary: dict) -> None:
    fig, axes = plt.subplots(1, len(summary["clusters"]), figsize=(12, 4.8), constrained_layout=True)
    if len(summary["clusters"]) == 1:
        axes = [axes]

    palette = ["#0f766e", "#b91c1c"]
    for ax, cluster in zip(axes, summary["clusters"]):
        histogram = cluster["digit_histogram"]
        xs = np.arange(10)
        ys = [histogram[str(i)] for i in range(10)]
        ax.bar(xs, ys, color=palette[cluster["cluster_id"] % len(palette)], width=0.75)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_title(f"Cluster {cluster['cluster_id']} ({cluster['size']:,} images)")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
    fig.suptitle("Digit composition of each cluster")
    fig.savefig(out_dir / "cluster_digit_histograms.png", bbox_inches="tight")
    plt.close(fig)


def sample_indices_for_cluster(
    rng: np.random.Generator,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_id: int,
    count: int,
) -> np.ndarray:
    cluster_mask = cluster_ids == cluster_id
    cluster_idx = np.where(cluster_mask)[0]
    chosen: list[int] = []
    for digit in range(10):
        digit_idx = cluster_idx[labels[cluster_idx] == digit]
        if len(digit_idx) == 0:
            continue
        take = min(max(count // 10, 1), len(digit_idx))
        chosen.extend(rng.choice(digit_idx, size=take, replace=False).tolist())
    remaining = count - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(cluster_idx, np.asarray(chosen, dtype=np.int64), assume_unique=False)
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            chosen.extend(extra.tolist())
    chosen = chosen[:count]
    return np.asarray(chosen, dtype=np.int64)


def save_example_grid(
    out_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    samples_per_cluster: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6), constrained_layout=True)
    for cluster_id in range(2):
        idx = sample_indices_for_cluster(rng, labels, cluster_ids, cluster_id, samples_per_cluster)
        idx = idx[:16]
        for cell, sample_idx in zip(axes[cluster_id], idx):
            cell.imshow(images[sample_idx], cmap="gray")
            cell.set_title(str(int(labels[sample_idx])), pad=2)
            cell.axis("off")
        for cell in axes[cluster_id][len(idx):]:
            cell.axis("off")
        axes[cluster_id, 0].set_ylabel(f"Cluster {cluster_id}", rotation=90, labelpad=10)
    fig.suptitle("Representative MNIST samples from each cluster")
    fig.savefig(out_dir / "cluster_examples.png", bbox_inches="tight")
    plt.close(fig)


def save_cluster_mean_images(out_dir: Path, images: np.ndarray, cluster_ids: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), constrained_layout=True)
    for cluster_id, ax in enumerate(axes):
        mean_img = images[cluster_ids == cluster_id].mean(axis=0)
        ax.imshow(mean_img, cmap="magma")
        ax.set_title(f"Cluster {cluster_id} mean image")
        ax.axis("off")
    fig.savefig(out_dir / "cluster_mean_images.png", bbox_inches="tight")
    plt.close(fig)


def write_text_summary(out_dir: Path, summary: dict) -> None:
    lines = []
    lines.append("MNIST clustering summary")
    lines.append("======================")
    lines.append("")
    lines.append(f"Num examples: {summary['num_examples']:,}")
    lines.append(f"Num clusters: {summary['num_clusters']}")
    lines.append(f"PCA explained variance ratio sum: {summary['pca_explained_variance_ratio_sum']:.4f}")
    lines.append("")
    for cluster in summary["clusters"]:
        lines.append(f"Cluster {cluster['cluster_id']}")
        lines.append(f"- Size: {cluster['size']:,}")
        lines.append(f"- Top digits: {', '.join(str(x) for x in cluster['top_digits'])}")
        histogram = cluster["digit_histogram"]
        lines.append("- Digit histogram:")
        for digit in range(10):
            lines.append(f"  - {digit}: {histogram[str(digit)]}")
        lines.append("")
    (out_dir / "cluster_summary.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    style()

    input_dir = Path(args.input_dir).resolve()
    report_dir = input_dir / "clustering"
    report_dir.mkdir(parents=True, exist_ok=True)

    images, labels, cluster_ids, embedding_2d, summary = load_artifacts(input_dir)

    save_cluster_scatter(report_dir, embedding_2d, cluster_ids, labels)
    save_digit_histogram(report_dir, summary)
    save_example_grid(report_dir, images, labels, cluster_ids, args.samples_per_cluster, args.seed)
    save_cluster_mean_images(report_dir, images, cluster_ids)
    write_text_summary(report_dir, summary)

    print(f"Saved clustering report to {report_dir}")


if __name__ == "__main__":
    main()
