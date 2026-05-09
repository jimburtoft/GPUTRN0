from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass
class NumpyDataset:
    images: np.ndarray
    labels: np.ndarray


def _resize_and_normalize(images: np.ndarray, image_size: int) -> np.ndarray:
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError(
            f"Expected MNIST images with 3 or 4 dimensions, got shape {images.shape}"
        )
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0
    if images.shape[1] != image_size or images.shape[2] != image_size:
        pad = (image_size - images.shape[1]) // 2
        images = np.pad(
            images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant"
        )
    return images


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def load_mnist_numpy(image_size: int = 32) -> tuple[NumpyDataset, NumpyDataset]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    cache_dir = Path.home() / ".cache" / "mnist"

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    for name, filename in files.items():
        _download_file(base_url + filename, cache_dir / filename)

    train_images = _read_idx_images(cache_dir / files["train_images"])
    train_labels = _read_idx_labels(cache_dir / files["train_labels"])
    test_images = _read_idx_images(cache_dir / files["test_images"])
    test_labels = _read_idx_labels(cache_dir / files["test_labels"])

    train_images = _resize_and_normalize(train_images, image_size)
    test_images = _resize_and_normalize(test_images, image_size)

    return (
        NumpyDataset(train_images, train_labels),
        NumpyDataset(test_images, test_labels),
    )


def filter_by_class_ids(
    dataset: NumpyDataset, class_ids: tuple[int, ...]
) -> NumpyDataset:
    mask = np.isin(dataset.labels, np.asarray(class_ids, dtype=np.int32))
    return NumpyDataset(dataset.images[mask], dataset.labels[mask])


def split_cluster_assignments(
    cluster_assignments_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.load(Path(cluster_assignments_path).expanduser().resolve())
    if cluster_ids.shape[0] != 70000:
        raise ValueError(
            f"Expected 70,000 cluster assignments for MNIST, got {cluster_ids.shape[0]}"
        )
    return cluster_ids[:60000].astype(np.int32), cluster_ids[60000:].astype(np.int32)


def filter_by_cluster_id(
    dataset: NumpyDataset,
    cluster_ids: np.ndarray,
    cluster_id: int,
) -> NumpyDataset:
    if dataset.labels.shape[0] != cluster_ids.shape[0]:
        raise ValueError(
            f"Dataset length {dataset.labels.shape[0]} does not match cluster id length {cluster_ids.shape[0]}"
        )
    mask = cluster_ids == np.int32(cluster_id)
    return NumpyDataset(dataset.images[mask], dataset.labels[mask])


def batch_iterator(
    dataset: NumpyDataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset.labels))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield {
            "images": jnp.asarray(dataset.images[batch_idx]),
            "labels": jnp.asarray(dataset.labels[batch_idx]),
        }
