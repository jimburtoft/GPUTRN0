from __future__ import annotations

from io import BytesIO

import numpy as np


def _load_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload)) as data:
        return {key: data[key] for key in data.files}


def encode_predict_request(x_t: np.ndarray, t: np.ndarray, y: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        x_t=np.asarray(x_t, dtype=np.float32),
        t=np.asarray(t, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
    )
    return buffer.getvalue()


def decode_predict_request(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = _load_npz(payload)
    return arrays["x_t"], arrays["t"], arrays["y"]


def encode_predict_response(velocity: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, velocity=np.asarray(velocity, dtype=np.float32))
    return buffer.getvalue()


def decode_predict_response(payload: bytes) -> np.ndarray:
    arrays = _load_npz(payload)
    return arrays["velocity"]
