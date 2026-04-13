from __future__ import annotations

import json

import numpy as np

from tpugpu.demo.app import _normalize_frame


def test_normalize_frame_returns_8bit_pixels() -> None:
    x_t = np.zeros((1, 32, 32, 1), dtype=np.float32)
    pixels = _normalize_frame(x_t)
    assert len(pixels) == 32 * 32
    assert min(pixels) >= 0
    assert max(pixels) <= 255


def test_step_payload_is_json_serializable() -> None:
    payload = {
        "type": "step",
        "step": 1,
        "steps": 40,
        "label": 2,
        "selected_expert": 0,
        "progress": 0.025,
        "frame": [0, 127, 255],
    }
    assert json.loads(json.dumps(payload))["selected_expert"] == 0
