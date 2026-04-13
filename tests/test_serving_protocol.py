from __future__ import annotations

import numpy as np

from tpugpu.serving.protocol import (
    decode_predict_request,
    decode_predict_response,
    encode_predict_request,
    encode_predict_response,
)


def test_predict_request_round_trip_preserves_arrays() -> None:
    x_t = np.random.default_rng(0).standard_normal((4, 32, 32, 1), dtype=np.float32)
    t = np.linspace(0.0, 0.75, 4, dtype=np.float32)
    y = np.asarray([0, 2, 5, 9], dtype=np.int32)

    payload = encode_predict_request(x_t, t, y)
    decoded_x_t, decoded_t, decoded_y = decode_predict_request(payload)

    np.testing.assert_allclose(decoded_x_t, x_t)
    np.testing.assert_allclose(decoded_t, t)
    np.testing.assert_array_equal(decoded_y, y)


def test_predict_response_round_trip_preserves_velocity() -> None:
    velocity = np.random.default_rng(1).standard_normal((4, 32, 32, 1), dtype=np.float32)
    payload = encode_predict_response(velocity)
    decoded_velocity = decode_predict_response(payload)
    np.testing.assert_allclose(decoded_velocity, velocity)
