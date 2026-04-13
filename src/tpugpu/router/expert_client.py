from __future__ import annotations

from urllib import request

import numpy as np

from tpugpu.serving.protocol import decode_predict_response, encode_predict_request


class ExpertClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def predict_velocity(self, x_t: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        payload = encode_predict_request(x_t, t, y)
        req = request.Request(
            f"{self.base_url}/predict",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            return decode_predict_response(response.read())
