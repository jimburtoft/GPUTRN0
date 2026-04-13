#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tpugpu.experts.inference import load_expert_state, predict_velocity_numpy
from tpugpu.serving.protocol import decode_predict_request, encode_predict_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an MNIST expert checkpoint over HTTP.")
    parser.add_argument("--expert-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_expert_state(
        args.expert_name,
        args.checkpoint_dir,
        batch_size=args.batch_size,
    )

    class ExpertHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            payload = json.dumps({"status": "ok", "expert_name": args.expert_name}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            x_t, t, y = decode_predict_request(payload)
            print(
                "expert_predict "
                f"expert={args.expert_name} "
                f"label={int(y[0]) if len(y) else 'na'} "
                f"t={float(t[0]) if len(t) else float('nan'):.4f} "
                f"batch={x_t.shape[0]} "
                f"shape={tuple(x_t.shape)} "
                f"x_mean={float(x_t.mean()):.6f} "
                f"x_std={float(x_t.std()):.6f}",
                flush=True,
            )
            velocity = predict_velocity_numpy(state, x_t, t, y)
            print(
                "expert_velocity "
                f"expert={args.expert_name} "
                f"label={int(y[0]) if len(y) else 'na'} "
                f"t={float(t[0]) if len(t) else float('nan'):.4f} "
                f"v_mean={float(velocity.mean()):.6f} "
                f"v_std={float(velocity.std()):.6f}",
                flush=True,
            )
            response = encode_predict_response(velocity)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((args.host, args.port), ExpertHandler)
    print(f"serving expert={args.expert_name} host={args.host} port={args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
