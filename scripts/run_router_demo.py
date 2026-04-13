#!/usr/bin/env python3
from __future__ import annotations

import argparse

import uvicorn

from tpugpu.demo import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TPUGPU router demo web app.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--expert-url-a", type=str, default="http://localhost:8000")
    parser.add_argument("--expert-url-b", type=str, default="http://localhost:8001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(expert_urls=(args.expert_url_a, args.expert_url_b))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
