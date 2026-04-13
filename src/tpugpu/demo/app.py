from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tpugpu.router.expert_client import ExpertClient
from tpugpu.router.inference import load_router_state, predict_expert_id

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _normalize_frame(x_t: np.ndarray) -> list[int]:
    image = np.asarray(x_t[0, :, :, 0], dtype=np.float32)
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return image.reshape(-1).tolist()


def _fallback_expert(step_idx: int, total_steps: int, label: int, strategy: str) -> int:
    if strategy == "alternating":
        return step_idx % 2
    if strategy == "switch_halfway":
        return 0 if step_idx < (total_steps // 2) else 1
    return 0 if label <= 4 else 1


async def _stream_demo_events(
    *,
    label: int,
    steps: int,
    seed: int,
    expert_urls: tuple[str, str],
    strategy: str,
    router_state,
) -> str:
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2_147_483_647))
    rng = np.random.default_rng(seed)
    x_t = rng.standard_normal((1, 32, 32, 1), dtype=np.float32)
    y = np.asarray([label], dtype=np.int32)
    clients = [ExpertClient(url, timeout_seconds=60.0) for url in expert_urls]
    router_mode = "learned" if router_state is not None else f"fallback:{strategy}"
    print(f"demo_start label={label} steps={steps} seed={seed} router_mode={router_mode}", flush=True)

    start_payload = {
        "type": "start",
        "label": label,
        "steps": steps,
        "strategy": strategy,
        "frame": _normalize_frame(x_t),
        "selected_expert": None,
        "progress": 0.0,
    }
    yield f"data: {json.dumps(start_payload)}\n\n"

    dt = 1.0 / steps
    last_selected_expert = None
    for step_idx in range(steps):
        t = np.full((1,), step_idx * dt, dtype=np.float32)
        if router_state is not None:
            selected_expert = predict_expert_id(router_state, x_t, t, y)
        else:
            selected_expert = _fallback_expert(step_idx, steps, label, strategy)
        last_selected_expert = selected_expert
        selected_url = expert_urls[selected_expert]
        print(
            "demo_step "
            f"step={step_idx + 1}/{steps} "
            f"label={label} "
            f"t={float(t[0]):.4f} "
            f"selected_expert={selected_expert} "
            f"selected_url={selected_url} "
            f"x_mean={float(x_t.mean()):.6f} "
            f"x_std={float(x_t.std()):.6f}",
            flush=True,
        )
        velocity = clients[selected_expert].predict_velocity(x_t, t, y)
        print(
            "demo_velocity "
            f"step={step_idx + 1}/{steps} "
            f"label={label} "
            f"selected_expert={selected_expert} "
            f"v_mean={float(velocity.mean()):.6f} "
            f"v_std={float(velocity.std()):.6f}",
            flush=True,
        )
        x_t = x_t + dt * velocity
        payload = {
            "type": "step",
            "step": step_idx + 1,
            "steps": steps,
            "label": label,
            "selected_expert": selected_expert,
            "progress": float((step_idx + 1) / steps),
            "frame": _normalize_frame(x_t),
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.03)

    done_payload = {
        "type": "done",
        "label": label,
        "steps": steps,
        "selected_expert": last_selected_expert,
        "progress": 1.0,
        "frame": _normalize_frame(x_t),
    }
    print(f"demo_done label={label} steps={steps} last_selected_expert={last_selected_expert}", flush=True)
    yield f"data: {json.dumps(done_payload)}\n\n"


def create_app(
    expert_urls: tuple[str, str] = ("http://localhost:8000", "http://localhost:8001"),
) -> FastAPI:
    app = FastAPI(title="TPUGPU Router Demo")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.expert_urls = expert_urls
    try:
        router_state, router_info = load_router_state("router_mnist_oracle", "./outputs/router_checkpoints")
        app.state.router_state = router_state
        app.state.router_info = router_info
    except FileNotFoundError:
        app.state.router_state = None
        app.state.router_info = None

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "router_loaded": "yes" if app.state.router_state is not None else "no",
        }

    @app.get("/api/demo/stream")
    async def stream_demo(
        label: int = 2,
        steps: int = 40,
        seed: int | None = None,
        strategy: str = "alternating",
    ) -> StreamingResponse:
        return StreamingResponse(
            _stream_demo_events(
                label=label,
                steps=steps,
                seed=seed,
                expert_urls=app.state.expert_urls,
                strategy=strategy,
                router_state=app.state.router_state,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
