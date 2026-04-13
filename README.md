# GPUTPU0

**Live Demo**: [http://34.141.174.148:8081/](http://34.141.174.148:8081/)

A minimal proof-of-concept demonstrating that a **Decentralized Diffusion Model (DDM)** can operate as one logical system across heterogeneous accelerators — specifically **TPU** and **GPU**.

## What It Does

Two diffusion model experts, each trained independently on a disjoint subset of MNIST, are served from different geographic regions on different hardware. A lightweight router selects which expert handles each denoising step at inference time, composing them into a single generation path.

- **Expert A** runs on a **TPU v6e** and is trained on digits 0–4
- **Expert B** runs on an **NVIDIA A100 GPU** and is trained on digits 5–9
- A **CPU-based router** in a third region selects experts per-step based on the noisy state, timestep, and class label

The system generates MNIST digits through a flow-matching denoising loop where each step may be routed to a different accelerator over the network.

## Architecture

```
x_t, t, label
      │
      ▼
  ┌─────────┐
  │  Router  │  (CPU, europe-west4)
  └────┬────┘
       │  selects expert per step
  ┌────┴────┐
  │         │
  ▼         ▼
┌─────┐  ┌─────┐
│ TPU │  │ GPU │
│Expert│  │Expert│  (us-east5 / asia-southeast1)
└──┬──┘  └──┬──┘
   │         │
   ▼         ▼
 velocity   velocity
      │
      ▼
  x_{t+1} = x_t + dt * velocity
```

## Model

- **Architecture**: Small class-conditional UNet (~1.65M params)
- **Objective**: Flow matching (`v* = x_0 - x_1`)
- **Conditioning**: Class label via learned embedding
- **Framework**: JAX + Flax (same codebase runs on both TPU and GPU)

## Repo Structure

```
src/tpugpu/
  experts/        # UNet model, training loop, inference helpers
  router/         # Router MLP, training, inference, expert HTTP client
  serving/        # Binary npz protocol for network transport
  demo/           # FastAPI app + SSE streaming + web UI
  data/           # MNIST loading and class-based filtering
  eval/           # Image grids, t-SNE, PCA-FID, training curves
scripts/
  train_expert_mnist.py      # Train one expert
  train_router_mnist.py      # Train the router
  serve_expert_mnist.py      # Serve an expert over HTTP
  sample_expert_mnist.py     # Generate samples from a checkpoint
  sample_distributed_mnist.py # Distributed sampling via remote expert
  run_router_demo.py         # Launch the web demo
```

## Setup

### Requirements

- Python 3.10+
- JAX with TPU or GPU backend
- Flax, Optax, Orbax
- FastAPI + Uvicorn (for the demo)

### Install

```bash
pip install -e .
pip install jax flax optax orbax-checkpoint matplotlib scikit-learn
pip install fastapi uvicorn
```

### Train Experts

```bash
# Expert A: digits 0-4
python scripts/train_expert_mnist.py \
  --expert-name expert_a --class-ids 0,1,2,3,4 \
  --num-epochs 50 --batch-size 64

# Expert B: digits 5-9
python scripts/train_expert_mnist.py \
  --expert-name expert_b --class-ids 5,6,7,8,9 \
  --num-epochs 50 --batch-size 64
```

### Train Router

```bash
python scripts/train_router_mnist.py \
  --router-name router_oracle \
  --expert-label-splits "0,1,2,3,4|5,6,7,8,9" \
  --num-epochs 10
```

### Serve Experts

```bash
# On the TPU machine
python scripts/serve_expert_mnist.py --expert-name expert_a --port 8000

# On the GPU machine
python scripts/serve_expert_mnist.py --expert-name expert_b --port 8000
```

### Run Demo

```bash
python scripts/run_router_demo.py --host 0.0.0.0 --port 8080
```

The demo streams a live denoising visualization where you can see which expert is selected at each step.

## Wire Protocol

Experts communicate over HTTP using a compact binary payload (numpy `.npz`):

- **Request**: `{x_t, t, y}` — noisy state, timestep, class label
- **Response**: `{velocity}` — predicted velocity field

This keeps per-step latency low enough for interactive demo streaming across regions.

## What This Proves

1. Two experts trained independently on disjoint data subsets
2. One expert on TPU, one on GPU — different hardware, different regions
3. A router selects experts during the denoising trajectory (per-step, not per-request)
4. The full system generates samples through one logical routed inference path
5. The combined system behaves like a routed DDM, not a static ensemble
