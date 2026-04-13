#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/tpugpu-router-venv"

sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  python3-pip \
  git \
  gh \
  jq \
  tree \
  zip \
  build-essential \
  tmux

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install jax flax optax tensorflow-datasets tensorflow-cpu orbax-checkpoint chex einops scikit-learn matplotlib fastapi uvicorn
pip install -e "${REPO_ROOT}"

python - <<'PY'
import jax
print("jax_version", jax.__version__)
print("devices", jax.devices())
print("backend", jax.default_backend())
PY
