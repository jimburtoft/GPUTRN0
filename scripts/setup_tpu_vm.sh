#!/usr/bin/env bash
set -euo pipefail

# TPU-first project bootstrap for the TPUGPU repo.
# Safe to run multiple times: it only installs missing system packages,
# recreates the project venv, installs Python dependencies, and runs a smoke test.

VENV_DIR="${HOME}/tpugpu-venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> TPUGPU TPU VM setup"
echo "Repo: $(pwd)"
echo "Venv: ${VENV_DIR}"

echo "==> Installing required system packages"
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  python3-pip \
  git \
  gh \
  build-essential \
  tmux \
  tree \
  jq \
  unzip \
  zip

echo "==> Creating virtualenv"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing TPU JAX stack and project dependencies"
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install \
  flax \
  matplotlib \
  optax \
  scikit-learn \
  tensorflow \
  tensorflow-datasets \
  orbax-checkpoint \
  chex \
  einops \
  pytest \
  rich

echo "==> Installing local package in editable mode"
pip install -e .

echo "==> Running smoke test"
python - <<'PY'
import jax
print("jax version:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY

cat <<EOF

Setup complete.

Activate the environment with:
  source ${VENV_DIR}/bin/activate

Quick TPU check:
  python -c "import jax; print(jax.devices()); print(jax.default_backend())"
EOF
