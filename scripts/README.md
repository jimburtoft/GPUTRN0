# Scripts

This directory will hold thin entrypoints for:

- training experts
- training the router
- generating routed samples
- running quick evaluations
- bootstrapping TPU VMs

The core logic should live under `src/tpugpu`, not here.

Available helper scripts:

- `setup_tpu_vm.sh`
  - installs the TPU-side Python environment
  - installs JAX TPU + core project dependencies
  - runs a smoke test
- `train_expert_mnist.py`
  - trains one small class-conditional MNIST expert in JAX
  - first runnable path for the TPU-side POC
