"""TPUGPU: a minimal routed DDM proof across TPU and GPU experts."""

# Apply JAX 0.7.0 / Flax compatibility patch (Neuron backend)
from tpugpu import compat as _compat  # noqa: F401
