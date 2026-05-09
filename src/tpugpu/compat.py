"""JAX 0.7.0 / Flax 0.8.4 compatibility patch for Neuron.

JAX 0.7.0 changed EvalTrace internals (main.level removed).
This patch must be imported before any Flax model operations.
"""

import flax.core.tracers as _ft


def _patched_trace_level(main):
    """Returns the level of the trace, or -inf if unavailable."""
    if main is None:
        return float("-inf")
    if hasattr(main, "level"):
        return main.level
    return float("-inf")


_ft.trace_level = _patched_trace_level
