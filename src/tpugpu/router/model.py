from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from tpugpu.experts.model import sinusoidal_embedding


class RouterMLP(nn.Module):
    num_experts: int
    num_classes: int = 10
    hidden_dim: int = 256
    time_embed_dim: int = 128

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x_flat = x_t.reshape((x_t.shape[0], -1))

        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = nn.Dense(self.time_embed_dim)(t_emb)
        t_emb = nn.swish(t_emb)
        t_emb = nn.Dense(self.time_embed_dim)(t_emb)

        class_emb = nn.Embed(self.num_classes, self.time_embed_dim)(y)

        h = jnp.concatenate([x_flat, t_emb, class_emb], axis=-1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.swish(h)
        logits = nn.Dense(self.num_experts)(h)
        return logits


@jax.jit
def router_predict_logits(params: dict, apply_fn, x_t: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return apply_fn({"params": params}, x_t, t, y)
