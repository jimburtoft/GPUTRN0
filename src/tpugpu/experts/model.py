from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def sinusoidal_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / max(half - 1, 1))
    args = timesteps[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = nn.GroupNorm()(x)
        h = nn.swish(h)
        h = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(h)

        cond_proj = nn.Dense(self.channels)(nn.swish(cond))
        h = h + cond_proj[:, None, None, :]

        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        h = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(h)

        if residual.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, kernel_size=(1, 1))(residual)
        return residual + h


class SmallConditionalUNet(nn.Module):
    hidden_channels: int = 64
    num_classes: int = 10
    out_channels: int = 1
    time_embed_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = nn.Dense(self.time_embed_dim)(t_emb)
        t_emb = nn.swish(t_emb)
        t_emb = nn.Dense(self.time_embed_dim)(t_emb)

        class_emb = nn.Embed(self.num_classes, self.time_embed_dim)(y)
        cond = t_emb + class_emb

        h0 = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        h1 = ResBlock(self.hidden_channels)(h0, cond)
        d1 = nn.avg_pool(h1, window_shape=(2, 2), strides=(2, 2))

        h2 = ResBlock(self.hidden_channels * 2)(d1, cond)
        d2 = nn.avg_pool(h2, window_shape=(2, 2), strides=(2, 2))

        mid = ResBlock(self.hidden_channels * 2)(d2, cond)
        mid = ResBlock(self.hidden_channels * 2)(mid, cond)

        u1 = jax_image_resize(mid, h2.shape[1:3])
        u1 = jnp.concatenate([u1, h2], axis=-1)
        u1 = ResBlock(self.hidden_channels * 2)(u1, cond)

        u2 = jax_image_resize(u1, h1.shape[1:3])
        u2 = jnp.concatenate([u2, h1], axis=-1)
        u2 = ResBlock(self.hidden_channels)(u2, cond)

        out = nn.GroupNorm()(u2)
        out = nn.swish(out)
        out = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(out)
        return out


def jax_image_resize(x: jnp.ndarray, shape: tuple[int, int]) -> jnp.ndarray:
    return jax.image.resize(x, (x.shape[0], shape[0], shape[1], x.shape[-1]), method="nearest")
