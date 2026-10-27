"""Gradient and integrated-gradient attribution methods."""

import jax
from jax import numpy as jp


def gradient_attribution(fn, *args) -> tuple[jax.Array, ...]:
  """Standard gradient attribution for a scalar function of one or more inputs."""
  if len(args) == 1:
    return (jax.grad(fn)(args[0]),)
  return jax.grad(fn, argnums=tuple(range(len(args))))(*args)


def integrated_gradients(
    fn,
    inputs: tuple[jax.Array, ...],
    baselines: tuple[jax.Array, ...],
    n_steps: int,
) -> tuple[jax.Array, ...]:
  """Integrated gradients attributions for a scalar function.

  Integrates gradients along straight-line paths from baselines to inputs.
  """
  if len(inputs) != len(baselines):
    raise ValueError('inputs and baselines must have the same arity')
  if n_steps < 1:
    raise ValueError('n_steps must be >= 1')

  alphas = jp.linspace(0.0, 1.0, n_steps + 1)[1:]

  def grads_at_alpha(alpha: jax.Array) -> tuple[jax.Array, ...]:
    scaled = tuple(
        base + alpha * (inp - base)
        for inp, base in zip(inputs, baselines)
    )
    if len(scaled) == 1:
      return (jax.grad(fn)(scaled[0]),)
    return jax.grad(fn, argnums=tuple(range(len(scaled))))(*scaled)

  batched_grads = jax.vmap(grads_at_alpha)(alphas)
  if not isinstance(batched_grads, tuple):
    batched_grads = (batched_grads,)
  mean_grads = tuple(jp.mean(g, axis=0) for g in batched_grads)
  return tuple(
      (inp - base) * grad
      for inp, base, grad in zip(inputs, baselines, mean_grads)
  )
