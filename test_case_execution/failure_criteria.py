"""Failure criteria evaluated on rollout traces."""

from typing import Callable

import jax
from jax import numpy as jp

from test_case_execution.types import RolloutTrace

FailureCriterion = Callable[[RolloutTrace], jax.Array]


def humanoid_fall_failure_criterion(trace: RolloutTrace) -> jax.Array:
  """Fail if the humanoid becomes unhealthy (falls) during the rollout."""
  return jp.any(trace.dones > 0)


def humanoid_fall_failed_at_horizon(
    dones: jax.Array, horizon: int
) -> jax.Array:
  """Fall by step ``horizon`` from a longer rollout. ``dones`` ends in time."""
  t = int(horizon)
  if t < 1:
    raise ValueError(f'horizon must be >= 1, got {horizon!r}')
  if t > dones.shape[-1]:
    raise ValueError(
        f'horizon {t} exceeds trace length {dones.shape[-1]}'
    )
  return jp.any(dones[..., :t] > 0, axis=-1)


def humanoid_low_progress_failure_criterion(
    trace: RolloutTrace,
    min_forward_dist: float = 1.0,
) -> jax.Array:
  """Fail if forward progress from start to end is below a threshold."""
  x_init = trace.metrics.x_position[0]
  x_final = trace.metrics.x_position[-1]
  return (x_final - x_init) < min_forward_dist


def hopper_fall_failure_criterion(trace: RolloutTrace) -> jax.Array:
  """Fail if Hopper becomes unhealthy (too low or too tilted)."""
  return jp.any(trace.dones > 0)


def walker2d_fall_failure_criterion(trace: RolloutTrace) -> jax.Array:
  """Fail if Walker2d becomes unhealthy (height or torso angle)."""
  return jp.any(trace.dones > 0)


def ant_fall_failure_criterion(trace: RolloutTrace) -> jax.Array:
  """Fail if the ant becomes unhealthy (torso height) during the rollout."""
  return jp.any(trace.dones > 0)


def prefix_failed_matrix(
    environment: str,
    traces: RolloutTrace,
    horizon: int,
) -> jax.Array:
  """Per-policy failure on the length-``horizon`` prefix of ``traces``.

  ``traces`` may be a single-policy batch ``(N, T, ...)`` or stacked policies
  ``(P, N, T, ...)``.
  """
  if environment in ('humanoid', 'hopper', 'walker2d', 'ant'):
    return humanoid_fall_failed_at_horizon(traces.dones, horizon)
  raise ValueError(
      f'Prefix failure is not implemented for environment {environment!r}'
  )


def compose_failure_criteria(
    *criteria: FailureCriterion,
    mode: str = 'any',
) -> FailureCriterion:
  """Combine multiple failure criteria with logical any/all."""

  def combined(trace: RolloutTrace) -> jax.Array:
    outcomes = jp.stack([fn(trace) for fn in criteria])
    if mode == 'any':
      return jp.any(outcomes)
    if mode == 'all':
      return jp.all(outcomes)
    raise ValueError(f'Unknown mode: {mode}')

  return combined
