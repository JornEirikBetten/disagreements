"""Data types for feature attribution results."""

import jax
from flax import struct


@struct.dataclass
class AttributionConfig:
  """Hyperparameters for attribution extraction."""

  ig_steps: int = 32
  reduction: str = 'sum'  # 'sum' or 'l2' — how policy actions become a scalar


@struct.dataclass
class SuiteAttributionResult:
  """Attributions for P policies on N test cases from a TestSuite.

  All attribution arrays use shape (P, N, ...) unless noted.
  """

  actions: jax.Array
  action_loc: jax.Array
  action_scale: jax.Array
  values: jax.Array
  observations: jax.Array
  observation_gradient: jax.Array
  observation_integrated_gradient: jax.Array
  state_gradient_qpos: jax.Array
  state_gradient_qvel: jax.Array
  state_integrated_gradient_qpos: jax.Array
  state_integrated_gradient_qvel: jax.Array
  state_importance_qpos: jax.Array
  state_importance_qvel: jax.Array
