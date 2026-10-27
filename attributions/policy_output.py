"""Differentiable policy outputs for attribution."""

from typing import Any

import jax
from jax import numpy as jp


def policy_action_mode(network: Any, params, obs: jax.Array):
  """Deterministic policy action (distribution mode) for an observation."""
  logits = network.policy_network.apply(params[0], params[1], obs)
  return network.parametric_action_distribution.mode(logits)


def policy_action_loc_scale(
    network: Any, params, obs: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Gaussian loc and scale parameters for an observation."""
  logits = network.policy_network.apply(params[0], params[1], obs)
  distribution = network.parametric_action_distribution.create_dist(logits)
  return distribution.loc, distribution.scale


def policy_value(network: Any, params, obs: jax.Array) -> jax.Array:
  """Scalar value-function estimate for an observation.

  PPO uses the learned V(s). SAC uses the maximum twin-Q value at the
  deterministic policy action: max_i Q_i(s, π(s)).
  """
  if hasattr(network, 'q_network'):
    action = policy_action_mode(network, params, obs)
    q_values = network.q_network.apply(params[0], params[2], obs, action)
    return jp.squeeze(jp.max(q_values, axis=-1))
  return jp.squeeze(network.value_network.apply(params[0], params[2], obs))


def policy_scalar_output(
    network: Any,
    params,
    obs: jax.Array,
    reduction: str = 'sum',
) -> jax.Array:
  """Reduce the deterministic policy action to a scalar for differentiation."""
  action = policy_action_mode(network, params, obs)
  if reduction == 'sum':
    return jp.sum(action)
  if reduction == 'l2':
    return jp.linalg.norm(action)
  raise ValueError(f'Unknown reduction: {reduction}')


def state_importance(network: Any, params, obs: jax.Array):
  """Calculate the importance of the state for the policy action."""
  logits = network.policy_network.apply(params[0], params[1], obs)
  return network.parametric_action_distribution.entropy(logits)
