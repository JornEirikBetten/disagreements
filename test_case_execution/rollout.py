"""Vectorized rollout of policies on test cases."""

from typing import Callable, Tuple

import jax
from jax import numpy as jp
from flax import struct

from brax.envs.base import Env, State
from test_case_execution.types import RolloutMetrics, RolloutTrace, TestCase

PolicyFn = Callable[[jax.Array, jax.Array], Tuple[jax.Array, object]]


@struct.dataclass
class _RolloutStep:
  action: jax.Array
  reward: jax.Array
  done: jax.Array
  metrics: RolloutMetrics
  qvel: jax.Array


def _metrics_from_state(metrics: dict) -> RolloutMetrics:
  return RolloutMetrics(
      forward_reward=metrics['forward_reward'],
      reward_linvel=metrics['reward_linvel'],
      reward_quadctrl=metrics['reward_quadctrl'],
      reward_alive=metrics['reward_alive'],
      x_position=metrics['x_position'],
      y_position=metrics['y_position'],
      distance_from_origin=metrics['distance_from_origin'],
      x_velocity=metrics['x_velocity'],
      y_velocity=metrics['y_velocity'],
  )


def _stack_rollout_steps(steps: _RolloutStep) -> RolloutTrace:
  return RolloutTrace(
      actions=steps.action,
      rewards=steps.reward,
      dones=steps.done,
      metrics=steps.metrics,
      qvel=steps.qvel,
  )


def rollout(
    env: Env,
    test_case: TestCase,
    policy_fn: PolicyFn,
    rollout_length: int,
) -> RolloutTrace:
  """Roll out a single policy on a single test case."""
  state = env.reset_from_initial_state(test_case.qpos, test_case.qvel)

  def step_fn(carry, _):
    state, key = carry
    key, act_key = jax.random.split(key)
    action, _ = policy_fn(state.obs, act_key)
    qvel = state.pipeline_state.qvel
    next_state = env.step(state, action)
    step = _RolloutStep(
        action=action,
        reward=next_state.reward,
        done=next_state.done,
        metrics=_metrics_from_state(next_state.metrics),
        qvel=qvel,
    )
    return (next_state, key), step

  _, steps = jax.lax.scan(
      step_fn, (state, test_case.policy_rng), (), rollout_length
  )
  return _stack_rollout_steps(steps)


def make_rollout_fn(
    env: Env,
    policy_fn: PolicyFn,
    rollout_length: int,
):
  """Build a jitted rollout function closed over env and policy."""

  def rollout_arrays(qpos, qvel, policy_rng):
    test_case = TestCase(qpos=qpos, qvel=qvel, policy_rng=policy_rng)
    return rollout(env, test_case, policy_fn, rollout_length)

  return jax.jit(rollout_arrays)


def make_batched_rollout_fn(
    env: Env,
    policy_fn: PolicyFn,
    rollout_length: int,
):
  """Build a jitted function that vmaps rollout over a batch of test cases."""

  rollout_arrays = make_rollout_fn(env, policy_fn, rollout_length)

  @jax.jit
  def batched_rollout(qpos, qvel, policy_rng):
    return jax.vmap(rollout_arrays)(qpos, qvel, policy_rng)

  return batched_rollout
