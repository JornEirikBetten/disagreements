"""Data types for test cases, rollouts, and execution results."""

from typing import Sequence

import jax
from jax import numpy as jp
from flax import struct


@struct.dataclass
class TestCase:
  """A single test case: initial state plus policy RNG."""

  qpos: jax.Array       # (nq,)
  qvel: jax.Array       # (nv,)
  policy_rng: jax.Array  # PRNGKey shape (2,)


@struct.dataclass
class TestSuite:
  """Batched test cases for vectorized execution."""

  qpos: jax.Array        # (N, nq)
  qvel: jax.Array        # (N, nv)
  policy_rng: jax.Array  # (N, 2)


@struct.dataclass
class RolloutMetrics:
  """Per-timestep environment metrics collected during rollout."""

  forward_reward: jax.Array
  reward_linvel: jax.Array
  reward_quadctrl: jax.Array
  reward_alive: jax.Array
  x_position: jax.Array
  y_position: jax.Array
  distance_from_origin: jax.Array
  x_velocity: jax.Array
  y_velocity: jax.Array


@struct.dataclass
class RolloutTrace:
  """Trajectory produced by rolling out a policy on a test case."""

  actions: jax.Array       # (T, nu)
  rewards: jax.Array       # (T,)
  dones: jax.Array         # (T,)
  metrics: RolloutMetrics  # each field (T,)
  qvel: jax.Array          # (T, nv) velocity at the step the action is applied


@struct.dataclass
class TestResult:
  """Outcome of executing one test case with one policy."""

  trace: RolloutTrace
  failed: jax.Array  # scalar bool


@struct.dataclass
class SuiteExecutionResult:
  """Batched results from executing a test suite with one policy."""

  traces: RolloutTrace   # batched on leading axis (N, ...)
  failed: jax.Array      # (N,) bool


def test_case_from_genome(
    dqpos: jax.Array,
    qvel: jax.Array,
    policy_rng: jax.Array,
    qpos0: jax.Array,
) -> TestCase:
  """Build a test case from a GA genome and default pose."""
  return TestCase(qpos=qpos0 + dqpos, qvel=qvel, policy_rng=policy_rng)


def stack_test_cases(cases: Sequence[TestCase]) -> TestSuite:
  """Stack individual test cases into a batched suite."""
  return TestSuite(
      qpos=jp.stack([c.qpos for c in cases]),
      qvel=jp.stack([c.qvel for c in cases]),
      policy_rng=jp.stack([c.policy_rng for c in cases]),
  )


def unstack_test_suite(suite: TestSuite) -> list[TestCase]:
  """Split a batched suite into individual test cases."""
  n = suite.qpos.shape[0]
  return [
      TestCase(
          qpos=suite.qpos[i],
          qvel=suite.qvel[i],
          policy_rng=suite.policy_rng[i],
      )
      for i in range(n)
  ]
