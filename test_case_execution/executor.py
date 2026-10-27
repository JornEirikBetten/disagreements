"""High-level API for executing test cases and suites."""

from typing import Callable, Sequence

import jax
from jax import numpy as jp

from brax.envs.base import Env
from test_case_execution.failure_criteria import FailureCriterion
from test_case_execution.rollout import PolicyFn, make_batched_rollout_fn
from test_case_execution.types import (
    SuiteExecutionResult,
    TestCase,
    TestResult,
    TestSuite,
)


def execute_test_case(
    env: Env,
    policy_fn: PolicyFn,
    test_case: TestCase,
    rollout_length: int,
    failure_fn: FailureCriterion,
) -> TestResult:
  """Execute a single test case with one policy."""
  batched = execute_test_suite(
      env, policy_fn, _singleton_suite(test_case), rollout_length, failure_fn
  )
  return TestResult(
      trace=_unbatch_trace(batched.traces, 0),
      failed=batched.failed[0],
  )


def execute_test_suite(
    env: Env,
    policy_fn: PolicyFn,
    suite: TestSuite,
    rollout_length: int,
    failure_fn: FailureCriterion,
) -> SuiteExecutionResult:
  """Execute all test cases in parallel for one policy."""
  batched_rollout = make_batched_rollout_fn(env, policy_fn, rollout_length)
  traces = batched_rollout(suite.qpos, suite.qvel, suite.policy_rng)
  failed = jax.vmap(failure_fn)(traces)
  return SuiteExecutionResult(traces=traces, failed=failed)


def execute_test_suite_for_policies(
    env: Env,
    policy_fns: Sequence[PolicyFn],
    suite: TestSuite,
    rollout_length: int,
    failure_fn: FailureCriterion,
) -> SuiteExecutionResult:
  """Execute a test suite for multiple policies.

  Returns batched results with leading axis (P, N, ...) where P is the
  number of policies and N is the number of test cases.
  """
  batched_rollout_fns = [
      make_batched_rollout_fn(env, policy_fn, rollout_length)
      for policy_fn in policy_fns
  ]

  failed_rows = []
  trace_rows = []
  for batched_rollout_fn in batched_rollout_fns:
    traces = batched_rollout_fn(suite.qpos, suite.qvel, suite.policy_rng)
    failed_rows.append(jax.vmap(failure_fn)(traces))
    trace_rows.append(traces)

  return SuiteExecutionResult(
      traces=jax.tree.map(lambda *xs: jp.stack(xs), *trace_rows),
      failed=jp.stack(failed_rows),
  )


def make_multi_policy_execute_fn(
    env: Env,
    policy_fns: Sequence[PolicyFn],
    rollout_length: int,
    failure_fn: FailureCriterion,
):
  """Build a cached executor that returns only the failure matrix (P, N)."""
  batched_rollout_fns = [
      make_batched_rollout_fn(env, policy_fn, rollout_length)
      for policy_fn in policy_fns
  ]

  def execute(suite: TestSuite) -> jax.Array:
    failed_rows = []
    for batched_rollout_fn in batched_rollout_fns:
      traces = batched_rollout_fn(suite.qpos, suite.qvel, suite.policy_rng)
      failed_rows.append(jax.vmap(failure_fn)(traces))
    return jp.stack(failed_rows)

  return execute


def make_simultaneous_policy_execute_fn(
    env: Env,
    policy_models: Sequence,
    rollout_length: int,
    failure_fn: FailureCriterion,
):
  """Roll out every policy and every case in one compiled program.

  Returns a failure matrix of shape ``(P, N)``. Policy parameters are stacked
  and mapped together with the case batch, so the GPU holds all of those
  rollouts at once instead of one policy at a time.
  """
  from policies.loader import stack_policy_params
  from test_case_execution.rollout import rollout

  if not policy_models:
    raise ValueError('At least one policy model is required.')
  algorithm = policy_models[0].algorithm
  network = policy_models[0].network
  stacked_params = stack_policy_params(policy_models)
  if algorithm == 'sac':
    from brax.training.agents.sac import networks as sac_networks
    make_inference = sac_networks.make_inference_fn(network)

    def action_fn(params, obs, key):
      infer = make_inference((params[0], params[1]), deterministic=True)
      action, _ = infer(obs, key)
      return action
  else:
    from brax.training.agents.ppo import networks as ppo_networks
    make_inference = ppo_networks.make_inference_fn(network)

    def action_fn(params, obs, key):
      infer = make_inference(params, deterministic=True)
      action, _ = infer(obs, key)
      return action

  def one_policy(params, qpos, qvel, policy_rng):
    def policy_fn(obs, key):
      return action_fn(params, obs, key), None

    def one_case(case_qpos, case_qvel, case_rng):
      trace = rollout(
          env,
          TestCase(qpos=case_qpos, qvel=case_qvel, policy_rng=case_rng),
          policy_fn,
          rollout_length,
      )
      return failure_fn(trace)

    return jax.vmap(one_case)(qpos, qvel, policy_rng)

  @jax.jit
  def execute_arrays(qpos, qvel, policy_rng):
    return jax.vmap(one_policy, in_axes=(0, None, None, None))(
        stacked_params, qpos, qvel, policy_rng
    )

  def execute(suite: TestSuite) -> jax.Array:
    return execute_arrays(suite.qpos, suite.qvel, suite.policy_rng)

  return execute


def make_multi_policy_rollout_fn(
    env: Env,
    policy_fns: Sequence[PolicyFn],
    rollout_length: int,
):
  """Build a cached executor that returns stacked traces ``(P, N, T, ...)``."""
  batched_rollout_fns = [
      make_batched_rollout_fn(env, policy_fn, rollout_length)
      for policy_fn in policy_fns
  ]

  def rollout_suite(suite: TestSuite):
    trace_rows = [
        batched_rollout_fn(suite.qpos, suite.qvel, suite.policy_rng)
        for batched_rollout_fn in batched_rollout_fns
    ]
    return jax.tree.map(lambda *xs: jp.stack(xs), *trace_rows)

  return rollout_suite


def difficulty(failed: jax.Array) -> jax.Array:
  """Fraction of policies that failed. Accepts (N,) or (P, N) arrays."""
  return jp.mean(failed.astype(jp.float32), axis=-1)


def _singleton_suite(test_case: TestCase) -> TestSuite:
  return TestSuite(
      qpos=test_case.qpos[None],
      qvel=test_case.qvel[None],
      policy_rng=test_case.policy_rng[None],
  )


def _unbatch_trace(traces, index: int):
  return jax.tree.map(lambda x: x[index], traces)
