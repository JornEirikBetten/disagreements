"""Extract feature attributions for policies on test suite initial states."""

from typing import Callable, Sequence

import jax
from jax import numpy as jp

from attributions.methods import gradient_attribution, integrated_gradients
from attributions.policy_output import (
    policy_action_loc_scale,
    policy_action_mode,
    policy_scalar_output,
    policy_value,
)
from attributions.types import AttributionConfig, SuiteAttributionResult
from policies.loader import PolicyModel, stack_policy_params
from test_case_execution.types import TestSuite


def observations_from_suite(env, suite: TestSuite) -> jax.Array:
  """Compute initial observations for each test case. Shape (N, obs_dim)."""
  return jax.vmap(
      lambda qpos, qvel: env.reset_from_initial_state(qpos, qvel).obs
  )(suite.qpos, suite.qvel)


def _observation_attributions(
    network,
    params,
    obs: jax.Array,
    baseline_obs: jax.Array,
    config: AttributionConfig,
) -> tuple[jax.Array, jax.Array]:
  """Gradient and IG attributions w.r.t. a single observation vector."""

  def scalar_fn(x):
    return policy_scalar_output(
        network, params, x, reduction=config.reduction
    )

  obs_grad = gradient_attribution(scalar_fn, obs)[0]
  obs_ig = integrated_gradients(
      scalar_fn, (obs,), (baseline_obs,), config.ig_steps
  )[0]
  return obs_grad, obs_ig


def _state_attributions(
    env,
    network,
    params,
    qpos: jax.Array,
    qvel: jax.Array,
    baseline_qpos: jax.Array,
    baseline_qvel: jax.Array,
    config: AttributionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Gradient and IG attributions w.r.t. qpos and qvel."""

  def scalar_fn(q, v):
    obs = env.reset_from_initial_state(q, v).obs
    return policy_scalar_output(
        network, params, obs, reduction=config.reduction
    )

  qpos_grad, qvel_grad = gradient_attribution(scalar_fn, qpos, qvel)
  qpos_ig, qvel_ig = integrated_gradients(
      scalar_fn,
      (qpos, qvel),
      (baseline_qpos, baseline_qvel),
      config.ig_steps,
  )
  return qpos_grad, qvel_grad, qpos_ig, qvel_ig


def _blank(width: int) -> jax.Array:
  return jp.zeros((width,))


def _selected_attributions_for_params(
    env,
    network,
    params,
    suite: TestSuite,
    baseline_qpos: jax.Array,
    baseline_qvel: jax.Array,
    config: AttributionConfig,
    components: frozenset[str],
) -> SuiteAttributionResult:
  """Integrated gradients for the named components, and nothing else."""
  known = frozenset({'state_ig', 'obs_ig'})
  unknown = components - known
  if unknown:
    raise ValueError(f'Unknown attribution components: {sorted(unknown)}')
  if not components:
    raise ValueError('components must name at least one attribution.')
  need_obs = 'obs_ig' in components
  observations = (
      observations_from_suite(env, suite) if need_obs else None
  )
  if need_obs:
    baseline_obs = env.reset_from_initial_state(
        baseline_qpos, baseline_qvel
    ).obs

  def per_case(qpos, qvel, obs):
    if need_obs:
      def obs_scalar(x):
        return policy_scalar_output(
            network, params, x, reduction=config.reduction
        )

      obs_ig = integrated_gradients(
          obs_scalar, (obs,), (baseline_obs,), config.ig_steps
      )[0]
    else:
      obs_ig = _blank(0)
    if 'state_ig' in components:
      def state_scalar(q, v):
        state_obs = env.reset_from_initial_state(q, v).obs
        return policy_scalar_output(
            network, params, state_obs, reduction=config.reduction
        )

      qpos_ig, qvel_ig = integrated_gradients(
          state_scalar,
          (qpos, qvel),
          (baseline_qpos, baseline_qvel),
          config.ig_steps,
      )
    else:
      qpos_ig = _blank(0)
      qvel_ig = _blank(0)
    return SuiteAttributionResult(
        actions=_blank(0),
        action_loc=_blank(0),
        action_scale=_blank(0),
        values=jp.zeros(()),
        observations=_blank(0),
        observation_gradient=_blank(0),
        observation_integrated_gradient=obs_ig,
        state_gradient_qpos=_blank(0),
        state_gradient_qvel=_blank(0),
        state_integrated_gradient_qpos=qpos_ig,
        state_integrated_gradient_qvel=qvel_ig,
        state_importance_qpos=_blank(0),
        state_importance_qvel=_blank(0),
    )

  if need_obs:
    return jax.vmap(per_case)(suite.qpos, suite.qvel, observations)
  return jax.vmap(lambda qpos, qvel: per_case(qpos, qvel, None))(
      suite.qpos, suite.qvel
  )


def _attributions_for_params(
    env,
    network,
    params,
    suite: TestSuite,
    baseline_qpos: jax.Array,
    baseline_qvel: jax.Array,
    config: AttributionConfig,
    components: frozenset[str] | None = None,
) -> SuiteAttributionResult:
  """Compute attributions for one policy across all test cases."""
  if components is not None:
    return _selected_attributions_for_params(
        env,
        network,
        params,
        suite,
        baseline_qpos,
        baseline_qvel,
        config,
        components,
    )
  observations = observations_from_suite(env, suite)
  baseline_obs = env.reset_from_initial_state(
      baseline_qpos, baseline_qvel
  ).obs

  def per_case(qpos, qvel, obs):
    obs_grad, obs_ig = _observation_attributions(
        network,
        params,
        obs,
        baseline_obs,
        config,
    )
    qpos_grad, qvel_grad, qpos_ig, qvel_ig = _state_attributions(
        env,
        network,
        params,
        qpos,
        qvel,
        baseline_qpos,
        baseline_qvel,
        config,
    )
    action = policy_action_mode(network, params, obs)
    action_loc, action_scale = policy_action_loc_scale(
        network, params, obs
    )
    value = policy_value(network, params, obs)
    return (
        action,
        action_loc,
        action_scale,
        value,
        obs,
        obs_grad,
        obs_ig,
        qpos_grad,
        qvel_grad,
        qpos_ig,
        qvel_ig,
        jp.abs(qpos_ig),
        jp.abs(qvel_ig),
    )

  results = jax.vmap(per_case)(
      suite.qpos, suite.qvel, observations
  )
  return SuiteAttributionResult(
      actions=results[0],
      action_loc=results[1],
      action_scale=results[2],
      values=results[3],
      observations=results[4],
      observation_gradient=results[5],
      observation_integrated_gradient=results[6],
      state_gradient_qpos=results[7],
      state_gradient_qvel=results[8],
      state_integrated_gradient_qpos=results[9],
      state_integrated_gradient_qvel=results[10],
      state_importance_qpos=results[11],
      state_importance_qvel=results[12],
  )


def _attributions_for_policy(
    env,
    policy_model: PolicyModel,
    suite: TestSuite,
    baseline_qpos: jax.Array,
    baseline_qvel: jax.Array,
    config: AttributionConfig,
) -> SuiteAttributionResult:
  """Compute attributions for one policy across all test cases."""
  return _attributions_for_params(
      env,
      policy_model.network,
      policy_model.params,
      suite,
      baseline_qpos,
      baseline_qvel,
      config,
  )


def extract_attributions(
    env,
    policy_models: Sequence[PolicyModel],
    suite: TestSuite,
    config: AttributionConfig | None = None,
    baseline_qpos: jax.Array | None = None,
    baseline_qvel: jax.Array | None = None,
) -> SuiteAttributionResult:
  """Extract attributions for all policies on a test suite.

  Policies are evaluated in parallel via ``vmap`` over stacked parameters.

  Returns:
    ``SuiteAttributionResult`` with leading policy axis (P, N, ...).
  """
  if config is None:
    config = AttributionConfig()
  if baseline_qpos is None:
    baseline_qpos = env.sys.qpos0
  if baseline_qvel is None:
    baseline_qvel = jp.zeros(env.sys.nv)

  if len(policy_models) == 1:
    return jax.tree.map(
        lambda x: x[None],
        _attributions_for_policy(
            env, policy_models[0], suite, baseline_qpos, baseline_qvel, config
        ),
    )

  network = policy_models[0].network
  stacked_params = stack_policy_params(policy_models)

  return jax.vmap(
      lambda params: _attributions_for_params(
          env,
          network,
          params,
          suite,
          baseline_qpos,
          baseline_qvel,
          config,
      )
  )(stacked_params)


# JAX specializes a jitted IG program on the case-batch axis. Keeping that
# axis fixed avoids a new CUDA compile every time the number of kept cases
# changes.
ATTRIBUTION_MICROBATCH = 128


def _suite_slice(suite: TestSuite, start: int, stop: int) -> TestSuite:
  return TestSuite(
      qpos=suite.qpos[start:stop],
      qvel=suite.qvel[start:stop],
      policy_rng=suite.policy_rng[start:stop],
  )


def _pad_suite(suite: TestSuite, batch: int) -> TestSuite:
  """Repeat the first case so the suite length is exactly ``batch``."""
  n = int(suite.qpos.shape[0])
  if n == batch:
    return suite
  if n < 1 or n > batch:
    raise ValueError(f'Cannot pad suite of length {n} to batch {batch}.')
  pad_n = batch - n

  def _pad(values: jax.Array) -> jax.Array:
    filler = jp.broadcast_to(values[:1], (pad_n,) + values.shape[1:])
    return jp.concatenate([values, filler], axis=0)

  return TestSuite(
      qpos=_pad(suite.qpos),
      qvel=_pad(suite.qvel),
      policy_rng=_pad(suite.policy_rng),
  )


def _trim_cases(result: SuiteAttributionResult, n: int) -> SuiteAttributionResult:
  return jax.tree.map(lambda values: values[:, :n], result)


def _concat_cases(parts: Sequence[SuiteAttributionResult]) -> SuiteAttributionResult:
  return jax.tree.map(lambda *values: jp.concatenate(values, axis=1), *parts)


def extract_in_fixed_batches(
    extract_fn: Callable[[TestSuite], SuiteAttributionResult],
    suite: TestSuite,
    *,
    microbatch: int = ATTRIBUTION_MICROBATCH,
) -> SuiteAttributionResult:
  """Run ``extract_fn`` on chunks of exactly ``microbatch`` cases.

  The last chunk is padded by repeating a real case, then trimmed, so the
  jitted program is only ever traced at one batch size.
  """
  if microbatch < 1:
    raise ValueError(f'microbatch must be >= 1, got {microbatch}')
  n = int(suite.qpos.shape[0])
  if n < 1:
    raise ValueError('Cannot extract attributions for an empty suite.')
  parts = []
  for start in range(0, n, microbatch):
    stop = min(start + microbatch, n)
    chunk = _pad_suite(_suite_slice(suite, start, stop), microbatch)
    result = extract_fn(chunk)
    n_keep = stop - start
    if n_keep != microbatch:
      result = _trim_cases(result, n_keep)
    parts.append(result)
  if len(parts) == 1:
    return parts[0]
  return _concat_cases(parts)


def make_batched_extract_attributions_fn(
    env,
    policy_models: Sequence[PolicyModel],
    config: AttributionConfig | None = None,
    baseline_qpos: jax.Array | None = None,
    baseline_qvel: jax.Array | None = None,
    microbatch: int = ATTRIBUTION_MICROBATCH,
    components: frozenset[str] | None = None,
) -> Callable[[TestSuite], SuiteAttributionResult]:
  """Build an IG extractor that vmaps policies and runs fixed case batches.

  The returned function accepts any suite length. Internally it calls one
  jitted program with ``microbatch`` cases (default 128), padding the tail,
  so a changing kept-count does not compile a new kernel.
  """
  if not policy_models:
    raise ValueError('At least one policy model is required.')
  if microbatch < 1:
    raise ValueError(f'microbatch must be >= 1, got {microbatch}')
  if config is None:
    config = AttributionConfig()
  if baseline_qpos is None:
    baseline_qpos = env.sys.qpos0
  if baseline_qvel is None:
    baseline_qvel = jp.zeros(env.sys.nv)

  network = policy_models[0].network
  stacked_params = stack_policy_params(policy_models)

  @jax.jit
  def extract_fixed(suite: TestSuite) -> SuiteAttributionResult:
    return jax.vmap(
        lambda params: _attributions_for_params(
            env,
            network,
            params,
            suite,
            baseline_qpos,
            baseline_qvel,
            config,
            components,
        )
    )(stacked_params)

  def extract(suite: TestSuite) -> SuiteAttributionResult:
    return extract_in_fixed_batches(
        extract_fixed, suite, microbatch=microbatch
    )

  return extract


def make_batched_first_step_actions_fn(
    env,
    policy_models: Sequence[PolicyModel],
) -> Callable[[TestSuite], jax.Array]:
  """Build a jitted first-step action extractor. Returns ``(P, N, nu)``."""
  if not policy_models:
    raise ValueError('At least one policy model is required.')
  network = policy_models[0].network
  stacked_params = stack_policy_params(policy_models)

  @jax.jit
  def first_actions(suite: TestSuite) -> jax.Array:
    observations = observations_from_suite(env, suite)

    def per_policy(params):
      return jax.vmap(
          lambda obs: policy_action_mode(network, params, obs)
      )(observations)

    return jax.vmap(per_policy)(stacked_params)

  return first_actions


def make_batched_policy_outputs_fn(
    env,
    policy_models: Sequence[PolicyModel],
    fields: frozenset[str] | None = None,
):
  """Initial-state actions, scales, and values for every policy.

  This is one forward pass per policy. It does not roll out a test and it
  does not compute integrated gradients. SAC rows of ``q_values`` are the
  twin critics at the deterministic action; PPO rows are NaN.

  ``fields`` limits the forward pass to the named outputs. ``None`` returns
  every output.

  Returns:
    Function from ``TestSuite`` to a dict. Present shapes are
    ``actions`` ``(P, N, nu)``, ``action_loc`` ``(P, N, nu)``,
    ``action_scale`` ``(P, N, nu)``, ``values`` ``(P, N)``,
    ``q_values`` ``(P, N, 2)``.
  """
  if not policy_models:
    raise ValueError('At least one policy model is required.')
  network = policy_models[0].network
  stacked_params = stack_policy_params(policy_models)
  sac = hasattr(network, 'q_network')
  wanted = None if fields is None else frozenset(fields)
  known = frozenset({
      'actions', 'action_loc', 'action_scale', 'values', 'q_values',
  })
  if wanted is not None:
    unknown = wanted - known
    if unknown:
      raise ValueError(f'Unknown policy output fields: {sorted(unknown)}')
    if not wanted:
      raise ValueError('fields must name at least one policy output.')

  def _selected(params, obs):
    logits = network.policy_network.apply(params[0], params[1], obs)
    distribution = network.parametric_action_distribution
    out = {}
    action = None
    if 'actions' in wanted or (
        sac and ('values' in wanted or 'q_values' in wanted)
    ):
      action = distribution.mode(logits)
    if 'actions' in wanted:
      out['actions'] = action
    if 'action_loc' in wanted or 'action_scale' in wanted:
      gaussian = distribution.create_dist(logits)
      if 'action_loc' in wanted:
        out['action_loc'] = gaussian.loc
      if 'action_scale' in wanted:
        out['action_scale'] = gaussian.scale
    if 'values' in wanted or 'q_values' in wanted:
      if sac:
        critics = network.q_network.apply(params[0], params[2], obs, action)
        critics = jp.reshape(critics, (2,))
        if 'q_values' in wanted:
          out['q_values'] = critics
        if 'values' in wanted:
          out['values'] = jp.max(critics)
      else:
        if 'values' in wanted:
          out['values'] = policy_value(network, params, obs)
        if 'q_values' in wanted:
          out['q_values'] = jp.full((2,), jp.nan)
    return out

  @jax.jit
  def outputs(suite: TestSuite) -> dict[str, jax.Array]:
    observations = observations_from_suite(env, suite)

    def per_policy(params):
      if wanted is None:
        def per_case(obs):
          action = policy_action_mode(network, params, obs)
          loc, scale = policy_action_loc_scale(network, params, obs)
          if sac:
            critics = network.q_network.apply(
                params[0], params[2], obs, action
            )
            critics = jp.reshape(critics, (2,))
            value = jp.max(critics)
          else:
            value = policy_value(network, params, obs)
            critics = jp.full((2,), jp.nan)
          return action, loc, scale, value, critics

        action, loc, scale, value, critics = jax.vmap(per_case)(observations)
        return {
            'actions': action,
            'action_loc': loc,
            'action_scale': scale,
            'values': value,
            'q_values': critics,
        }
      return jax.vmap(lambda obs: _selected(params, obs))(observations)

    return jax.vmap(per_policy)(stacked_params)

  return outputs
