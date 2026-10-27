"""Shared environment / policy / executor setup for search scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Sequence

import jax
from jax import numpy as jp

from attributions import AttributionConfig
from ga.ga import GAConfig

LOCOMOTION_ENVS = ('humanoid', 'hopper', 'walker2d', 'ant')


def policy_filename(environment: str, algorithm: str) -> str:
  del environment, algorithm
  return 'best_control_{id}/params.json'


@dataclass
class SearchBundle:
  """Everything needed to sample, evaluate, and (optionally) collect."""

  environment: str
  algorithm: str
  kind: str
  env: Any
  ga_config: GAConfig
  qpos0: jax.Array
  nq: int
  nv: int
  operators: dict
  execute_fn: Callable
  rollout_fn: Callable
  extract_attributions_fn: Callable
  policy_ids: list[int]
  attribution_config: AttributionConfig
  eval_policy_ids: list[int] | None = None
  policy_algorithms: list[str] | None = None
  eval_policy_algorithms: list[str] | None = None
  first_step_actions_fn: Callable | None = None
  parallel_execute_fn: Callable | None = None
  policy_outputs_fn: Callable | None = None


def _sys_pose_dims(env) -> tuple:
  """Default pose and tangent sizes for a Brax/MJX locomotion env."""
  sys = env.sys
  qpos0 = getattr(sys, 'qpos0', None)
  if qpos0 is None:
    qpos0 = sys.init_q
  nq = int(getattr(sys, 'nq', sys.q_size()))
  nv = int(getattr(sys, 'nv', sys.qd_size()))
  return qpos0, nq, nv


def _operators(module) -> dict:
  return dict(
      genomes_to_suite_fn=module.genomes_to_test_suite,
      random_genome_fn=module.random_genome,
      repair_genome_fn=module.repair_genome,
      crossover_genomes_fn=module.crossover_genomes,
      mutate_genome_fn=module.mutate_genome,
  )


def _locomotion_ga_config(environment: str, rollout_length: int) -> GAConfig:
  if environment == 'humanoid':
    return GAConfig(
        population_size=1000,
        n_generations=100,
        rollout_length=rollout_length,
        fitness_mode='maximize',
        elite_count=500,
        log_every=1,
    )
  if environment == 'hopper':
    # Match Brax Hopper reset_noise_scale=5e-3. Healthy torso angle is
    # only ±0.2 rad, so keep genome bounds tight.
    return GAConfig(
        population_size=1000,
        n_generations=100,
        rollout_length=rollout_length,
        fitness_mode='maximize',
        elite_count=500,
        noise_scale=5e-3,
        sigma_dqpos=2.5e-3,
        sigma_qvel=2.5e-3,
        max_dqpos=0.2,
        max_qvel=0.2,
        log_every=1,
    )
  if environment == 'walker2d':
    # Healthy height [0.8, 2.0] m and torso ±1 rad. Hopper-tight ±0.2
    # never reaches the last surviving policies.
    return GAConfig(
        population_size=1000,
        n_generations=100,
        rollout_length=rollout_length,
        fitness_mode='maximize',
        elite_count=500,
        noise_scale=5e-3,
        sigma_dqpos=5e-2,
        sigma_qvel=5e-2,
        max_dqpos=1.0,
        max_qvel=1.0,
        log_every=1,
    )
  if environment == 'ant':
    # Match Brax Ant reset_noise_scale=0.1. Init torso height is 0.55 m
    # and the healthy band is [0.2, 1.0], so the clip stays inside that
    # band at generation 0. The free joint is renormalized by the mutator.
    return GAConfig(
        population_size=1000,
        n_generations=100,
        rollout_length=rollout_length,
        fitness_mode='maximize',
        elite_count=500,
        noise_scale=0.1,
        sigma_dqpos=0.05,
        sigma_qvel=0.05,
        max_dqpos=0.25,
        max_qvel=0.5,
        log_every=1,
    )
  raise ValueError(f'Unknown locomotion environment {environment!r}')


def _not_built(name: str) -> Callable:
  """Callable that fails if a search asks for a bundle piece it disabled."""

  def _call(*_args, **_kwargs):
    raise RuntimeError(f'{name} was not built for this search bundle.')

  return _call


def make_search_bundle(
    environment: str,
    algorithm: str,
    n_policies: int,
    rollout_length: int,
    *,
    with_attributions: bool = True,
    policy_ids: Sequence[int] | None = None,
    with_executor: bool = True,
    with_policies: bool = True,
    with_policy_outputs: bool = False,
    policy_output_fields: frozenset[str] | None = None,
    attribution_components: frozenset[str] | None = None,
) -> SearchBundle:
  """Load env, policies, and executors for one environment/algorithm.

  ``with_executor`` builds the rollout that scores pass/fail. Feature search
  leaves it off. ``with_policies`` loads the search checkpoints; a state-only
  objective does not need them. ``with_policy_outputs`` adds the initial-state
  action/value forward pass used as a fitness feature. ``policy_output_fields``
  and ``attribution_components`` limit that pass to the objective; ``None``
  measures every output.
  """
  if algorithm not in ('ppo', 'sac'):
    raise ValueError(f'Unknown algorithm {algorithm!r}; expected ppo or sac')
  if n_policies < 2:
    raise ValueError(f'n_policies must be >= 2, got {n_policies}')
  if rollout_length < 1:
    raise ValueError(f'rollout_length must be >= 1, got {rollout_length}')
  if not with_policies and (
      with_attributions or with_executor or with_policy_outputs
  ):
    raise ValueError(
        'with_policies=False cannot be combined with rollouts, '
        'attributions, or policy outputs.'
    )
  if policy_ids is not None:
    policy_ids = [int(i) for i in policy_ids]
    if len(policy_ids) != n_policies:
      raise ValueError(
          f'policy_ids has length {len(policy_ids)}, expected n_policies='
          f'{n_policies}'
      )

  if environment in LOCOMOTION_ENVS:
    return _make_locomotion_bundle(
        environment,
        algorithm,
        n_policies,
        rollout_length,
        with_attributions=with_attributions,
        policy_ids=policy_ids,
        with_executor=with_executor,
        with_policies=with_policies,
        with_policy_outputs=with_policy_outputs,
        policy_output_fields=policy_output_fields,
        attribution_components=attribution_components,
    )
  raise ValueError(
      f'Unknown environment {environment!r}; expected one of {LOCOMOTION_ENVS}'
  )


def _load_locomotion_env(environment: str):
  """Environment, genetic operators, and failure criterion for one body."""
  if environment == 'humanoid':
    from envs.humanoid import make_humanoid_env
    from test_case_execution import humanoid_fall_failure_criterion

    return make_humanoid_env(), {}, humanoid_fall_failure_criterion
  if environment == 'hopper':
    from envs.hopper import make_hopper_env
    from mutators import hopper as hopper_mutators
    from test_case_execution import hopper_fall_failure_criterion

    return make_hopper_env(), _operators(hopper_mutators), hopper_fall_failure_criterion
  if environment == 'walker2d':
    from envs.walker2d import make_walker2d_env
    from mutators import walker2d as walker2d_mutators
    from test_case_execution import walker2d_fall_failure_criterion

    return (
        make_walker2d_env(),
        _operators(walker2d_mutators),
        walker2d_fall_failure_criterion,
    )
  if environment == 'ant':
    from envs.ant import make_ant_env
    from mutators import ant as ant_mutators
    from test_case_execution import ant_fall_failure_criterion

    operators = _operators(ant_mutators)
    return make_ant_env(), operators, ant_fall_failure_criterion
  raise ValueError(f'Unknown locomotion environment {environment!r}')


def _make_locomotion_bundle(
    environment: str,
    algorithm: str,
    n_policies: int,
    rollout_length: int,
    *,
    with_attributions: bool,
    policy_ids: Sequence[int] | None = None,
    with_executor: bool = True,
    with_policies: bool = True,
    with_policy_outputs: bool = False,
    policy_output_fields: frozenset[str] | None = None,
    attribution_components: frozenset[str] | None = None,
) -> SearchBundle:
  from attributions.extract import (
      make_batched_extract_attributions_fn,
      make_batched_first_step_actions_fn,
      make_batched_policy_outputs_fn,
  )
  from policies.loader import load_policy_models_from_dir
  from test_case_execution.executor import (
      make_multi_policy_execute_fn,
      make_multi_policy_rollout_fn,
      make_simultaneous_policy_execute_fn,
  )

  ga_config = _locomotion_ga_config(environment, rollout_length)
  env, operators, failure_fn = _load_locomotion_env(environment)

  qpos0, nq, nv = _sys_pose_dims(env)
  if environment == 'ant':
    # qpos0 is the XML pose (ankles at 0, outside their limits). Training
    # and Ant.reset start from init_q.
    qpos0 = env.sys.init_q
  if policy_ids is None:
    policy_ids = list(range(1, n_policies + 1))
  else:
    policy_ids = [int(i) for i in policy_ids]
  attribution_config = AttributionConfig(ig_steps=16, reduction='sum')
  if not with_policies:
    return SearchBundle(
        environment=environment,
        algorithm=algorithm,
        kind='locomotion',
        env=env,
        ga_config=ga_config,
        qpos0=qpos0,
        nq=nq,
        nv=nv,
        operators=operators,
        execute_fn=_not_built('execute_fn'),
        rollout_fn=_not_built('rollout_fn'),
        extract_attributions_fn=_not_built('extract_attributions_fn'),
        policy_ids=policy_ids,
        attribution_config=attribution_config,
        first_step_actions_fn=None,
        parallel_execute_fn=None,
        policy_outputs_fn=None,
    )
  policy_dir = os.path.join('trained_policies', environment, algorithm)
  policy_models = load_policy_models_from_dir(
      policy_dir,
      observation_size=env.observation_size,
      action_size=env.action_size,
      policy_ids=policy_ids,
      filename_template=policy_filename(environment, algorithm),
      algorithm=algorithm,
  )
  if with_executor:
    policy_fns = [model.policy_fn for model in policy_models]
    execute_fn = make_multi_policy_execute_fn(
        env, policy_fns, ga_config.rollout_length, failure_fn
    )
    parallel_execute_fn = make_simultaneous_policy_execute_fn(
        env, policy_models, ga_config.rollout_length, failure_fn
    )
    rollout_fn = make_multi_policy_rollout_fn(
        env, policy_fns, ga_config.rollout_length
    )
    first_step_actions_fn = make_batched_first_step_actions_fn(
        env, policy_models
    )
  else:
    execute_fn = _not_built('execute_fn')
    parallel_execute_fn = None
    rollout_fn = _not_built('rollout_fn')
    first_step_actions_fn = None
  extract_fn = (
      make_batched_extract_attributions_fn(
          env,
          policy_models,
          attribution_config,
          components=attribution_components,
      )
      if with_attributions
      else (lambda suite: None)
  )
  policy_outputs_fn = (
      make_batched_policy_outputs_fn(
          env, policy_models, fields=policy_output_fields
      )
      if with_policy_outputs
      else None
  )
  return SearchBundle(
      environment=environment,
      algorithm=algorithm,
      kind='locomotion',
      env=env,
      ga_config=ga_config,
      qpos0=qpos0,
      nq=nq,
      nv=nv,
      operators=operators,
      execute_fn=execute_fn,
      rollout_fn=rollout_fn,
      extract_attributions_fn=extract_fn,
      policy_ids=policy_ids,
      attribution_config=attribution_config,
      first_step_actions_fn=first_step_actions_fn,
      parallel_execute_fn=parallel_execute_fn,
      policy_outputs_fn=policy_outputs_fn,
  )


MIXED_ALGORITHM = 'ppo_sac'


def make_mixed_search_bundle(
    environment: str,
    rollout_length: int,
    search_members: Sequence[dict],
    eval_members: Sequence[dict] | None = None,
    *,
    with_attributions: bool = True,
    with_executor: bool = True,
    with_policy_outputs: bool = False,
    policy_output_fields: frozenset[str] | None = None,
    attribution_components: frozenset[str] | None = None,
) -> SearchBundle:
  """Load a PPO-then-SAC ensemble for one locomotion environment.

  ``search_members`` and ``eval_members`` are records from
  ``split_search_eval_mixed``. Policies are executed independently and
  attributions are computed per algorithm, then concatenated on the policy
  axis, because PPO and SAC parameters cannot share one stacked network.
  """
  from attributions.extract import (
      make_batched_extract_attributions_fn,
      make_batched_first_step_actions_fn,
      make_batched_policy_outputs_fn,
  )
  from policies.loader import load_policy_models_from_dir
  from test_case_execution.executor import (
      make_multi_policy_execute_fn,
      make_multi_policy_rollout_fn,
      make_simultaneous_policy_execute_fn,
  )

  if environment not in LOCOMOTION_ENVS:
    raise ValueError(
        f'Mixed PPO/SAC search supports locomotion envs {LOCOMOTION_ENVS}; '
        f'got {environment!r}'
    )
  if rollout_length < 1:
    raise ValueError(f'rollout_length must be >= 1, got {rollout_length}')
  if len(search_members) < 2:
    raise ValueError(
        f'Need at least 2 search policies, got {len(search_members)}'
    )

  ga_config = _locomotion_ga_config(environment, rollout_length)
  env, operators, failure_fn = _load_locomotion_env(environment)
  qpos0, nq, nv = _sys_pose_dims(env)
  if environment == 'ant':
    qpos0 = env.sys.init_q

  policy_models = []
  policy_ids: list[int] = []
  policy_algorithms: list[str] = []
  for member in search_members:
    algorithm = str(member['algorithm'])
    policy_id = int(member['policy_id'])
    policy_dir = os.path.join('trained_policies', environment, algorithm)
    policy_models.extend(
        load_policy_models_from_dir(
            policy_dir,
            observation_size=env.observation_size,
            action_size=env.action_size,
            policy_ids=[policy_id],
            filename_template=policy_filename(environment, algorithm),
            algorithm=algorithm,
        )
    )
    policy_ids.append(policy_id)
    policy_algorithms.append(algorithm)

  # Stack within each consecutive same-algorithm block so PPO and SAC stay
  # on their own networks, then concatenate back to member order.
  blocks: list[tuple[str, list]] = []
  for model, algorithm in zip(policy_models, policy_algorithms):
    if blocks and blocks[-1][0] == algorithm:
      blocks[-1][1].append(model)
    else:
      blocks.append((algorithm, [model]))

  attribution_config = AttributionConfig(ig_steps=16, reduction='sum')
  if with_executor:
    policy_fns = [model.policy_fn for model in policy_models]
    execute_fn = make_multi_policy_execute_fn(
        env, policy_fns, ga_config.rollout_length, failure_fn
    )
    # PPO and SAC stay on separate stacked rollouts, then join on the policy axis.
    block_executors = [
        make_simultaneous_policy_execute_fn(
            env, models, ga_config.rollout_length, failure_fn
        )
        for _, models in blocks
    ]

    def parallel_execute_fn(suite):
      return jp.concatenate([fn(suite) for fn in block_executors], axis=0)

    rollout_fn = make_multi_policy_rollout_fn(
        env, policy_fns, ga_config.rollout_length
    )
    action_fns = [
        make_batched_first_step_actions_fn(env, models) for _, models in blocks
    ]

    def first_step_actions_fn(suite):
      return jp.concatenate([fn(suite) for fn in action_fns], axis=0)
  else:
    execute_fn = _not_built('execute_fn')
    parallel_execute_fn = None
    rollout_fn = _not_built('rollout_fn')
    first_step_actions_fn = None

  if with_attributions:
    extractors = [
        make_batched_extract_attributions_fn(
            env,
            models,
            attribution_config,
            components=attribution_components,
        )
        for _, models in blocks
    ]

    def extract_fn(suite):
      parts = [fn(suite) for fn in extractors]
      return jax.tree.map(lambda *xs: jp.concatenate(xs, axis=0), *parts)
  else:
    def extract_fn(suite):
      del suite
      return None

  if with_policy_outputs:
    output_fns = [
        make_batched_policy_outputs_fn(
            env, models, fields=policy_output_fields
        )
        for _, models in blocks
    ]

    def policy_outputs_fn(suite):
      parts = [fn(suite) for fn in output_fns]
      keys = parts[0].keys()
      return {
          key: jp.concatenate([part[key] for part in parts], axis=0)
          for key in keys
      }
  else:
    policy_outputs_fn = None

  eval_policy_ids = None
  eval_policy_algorithms = None
  if eval_members is not None:
    eval_policy_ids = [int(member['policy_id']) for member in eval_members]
    eval_policy_algorithms = [str(member['algorithm']) for member in eval_members]

  return SearchBundle(
      environment=environment,
      algorithm=MIXED_ALGORITHM,
      kind='locomotion',
      env=env,
      ga_config=ga_config,
      qpos0=qpos0,
      nq=nq,
      nv=nv,
      operators=operators,
      execute_fn=execute_fn,
      rollout_fn=rollout_fn,
      extract_attributions_fn=extract_fn,
      policy_ids=policy_ids,
      attribution_config=attribution_config,
      eval_policy_ids=eval_policy_ids,
      policy_algorithms=policy_algorithms,
      eval_policy_algorithms=eval_policy_algorithms,
      first_step_actions_fn=first_step_actions_fn,
      parallel_execute_fn=parallel_execute_fn,
      policy_outputs_fn=policy_outputs_fn,
  )


