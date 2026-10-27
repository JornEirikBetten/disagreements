"""Load trained PPO and SAC policies for evaluation."""

import os
from dataclasses import dataclass
from typing import Any, Sequence

import jax
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from test_case_execution.rollout import PolicyFn

from policies.ranking import sac_checkpoint_has_action_value

SUPPORTED_ALGORITHMS = ('ppo', 'sac')


@dataclass(frozen=True)
class PolicyModel:
  """Loaded policy with parameters and network definitions."""

  params: Any
  network: Any
  policy_fn: PolicyFn
  model_path: str
  algorithm: str = 'ppo'

  @property
  def ppo_network(self):
    """Backward-compatible alias for :attr:`network`."""
    return self.network


def infer_algorithm(path: str) -> str:
  """Infer ``'ppo'`` or ``'sac'`` from a checkpoint or directory path."""
  for part in reversed(os.path.normpath(path).split(os.sep)):
    lowered = part.lower()
    if lowered in SUPPORTED_ALGORITHMS:
      return lowered
  return 'ppo'


def _to_ppo_inference_params(params):
  """Convert saved PPO checkpoints to ``(normalizer, policy, value)``."""
  if len(params) == 3:
    return params
  normalizer_params, network_params = params
  if isinstance(network_params, ppo_losses.PPONetworkParams):
    return normalizer_params, network_params.policy, network_params.value
  raise ValueError(
      f'Unexpected PPO params structure: len={len(params)}, '
      f'network_params type={type(network_params)}'
  )


def _to_sac_inference_params(params, model_path: str):
  """Convert saved SAC checkpoints to ``(normalizer, policy, q)``.

  Brax's default ``sac.train`` return value is ``(normalizer, policy)``.
  Collection of max-Q values requires the critic, so a third element is
  accepted when present, as is a sibling ``q_params.json``.
  """
  if len(params) >= 3:
    return params[0], params[1], params[2]
  if len(params) != 2:
    raise ValueError(
        f'Unexpected SAC params structure: len={len(params)} in {model_path}'
    )

  normalizer_params, policy_params = params
  if isinstance(policy_params, dict) and 'q' in policy_params:
    q_params = policy_params['q']
    actor_params = policy_params.get('policy', policy_params)
    return normalizer_params, actor_params, q_params

  q_path = os.path.join(os.path.dirname(model_path), 'q_params.json')
  if os.path.exists(q_path):
    return normalizer_params, policy_params, model.load_params(q_path)

  raise ValueError(
      f'SAC checkpoint {model_path} has no Q-network parameters. '
      'Brax saves only (normalizer, policy). Re-save as '
      '(normalizer, policy, q_params) or write q_params.json next to '
      'params.json so values can be max_i Q_i(s, π(s)).'
  )


def _make_ppo_network(observation_size: int, action_size: int):
  return ppo_networks.make_ppo_networks(
      observation_size=observation_size,
      action_size=action_size,
      preprocess_observations_fn=running_statistics.normalize,
  )


def _make_sac_network(observation_size: int, action_size: int):
  return sac_networks.make_sac_networks(
      observation_size=observation_size,
      action_size=action_size,
      preprocess_observations_fn=running_statistics.normalize,
  )


def load_policy_model(
    model_path: str,
    observation_size: int,
    action_size: int,
    algorithm: str | None = None,
) -> PolicyModel:
  """Load a policy with params and networks for inference and attribution."""
  if algorithm is None:
    algorithm = infer_algorithm(model_path)
  if algorithm not in SUPPORTED_ALGORITHMS:
    raise ValueError(
        f'Unknown algorithm {algorithm!r}; expected one of {SUPPORTED_ALGORITHMS}'
    )

  raw_params = model.load_params(model_path)
  if algorithm == 'sac':
    params = _to_sac_inference_params(raw_params, model_path)
    network = _make_sac_network(observation_size, action_size)
    inference_params = (params[0], params[1])
    make_inference_fn = sac_networks.make_inference_fn(network)
  else:
    params = _to_ppo_inference_params(raw_params)
    network = _make_ppo_network(observation_size, action_size)
    inference_params = params
    make_inference_fn = ppo_networks.make_inference_fn(network)

  policy_fn = jax.jit(make_inference_fn(inference_params, deterministic=True))
  return PolicyModel(
      params=params,
      network=network,
      policy_fn=policy_fn,
      model_path=model_path,
      algorithm=algorithm,
  )


def load_policy_models_from_dir(
    policy_dir: str,
    observation_size: int,
    action_size: int,
    policy_ids: Sequence[int] | None = None,
    filename_template: str = 'best_control_{id}/params.json',
    algorithm: str | None = None,
) -> list[PolicyModel]:
  """Load multiple policy models from a directory of checkpoints."""
  if algorithm is None:
    algorithm = infer_algorithm(policy_dir)
  if policy_ids is None:
    policy_ids = _discover_policy_ids(
        policy_dir, filename_template, algorithm=algorithm
    )

  models = []
  for policy_id in policy_ids:
    rel_path = filename_template.format(id=policy_id)
    model_path = os.path.join(policy_dir, rel_path)
    models.append(
        load_policy_model(
            model_path,
            observation_size,
            action_size,
            algorithm=algorithm,
        )
    )
  return models


def stack_policy_params(policy_models: Sequence[PolicyModel]):
  """Stack policy params into a pytree with leading policy axis (P, ...)."""
  if not policy_models:
    raise ValueError('No policy models to stack.')
  algorithms = {model.algorithm for model in policy_models}
  if len(algorithms) > 1:
    raise ValueError(
        f'Cannot stack mixed algorithm checkpoints: {sorted(algorithms)}'
    )
  return jax.tree.map(
      lambda *xs: jp.stack(xs),
      *[model.params for model in policy_models],
  )


def load_policy(
    model_path: str,
    observation_size: int,
    action_size: int,
    algorithm: str | None = None,
) -> PolicyFn:
  """Load a single jitted inference function from a checkpoint."""
  return load_policy_model(
      model_path, observation_size, action_size, algorithm=algorithm
  ).policy_fn


def load_policies_from_dir(
    policy_dir: str,
    observation_size: int,
    action_size: int,
    policy_ids: Sequence[int] | None = None,
    filename_template: str = 'best_control_{id}/params.json',
    algorithm: str | None = None,
) -> list[PolicyFn]:
  """Load multiple policies from a directory of checkpoints."""
  return [
      model.policy_fn
      for model in load_policy_models_from_dir(
          policy_dir,
          observation_size,
          action_size,
          policy_ids=policy_ids,
          filename_template=filename_template,
          algorithm=algorithm,
      )
  ]


def _discover_policy_ids(
    policy_dir: str,
    filename_template: str,
    algorithm: str | None = None,
) -> list[int]:
  del filename_template  # discovery uses dirname layout from visualize script.
  if algorithm is None:
    algorithm = infer_algorithm(policy_dir)
  ids = []
  for entry in os.listdir(policy_dir):
    if not entry.startswith('best_control_'):
      continue
    suffix = entry.removeprefix('best_control_')
    if not suffix.isdigit():
      continue
    params_path = os.path.join(policy_dir, entry, 'params.json')
    if not os.path.isfile(params_path):
      continue
    if algorithm == 'sac' and not sac_checkpoint_has_action_value(params_path):
      continue
    ids.append(int(suffix))
  return sorted(ids)
