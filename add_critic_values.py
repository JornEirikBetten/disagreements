"""Write both SAC critic values onto finished search collections.

Usage:
  python add_critic_values.py
  python add_critic_values.py --environment hopper --algorithm sac

For every kept initial state, each SAC policy in the search ensemble is
evaluated at the action already stored for that state. The twin Q-network
returns both critics, and those two numbers are saved on the generation
record as ``q_values`` with shape ``(P, N, 2)``. The policy axis follows the
search ensemble. PPO rows are NaN, because a PPO policy has no twin critic.
The last axis is the Q-network's critic order, the same order ``min`` and
``max`` use.

The existing ``values`` column is left as it was. It is the larger of the
two critics, and the script checks the new array against that column before
replacing a shard.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
from dataclasses import replace

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import numpy as np
from jax import numpy as jp

from ga.collect import (
    generation_shard_dir,
    load_collection,
    save_generation_shard,
)
from ga.env_setup import (
    LOCOMOTION_ENVS,
    MIXED_ALGORITHM,
    _load_locomotion_env,
    policy_filename,
)
from policies.loader import load_policy_model, stack_policy_params

ALGORITHMS = ('sac', MIXED_ALGORITHM)
ROLLOUT_LENGTHS = {
    'ant': 15,
    'humanoid': 15,
    'walker2d': 15,
    'hopper': 15,
}
_COLLECTION_NAME = re.compile(r'^experiment_(\d+)_ga_collection\.pkl$')
# Fixed case batch so the critic forward is compiled once per ensemble.
_CASE_BATCH = 2048


def _collection_dir(environment: str, algorithm: str) -> str:
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'difficulty_bins',
      f'rollout_length={ROLLOUT_LENGTHS[environment]}',
  )


def _discover_experiments(environment: str, algorithm: str) -> list[tuple[int, str]]:
  directory = _collection_dir(environment, algorithm)
  if not os.path.isdir(directory):
    return []
  found = []
  for name in os.listdir(directory):
    match = _COLLECTION_NAME.match(name)
    if not match:
      continue
    path = os.path.join(directory, name)
    if os.path.isdir(generation_shard_dir(path)):
      found.append((int(match.group(1)), path))
  found.sort()
  return found


def _search_members(path: str, algorithm: str, policy_ids: list[int]) -> list[dict]:
  """Search-ensemble members in the order of the failure matrix."""
  init_path = os.path.splitext(path)[0] + '_init.json'
  if not os.path.isfile(init_path):
    raise FileNotFoundError(f'No init file for {path}.')
  with open(init_path) as handle:
    selection = json.load(handle).get('policy_selection', {})
  if algorithm == MIXED_ALGORITHM:
    members = selection.get('search')
    if not members:
      raise ValueError(f'{init_path} has no search ensemble.')
    parsed = []
    for member in members:
      name = str(member['algorithm'])
      if name not in ('ppo', 'sac'):
        raise ValueError(
            f'{init_path} search member has algorithm {name!r}.'
        )
      parsed.append({
          'algorithm': name,
          'policy_id': int(member['policy_id']),
      })
  else:
    ids = [int(policy_id) for policy_id in policy_ids]
    if not ids:
      ids = [int(policy_id) for policy_id in selection.get('search_policy_ids', [])]
    parsed = [{'algorithm': algorithm, 'policy_id': policy_id} for policy_id in ids]
  if policy_ids and [member['policy_id'] for member in parsed] != list(policy_ids):
    raise ValueError(
        f'{path} policy ids {list(policy_ids)} do not match the search '
        f'ensemble {[member["policy_id"] for member in parsed]}.'
    )
  return parsed


def _has_critics(record, n_policies: int) -> bool:
  q_values = getattr(record, 'q_values', None)
  if q_values is None:
    return False
  q_values = np.asarray(q_values)
  return q_values.shape == (int(n_policies), int(record.n_cases), 2)


def _make_observation_fn(env):
  """Jitted reset. Callers pad every batch to the same length, so this
  function is traced once per environment.
  """

  @jax.jit
  def observations(qpos: jax.Array, qvel: jax.Array) -> jax.Array:
    def one(q, v):
      return env.reset_from_initial_state(q, v).obs

    return jax.vmap(one)(qpos, qvel)

  return observations


def _make_critic_fn(models):
  """Both critics for every SAC policy. Output shape ``(P_sac, batch, 2)``."""
  network = models[0].network
  stacked = stack_policy_params(models)

  @jax.jit
  def critic_values(obs: jax.Array, action: jax.Array) -> jax.Array:
    def one_policy(params, policy_action):
      return network.q_network.apply(
          params[0], params[2], obs, policy_action
      )

    return jax.vmap(one_policy)(stacked, action)

  return critic_values


def _pad_axis(array: np.ndarray, length: int, axis: int) -> np.ndarray:
  """Repeat the first row along ``axis`` until the length is ``length``."""
  n = int(array.shape[axis])
  if n == length:
    return array
  if n < 1 or n > length:
    raise ValueError(f'Cannot pad axis of length {n} to {length}.')
  index = [slice(None)] * array.ndim
  index[axis] = slice(0, 1)
  filler = np.repeat(array[tuple(index)], length - n, axis=axis)
  return np.concatenate([array, filler], axis=axis)


def _critics_for_cases(observe, critic_fn, qpos, qvel, action, batch: int) -> np.ndarray:
  """``action`` is ``(P_sac, N, action_dim)``. Returns ``(P_sac, N, 2)``.

  Every forward pass has ``batch`` cases. Padding is discarded.
  """
  n = int(qpos.shape[0])
  n_policies = int(action.shape[0])
  if n == 0:
    return np.zeros((n_policies, 0, 2), dtype=np.float32)
  parts = []
  for start in range(0, n, batch):
    stop = min(start + batch, n)
    width = stop - start
    qpos_b = _pad_axis(np.asarray(qpos[start:stop]), batch, 0)
    qvel_b = _pad_axis(np.asarray(qvel[start:stop]), batch, 0)
    action_b = _pad_axis(np.asarray(action[:, start:stop]), batch, 1)
    obs = observe(jp.asarray(qpos_b), jp.asarray(qvel_b))
    values = critic_fn(obs, jp.asarray(action_b))
    parts.append(np.asarray(values, dtype=np.float32)[:, :width])
    del obs, values
  return np.concatenate(parts, axis=1)


# Recomputing the same critic in float32 moves the stored max by a few
# hundredths, which is past rtol=1e-3 on a small Q. A wrong action or a
# wrong network moves these values by orders of magnitude.
_STORED_MAX_RTOL = 1e-2
_STORED_MAX_ATOL = 1.0


def _check_against_stored_max(q_values: np.ndarray, stored: np.ndarray, sac_index: list[int]) -> None:
  """The stored value column is the larger critic. A mismatch means the
  forward pass did not see the same action or the same network.
  """
  predicted = np.max(q_values[sac_index], axis=-1)
  recorded = np.asarray(stored, dtype=np.float32)[sac_index]
  gap = np.abs(predicted - recorded)
  allowed = _STORED_MAX_ATOL + _STORED_MAX_RTOL * np.abs(recorded)
  if not np.any(gap > allowed):
    return
  policy, case = np.unravel_index(int(np.argmax(gap - allowed)), gap.shape)
  raise ValueError(
      'SAC critic max does not match the stored value column '
      f'(policy {int(policy)}, case {int(case)}: '
      f'predicted {predicted[policy, case]:.6g}, '
      f'stored {recorded[policy, case]:.6g}, '
      f'gap {gap[policy, case]:.4g}).'
  )


def annotate_experiment(
    environment: str,
    algorithm: str,
    path: str,
    *,
    env,
    observe,
    batch: int,
    overwrite: bool,
) -> None:
  if algorithm == 'ppo':
    print(f'{environment}/ppo has no twin critics; skipping {path}', flush=True)
    return
  init_path = os.path.splitext(path)[0] + '_init.json'
  if not os.path.isfile(init_path):
    print(f'No init file for {path}; skipping', flush=True)
    return
  collection = load_collection(path, load_shards=False)
  members = _search_members(path, algorithm, list(collection.policy_ids))
  sac_index = [
      index for index, member in enumerate(members) if member['algorithm'] == 'sac'
  ]
  if not sac_index:
    print(f'No SAC policies in {path}; skipping', flush=True)
    return

  models = []
  for index in sac_index:
    member = members[index]
    model_path = os.path.join(
        'trained_policies',
        environment,
        'sac',
        policy_filename(environment, 'sac').format(id=member['policy_id']),
    )
    models.append(
        load_policy_model(
            model_path,
            observation_size=env.observation_size,
            action_size=env.action_size,
            algorithm='sac',
        )
    )
  critic_fn = _make_critic_fn(models)
  n_policies = len(members)
  shard_dir = generation_shard_dir(path)
  manifest = json.load(open(os.path.join(shard_dir, 'manifest.json')))
  print(
      f'{environment}/{algorithm} {os.path.basename(path)}: '
      f'{len(sac_index)} SAC critics, {len(manifest.get("shards", []))} generations',
      flush=True,
  )
  failed_generations = []
  for entry in manifest.get('shards', []):
    generation = int(entry['generation'])
    shard_path = os.path.join(shard_dir, entry['file'])
    record = None
    try:
      with open(shard_path, 'rb') as handle:
        record = pickle.load(handle)
      generation = int(record.generation)
      if not overwrite and _has_critics(record, n_policies):
        print(f'  generation {generation}: already stored', flush=True)
        continue
      sac_q = _critics_for_cases(
          observe,
          critic_fn,
          np.asarray(record.suite.qpos),
          np.asarray(record.suite.qvel),
          np.asarray(record.actions)[sac_index],
          batch,
      )
      q_values = np.full((n_policies, record.n_cases, 2), np.nan, dtype=np.float32)
      q_values[np.asarray(sac_index)] = sac_q
      _check_against_stored_max(q_values, record.values, sac_index)
      if not hasattr(record, 'q_values'):
        record.q_values = None
      record = replace(record, q_values=q_values)
      save_generation_shard(record, path)
      print(
          f'  generation {generation}: wrote q_values {q_values.shape}',
          flush=True,
      )
      del sac_q, q_values
    except Exception as exc:
      failed_generations.append(generation)
      print(f'  generation {generation}: {exc}', flush=True)
    finally:
      del record
  del models, critic_fn, collection
  gc.collect()
  if failed_generations:
    shown = ', '.join(str(generation) for generation in failed_generations[:8])
    extra = '' if len(failed_generations) <= 8 else ', ...'
    raise RuntimeError(
        f'{len(failed_generations)} generations were not annotated '
        f'({shown}{extra}).'
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--environment',
      choices=LOCOMOTION_ENVS,
      action='append',
      dest='environments',
  )
  parser.add_argument(
      '--algorithm',
      choices=('sac', MIXED_ALGORITHM, 'ppo'),
      action='append',
      dest='algorithms',
  )
  parser.add_argument('--experiment', type=int, action='append', dest='experiments')
  parser.add_argument('--batch', type=int, default=_CASE_BATCH)
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help='Recompute shards that already store both critics.',
  )
  args = parser.parse_args()
  if args.batch < 1:
    raise ValueError(f'batch must be >= 1, got {args.batch}')
  args.environments = args.environments or list(LOCOMOTION_ENVS)
  args.algorithms = args.algorithms or list(ALGORITHMS)
  return args


def main() -> int:
  args = parse_args()
  failed = False
  for environment in args.environments:
    env = None
    observe = None
    for algorithm in args.algorithms:
      experiments = _discover_experiments(environment, algorithm)
      if args.experiments is not None:
        wanted = set(args.experiments)
        experiments = [item for item in experiments if item[0] in wanted]
      if not experiments:
        print(f'No shards for {environment}/{algorithm}', flush=True)
        continue
      if algorithm == 'ppo':
        for _exp_number, path in experiments:
          print(f'{environment}/ppo has no twin critics; skipping {path}', flush=True)
        continue
      if observe is None:
        env, _, _ = _load_locomotion_env(environment)
        observe = _make_observation_fn(env)
      for _exp_number, path in experiments:
        try:
          annotate_experiment(
              environment,
              algorithm,
              path,
              env=env,
              observe=observe,
              batch=args.batch,
              overwrite=args.overwrite,
          )
        except Exception as exc:
          failed = True
          print(f'Failed {path}: {exc}', flush=True)
          continue
    del env, observe
    gc.collect()
    jax.clear_caches()
  return 1 if failed else 0


if __name__ == '__main__':
  raise SystemExit(main())
