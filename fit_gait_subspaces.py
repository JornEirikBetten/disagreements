"""Fit and save a PCA gait subspace for every locomotion policy.

Usage:
  python fit_gait_subspaces.py
  python fit_gait_subspaces.py humanoid ppo
  python fit_gait_subspaces.py hopper ppo 1

Rolls out each policy from the default (unperturbed) reset, keeps successful
traces after a 1 s burn-in, fits PCA, and writes:

  datasets/gait_subspaces/{env}/{alg}/policy_{id}.npz

Load later with ``analysis.gait_subspace.load_gait_subspace_for_policy``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import numpy as np

from analysis.gait_subspace import (
    DEFAULT_ROOT,
    GaitSubspace,
    dominant_frequency_hz,
    fit_gait_pca,
    gait_subspace_path,
    load_gait_subspace,
    save_gait_subspace,
)
from brax.io import model as brax_model
from policies.loader import (
    _make_ppo_network,
    _make_sac_network,
    _to_ppo_inference_params,
)
from policies.ranking import local_policy_ids

LOCOMOTION_ENVS = ('humanoid', 'ant', 'walker2d', 'hopper')
_FALL_ENVS = ('humanoid', 'ant', 'hopper', 'walker2d')
ALGORITHMS = ('ppo', 'sac')
NOMINAL_DURATION_S = 5.0
BURNIN_S = 1.0
N_ROLLOUTS = 16
INDEX_NAME = 'index.csv'


def _make_env(environment: str):
  if environment == 'humanoid':
    from envs.humanoid import make_humanoid_env
    return make_humanoid_env()
  if environment == 'hopper':
    from envs.hopper import make_hopper_env
    return make_hopper_env()
  if environment == 'walker2d':
    from envs.walker2d import make_walker2d_env
    return make_walker2d_env()
  if environment == 'ant':
    from envs.ant import make_ant_env
    return make_ant_env()
  raise ValueError(f'Unknown environment {environment!r}')


def load_actor_inference_params(model_path: str, algorithm: str):
  """Load only the actor. SAC gait rollouts do not need Q-network weights."""
  raw = brax_model.load_params(model_path)
  if algorithm == 'ppo':
    return _to_ppo_inference_params(raw)
  if len(raw) < 2:
    raise ValueError(
        f'Unexpected SAC params structure: len={len(raw)} in {model_path}'
    )
  normalizer, policy_params = raw[0], raw[1]
  if isinstance(policy_params, dict) and 'policy' in policy_params:
    policy_params = policy_params['policy']
  return normalizer, policy_params


def _make_inference_fn(algorithm: str, observation_size: int, action_size: int):
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.sac import networks as sac_networks

  if algorithm == 'sac':
    network = _make_sac_network(observation_size, action_size)
    return sac_networks.make_inference_fn(network)
  network = _make_ppo_network(observation_size, action_size)
  return ppo_networks.make_inference_fn(network)


def make_nominal_collect_fn(env, make_inference_fn, rollout_length: int):
  """Jit once per (env, algorithm): params are traced inputs, not constants."""

  def collect(inference_params, rngs):
    policy_fn = make_inference_fn(inference_params, deterministic=True)

    def one_rollout(rng):
      reset_rng, act_rng = jax.random.split(rng)
      state = env.reset(reset_rng)

      def step_fn(carry, _):
        state, key = carry
        key, act_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, act_key)
        next_state = env.step(state, action)
        return (next_state, key), (action, next_state.done)

      _, (actions, dones) = jax.lax.scan(
          step_fn, (state, act_rng), None, rollout_length
      )
      return actions, dones

    return jax.vmap(one_rollout)(rngs)

  return jax.jit(collect)


def _successful_mask(environment: str, dones: np.ndarray) -> np.ndarray:
  if environment in _FALL_ENVS:
    return ~np.any(dones > 0, axis=-1)
  raise ValueError(f'Unknown environment {environment!r}')


def _first_fail_index(dones_row: np.ndarray) -> int:
  hits = np.flatnonzero(dones_row > 0)
  if hits.size == 0:
    return dones_row.shape[0]
  return int(hits[0])


def stack_nominal_actions(
    environment: str,
    actions: np.ndarray,
    dones: np.ndarray,
    burnin_steps: int,
) -> tuple[np.ndarray, int, bool]:
  """Keep post-burn-in actions from successful rollouts.

  If none succeed, fall back to healthy prefixes long enough to contain gait.
  """
  success = _successful_mask(environment, dones)
  chunks = []
  min_keep = burnin_steps + 20
  for i, ok in enumerate(success):
    if ok:
      chunks.append(actions[i, burnin_steps:])
  if chunks:
    stacked = np.concatenate(chunks, axis=0)
    return stacked, int(np.sum(success)), False

  for i in range(actions.shape[0]):
    end = _first_fail_index(dones[i])
    if end >= min_keep:
      chunks.append(actions[i, burnin_steps:end])
  if not chunks:
    empty = np.zeros((0, actions.shape[-1]), dtype=np.float64)
    return empty, 0, True
  stacked = np.concatenate(chunks, axis=0)
  return stacked, 0, True


def _stride_hz_from_gait(
    actions_traj: np.ndarray, subspace: GaitSubspace, dt: float
) -> float | None:
  """Peak frequency of the first gait coordinate (not ||a_gait||)."""
  centered = np.asarray(actions_traj, dtype=np.float64) - subspace.mean_action
  scores = centered @ subspace.W_gait[:, 0]
  return dominant_frequency_hz(scores, dt)


def _index_path(root: str) -> str:
  return os.path.join(root, INDEX_NAME)


def _write_index(root: str, rows: list[dict]) -> None:
  if not rows:
    return
  path = _index_path(root)
  fieldnames = list(rows[0].keys())
  existing = []
  if os.path.isfile(path):
    with open(path, newline='') as handle:
      existing = list(csv.DictReader(handle))
  keyed = {
      (row['environment'], row['algorithm'], str(row['policy_id'])): row
      for row in existing
  }
  for row in rows:
    keyed[(row['environment'], row['algorithm'], str(row['policy_id']))] = row
  merged = [keyed[key] for key in sorted(keyed)]
  os.makedirs(root, exist_ok=True)
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)


def _row_from_subspace(
    subspace: GaitSubspace, path: str, status: str
) -> dict:
  ratios = subspace.full_explained_variance_ratio
  return {
      'environment': subspace.environment,
      'algorithm': subspace.algorithm,
      'policy_id': str(subspace.policy_id),
      'k': str(subspace.k),
      'cumulative_explained': f'{subspace.cumulative_explained:.6f}',
      'var_pc1': f'{float(ratios[0]):.6f}' if ratios.size else '',
      'var_pc2': f'{float(ratios[1]):.6f}' if ratios.size > 1 else '',
      'var_pc3': f'{float(ratios[2]):.6f}' if ratios.size > 2 else '',
      'n_successful_rollouts': str(subspace.n_successful_rollouts),
      'n_samples': str(subspace.n_samples),
      'used_partial_rollouts': str(subspace.used_partial_rollouts).lower(),
      'stride_hz': (
          f'{subspace.stride_hz:.4f}' if subspace.stride_hz is not None else ''
      ),
      'status': status,
      'path': path,
  }


def _failed_row(environment, algorithm, policy_id, status) -> dict:
  return {
      'environment': environment,
      'algorithm': algorithm,
      'policy_id': str(policy_id),
      'k': '',
      'cumulative_explained': '',
      'var_pc1': '',
      'var_pc2': '',
      'var_pc3': '',
      'n_successful_rollouts': '0',
      'n_samples': '0',
      'used_partial_rollouts': 'false',
      'stride_hz': '',
      'status': status,
      'path': '',
  }


def fit_one_policy(
    environment: str,
    algorithm: str,
    policy_id: int,
    env,
    collect_fn,
    rollout_length: int,
    burnin_steps: int,
    n_rollouts: int,
    root: str,
) -> dict:
  policy_dir = os.path.join('trained_policies', environment, algorithm)
  model_path = os.path.join(
      policy_dir, 'best_control_{id}/params.json'.format(id=policy_id)
  )
  inference_params = load_actor_inference_params(model_path, algorithm)
  rngs = jax.random.split(jax.random.PRNGKey(int(policy_id)), n_rollouts)
  actions, dones = collect_fn(inference_params, rngs)
  actions_np = np.asarray(actions)
  dones_np = np.asarray(dones)
  stacked, n_success, used_partial = stack_nominal_actions(
      environment, actions_np, dones_np, burnin_steps
  )
  out_path = gait_subspace_path(environment, algorithm, policy_id, root=root)
  if stacked.shape[0] < 2:
    return _failed_row(
        environment, algorithm, policy_id, 'no_nominal_actions'
    )

  mean, w_gait, ratios, k = fit_gait_pca(stacked)
  subspace = GaitSubspace(
      mean_action=mean,
      W_gait=w_gait,
      explained_variance_ratio=ratios[:k],
      full_explained_variance_ratio=ratios,
      n_samples=int(stacked.shape[0]),
      n_successful_rollouts=n_success,
      n_rollouts=n_rollouts,
      environment=environment,
      algorithm=algorithm,
      policy_id=int(policy_id),
      dt=float(env.dt),
      burnin_steps=burnin_steps,
      rollout_length=rollout_length,
      used_partial_rollouts=used_partial,
  )
  success_idx = np.flatnonzero(
      _successful_mask(environment, dones_np)
  )
  traj_i = int(success_idx[0]) if success_idx.size else 0
  end = (
      rollout_length
      if success_idx.size
      else _first_fail_index(dones_np[traj_i])
  )
  traj = actions_np[traj_i, burnin_steps:end]
  if traj.shape[0] >= 8:
    subspace.stride_hz = _stride_hz_from_gait(traj, subspace, float(env.dt))
  save_gait_subspace(out_path, subspace)
  status = 'ok_partial' if used_partial else 'ok'
  return _row_from_subspace(subspace, out_path, status)


def fit_combo(
    environment: str,
    algorithm: str,
    policy_ids: list[int],
    root: str,
    overwrite: bool,
    n_rollouts: int,
) -> list[dict]:
  rows = []
  pending = []
  for policy_id in policy_ids:
    path = gait_subspace_path(environment, algorithm, policy_id, root=root)
    if os.path.isfile(path) and not overwrite:
      subspace = load_gait_subspace(path)
      rows.append(_row_from_subspace(subspace, path, 'skipped_existing'))
      continue
    pending.append(policy_id)
  if not pending:
    _write_index(root, rows)
    print(
        f'[{environment} {algorithm}] {len(rows)} subspaces already present',
        flush=True,
    )
    return rows

  env = _make_env(environment)
  dt = float(env.dt)
  burnin_steps = max(1, int(round(BURNIN_S / dt)))
  rollout_length = max(burnin_steps + 20, int(round(NOMINAL_DURATION_S / dt)))
  make_inf = _make_inference_fn(
      algorithm, env.observation_size, env.action_size
  )
  print(
      f'[{environment} {algorithm}] dt={dt:.4f}s  '
      f'T={rollout_length}  burn-in={burnin_steps}  '
      f'policies={len(pending)}/{len(policy_ids)} to fit  compiling collector…',
      flush=True,
  )
  collect_fn = make_nominal_collect_fn(env, make_inf, rollout_length)

  t0 = time.time()
  for i, policy_id in enumerate(pending):
    started = time.time()
    try:
      row = fit_one_policy(
          environment,
          algorithm,
          policy_id,
          env,
          collect_fn,
          rollout_length,
          burnin_steps,
          n_rollouts,
          root,
      )
    except Exception as exc:  # noqa: BLE001 — keep the rest of the sweep going
      row = _failed_row(
          environment, algorithm, policy_id, f'error:{type(exc).__name__}'
      )
      print(f'  {environment} {algorithm} policy {policy_id}: {exc}', flush=True)
    rows.append(row)
    elapsed = time.time() - started
    total = time.time() - t0
    print(
        f'  {environment} {algorithm} policy {policy_id}: '
        f'{row["status"]} k={row["k"]} '
        f'explained={row["cumulative_explained"] or "n/a"} '
        f'success={row["n_successful_rollouts"]} '
        f'({elapsed:.1f}s, {i + 1}/{len(pending)}, {total:.0f}s total)',
        flush=True,
    )
  _write_index(root, rows)
  return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Fit PCA gait subspaces for locomotion policies.',
  )
  parser.add_argument(
      'environment',
      nargs='?',
      choices=LOCOMOTION_ENVS,
      help='If omitted, fit all locomotion environments.',
  )
  parser.add_argument(
      'algorithm',
      nargs='?',
      choices=ALGORITHMS,
      help='If omitted, fit PPO and SAC.',
  )
  parser.add_argument(
      'policy_id',
      nargs='?',
      type=int,
      help='If omitted, fit every local checkpoint.',
  )
  parser.add_argument(
      '--root',
      default=DEFAULT_ROOT,
      help='Output directory for npz files and index.csv.',
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help='Refit even when a subspace file already exists.',
  )
  parser.add_argument(
      '--n-rollouts',
      type=int,
      default=N_ROLLOUTS,
      help='Unperturbed rollouts per policy (default 16).',
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  envs = (args.environment,) if args.environment else LOCOMOTION_ENVS
  algs = (args.algorithm,) if args.algorithm else ALGORITHMS
  if args.policy_id is not None and (args.environment is None or args.algorithm is None):
    print(
        'policy_id requires environment and algorithm, e.g. '
        '`python fit_gait_subspaces.py humanoid ppo 1`',
        file=sys.stderr,
    )
    return 2

  all_rows = []
  for environment in envs:
    for algorithm in algs:
      ids = local_policy_ids(environment, algorithm)
      if args.policy_id is not None:
        if args.policy_id not in ids:
          print(
              f'No checkpoint trained_policies/{environment}/{algorithm}/'
              f'best_control_{args.policy_id}/params.json',
              file=sys.stderr,
          )
          return 1
        ids = [args.policy_id]
      if not ids:
        print(f'No local policies for {environment} {algorithm}', flush=True)
        continue
      all_rows.extend(
          fit_combo(
              environment,
              algorithm,
              ids,
              root=args.root,
              overwrite=args.overwrite,
              n_rollouts=args.n_rollouts,
          )
      )

  ok = sum(1 for row in all_rows if row['status'].startswith('ok'))
  skipped = sum(1 for row in all_rows if row['status'] == 'skipped_existing')
  failed = len(all_rows) - ok - skipped
  print(
      f'Done: {ok} fitted, {skipped} already present, {failed} failed '
      f'({len(all_rows)} total). Index: {_index_path(args.root)}',
      flush=True,
  )
  return 1 if failed else 0


if __name__ == '__main__':
  raise SystemExit(main())
