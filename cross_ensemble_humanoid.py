"""Failure rates of other ensembles on the Humanoid tests from the PPO search.

The cases are the initial states collected by the finished Humanoid PPO
difficulty-bin searches (five experiments, rollout length 15). Exact repeats
inside one experiment are collapsed. PPO evaluation rates are the held-out
rates already stored on those shards. This script rolls out the other four
ensembles on the same states and the same policy RNGs:

  SAC search, SAC eval, PPO+SAC search, PPO+SAC eval

Ensemble membership is the split recorded in each algorithm's ``*_init.json``.
Those ids are identical across the five experiments. Results are cached under
``figures/ensemble_transfer/`` so a rerun resumes a finished ensemble and a
partially scored one.

Usage:
  python cross_ensemble_humanoid.py
  python cross_ensemble_humanoid.py --experiment 1
  python cross_ensemble_humanoid.py --limit 600
  python cross_ensemble_humanoid.py --plot-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from add_heldout_fail_rates import (
    _CASE_BATCH,
    _discover_experiments,
    _pad_suite,
    _suite_from_numpy,
)
from ga.collect import iter_generation_shards
from ga.env_setup import MIXED_ALGORITHM, make_mixed_search_bundle, make_search_bundle

SOURCE_ALGORITHM = 'ppo'
ENVIRONMENTS = ('ant', 'humanoid', 'walker2d', 'hopper')
ROLLOUT_LENGTHS = {
    'ant': 15,
    'humanoid': 15,
    'walker2d': 15,
    'hopper': 15,
}
N_POLICIES = 16
OUTPUT_DIR = os.path.join('figures', 'ensemble_transfer')
_CASES_VERSION = 'ppo-cases-1'
# The first Humanoid cache was written before the version string was shared.
_CASES_VERSIONS = frozenset((_CASES_VERSION, 'humanoid-ppo-cases-1'))

# (cache key, legend, source algorithm, init-json role)
# PPO eval is read from the source shards and is not rolled out again.
ENSEMBLES = (
    ('ppo_eval', 'PPO eval', None, None),
    ('sac_search', 'SAC search', 'sac', 'search'),
    ('sac_eval', 'SAC eval', 'sac', 'eval'),
    ('ppo_sac_search', 'PPO+SAC search', MIXED_ALGORITHM, 'search'),
    ('ppo_sac_eval', 'PPO+SAC eval', MIXED_ALGORITHM, 'eval'),
)

_COLORS = {
    'ppo_eval': '#1f4e79',
    'sac_search': '#e07a3d',
    'sac_eval': '#c0392b',
    'ppo_sac_search': '#1e8449',
    'ppo_sac_eval': '#6c3483',
}
_LINESTYLES = {
    'ppo_eval': '-',
    'sac_search': '--',
    'sac_eval': ':',
    'ppo_sac_search': '-.',
    'ppo_sac_eval': (0, (1, 1)),
}
_MARKERS = {
    'ppo_eval': 'o',
    'sac_search': 's',
    'sac_eval': '^',
    'ppo_sac_search': 'D',
    'ppo_sac_eval': 'v',
}


def _init_path(collection_path: str) -> str:
  if not collection_path.endswith('.pkl'):
    raise ValueError(f'Expected a collection pickle, got {collection_path}')
  return collection_path[:-4] + '_init.json'


def _member_pairs(selection: dict, algorithm: str, role: str) -> tuple[tuple[str, int], ...]:
  if algorithm == MIXED_ALGORITHM:
    members = selection[role]
    return tuple(
        (str(member['algorithm']), int(member['policy_id'])) for member in members
    )
  key = 'search_policy_ids' if role == 'search' else 'eval_policy_ids'
  return tuple((algorithm, int(policy_id)) for policy_id in selection[key])


def load_ensemble_members(
    environment: str, algorithm: str, role: str
) -> tuple[tuple[str, int], ...]:
  """Policy ids from the finished searches. They do not depend on the experiment."""
  experiments = _discover_experiments(environment, algorithm)
  if not experiments:
    raise SystemExit(
        f'No finished {environment}/{algorithm} difficulty-bin search.'
    )
  reference = None
  for _number, path in experiments:
    with open(_init_path(path)) as handle:
      meta = json.load(handle)
    pairs = _member_pairs(meta['policy_selection'], algorithm, role)
    if len(pairs) != N_POLICIES:
      raise ValueError(
          f'{path} {role} ensemble has {len(pairs)} policies, expected {N_POLICIES}'
      )
    if reference is None:
      reference = pairs
    elif pairs != reference:
      raise ValueError(
          f'{algorithm} {role} membership differs across experiments.'
      )
  return reference


def _atomic_npy(path: str, values: np.ndarray) -> None:
  temporary = path + '.tmp'
  with open(temporary, 'wb') as handle:
    np.save(handle, np.asarray(values))
  os.replace(temporary, path)


def _distinct_index(qpos: np.ndarray, qvel: np.ndarray, experiment: np.ndarray) -> np.ndarray:
  """First occurrence of each (q, qdot) inside an experiment."""
  rows = np.ascontiguousarray(np.concatenate([qpos, qvel], axis=1))
  width = int(rows.dtype.itemsize * rows.shape[1])
  keys = rows.view(np.dtype((np.void, width))).reshape(rows.shape[0])
  kept = []
  for value in np.unique(experiment):
    index = np.flatnonzero(experiment == value)
    _unique, first = np.unique(keys[index], return_index=True)
    kept.append(index[first])
  if not kept:
    return np.zeros(0, dtype=np.int64)
  return np.sort(np.concatenate(kept).astype(np.int64))


def _tag(experiment_ids: tuple[int, ...] | None, limit: int | None) -> str:
  experiments = 'all' if experiment_ids is None else 'e' + '-'.join(
      str(number) for number in experiment_ids
  )
  limited = '' if limit is None else f'_limit{int(limit)}'
  return f'{experiments}{limited}'


def _cases_path(environment: str, tag: str, source: str = 'ppo') -> str:
  return os.path.join(OUTPUT_DIR, f'{environment}_{source}_cases_{tag}.npz')


def _rates_path(
    environment: str, tag: str, ensemble: str, source: str = 'ppo'
) -> str:
  return os.path.join(
      OUTPUT_DIR, f'{environment}_{source}_{ensemble}_{tag}.npy'
  )


def _with_difficulty_aliases(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Expose the search quota and the stored eval rate under stable names."""
  if 'search_difficulty' not in cases and 'ppo_search' in cases:
    cases['search_difficulty'] = cases['ppo_search']
    cases['source_eval'] = cases['ppo_eval']
  return cases


def load_search_cases(
    environment: str,
    source: str,
    experiment_ids: tuple[int, ...] | None,
    limit: int | None,
) -> dict[str, np.ndarray]:
  """States, policy RNGs, and stored eval rates from one algorithm's searches."""
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  tag = _tag(experiment_ids, limit)
  path = _cases_path(environment, tag, source)
  if os.path.isfile(path):
    with np.load(path) as cached:
      if str(cached['version']) in _CASES_VERSIONS or str(cached['version']) == 'search-cases-1':
        print(f'cases: {path} ({int(cached["qpos"].shape[0])} states)', flush=True)
        return _with_difficulty_aliases({key: cached[key] for key in cached.files})

  experiments = _discover_experiments(environment, source)
  if experiment_ids is not None:
    wanted = set(experiment_ids)
    experiments = [item for item in experiments if item[0] in wanted]
    missing = wanted.difference(number for number, _path in experiments)
    if missing:
      raise SystemExit(
          f'Finished {environment} {source} experiments missing: {sorted(missing)}'
      )
  if not experiments:
    raise SystemExit(f'No finished {environment}/{source} difficulty-bin search.')

  qpos_parts = []
  qvel_parts = []
  rng_parts = []
  eval_parts = []
  search_parts = []
  experiment_parts = []
  n_read = 0
  for number, collection_path in experiments:
    print(f'loading {environment} {source} experiment {number}', flush=True)
    for record in iter_generation_shards(collection_path):
      rate = getattr(record, 'eval_fail_rate', None)
      if rate is None:
        raise SystemExit(
            f'Experiment {number} generation {record.generation} has no '
            f'eval_fail_rate. Run add_heldout_fail_rates.py for {environment}/{source}.'
        )
      qpos = np.asarray(record.suite.qpos)
      qvel = np.asarray(record.suite.qvel)
      policy_rng = np.asarray(record.suite.policy_rng)
      rate = np.asarray(rate, dtype=np.float64)
      difficulty = np.asarray(record.difficulty, dtype=np.float64)
      n_cases = int(qpos.shape[0])
      if rate.shape != (n_cases,) or difficulty.shape != (n_cases,):
        raise ValueError(
            f'Experiment {number} generation {record.generation}: '
            f'{n_cases} states, eval {rate.shape}, difficulty {difficulty.shape}'
        )
      if policy_rng.shape[0] != n_cases:
        raise ValueError(
            f'Experiment {number} generation {record.generation}: '
            f'policy_rng length {policy_rng.shape[0]} != {n_cases}'
        )
      qpos_parts.append(qpos)
      qvel_parts.append(qvel)
      rng_parts.append(policy_rng)
      eval_parts.append(rate)
      search_parts.append(difficulty)
      experiment_parts.append(np.full(n_cases, int(number), dtype=np.int32))
      n_read += n_cases
      del record
      if limit is not None and n_read >= int(limit):
        break
    gc.collect()
    if limit is not None and n_read >= int(limit):
      break

  qpos = np.concatenate(qpos_parts, axis=0)
  qvel = np.concatenate(qvel_parts, axis=0)
  policy_rng = np.concatenate(rng_parts, axis=0)
  source_eval = np.concatenate(eval_parts, axis=0)
  search_difficulty = np.concatenate(search_parts, axis=0)
  experiment = np.concatenate(experiment_parts, axis=0)
  if limit is not None and int(qpos.shape[0]) > int(limit):
    qpos = qpos[:int(limit)]
    qvel = qvel[:int(limit)]
    policy_rng = policy_rng[:int(limit)]
    source_eval = source_eval[:int(limit)]
    search_difficulty = search_difficulty[:int(limit)]
    experiment = experiment[:int(limit)]

  index = _distinct_index(qpos, qvel, experiment)
  n_raw = int(qpos.shape[0])
  if int(index.size) != n_raw:
    print(
        f'  distinct states: {int(index.size)}/{n_raw} '
        f'(dropped {n_raw - int(index.size)} exact repeats within experiment)',
        flush=True,
    )
    qpos = qpos[index]
    qvel = qvel[index]
    policy_rng = policy_rng[index]
    source_eval = source_eval[index]
    search_difficulty = search_difficulty[index]
    experiment = experiment[index]

  source_eval = np.asarray(source_eval, dtype=np.float64)
  search_difficulty = np.asarray(search_difficulty, dtype=np.float64)
  payload = {
      'version': np.asarray(_CASES_VERSION if source == 'ppo' else 'search-cases-1'),
      'qpos': qpos,
      'qvel': qvel,
      'policy_rng': policy_rng,
      'experiment': experiment,
      'source_eval': source_eval,
      'search_difficulty': search_difficulty,
  }
  # The PPO caches written earlier use these two names.
  if source == 'ppo':
    payload['ppo_eval'] = source_eval
    payload['ppo_search'] = search_difficulty
  temporary = path + '.tmp.npz'
  np.savez_compressed(temporary, **payload)
  os.replace(temporary, path)
  print(f'cases: wrote {path} ({int(qpos.shape[0])} states)', flush=True)
  return payload


def load_ppo_cases(
    environment: str,
    experiment_ids: tuple[int, ...] | None,
    limit: int | None,
) -> dict[str, np.ndarray]:
  """States, policy RNGs, and stored PPO eval rates from the PPO searches."""
  return load_search_cases(environment, 'ppo', experiment_ids, limit)


def _execute_fn(environment: str, members: tuple[tuple[str, int], ...]):
  rollout_length = ROLLOUT_LENGTHS[environment]
  algorithms = {algorithm for algorithm, _policy_id in members}
  if algorithms == {'ppo'} or algorithms == {'sac'}:
    algorithm = next(iter(algorithms))
    policy_ids = [policy_id for _algorithm, policy_id in members]
    bundle = make_search_bundle(
        environment,
        algorithm,
        len(policy_ids),
        rollout_length,
        with_attributions=False,
        policy_ids=policy_ids,
    )
  else:
    bundle = make_mixed_search_bundle(
        environment,
        rollout_length,
        [
            {'algorithm': algorithm, 'policy_id': policy_id}
            for algorithm, policy_id in members
        ],
        with_attributions=False,
    )
  if bundle.parallel_execute_fn is None:
    raise RuntimeError('Search bundle has no simultaneous executor.')
  return bundle.parallel_execute_fn


def score_ensemble(
    environment: str,
    name: str,
    members: tuple[tuple[str, int], ...],
    cases: dict[str, np.ndarray],
    tag: str,
    batch: int,
    source: str = 'ppo',
) -> np.ndarray:
  """Fail rate of one ensemble. A finished cache is reused; a partial cache resumes."""
  path = _rates_path(environment, tag, name, source)
  n_cases = int(cases['qpos'].shape[0])
  if os.path.isfile(path):
    rates = np.load(path)
    if rates.shape == (n_cases,):
      print(
          f'{name}: cached  mean={float(np.mean(rates)):.4f}  ({path})',
          flush=True,
      )
      return np.asarray(rates, dtype=np.float64)
    print(f'{name}: cache length {rates.shape} != {n_cases}, rescoring', flush=True)

  partial_path = path + '.partial.npy'
  done = 0
  chunks: list[np.ndarray] = []
  if os.path.isfile(partial_path):
    previous = np.load(partial_path)
    if previous.ndim == 1 and 0 < int(previous.shape[0]) <= n_cases:
      done = int(previous.shape[0])
      chunks.append(np.asarray(previous, dtype=np.float64))
      print(f'{name}: resume at case {done}/{n_cases}', flush=True)

  print(
      f'{name}: rolling out {len(members)} policies on {n_cases - done} cases',
      flush=True,
  )
  execute_fn = _execute_fn(environment, members)
  qpos = cases['qpos']
  qvel = cases['qvel']
  policy_rng = cases['policy_rng']
  remaining = n_cases - done
  n_batches = (remaining + batch - 1) // batch if remaining else 0
  batch_index = 0
  while done < n_cases:
    stop = min(done + batch, n_cases)
    n_keep = stop - done
    batch_index += 1
    print(
        f'  {name} batch {batch_index}/{n_batches}  cases {done}:{stop}',
        flush=True,
    )
    chunk = _pad_suite(
        _suite_from_numpy(qpos, qvel, policy_rng, done, stop), batch
    )
    failed = np.asarray(execute_fn(chunk))
    if failed.ndim != 2 or int(failed.shape[1]) != batch:
      raise ValueError(
          f'{name} failure matrix has shape {failed.shape}, expected (P, {batch})'
      )
    if int(failed.shape[0]) != len(members):
      raise ValueError(
          f'{name} scored {failed.shape[0]} policies, expected {len(members)}'
      )
    rate = np.mean(np.asarray(failed[:, :n_keep], dtype=np.float64), axis=0)
    chunks.append(rate)
    done = stop
    _atomic_npy(partial_path, np.concatenate(chunks))

  rates = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
  if rates.shape != (n_cases,):
    raise ValueError(f'{name} produced {rates.shape}, expected ({n_cases},)')
  _atomic_npy(path, rates)
  if os.path.isfile(partial_path):
    os.remove(partial_path)
  print(f'{name}: mean={float(np.mean(rates)):.4f}  wrote {path}', flush=True)
  return rates


def fail_histogram(rates: np.ndarray) -> np.ndarray:
  """Fraction of tests at each fail count k/16."""
  counts = np.rint(np.asarray(rates, dtype=np.float64) * N_POLICIES).astype(np.int64)
  counts = np.clip(counts, 0, N_POLICIES)
  histogram = np.bincount(counts, minlength=N_POLICIES + 1).astype(np.float64)
  total = float(histogram.sum())
  if total <= 0:
    raise ValueError('Cannot histogram an empty rate vector.')
  return histogram / total


def plot_distributions(
    series: list[tuple[str, str, np.ndarray]],
    tag: str,
    n_experiments: int,
) -> str:
  levels = np.arange(N_POLICIES + 1) / N_POLICIES
  figure, axis = plt.subplots(figsize=(7.2, 4.4))
  shares = []
  for key, label, rates in series:
    share = fail_histogram(rates)
    shares.append(share)
    mean = float(np.mean(rates))
    axis.plot(
        levels,
        share,
        linestyle=_LINESTYLES[key],
        marker=_MARKERS[key],
        markersize=4.5,
        linewidth=1.6,
        color=_COLORS[key],
        label=f'{label}  (mean {mean:.2f})',
    )
  axis.set_xlabel('Fail rate')
  axis.set_ylabel('Fraction of tests')
  axis.set_xlim(-0.02, 1.02)
  axis.set_xticks(np.arange(0, N_POLICIES + 1, 2) / N_POLICIES)
  axis.set_ylim(bottom=0)
  axis.legend(frameon=False, fontsize=8)
  axis.set_title(
      f'Humanoid tests from the PPO search ({n_experiments} experiments)'
  )
  figure.tight_layout()
  pdf_path = os.path.join(OUTPUT_DIR, f'humanoid_ppo_fail_distributions_{tag}.pdf')
  csv_path = os.path.join(OUTPUT_DIR, f'humanoid_ppo_fail_distributions_{tag}.csv')
  figure.savefig(pdf_path)
  plt.close(figure)
  header = 'fail_rate,' + ','.join(key for key, _label, _rates in series)
  lines = [header]
  for index, level in enumerate(levels):
    cells = ','.join(f'{share[index]:.6f}' for share in shares)
    lines.append(f'{level:.4f},{cells}')
  with open(csv_path, 'w') as handle:
    handle.write('\n'.join(lines) + '\n')
  print(f'plot: {pdf_path}', flush=True)
  return pdf_path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--experiment',
      type=int,
      action='append',
      help='PPO experiment number. Repeat to select several. Default: all finished.',
  )
  parser.add_argument(
      '--limit',
      type=int,
      default=None,
      help='Read only the first this many cases, in generation order. For a smoke test.',
  )
  parser.add_argument('--batch', type=int, default=_CASE_BATCH)
  parser.add_argument(
      '--plot-only',
      action='store_true',
      help='Plot cached rates. Does not roll out.',
  )
  args = parser.parse_args()
  if args.batch < 1:
    raise SystemExit(f'--batch must be >= 1, got {args.batch}')
  if args.limit is not None and args.limit < 1:
    raise SystemExit(f'--limit must be >= 1, got {args.limit}')

  experiment_ids = None if not args.experiment else tuple(args.experiment)
  tag = _tag(experiment_ids, args.limit)
  environment = 'humanoid'
  cases = load_ppo_cases(environment, experiment_ids, args.limit)
  n_experiments = int(np.unique(cases['experiment']).size)

  series = []
  for key, label, algorithm, role in ENSEMBLES:
    if key == 'ppo_eval':
      rates = np.asarray(cases['ppo_eval'], dtype=np.float64)
      _atomic_npy(_rates_path(environment, tag, key), rates)
      print(f'ppo_eval: stored mean={float(np.mean(rates)):.4f}', flush=True)
    elif args.plot_only:
      path = _rates_path(environment, tag, key)
      if not os.path.isfile(path):
        raise SystemExit(f'--plot-only needs {path}')
      rates = np.asarray(np.load(path), dtype=np.float64)
    else:
      members = load_ensemble_members(environment, algorithm, role)
      rates = score_ensemble(
          environment, key, members, cases, tag, args.batch
      )
    series.append((key, label, rates))

  plot_distributions(series, tag, n_experiments)


if __name__ == '__main__':
  main()
