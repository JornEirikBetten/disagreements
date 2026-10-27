"""Add held-out fail rates to finished difficulty-bin collections.

Usage:
  python add_heldout_fail_rates.py
  python add_heldout_fail_rates.py --environment hopper --algorithm ppo

Reads finished searches under

  datasets/ga_data/{env}/{alg}/difficulty_bins/rollout_length={T}/

The search entry points call this scoring stage automatically after the GA
stops; this module also remains a standalone backfill command when
``--skip-heldout-eval`` was used.

The held-out policies are the eval ensemble in each experiment's
``*_init.json``. Single-algorithm searches store ``eval_policy_ids``.
PPO+SAC searches store ``policy_selection.eval``, eight PPO and eight SAC
policies, and those are the policies that score the cases. Each kept
initial state is rolled out with those policies, and the fraction that
terminate is written onto the
generation record as ``eval_fail_rate`` with shape ``(N,)`` and values in
``[0, 1]``. The exact per-policy mask is stored as ``eval_failed`` with shape
``(P_E, N)``; ``eval_fail_rate`` is checked against its column mean. A value
of ``1`` means every held-out policy failed.

The sidecar records the complete held-out failure-count histogram and its
conditional distribution at every search-failure-rate level. These are
descriptive diagnostics only: evaluation outcomes never fill a quota or
affect which genomes breed.

Sharded archives are read one generation at a time only to collect the initial
states. Those states are then scored as one pool of 34,000 cases, in
batches of 6,000. The compiled rollout always sees exactly 6,000 cases and every held-out
policy at once. The last batch is padded and the padding is discarded. Rates are written back
onto the generation each case
came from, in the original case order.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
from jax import numpy as jp

from ga.collect import generation_shard_dir, iter_generation_shards, load_collection
from ga.env_setup import (
    LOCOMOTION_ENVS,
    MIXED_ALGORITHM,
    make_mixed_search_bundle,
    make_search_bundle,
)
from test_case_execution.types import TestSuite

ALGORITHMS = ('ppo', 'sac', MIXED_ALGORITHM)
ROLLOUT_LENGTHS = {
    'ant': 15,
    'humanoid': 15,
    'walker2d': 15,
    'hopper': 15,
}
_COLLECTION_NAME = re.compile(r'^experiment_(\d+)_ga_collection\.pkl$')
# One compiled case-batch. A finished experiment is 34,000 cases, so this
# runs as five full batches and one short tail padded up to the same size.
_CASE_BATCH = 6000


def _collection_dir(environment: str, algorithm: str) -> str:
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'difficulty_bins',
      f'rollout_length={ROLLOUT_LENGTHS[environment]}',
  )


def _bins_full(path: str) -> bool:
  init_path = os.path.splitext(path)[0] + '_init.json'
  if not os.path.isfile(init_path):
    return False
  with open(init_path) as handle:
    meta = json.load(handle)
  return bool(meta.get('bins', {}).get('all_full'))


def _discover_experiments(environment: str, algorithm: str) -> list[tuple[int, str]]:
  directory = _collection_dir(environment, algorithm)
  if not os.path.isdir(directory):
    raise FileNotFoundError(
        f'No difficulty-bin directory at {directory}. '
        'Run search_for_tests/binned_ga_search.py first.'
    )
  found = []
  skipped = []
  for name in os.listdir(directory):
    match = _COLLECTION_NAME.match(name)
    if not match:
      continue
    path = os.path.join(directory, name)
    if _bins_full(path):
      found.append((int(match.group(1)), path))
    else:
      skipped.append(name)
  found.sort()
  if skipped:
    print(f'Skipping unfinished bins: {", ".join(sorted(skipped))}', flush=True)
  if not found:
    raise FileNotFoundError(
        f'No finished experiment_*_ga_collection.pkl files in {directory}'
    )
  return found


def _eval_members(path: str, algorithm: str) -> list[dict]:
  """Held-out policies for one experiment, in rollout order.

  PPO+SAC experiments store one record per policy, including which
  algorithm it belongs to. Single-algorithm experiments store ids only.
  """
  init_path = os.path.splitext(path)[0] + '_init.json'
  with open(init_path) as handle:
    meta = json.load(handle)
  selection = meta.get('policy_selection', {})
  if algorithm == MIXED_ALGORITHM:
    members = selection.get('eval')
    if not members:
      raise ValueError(f'{init_path} has no held-out eval ensemble.')
    parsed = []
    for member in members:
      name = str(member['algorithm'])
      if name not in ('ppo', 'sac'):
        raise ValueError(
            f'{init_path} eval member has algorithm {name!r}, expected ppo or sac.'
        )
      parsed.append({
          'algorithm': name,
          'policy_id': int(member['policy_id']),
      })
    return parsed
  ids = selection.get('eval_policy_ids')
  if not ids:
    raise ValueError(f'{init_path} has no held-out eval_policy_ids.')
  return [
      {'algorithm': algorithm, 'policy_id': int(policy_id)} for policy_id in ids
  ]


def _has_evaluation(record, n_eval_policies: int) -> bool:
  """Whether a record has an aligned exact mask and its derived rate."""
  rate = getattr(record, 'eval_fail_rate', None)
  failed = getattr(record, 'eval_failed', None)
  if rate is None or failed is None:
    return False
  rate = np.asarray(rate, dtype=np.float64)
  failed = np.asarray(failed, dtype=bool)
  expected_mask = (int(n_eval_policies), int(record.n_cases))
  if rate.shape != (int(record.n_cases),) or failed.shape != expected_mask:
    return False
  return bool(
      np.allclose(
          rate,
          np.mean(failed, axis=0),
          atol=1e-12,
          rtol=0.0,
      )
  )


def _suite_from_numpy(qpos, qvel, policy_rng, start: int, stop: int) -> TestSuite:
  return TestSuite(
      qpos=jp.asarray(qpos[start:stop]),
      qvel=jp.asarray(qvel[start:stop]),
      policy_rng=jp.asarray(policy_rng[start:stop]),
  )


def _pad_suite(suite: TestSuite, batch: int) -> TestSuite:
  """Repeat the first case until the suite length is exactly ``batch``."""
  n = int(suite.qpos.shape[0])
  if n == batch:
    return suite
  if n < 1 or n > batch:
    raise ValueError(f'Cannot pad a suite of length {n} to {batch}.')
  pad_n = batch - n

  def _pad(values):
    filler = jp.broadcast_to(values[:1], (pad_n,) + values.shape[1:])
    return jp.concatenate([values, filler], axis=0)

  return TestSuite(
      qpos=_pad(suite.qpos),
      qvel=_pad(suite.qvel),
      policy_rng=_pad(suite.policy_rng),
  )


def _heldout_failures(execute_fn, suite: TestSuite, batch: int) -> np.ndarray:
  """Exact held-out failure mask, with one column per original case.

  ``execute_fn`` is only ever called with exactly ``batch`` cases. Padding
  is appended after the real cases and removed before the mask is stored.
  """
  qpos = np.asarray(suite.qpos)
  qvel = np.asarray(suite.qvel)
  policy_rng = np.asarray(suite.policy_rng)
  n_cases = int(qpos.shape[0])
  if n_cases < 1:
    raise ValueError('Cannot score an empty suite.')
  if batch < 1:
    raise ValueError(f'batch must be >= 1, got {batch}')
  n_batches = (n_cases + batch - 1) // batch
  parts = []
  for batch_index, start in enumerate(range(0, n_cases, batch)):
    stop = min(start + batch, n_cases)
    n_keep = stop - start
    print(
        f'  batch {batch_index + 1}/{n_batches}  '
        f'cases {start}:{stop}',
        flush=True,
    )
    chunk = _pad_suite(
        _suite_from_numpy(qpos, qvel, policy_rng, start, stop), batch
    )
    if int(chunk.qpos.shape[0]) != batch:
      raise ValueError(
          f'Executor batch has length {chunk.qpos.shape[0]}, expected {batch}'
      )
    failed = np.asarray(execute_fn(chunk))
    if failed.ndim != 2 or int(failed.shape[1]) != batch:
      raise ValueError(
          f'Failure matrix has shape {failed.shape}, expected (P, {batch})'
      )
    # Columns [0, n_keep) are cases [start, stop), in that order.
    parts.append(np.asarray(failed[:, :n_keep], dtype=bool))
  mask = np.concatenate(parts, axis=1)
  if mask.ndim != 2 or int(mask.shape[1]) != n_cases:
    raise ValueError(
        f'Held-out failure mask has shape {mask.shape}, expected (P, {n_cases})'
    )
  return mask


def _write_record(path: str, record) -> None:
  temporary = path + '.tmp'
  with open(temporary, 'wb') as handle:
    pickle.dump(record, handle)
  os.replace(temporary, path)


def _annotate_init(
    path: str,
    members: list[dict],
    pattern_summary: dict,
    rollout_length: int,
) -> None:
  init_path = os.path.splitext(path)[0] + '_init.json'
  with open(init_path) as handle:
    meta = json.load(handle)
  meta['eval_fail_rate'] = {
      'policy_ids': [member['policy_id'] for member in members],
      'algorithms': [member['algorithm'] for member in members],
      'rollout_length': int(rollout_length),
      'role': 'held_out_after_search',
      'used_for_search_selection': False,
      'stored_cases_unique': bool(
          meta.get('bins', {}).get('unique_states', False)
          or meta.get('unique_states', False)
      ),
      'mask_field': 'eval_failed',
      'rate_field': 'eval_fail_rate',
      'definition': (
          'Fraction of held-out policies that fail the case. '
          'Values lie in [0, 1]; 1 means every held-out policy failed.'
      ),
      'pattern_summary': pattern_summary,
  }
  with open(init_path, 'w') as handle:
    json.dump(meta, handle, indent=2)


def _stored_provenance_matches(
    path: str,
    members: list[dict],
    rollout_length: int,
) -> bool:
  """Only trust an existing mask if it names this exact ordered ensemble."""
  init_path = os.path.splitext(path)[0] + '_init.json'
  with open(init_path) as handle:
    meta = json.load(handle)
  stored = meta.get('eval_fail_rate', {})
  return (
      stored.get('policy_ids')
      == [int(member['policy_id']) for member in members]
      and stored.get('algorithms')
      == [str(member['algorithm']) for member in members]
      and stored.get('rollout_length') == int(rollout_length)
      and stored.get('mask_field') == 'eval_failed'
      and stored.get('rate_field') == 'eval_fail_rate'
  )


def _copy_case_arrays(record):
  suite = record.suite
  difficulty = np.asarray(record.difficulty, dtype=np.float64)
  qpos = np.asarray(suite.qpos)
  if int(difficulty.shape[0]) != int(qpos.shape[0]):
    raise ValueError(
        f'Generation {record.generation} difficulty length '
        f'{difficulty.shape[0]} != {qpos.shape[0]} cases'
    )
  return (
      qpos,
      np.asarray(suite.qvel),
      np.asarray(suite.policy_rng),
      difficulty,
  )


def _load_experiment_states(
    path: str,
    *,
    n_eval_policies: int,
    trust_existing: bool,
):
  """Initial states in generation order, without keeping attribution arrays.

  Returns ``qpos``, ``qvel``, ``policy_rng``, the search-set difficulty, and
  one span per generation so masks and rates can be written back onto the
  same rows. Returns ``None`` only when every record has a valid exact mask
  and the sidecar provenance matches the requested evaluation ensemble.
  """
  shard_dir = generation_shard_dir(path)
  qpos_parts = []
  qvel_parts = []
  rng_parts = []
  difficulty_parts = []
  spans = []
  if os.path.isdir(shard_dir):
    manifest_path = os.path.join(shard_dir, 'manifest.json')
    with open(manifest_path) as handle:
      manifest = json.load(handle)
    entries = manifest.get('shards', [])
    already = []
    for entry in entries:
      shard_path = os.path.join(shard_dir, entry['file'])
      with open(shard_path, 'rb') as handle:
        record = pickle.load(handle)
      already.append(
          trust_existing and _has_evaluation(record, n_eval_policies)
      )
      qpos, qvel, policy_rng, difficulty = _copy_case_arrays(record)
      if int(qpos.shape[0]) != int(record.n_cases):
        raise ValueError(
            f'{shard_path} suite length {qpos.shape[0]} != '
            f'{record.n_cases} cases'
        )
      spans.append({
          'path': shard_path,
          'generation': int(record.generation),
          'n_cases': int(record.n_cases),
      })
      qpos_parts.append(qpos)
      qvel_parts.append(qvel)
      rng_parts.append(policy_rng)
      difficulty_parts.append(difficulty)
      del record
      gc.collect()
    if already and all(already):
      return None
  else:
    collection = load_collection(path, load_shards=True)
    if (
        trust_existing
        and collection.generations
        and all(
            _has_evaluation(record, n_eval_policies)
            for record in collection.generations
        )
    ):
      del collection
      gc.collect()
      return None
    for record in collection.generations:
      qpos, qvel, policy_rng, difficulty = _copy_case_arrays(record)
      spans.append({
          'generation': int(record.generation),
          'n_cases': int(qpos.shape[0]),
      })
      qpos_parts.append(qpos)
      qvel_parts.append(qvel)
      rng_parts.append(policy_rng)
      difficulty_parts.append(difficulty)
    collection.generations.clear()
    del collection
    gc.collect()
  if not spans:
    raise ValueError(f'No generations in {path}')
  return (
      np.concatenate(qpos_parts, axis=0),
      np.concatenate(qvel_parts, axis=0),
      np.concatenate(rng_parts, axis=0),
      np.concatenate(difficulty_parts, axis=0),
      spans,
  )


def _write_evaluation(
    path: str,
    eval_failed: np.ndarray,
    spans: list[dict],
) -> np.ndarray:
  """Store exact masks and their derived rates, preserving case alignment."""
  eval_failed = np.asarray(eval_failed, dtype=bool)
  if eval_failed.ndim != 2 or int(eval_failed.shape[0]) < 1:
    raise ValueError(
        f'eval_failed must have shape (P_E, N), got {eval_failed.shape}'
    )
  rates = np.mean(eval_failed, axis=0, dtype=np.float64)
  offset = 0
  shard_dir = generation_shard_dir(path)
  sharded = os.path.isdir(shard_dir)
  collection = None if sharded else load_collection(path, load_shards=True)
  by_generation = {}
  if collection is not None:
    by_generation = {
        int(record.generation): record for record in collection.generations
    }
  for span in spans:
    n_cases = int(span['n_cases'])
    failed_piece = np.asarray(
        eval_failed[:, offset:offset + n_cases], dtype=bool
    )
    rate_piece = np.asarray(
        rates[offset:offset + n_cases], dtype=np.float64
    )
    expected_mask = (int(eval_failed.shape[0]), n_cases)
    if failed_piece.shape != expected_mask or rate_piece.shape != (n_cases,):
      raise ValueError(
          f'Generation {span["generation"]} expected mask {expected_mask} '
          f'and {n_cases} rates; got {failed_piece.shape} and '
          f'{rate_piece.shape}'
      )
    if sharded:
      with open(span['path'], 'rb') as handle:
        record = pickle.load(handle)
      if int(record.n_cases) != n_cases:
        raise ValueError(
            f'{span["path"]} has {record.n_cases} cases, span says {n_cases}'
        )
      record.eval_failed = failed_piece
      record.eval_fail_rate = rate_piece
      _write_record(span['path'], record)
      del record
      gc.collect()
    else:
      record = by_generation[int(span['generation'])]
      if int(record.n_cases) != n_cases:
        raise ValueError(
            f'Generation {span["generation"]} has {record.n_cases} cases, '
            f'span says {n_cases}'
        )
      record.eval_failed = failed_piece
      record.eval_fail_rate = rate_piece
    offset += n_cases
  if offset != int(eval_failed.shape[1]):
    raise ValueError(
        f'Wrote {offset} cases, but the experiment produced '
        f'{eval_failed.shape[1]}'
    )
  if collection is not None:
    with open(path, 'wb') as handle:
      pickle.dump(collection, handle)
    del collection
    gc.collect()
  return rates


def _evaluation_pattern_summary(
    difficulty: np.ndarray,
    rates: np.ndarray,
    n_eval_policies: int,
) -> dict:
  """Held-out-rate distribution overall and at every search-rate level."""
  difficulty = np.asarray(difficulty, dtype=np.float64)
  rates = np.asarray(rates, dtype=np.float64)
  if difficulty.shape != rates.shape:
    raise ValueError(
        f'Search difficulty shape {difficulty.shape} != '
        f'held-out rates {rates.shape}'
    )
  if n_eval_policies < 1:
    raise ValueError(
        f'n_eval_policies must be >= 1, got {n_eval_policies}'
    )
  failed_count = np.rint(rates * n_eval_policies).astype(np.int32)
  reconstructed = failed_count.astype(np.float64) / float(n_eval_policies)
  if np.any(failed_count < 0) or np.any(failed_count > n_eval_policies):
    raise ValueError('Held-out fail rates imply an invalid failure count.')
  if not np.allclose(rates, reconstructed, atol=1e-6, rtol=0.0):
    raise ValueError(
        'Held-out fail rates are not multiples of 1 / n_eval_policies.'
    )

  def _describe(mask: np.ndarray) -> dict:
    selected_rates = rates[mask]
    selected_counts = failed_count[mask]
    histogram = np.bincount(
        selected_counts, minlength=n_eval_policies + 1
    )
    return {
        'n_cases': int(selected_rates.size),
        'mean': float(np.mean(selected_rates)),
        'std': float(np.std(selected_rates)),
        'min': float(np.min(selected_rates)),
        'max': float(np.max(selected_rates)),
        'counts_by_failed_policy': [int(value) for value in histogram],
    }

  keys = np.round(difficulty, decimals=6)
  finite = np.isfinite(keys)
  levels = np.unique(keys[finite])
  order = np.argsort(np.where(levels < 0.0, 1.0, levels))
  by_search_difficulty = []
  for level in levels[order]:
    at_level = finite & np.isclose(keys, level)
    row = {
        'search_difficulty': float(level),
        'search_label': 'all-fail' if level < 0.0 else f'{level:.6f}',
    }
    row.update(_describe(at_level))
    by_search_difficulty.append(row)
  if np.any(~finite):
    # Feature search does not roll out the search ensemble, so the stored
    # search difficulty is NaN. The held-out rates still describe the archive.
    row = {
        'search_difficulty': None,
        'search_label': 'not-estimated',
    }
    row.update(_describe(~finite))
    by_search_difficulty.append(row)
  return {
      'n_eval_policies': int(n_eval_policies),
      'overall': _describe(np.ones(rates.shape, dtype=bool)),
      'by_search_difficulty': by_search_difficulty,
  }


def _report_evaluation_patterns(summary: dict) -> None:
  """Print the durable held-out distribution stored in the sidecar JSON."""
  overall = summary['overall']
  print(
      '  held-out pattern over all stored unique cases: '
      f'n={overall["n_cases"]} mean={overall["mean"]:.3f} '
      f'std={overall["std"]:.3f} '
      f'range=[{overall["min"]:.3f}, {overall["max"]:.3f}]',
      flush=True,
  )
  histogram = overall['counts_by_failed_policy']
  nonempty = [
      f'{n_failed}:{n_cases}'
      for n_failed, n_cases in enumerate(histogram)
      if n_cases
  ]
  print(
      '  held-out failed-policy histogram (failed:n_cases): '
      + ' '.join(nonempty),
      flush=True,
  )
  print(
      f'  {"search difficulty":>18}  {"eval mean":>9}  {"eval std":>8}  '
      f'{"eval range":>15}  {"n":>7}',
      flush=True,
  )
  for row in summary['by_search_difficulty']:
    print(
        f'  {row["search_label"]:>18}  {row["mean"]:9.3f}  '
        f'{row["std"]:8.3f}  '
        f'[{row["min"]:.3f}, {row["max"]:.3f}]  '
        f'{row["n_cases"]:7d}',
        flush=True,
    )


def annotate_experiment(
    environment: str,
    algorithm: str,
    path: str,
    *,
    batch: int,
    rollout_length: int | None = None,
) -> None:
  members = _eval_members(path, algorithm)
  eval_ids = [member['policy_id'] for member in members]
  if rollout_length is None:
    rollout_length = ROLLOUT_LENGTHS[environment]
  rollout_length = int(rollout_length)
  if rollout_length < 1:
    raise ValueError(
        f'rollout_length must be >= 1, got {rollout_length}'
    )
  labeled = [
      f'{member["algorithm"]}:{member["policy_id"]}' for member in members
  ]
  print(
      f'{environment}/{algorithm} {os.path.basename(path)}  '
      f'held-out P={len(members)} T={rollout_length}  '
      f'fixed batch={batch} ids={labeled}',
      flush=True,
  )
  trust_existing = _stored_provenance_matches(
      path, members, rollout_length
  )
  loaded = _load_experiment_states(
      path,
      n_eval_policies=len(members),
      trust_existing=trust_existing,
  )
  if loaded is None:
    print(
        '  exact eval_failed mask already matches this ensemble; skipping',
        flush=True,
    )
    return
  if not trust_existing:
    print(
        '  no matching exact-mask provenance; scoring all stored cases',
        flush=True,
    )
  if algorithm == MIXED_ALGORITHM:
    bundle = make_mixed_search_bundle(
        environment,
        rollout_length,
        members,
        with_attributions=False,
    )
  else:
    bundle = make_search_bundle(
        environment,
        algorithm,
        len(eval_ids),
        rollout_length,
        with_attributions=False,
        policy_ids=eval_ids,
    )
  if bundle.parallel_execute_fn is None:
    raise RuntimeError(
        f'{environment} has no simultaneous policy executor.'
    )
  qpos, qvel, policy_rng, difficulty, spans = loaded
  n_cases = int(qpos.shape[0])
  n_batches = (n_cases + batch - 1) // batch
  print(
      f'  {n_cases} cases, {n_batches} batches of {batch}',
      flush=True,
  )
  suite = TestSuite(qpos=qpos, qvel=qvel, policy_rng=policy_rng)
  eval_failed = _heldout_failures(
      bundle.parallel_execute_fn, suite, batch
  )
  del suite, qpos, qvel, policy_rng
  gc.collect()
  rates = _write_evaluation(path, eval_failed, spans)
  pattern_summary = _evaluation_pattern_summary(
      difficulty, rates, len(members)
  )
  _annotate_init(
      path,
      members,
      pattern_summary,
      rollout_length,
  )
  print(
      f'  wrote eval_failed and eval_fail_rate for {n_cases} cases',
      flush=True,
  )
  _report_evaluation_patterns(pattern_summary)
  del bundle
  gc.collect()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Add held-out policy fail rates to difficulty-bin collections.',
  )
  parser.add_argument('--environment', choices=LOCOMOTION_ENVS)
  parser.add_argument('--algorithm', choices=ALGORITHMS)
  parser.add_argument(
      '--experiment',
      type=int,
      help='If set, annotate only this experiment number.',
  )
  parser.add_argument(
      '--batch',
      type=int,
      default=_CASE_BATCH,
      help=(
          'Fixed case batch compiled once and reused for every generation '
          '(default: 6000).'
      ),
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.batch < 1:
    raise ValueError(f'batch must be >= 1, got {args.batch}')
  environments = (args.environment,) if args.environment else LOCOMOTION_ENVS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  explicit = args.environment is not None and args.algorithm is not None
  for environment in environments:
    for algorithm in algorithms:
      if not os.path.isdir(_collection_dir(environment, algorithm)):
        if explicit:
          raise FileNotFoundError(
              f'No difficulty-bin directory at '
              f'{_collection_dir(environment, algorithm)}.'
          )
        print(
            f'Skipping {environment}/{algorithm}: no difficulty-bin directory',
            flush=True,
        )
        continue
      for exp_number, path in _discover_experiments(environment, algorithm):
        if args.experiment is not None and exp_number != args.experiment:
          continue
        annotate_experiment(environment, algorithm, path, batch=args.batch)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
