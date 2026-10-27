"""Pool difficulty-balanced GA experiments into predictor datasets.

Usage:
  python build_predictor_datasets.py

Reads finished binned searches under

  datasets/ga_data/{env}/{alg}/difficulty_bins/rollout_length={T}/

with T=15 for Humanoid, Hopper, Walker2d, and Ant.
Only experiments whose bins are all full are included, so each difficulty
level contributes the same number of cases. Generation shards are read one
at a time.

Criticality disagreement is the mean pairwise absolute difference of
``sum_i log σ_i`` on the search ensemble. The same scale vector also
contributes the absolute difference of its Euclidean magnitude and the
cosine distance of its direction.

Each encounter becomes one row with the usual disagreement features, plus:

  * state-IG top-k cosine distance for every k = 1, ..., nq+nv
  * observation-IG top-k cosine distance for every k = 1, ..., D,
    where D is the observation dimension

The observation curve is the one that matters in high dimension: full
cosine distance is kept as its last column, not as hundreds of separate
model inputs.

Writes ``datasets/{environment}/{algorithm}/predictor_data_balanced.pkl``.
"""

from __future__ import annotations

import argparse
import json
import os
import re

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import gc

import numpy as np

from analysis.disagreement import (
    standardized_value_disagreement_np,
    value_estimates_np,
)
from analysis.gait_subspace import load_gait_subspaces_for_policies
from analysis.predictor_dataset import (
    concat_predictor_datasets,
    predictor_dataset_from_generation,
    predictor_dataset_path,
    save_predictor_dataset,
)
from ga.collect import (
    generation_shard_dir,
    iter_generation_shards,
    load_collection,
)

ENVIRONMENTS = ('ant', 'humanoid', 'walker2d', 'hopper')
ALGORITHMS = ('ppo', 'sac', 'ppo_sac')
ROLLOUT_LENGTHS = {
    'ant': 15,
    'humanoid': 15,
    'walker2d': 15,
    'hopper': 15,
}
_COLLECTION_NAME = re.compile(r'^experiment_(\d+)_ga_collection\.pkl$')


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


def _search_policy_algorithms(path: str, policy_ids: list[int]) -> list[str]:
  """Algorithm of each search policy, in collection order.

  Mixed collections store only integer ids, and PPO and SAC can share an
  id. The sidecar records which algorithm each row of the ensemble is.
  """
  init_path = os.path.splitext(path)[0] + '_init.json'
  with open(init_path) as handle:
    meta = json.load(handle)
  search = meta['policy_selection']['search']
  algorithms = [str(member['algorithm']) for member in search]
  ids = [int(member['policy_id']) for member in search]
  if ids != [int(policy_id) for policy_id in policy_ids]:
    raise ValueError(
        f'{path}: init.json search ids {ids} do not match collection '
        f'policy_ids {list(policy_ids)}'
    )
  return algorithms


def _iter_records(path: str):
  """Yield generation records without assembling a sharded archive."""
  shard_dir = generation_shard_dir(path)
  if os.path.isdir(shard_dir):
    collection = load_collection(path, load_shards=False)
    return collection, iter_generation_shards(path)
  collection = load_collection(path, load_shards=True)
  return collection, iter(collection.generations)


def build_combo(environment: str, algorithm: str) -> str:
  experiments = _discover_experiments(environment, algorithm)
  parts = []
  for exp_number, path in experiments:
    print(f'Loading {path}', flush=True)
    collection, records = _iter_records(path)
    policy_ids = list(collection.policy_ids)
    policy_algorithms = None
    if algorithm == 'ppo_sac':
      policy_algorithms = _search_policy_algorithms(path, policy_ids)
    subspaces = load_gait_subspaces_for_policies(
        environment, algorithm, policy_ids, algorithms=policy_algorithms
    )
    generation_parts = []
    value_pieces = []
    for record in records:
      value_pieces.append(
          value_estimates_np(
              record.values,
              getattr(record, 'q_values', None),
              algorithm=algorithm,
          )
      )
      built = predictor_dataset_from_generation(
          record,
          environment,
          algorithm,
          policy_ids,
          experiment=exp_number,
          subspaces=subspaces,
      )
      generation = int(record.generation)
      built.names = np.asarray(
          [
              f'exp{exp_number}_gen{generation:03d}_case{int(i):03d}'
              for i in built.case_index
          ],
          dtype=object,
      )
      generation_parts.append(built)
      del record, built
      gc.collect()
    part = concat_predictor_datasets(generation_parts)
    # Each generation was standardized on its own kept cases. Replace that
    # with one z-score per policy across this experiment.
    part.value_disagreement = standardized_value_disagreement_np(
        np.concatenate(value_pieces, axis=1)
    )
    del generation_parts, value_pieces, collection
    gc.collect()
    parts.append(part)
    print(
        f'  experiment {exp_number}: {part.n_cases} encounters  '
        f'P={part.n_policies} ids={part.policy_ids}  '
        f'obs_topk_D={part.observation_ig_topk_disagreement.shape[1]}  '
        f'state_topk_D={part.feature_attribution_topk_disagreement.shape[1]}',
        flush=True,
    )
  dataset = concat_predictor_datasets(parts)
  output_path = predictor_dataset_path(environment, algorithm, balanced=True)
  save_predictor_dataset(dataset, output_path)
  print(
      f'Saved {output_path}  N={dataset.n_cases}  P={dataset.n_policies}  '
      f'experiments={sorted(set(int(x) for x in dataset.experiment))}'
  )
  return output_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Build pooled predictor datasets from difficulty-balanced GA data.',
  )
  parser.add_argument(
      '--environment',
      choices=ENVIRONMENTS,
      help='If omitted, build all locomotion environments.',
  )
  parser.add_argument(
      '--algorithm',
      choices=ALGORITHMS,
      help='If omitted, build PPO and SAC.',
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  environments = (args.environment,) if args.environment else ENVIRONMENTS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  for environment in environments:
    for algorithm in algorithms:
      build_combo(environment, algorithm)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
