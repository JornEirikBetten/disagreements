"""Shared command-line plumbing for the binned GA search entry points.

Two scripts drive the binned search: ``binned_ga_search.py`` for a single
algorithm and ``binned_ga_search_mixed.py`` for a joint PPO/SAC ensemble.
Both run the same pipeline and differ only in how the ensembles are picked:

  1. split trained checkpoints into a search ensemble and a disjoint
     held-out evaluation ensemble              (differs between the two)
  2. build the rollout bundle for the search ensemble   (differs)
  3. seed generation 0, evolve until every search-rate bin is full, and
     record what was searched                           (shared, lives here)
  4. after search selection is over, score every stored unique state with
     the held-out ensemble                              (shared, lives here)

Stage 3 is what defines the dataset: the bin quotas, the immigrant rate, the
checkpoint cadence, and the metadata sidecar that every downstream analysis
reads. Keeping it in one place is what stops the two searches from quietly
producing datasets that are not comparable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import jax

from ga.binned import DEFAULT_BIN_CAPACITY, run_binned_ga_search
from ga.collect import generation_shard_dir, save_collection, summarize_collection
from ga.stratified_init import stratified_initial_population

# Bins are indexed by how many search policies failed: one all-fail bin plus
# one bin per achievable fail count below P.
BINS_PER_POLICY_COUNT = 1

_RNG_SEED_BASE = 42


def experiment_seed(experiment_number: int) -> int:
  """PRNG seed for one experiment number.

  The initial population, parent tournaments, crossover, mutation, and
  immigrants are drawn from ``jax.random.PRNGKey`` of this seed. The same
  experiment number rebuilds those draws.
  """
  experiment_number = int(experiment_number)
  if experiment_number < 0:
    raise ValueError(
        f'experiment number must be >= 0, got {experiment_number}'
    )
  return _RNG_SEED_BASE + experiment_number

# Budget multiple over the generations a perfect run would need. Yield per
# generation drops sharply once the easy bins fill, so the search needs
# several times the ideal count to finish the hard bins.
GENERATION_BUDGET_FACTOR = 4
MIN_GENERATIONS = 100

DEFAULT_EVAL_BATCH = 6_000


def add_search_arguments(parser: argparse.ArgumentParser) -> None:
  """Options that control the search itself, shared by both entry points."""
  parser.add_argument(
      '--bin-capacity',
      type=int,
      default=DEFAULT_BIN_CAPACITY,
      help=(
          'Distinct test cases to keep at each search-fail-rate level '
          '(default: 10000).'
      ),
  )
  parser.add_argument(
      '--max-generations',
      type=int,
      default=None,
      help=(
          'Evolution steps after generation 0. Default: four times the '
          'generations needed to fill every bin at 100%% yield.'
      ),
  )
  parser.add_argument(
      '--immigrant-fraction',
      type=float,
      default=0.25,
      help='Fraction of each new generation replaced by random genomes.',
  )
  parser.add_argument(
      '--save-every',
      type=int,
      default=1,
      help=(
          'Rewrite the sidecar pickle every N generations (0 disables). '
          'Kept cases are always dumped to a generation shard immediately.'
      ),
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help=(
          'Replace an existing experiment at the same output path. Without '
          'this flag the search fails rather than mixing old generation '
          'shards into a new run.'
      ),
  )
  parser.add_argument(
      '--no-stratified-init',
      action='store_true',
      help='Start from a random population instead of stratified init.',
  )
  parser.add_argument(
      '--skip-heldout-eval',
      action='store_true',
      help=(
          'Do not score the stored unique cases with the held-out ensemble '
          'after the search. By default the exact eval_failed mask, its '
          'eval_fail_rate mean, and its distribution are written to the '
          'generation shards and metadata.'
      ),
  )
  parser.add_argument(
      '--allow-unequal-eval',
      action='store_true',
      help=(
          'Allow fewer held-out than search policies. Confirmatory runs '
          'fail by default because unequal ensembles have different target '
          'resolution. This flag does not allow overlap or duplicate ids.'
      ),
  )
  parser.add_argument(
      '--eval-batch',
      type=int,
      default=DEFAULT_EVAL_BATCH,
      help=(
          'Fixed case batch for post-search held-out scoring '
          f'(default: {DEFAULT_EVAL_BATCH}).'
      ),
  )


def validate_search_arguments(args: argparse.Namespace) -> None:
  """Reject settings that would run for hours and then produce nothing."""
  if args.rollout_length < 1:
    raise ValueError(f'rollout_length must be >= 1, got {args.rollout_length}')
  if args.bin_capacity < 1:
    raise ValueError(f'bin_capacity must be >= 1, got {args.bin_capacity}')
  if args.max_generations is not None and args.max_generations < 1:
    raise ValueError(
        f'max_generations must be >= 1, got {args.max_generations}'
    )
  if not 0.0 <= args.immigrant_fraction < 1.0:
    raise ValueError(
        'immigrant_fraction must be in [0, 1); at 1.0 every generation is '
        f'discarded and replaced, got {args.immigrant_fraction}'
    )
  if args.save_every < 0:
    raise ValueError(f'save_every must be >= 0, got {args.save_every}')
  if args.eval_batch < 1:
    raise ValueError(f'eval_batch must be >= 1, got {args.eval_batch}')


def collection_path(
    environment: str, algorithm: str, exp_number: int, rollout_length: int
) -> str:
  """Where one experiment's collection pickle lives."""
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'difficulty_bins',
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(exp_number)}_ga_collection.pkl',
  )


def init_metadata_path(path: str) -> str:
  """Sidecar JSON describing how a collection was produced."""
  return os.path.splitext(path)[0] + '_init.json'


def default_max_generations(
    n_policies: int, bin_capacity: int, population_size: int
) -> int:
  """Generation budget for filling every bin, with slack for falling yield."""
  n_bins = int(n_policies) + BINS_PER_POLICY_COUNT
  need = n_bins * int(bin_capacity)
  per_gen = max(int(population_size), 1)
  ideal_generations = (need + per_gen - 1) // per_gen
  return max(MIN_GENERATIONS, int(ideal_generations) * GENERATION_BUDGET_FACTOR)


def _write_metadata(init_info: dict, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w') as handle:
    json.dump(init_info, handle, indent=2)


def run_binned_search(
    bundle,
    args: argparse.Namespace,
    *,
    n_policies: int,
    exp_number: int,
    selection: dict,
) -> dict:
  """Seed, evolve, and record one binned search. Returns the bin fill info.

  ``selection`` is the record of which checkpoints went to the search
  ensemble and which were held out; it is written to the sidecar verbatim.
  """
  max_generations = args.max_generations
  if max_generations is None:
    max_generations = default_max_generations(
        n_policies, args.bin_capacity, bundle.ga_config.population_size
    )
  output_path = collection_path(
      bundle.environment, bundle.algorithm, exp_number, args.rollout_length
  )
  meta_path = init_metadata_path(output_path)
  shard_path = generation_shard_dir(output_path)
  existing = [
      path
      for path in (output_path, meta_path, shard_path)
      if os.path.exists(path)
  ]
  if existing and not args.overwrite:
    paths = ', '.join(existing)
    raise FileExistsError(
        f'Experiment output already exists: {paths}. Pass --overwrite to '
        'replace the collection and all of its generation shards.'
    )
  if args.overwrite:
    for path in (output_path, meta_path):
      if os.path.isfile(path):
        os.remove(path)
    if os.path.isdir(shard_path):
      shutil.rmtree(shard_path)
  n_bins = n_policies + BINS_PER_POLICY_COUNT
  print(
      f'Binned GA {bundle.environment}/{bundle.algorithm} '
      f'exp {exp_number} P={n_policies} T={args.rollout_length}  '
      f'{n_bins} bins x {args.bin_capacity}  '
      f'max_generations={max_generations}'
  )
  print(f'Saving to {output_path}')
  print(f'Generation shards in {generation_shard_dir(output_path)}')

  rng = jax.random.PRNGKey(experiment_seed(exp_number))
  init_info: dict = {'policy_selection': selection, 'stratified_init': False}
  population = None
  if not args.no_stratified_init:
    rng, population, sampler, stratified_info = stratified_initial_population(
        rng, bundle, n_policies=n_policies
    )
    init_info.update(stratified_info)
    init_info['stratified_init'] = True
    print(
        f'Using sampler noise={sampler.noise_scale:.3f} '
        f'max_dqpos={sampler.max_dqpos:.3f}'
    )

  # Written before the search as well as after it. A run that is interrupted
  # part way through still leaves its generation shards paired with the
  # policy ids they were scored against, which is the one thing
  # add_heldout_fail_rates.py cannot reconstruct from the shards alone.
  _write_metadata(init_info, meta_path)

  collection, bin_info = run_binned_ga_search(
      bundle,
      rng,
      n_policies=n_policies,
      bin_capacity=args.bin_capacity,
      max_generations=max_generations,
      immigrant_fraction=args.immigrant_fraction,
      initial_population=population,
      save_path=output_path,
      save_every=args.save_every,
  )
  save_collection(collection, output_path)
  init_info['bins'] = bin_info
  _write_metadata(init_info, meta_path)
  print(f'Saved {output_path}')
  print(f'Saved {meta_path}')
  summarize_collection(collection)
  print(f'Bin fill: {bin_info["counts"]}  all_full={bin_info["all_full"]}')
  n_unique = bin_info.get('n_unique_states')
  if n_unique is None:
    n_unique = sum(int(count) for count in bin_info['counts'])
  print(
      f'Unique states: {n_unique}  '
      f'duplicates skipped: {bin_info.get("n_duplicate_skipped", 0)}'
  )
  return bin_info


def score_heldout_after_search(
    args: argparse.Namespace,
    *,
    environment: str,
    algorithm: str,
    exp_number: int,
) -> None:
  """Attach held-out fail rates after a search without guiding that search.

  Call this only after the caller has deleted its search bundle. Held-out
  policies are loaded afresh by ``annotate_experiment`` so their outcomes
  cannot affect GA selection or the search-difficulty quotas.
  """
  if args.skip_heldout_eval:
    print(
        'Skipped held-out scoring; run add_heldout_fail_rates.py later.',
        flush=True,
    )
    return
  from ga.env_setup import LOCOMOTION_ENVS

  if environment not in LOCOMOTION_ENVS:
    print(
        f'Automatic held-out scoring is not implemented for {environment}; '
        'the search collection is still complete.',
        flush=True,
    )
    return
  output_path = collection_path(
      environment, algorithm, exp_number, args.rollout_length
  )
  # The search executor and attribution kernels are no longer needed. Clear
  # their cached executables before loading the disjoint evaluation ensemble.
  gc.collect()
  jax.clear_caches()
  from add_heldout_fail_rates import annotate_experiment

  print(
      'Scoring every stored unique case with the held-out evaluation '
      'ensemble (these rates did not guide the search).',
      flush=True,
  )
  annotate_experiment(
      environment,
      algorithm,
      output_path,
      batch=args.eval_batch,
      rollout_length=args.rollout_length,
  )
