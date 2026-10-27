"""GA search that optimizes disagreement features.

Usage (from the repository root):

  python search_for_tests/feature_ga_search.py \\
      ENVIRONMENT ALGORITHM EXP_NUMBER ROLLOUT_LENGTH P \\
      --features +action_disagreement

  python search_for_tests/feature_ga_search.py hopper ppo_sac 1 15 16 \\
      --features +criticality_disagreement --failure-rate-every 25

  python search_for_tests/feature_ga_search.py humanoid ppo 9 15 16 \\
      --features +criticality_disagreement --start-generation 199 \\
      --max-generations 250

``ppo_sac`` is the mixed search ensemble: half the named P from PPO and half
from SAC, alternating top performers within each algorithm. Difficulty is
still the 32-policy evaluation ensemble.

Fitness is the requested disagreement at its absolute value. Several
disagreements are summed in their own units. ``+`` / ``:max`` pushes the
feature up. ``-`` / ``:min`` pushes it down. A disagreement feature is the
ensemble's initial-state output (actions, scales, values, or integrated
gradients).

Generation 0 draws 10000 elites. Every later generation scores those elites
plus 10000 new candidates and keeps the best distinct elites. A candidate
whose initial state is already outside the environment's healthy set is
discarded and is not stored. Every candidate also takes a zero-action
rollout of 3 steps, screened in batches of 10000. A state that leaves the
healthy set on any of those steps is discarded. Hopper and Walker2d also
discard a candidate whose limited joints start more than 0.18 rad past
their stops. The elite set is the highest-fitness distinct states, up
to 10000. A candidate that passes those screens
is rolled out with the search ensemble. One that fails every search
policy is discarded. The generation line reports how many failed the
zero-action rollout and how many failed every search policy, and the run
summary reports the totals. Every
other new initial state is stored once.
A copied elite is not stored a second time.
The search stops when the last 5 generations together replaced fewer than
1000 elites. ``--max-generations`` is only a ceiling.

``ROLLOUT_LENGTH`` is not a search horizon. Every fifth generation, including
generation 0 and the final generation, the current elite suite is scored.
Its search failures are the ones already computed while screening those
elites, and only the shared evaluation ensemble is rolled out again.
That evaluation ensemble is the 16 PPO evaluation
policies together with the 16 SAC evaluation policies, 32 policies, the
same set for every search. Difficulty is the fraction of those 32 that
fail. That suite is the distinct elites, at most 10000, not the
candidates and not every unique state stored during the search. The
failure-rate table also records how many candidates were discarded, how
many tests were replaced, and how many new tests were stored since the
previous row. Each row also records the fraction of elites that failed
exactly 0, 1, ..., P policies, once for the search ensemble and once for
the evaluation ensemble. Those rates do not enter
the fitness. ``--skip-heldout-eval`` leaves the rates blank and still
records the turnover counts.

Each generation archive stores only the kept initial states, ``qpos`` and
``qvel``. Fitness by generation stays in the collection sidecar, and the
suite failure rates stay in the failure-rate table. The collection path is:
  datasets/ga_data/{env}/{alg}/feature_search/{objectives}/rollout_length={T}/experiment_{n}_ga_collection.pkl

The finished suite and its failure rates are written to:
  test_suites/{objectives}/{env}/{alg}/rollout_length={T}/experiment_{n}/suite.pkl
  test_suites/{objectives}/{env}/{alg}/rollout_length={T}/experiment_{n}/failure_rates.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)
if __name__ == '__main__':
  os.chdir(_ROOT)

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Return device memory when a rollout finishes. The default allocator keeps
# its high-water mark, and a later 16-policy humanoid rollout then kills WSL.
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax
import numpy as np

from ga.collect import generation_shard_dir
from ga.env_setup import (
    LOCOMOTION_ENVS,
    make_mixed_search_bundle,
    make_search_bundle,
)
from ga.feature_objectives import (
    feature_catalog,
    measurement_plan,
    parse_feature_specs,
    requirements,
)
from ga.feature_search import (
    DEFAULT_ELITE_REPLACE_LIMIT,
    DEFAULT_ELITE_REPLACE_WINDOW,
    DEFAULT_FAILURE_RATE_EVERY,
    DEFAULT_MAX_GENERATIONS,
    ELITE_STOP,
    SUITE_REPORT,
    append_suite_report,
    feature_collection_path,
    feature_suite_paths,
    load_feature_resume,
    record_failure_count_fractions,
    run_feature_ga_search,
    suite_report_fieldnames,
    write_feature_suite,
)
from ga.search_cli import experiment_seed, init_metadata_path
from policies.ranking import (
    split_search_eval_by_performance,
    split_search_eval_mixed,
)

# The study's difficulty is the union of the two evaluation ensembles.
_SHARED_EVAL_PER_ALGORITHM = 16


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'GA search that maximizes or minimizes disagreement features, '
          'without rolling out the search ensemble.'
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          'Examples:\n'
          '  --features +action_disagreement\n'
          '  --features value_disagreement:min\n\n'
          + feature_catalog()
      ),
  )
  parser.add_argument('environment')
  parser.add_argument(
      'algorithm',
      choices=('ppo', 'sac', 'ppo_sac'),
      help=(
          'Search ensemble. ppo_sac uses P/2 PPO policies and P/2 SAC '
          'policies.'
      ),
  )
  parser.add_argument('exp_number', type=int)
  parser.add_argument('rollout_length', type=int)
  parser.add_argument('n_policies', type=int)
  parser.add_argument(
      '--features',
      nargs='+',
      required=True,
      metavar='FEATURE',
      help=(
          'One or more fitness features. Each needs a direction: '
          '+name, -name, name:max, or name:min. Disagreements are scored '
          'at their absolute values. '
          + feature_catalog()
      ),
  )
  parser.add_argument(
      '--elite-replace-limit',
      type=int,
      default=DEFAULT_ELITE_REPLACE_LIMIT,
      help=(
          'Stop when the last --elite-replace-window generations together '
          'replaced fewer than this many elites '
          f'(default: {DEFAULT_ELITE_REPLACE_LIMIT}).'
      ),
  )
  parser.add_argument(
      '--elite-replace-window',
      type=int,
      default=DEFAULT_ELITE_REPLACE_WINDOW,
      help=(
          'How many generations of elite replacement to add up '
          f'(default: {DEFAULT_ELITE_REPLACE_WINDOW}).'
      ),
  )
  parser.add_argument(
      '--failure-rate-every',
      type=int,
      default=DEFAULT_FAILURE_RATE_EVERY,
      help=(
          'Roll out the current elite suite every N generations, including '
          'generation 0 and the final generation. The search ensemble '
          'supplies the search fail rate. Difficulty is the fail rate of '
          'the 32-policy evaluation ensemble (16 PPO and 16 SAC evaluation '
          'policies). The suite is the distinct elites, at most 10000. The '
          'CSV records those mean failure rates and the turnover since the '
          f'previous row (default: {DEFAULT_FAILURE_RATE_EVERY}).'
      ),
  )
  parser.add_argument(
      '--start-generation',
      type=int,
      default=None,
      help=(
          'Continue an existing run after this generation. It has to be '
          'the last generation stored in the collection checkpoint, and '
          'that generation\'s state shard has to exist. The saved '
          'population is scored again so selection can continue, then the '
          'search proceeds at the next generation. Existing shards and the '
          'failure-rate table are kept. --max-generations is still the '
          'ceiling, counting from generation 0.'
      ),
  )
  parser.add_argument(
      '--max-generations',
      type=int,
      default=None,
      help=(
          'Ceiling on generations, including generation 0. The search '
          'stops earlier once fewer than --elite-replace-limit elites '
          'were replaced across the last --elite-replace-window generations '
          f'(default: {DEFAULT_MAX_GENERATIONS}).'
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
      help='Rewrite the sidecar pickle every N generations (0 disables).',
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help=(
          'Replace an existing experiment, including its suite and '
          'failure-rate table.'
      ),
  )
  parser.add_argument(
      '--skip-heldout-eval',
      action='store_true',
      help=(
          'Do not roll out the search or evaluation ensembles. The suite '
          'and the CSV are still written; the failure-rate columns are '
          'left blank.'
      ),
  )
  parser.add_argument(
      '--allow-unequal-eval',
      action='store_true',
      help=(
          'Allow fewer held-out than search policies. This flag does not '
          'allow overlap or duplicate ids.'
      ),
  )
  parser.add_argument(
      '--eval-batch',
      type=int,
      default=None,
      help=(
          'Cases rolled out together by one 16-policy ensemble. Humanoid '
          'defaults to 2000 because 6000 cases across 16 policies exhausted '
          'the 16 GB WSL limit. Other environments default to 6000.'
      ),
  )
  return parser.parse_args()


def select_policy_ensembles(environment: str, algorithm: str, n_policies: int):
  """Same split as binned_ga_search.select_policy_ensembles."""
  search_ids, eval_ids, assigned = split_search_eval_by_performance(
      environment, algorithm, n_policies
  )
  method = 'alternate_highest_performance'
  metric = 'best_eval_episode_reward'
  return search_ids, eval_ids, {
      'method': method,
      'metric': metric,
      'search_policy_ids': search_ids,
      'eval_policy_ids': eval_ids,
      'assigned': assigned,
  }


def select_mixed_policy_ensembles(environment: str, n_policies: int):
  """PPO+SAC search set: n_policies/2 from each algorithm."""
  per_algorithm = n_policies // 2
  search_members, eval_members, assigned = split_search_eval_mixed(
      environment, per_algorithm
  )
  return search_members, eval_members, {
      'method': 'alternate_highest_performance_by_algorithm',
      'metric': 'best_eval_episode_reward',
      'per_algorithm': int(per_algorithm),
      'search': search_members,
      'eval': eval_members,
      'assigned': assigned,
  }


def _member_label(member: dict) -> str:
  return f'{member["algorithm"]}:{int(member["policy_id"])}'


def shared_difficulty_ensemble(
    environment: str,
) -> tuple[list[int], list[int]]:
  """PPO and SAC evaluation policies that define difficulty.

  Each list is the held-out half of that algorithm's alternating split at
  ``_SHARED_EVAL_PER_ALGORITHM``. Together they are the 32-policy set used
  for every search in the environment.
  """
  members = {}
  for algorithm in ('ppo', 'sac'):
    _search_ids, eval_ids, _assigned = split_search_eval_by_performance(
        environment, algorithm, _SHARED_EVAL_PER_ALGORITHM
    )
    if len(eval_ids) != _SHARED_EVAL_PER_ALGORITHM:
      raise ValueError(
          f'{environment}/{algorithm} evaluation ensemble has '
          f'{len(eval_ids)} policies, expected {_SHARED_EVAL_PER_ALGORITHM}.'
      )
    members[algorithm] = [int(policy_id) for policy_id in eval_ids]
  return members['ppo'], members['sac']


def validate_policy_ensembles(
    search_ids: list[int],
    eval_ids: list[int],
    *,
    n_policies: int,
    allow_unequal_eval: bool,
) -> None:
  if len(search_ids) != n_policies:
    raise ValueError(
        f'Expected {n_policies} search policies, got {len(search_ids)}.'
    )
  if len(set(search_ids)) != len(search_ids):
    raise ValueError(f'Search ensemble contains duplicate ids: {search_ids}')
  if len(set(eval_ids)) != len(eval_ids):
    raise ValueError(f'Eval ensemble contains duplicate ids: {eval_ids}')
  overlap = sorted(set(search_ids) & set(eval_ids))
  if overlap:
    raise ValueError(f'Search and eval ensembles overlap: {overlap}')
  if len(eval_ids) != n_policies and not allow_unequal_eval:
    raise ValueError(
        f'Need {n_policies} held-out policies to match the search ensemble; '
        f'found {len(eval_ids)}. Add another usable checkpoint, reduce P, '
        'or pass --allow-unequal-eval to record this exception explicitly.'
    )


def _validate(args: argparse.Namespace) -> None:
  if args.rollout_length < 1:
    raise ValueError(f'rollout_length must be >= 1, got {args.rollout_length}')
  if args.n_policies < 2:
    raise ValueError(f'n_policies must be >= 2, got {args.n_policies}')
  if args.algorithm == 'ppo_sac' and args.n_policies % 2:
    raise ValueError(
        'ppo_sac splits P evenly between PPO and SAC, so n_policies must '
        f'be even, got {args.n_policies}'
    )
  if args.elite_replace_limit < 1:
    raise ValueError(
        'elite_replace_limit must be >= 1, got '
        f'{args.elite_replace_limit}'
    )
  if args.elite_replace_window < 1:
    raise ValueError(
        'elite_replace_window must be >= 1, got '
        f'{args.elite_replace_window}'
    )
  if args.failure_rate_every < 1:
    raise ValueError(
        f'failure_rate_every must be >= 1, got {args.failure_rate_every}'
    )
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
  if args.start_generation is not None and args.start_generation < 0:
    raise ValueError(
        f'start_generation must be >= 0, got {args.start_generation}'
    )
  if (
      args.start_generation is not None
      and args.max_generations is not None
      and args.start_generation + 1 >= args.max_generations
  ):
    raise ValueError(
        f'start_generation {args.start_generation} is already at the '
        f'max_generations ceiling {args.max_generations}. Raise '
        '--max-generations to continue past that generation.'
    )
  if args.eval_batch < 1:
    raise ValueError(f'eval_batch must be >= 1, got {args.eval_batch}')


def _write_metadata(payload: dict, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w') as handle:
    json.dump(payload, handle, indent=2)


def _rates_ready_to_continue(rates_path: str, generation: int) -> None:
  """The failure-rate table may end at this generation, and not past it."""
  if not os.path.isfile(rates_path):
    return
  last = None
  with open(rates_path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      if row.get('generation') in (None, ''):
        continue
      last = int(row['generation'])
  if last is None:
    return
  if last > generation:
    raise FileExistsError(
        f'{rates_path} already records generation {last}, which is past '
        f'the start generation {generation}.'
    )


def _existing_output(path: str, suite_path: str, rates_path: str) -> list[str]:
  meta_path = init_metadata_path(path)
  shard_path = generation_shard_dir(path)
  return [
      candidate
      for candidate in (path, meta_path, shard_path, suite_path, rates_path)
      if os.path.exists(candidate)
  ]


def _prepare_output(
    path: str,
    suite_path: str,
    rates_path: str,
    overwrite: bool,
) -> None:
  meta_path = init_metadata_path(path)
  shard_path = generation_shard_dir(path)
  existing = _existing_output(path, suite_path, rates_path)
  if existing and not overwrite:
    raise FileExistsError(
        f'Experiment output already exists: {", ".join(existing)}. '
        'Pass --overwrite to replace the collection, the suite, and the '
        'failure-rate table.'
    )
  if overwrite:
    for candidate in (path, meta_path, suite_path, rates_path):
      if os.path.isfile(candidate):
        os.remove(candidate)
    if os.path.isdir(shard_path):
      shutil.rmtree(shard_path)


def default_eval_batch(environment: str) -> int:
  """Case batch for one 16-policy rollout.

  Humanoid's simultaneous rollout at 6000 cases compiled and ran, then
  killed the 16 GB WSL VM on a later suite measurement. 2000 cases keeps
  that rollout's footprint three times smaller.
  """
  if environment == 'humanoid':
    return 2000
  return 6_000


def main() -> int:
  args = parse_args()
  if args.eval_batch is None:
    args.eval_batch = default_eval_batch(args.environment)
  _validate(args)
  objectives = parse_feature_specs(args.features)
  output_path = feature_collection_path(
      args.environment,
      args.algorithm,
      args.exp_number,
      args.rollout_length,
      objectives,
  )
  suite_path, rates_path = feature_suite_paths(
      args.environment,
      args.algorithm,
      args.exp_number,
      args.rollout_length,
      objectives,
  )
  resume = None
  if args.start_generation is not None:
    resume = load_feature_resume(output_path, args.start_generation)
    _rates_ready_to_continue(rates_path, int(resume.generation))
    print(
        f'Continuing from generation {resume.generation} '
        f'({len(resume.seen)} unique states).',
        flush=True,
    )
    if args.overwrite:
      print(
          '--start-generation keeps the existing collection, shards, '
          'and failure-rate table.',
          flush=True,
      )
  elif not args.overwrite:
    existing = _existing_output(output_path, suite_path, rates_path)
    if existing:
      print(
          f'Skipping {args.environment}/{args.algorithm} '
          f'experiment {args.exp_number}: output already exists '
          f'({", ".join(existing)}).',
          flush=True,
      )
      return 0
  needs = requirements(
      objectives, discrete_actions=False
  )
  needs_policies = bool(needs & {'policy', 'ig', 'gait'})
  mixed = args.algorithm == 'ppo_sac'
  search_members = None
  if mixed:
    search_members, eval_members, selection = select_mixed_policy_ensembles(
        args.environment, args.n_policies
    )
    search_ids = [int(member['policy_id']) for member in search_members]
    eval_ids = [int(member['policy_id']) for member in eval_members]
  else:
    search_ids, eval_ids, selection = select_policy_ensembles(
        args.environment, args.algorithm, args.n_policies
    )
    validate_policy_ensembles(
        search_ids,
        eval_ids,
        n_policies=args.n_policies,
        allow_unequal_eval=args.allow_unequal_eval,
    )
  ppo_eval_ids = sac_eval_ids = None
  if args.environment in LOCOMOTION_ENVS:
    ppo_eval_ids, sac_eval_ids = shared_difficulty_ensemble(args.environment)
  if mixed:
    per_algorithm = args.n_policies // 2
    print(
        f'Mixed search ensemble ({per_algorithm} PPO + {per_algorithm} SAC, '
        f'P={args.n_policies}): '
        + ' '.join(_member_label(member) for member in search_members)
    )
  else:
    print(
        f'Policy ensembles (P={args.n_policies}): '
        f'search={search_ids}  matched eval={eval_ids}'
    )
  if ppo_eval_ids is not None:
    print(
        'Difficulty ensemble: '
        f'{len(ppo_eval_ids)} PPO eval + {len(sac_eval_ids)} SAC eval '
        f'= {len(ppo_eval_ids) + len(sac_eval_ids)} policies'
    )
  print(f'Held-out rollout batch: {args.eval_batch} cases', flush=True)
  if not needs_policies:
    print(
        'State features only: search checkpoints are not loaded for '
        'selection.'
    )

  policy_fields, attribution_components = measurement_plan(
      objectives, discrete_actions=False
  )
  if mixed:
    bundle = make_mixed_search_bundle(
        args.environment,
        args.rollout_length,
        search_members,
        eval_members,
        with_attributions='ig' in needs,
        with_executor=False,
        with_policy_outputs=bool(needs & {'policy', 'gait'}),
        policy_output_fields=policy_fields or None,
        attribution_components=attribution_components or None,
    )
  else:
    bundle = make_search_bundle(
        args.environment,
        args.algorithm,
        args.n_policies,
        args.rollout_length,
        with_attributions='ig' in needs,
        with_executor=False,
        with_policies=needs_policies,
        with_policy_outputs=bool(needs & {'policy', 'gait'}),
        policy_ids=search_ids,
        policy_output_fields=policy_fields or None,
        attribution_components=attribution_components or None,
    )
  bundle.eval_policy_ids = eval_ids
  if resume is None:
    _prepare_output(output_path, suite_path, rates_path, args.overwrite)
  meta_path = init_metadata_path(output_path)
  metadata = {
      'search_kind': 'feature',
      'executed_search_tests': bool(needs_policies),
      'unique_states': True,
      'rollout_length_note': (
          'rollout_length is the horizon of the search-ensemble rollout. '
          'Every candidate that passes the health screens is rolled out, '
          'and a candidate that fails every search policy is discarded. '
          'The suite report reuses those search failures for the elites. '
          'Every failure_rate_every generations it rolls only the '
          'evaluation ensemble, on the same horizon. That suite is the '
          'distinct elites, at most 10000. Difficulty is the fail rate of '
          'the 32-policy evaluation ensemble. The failure rate is not the '
          'fitness.'
      ),
      'difficulty_ensemble': None if ppo_eval_ids is None else {
          'n_policies': len(ppo_eval_ids) + len(sac_eval_ids),
          'ppo_eval_policy_ids': ppo_eval_ids,
          'sac_eval_policy_ids': sac_eval_ids,
      },
      'failure_rate_every': int(args.failure_rate_every),
      'suite_report': SUITE_REPORT,
      'rollout_length': int(args.rollout_length),
      'elite_replace_limit': int(args.elite_replace_limit),
      'elite_replace_window': int(args.elite_replace_window),
      'elite_stop': ELITE_STOP,
      'features': [
          {'name': item.name, 'direction': item.direction_name}
          for item in objectives
      ],
      'policy_selection': selection,
  }
  _write_metadata(metadata, meta_path)
  print(f'Saving to {output_path}')

  rng = jax.random.PRNGKey(experiment_seed(args.exp_number))
  reporter = None
  on_suite_checkpoint = None
  if bundle.environment in LOCOMOTION_ENVS:
    reporter = _SuiteReporter(
        args,
        bundle.environment,
        bundle.algorithm,
        search_ids,
        ppo_eval_ids,
        sac_eval_ids,
        rates_path,
        search_members=search_members,
    )
    on_suite_checkpoint = reporter.record
    if resume is not None and os.path.isfile(rates_path):
      reporter._header = False
  screen_search_all_fail = None
  if reporter is not None and needs_policies:
    screen_search_all_fail = reporter.search_all_fail
  try:
    _collection, info = run_feature_ga_search(
        bundle,
        rng,
        objectives,
        elite_replace_limit=args.elite_replace_limit,
        elite_replace_window=args.elite_replace_window,
        failure_rate_every=args.failure_rate_every,
        max_generations=args.max_generations,
        immigrant_fraction=args.immigrant_fraction,
        save_path=output_path,
        save_every=args.save_every,
        on_suite_checkpoint=on_suite_checkpoint,
        screen_search_all_fail=screen_search_all_fail,
        resume=resume,
    )
  finally:
    if reporter is not None:
      reporter.close()
  metadata['feature_search'] = {
      key: value for key, value in info.items() if key != 'shard_dir'
  }
  _write_metadata(metadata, meta_path)
  print(
      f'Elites: {info["n_elites"]}/{info["elite_count"]}  '
      f'unique states: {info["n_unique_states"]}  '
      f'duplicates skipped: {info["n_duplicate_skipped"]}  '
      f'zero_action_fails={info["zero_action_fails"]}  '
      f'search_all_fails={info["search_all_fails"]}  '
      f'stopped_because={info["stopped_because"]}  '
      f'elite_replacements_in_window={info["elite_replace_window_total"]}'
  )

  environment = bundle.environment
  algorithm = bundle.algorithm
  del bundle
  gc.collect()
  jax.clear_caches()
  _export_suite(
      environment,
      suite_path,
      rates_path,
      metadata,
      meta_path,
      None if reporter is None else reporter.qpos,
      None if reporter is None else reporter.qvel,
      None if reporter is None else reporter.policy_rng,
      args.skip_heldout_eval,
  )
  return 0


def _case_keys(qpos, qvel, policy_rng) -> list[bytes]:
  """Identity of each case from its qpos, qvel, and policy_rng."""
  arrays = [np.asarray(qpos), np.asarray(qvel), np.asarray(policy_rng)]
  n_cases = int(arrays[0].shape[0])
  packed = []
  for array in arrays:
    if int(array.shape[0]) != n_cases:
      raise ValueError(
          f'Case arrays disagree on length: {int(array.shape[0])} vs {n_cases}.'
      )
    flat = np.ascontiguousarray(array.reshape(n_cases, -1))
    width = int(flat.dtype.itemsize * flat.shape[1])
    packed.append(flat.view(np.dtype((np.void, width))).reshape(n_cases))
  return [
      b''.join(bytes(part[index]) for part in packed)
      for index in range(n_cases)
  ]


class _SuiteReporter:
  """Append a CSV row each time the search measures the current suite."""

  def __init__(
      self, args, environment, algorithm, search_ids, ppo_eval_ids, sac_eval_ids,
      rates_path, search_members=None,
  ):
    self.args = args
    self.environment = environment
    self.algorithm = algorithm
    self.search_ids = list(search_ids)
    self.search_members = search_members
    self.ppo_eval_ids = list(ppo_eval_ids)
    self.sac_eval_ids = list(sac_eval_ids)
    self.rates_path = rates_path
    self.qpos = None
    self.qvel = None
    self.policy_rng = None
    self._header = True
    self._columns = None
    self._executors = []
    self._heldout = None
    self._search_failed = None
    self._search_index = None

  def record(self, report: dict, qpos, qvel, policy_rng) -> None:
    """Score the current suite, then append the row."""
    self.qpos = np.array(qpos, copy=True)
    self.qvel = np.array(qvel, copy=True)
    self.policy_rng = np.array(policy_rng, copy=True)
    n_cases = int(self.qpos.shape[0])
    if n_cases > 10_000 or n_cases != int(report['n_cases']):
      raise RuntimeError(
          f'Checkpoint asked for {report["n_cases"]} cases, but the current '
          f'suite has {n_cases}.'
      )
    if not self.args.skip_heldout_eval:
      search_failed, eval_failed = self._score(
          self.qpos, self.qvel, self.policy_rng
      )
      report['search_fail_rate'] = float(np.mean(search_failed))
      report['eval_fail_rate'] = float(np.mean(eval_failed))
      record_failure_count_fractions(report, search_failed, eval_failed)
    if self._columns is None:
      self._columns = suite_report_fieldnames(report)
    append_suite_report(
        self.rates_path,
        report,
        header=self._header,
        columns=self._columns,
    )
    self._header = False

  def close(self) -> None:
    self._executors.clear()
    self._search_failed = None
    self._search_index = None
    gc.collect()
    jax.clear_caches()

  def search_failures(self, qpos, qvel, policy_rng) -> np.ndarray:
    """Failure matrix of every search policy. Shape ``(P, N)``."""
    from test_case_execution.types import TestSuite

    self._ensure_search_executor()
    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    policy_rng = np.asarray(policy_rng)
    n_cases = int(qpos.shape[0])
    print(f'  search screen  candidates={n_cases}', flush=True)
    failed = self._heldout(
        self._executors[0],
        TestSuite(qpos=qpos, qvel=qvel, policy_rng=policy_rng),
        self.args.eval_batch,
    )
    failed = np.asarray(failed, dtype=bool)
    if failed.ndim != 2 or int(failed.shape[1]) != n_cases or int(failed.shape[0]) < 1:
      raise RuntimeError(
          f'Search failure matrix has shape {failed.shape}, '
          f'expected (P, {n_cases}) with P >= 1.'
      )
    self._remember_search_failures(qpos, qvel, policy_rng, failed)
    return failed

  def store_search_failures(self, qpos, qvel, policy_rng, failed) -> None:
    """Remember a failure matrix that was already rolled out."""
    qpos = np.asarray(qpos)
    failed = np.asarray(failed, dtype=bool)
    n_cases = int(qpos.shape[0])
    if (
        failed.ndim != 2
        or int(failed.shape[1]) != n_cases
        or int(failed.shape[0]) < 1
    ):
      raise RuntimeError(
          f'Search failure matrix has shape {failed.shape}, '
          f'expected (P, {n_cases}) with P >= 1.'
      )
    self._remember_search_failures(qpos, qvel, policy_rng, failed)

  def search_all_fail(self, qpos, qvel, policy_rng) -> np.ndarray:
    """True when every search policy fails that candidate."""
    return np.all(self.search_failures(qpos, qvel, policy_rng), axis=0)

  def _remember_search_failures(self, qpos, qvel, policy_rng, failed) -> None:
    """Keep this generation's search failures so the suite report can reuse them."""
    failed = np.asarray(failed, dtype=bool)
    keys = _case_keys(qpos, qvel, policy_rng)
    self._search_failed = failed
    self._search_index = {key: index for index, key in enumerate(keys)}

  def _cached_search_failures(self, qpos, qvel, policy_rng):
    """Search failure columns for these elites, or None if no screen ran."""
    if self._search_failed is None or self._search_index is None:
      return None
    keys = _case_keys(qpos, qvel, policy_rng)
    try:
      columns = [self._search_index[key] for key in keys]
    except KeyError:
      raise RuntimeError(
          'An elite has no search-screen failure row from this generation.'
      ) from None
    return np.ascontiguousarray(self._search_failed[:, columns])

  def _ensure_search_executor(self) -> None:
    from add_heldout_fail_rates import _heldout_failures

    if self._executors:
      if self._heldout is None:
        self._heldout = _heldout_failures
      return
    if self.search_members is not None:
      self._executors.append(
          _load_mixed_executor(
              self.environment, self.search_members, self.args.rollout_length,
          )
      )
    else:
      self._executors.append(
          _load_executor(
              self.environment, self.algorithm, self.search_ids,
              self.args.rollout_length,
          )
      )
    self._heldout = _heldout_failures

  def _ensure_eval_executors(self) -> None:
    self._ensure_search_executor()
    if len(self._executors) >= 3:
      return
    self._executors.append(
        _load_executor(
            self.environment, 'ppo', self.ppo_eval_ids,
            self.args.rollout_length,
        )
    )
    self._executors.append(
        _load_executor(
            self.environment, 'sac', self.sac_eval_ids,
            self.args.rollout_length,
        )
    )

  def _score(self, qpos, qvel, policy_rng):
    from test_case_execution.types import TestSuite

    self._ensure_eval_executors()
    suite = TestSuite(
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        policy_rng=np.asarray(policy_rng),
    )
    search_failed = self._cached_search_failures(qpos, qvel, policy_rng)
    if search_failed is None:
      search_failed = self._heldout(
          self._executors[0], suite, self.args.eval_batch
      )
      gc.collect()
    else:
      print(
          f'  search failures reused  elites={int(search_failed.shape[1])}',
          flush=True,
      )
    ppo_failed = self._heldout(
        self._executors[1], suite, self.args.eval_batch
    )
    gc.collect()
    sac_failed = self._heldout(
        self._executors[2], suite, self.args.eval_batch
    )
    gc.collect()
    if ppo_failed.shape[1] != sac_failed.shape[1]:
      raise RuntimeError(
          f'PPO eval mask {ppo_failed.shape} and SAC eval mask '
          f'{sac_failed.shape} cover different cases.'
      )
    eval_failed = np.concatenate([ppo_failed, sac_failed], axis=0)
    n_cases = int(np.asarray(qpos).shape[0])
    if (
        int(search_failed.shape[1]) != n_cases
        or int(eval_failed.shape[1]) != n_cases
    ):
      raise RuntimeError(
          f'Search failures {search_failed.shape} and evaluation failures '
          f'{eval_failed.shape} do not both cover {n_cases} elites.'
      )
    return search_failed, eval_failed


def _load_mixed_executor(environment, search_members, rollout_length):
  """Failure matrix ``(P, N)`` for the mixed PPO-then-SAC search ensemble."""
  bundle = make_mixed_search_bundle(
      environment,
      rollout_length,
      search_members,
      with_attributions=False,
      with_executor=True,
      with_policy_outputs=False,
  )
  execute = bundle.parallel_execute_fn
  if execute is None:
    raise RuntimeError(f'{environment} has no mixed policy executor.')
  return execute


def _load_executor(environment, algorithm, policy_ids, rollout_length):
  """Failure matrix ``(P, N)`` for one ensemble. The closure owns the policies."""
  bundle = make_search_bundle(
      environment,
      algorithm,
      len(policy_ids),
      rollout_length,
      with_attributions=False,
      with_executor=True,
      with_policies=True,
      with_policy_outputs=False,
      policy_ids=policy_ids,
  )
  execute = bundle.parallel_execute_fn
  if execute is None:
    raise RuntimeError(f'{environment} has no simultaneous policy executor.')
  return execute


def _export_suite(
    environment: str,
    suite_path: str,
    rates_path: str,
    metadata: dict,
    meta_path: str,
    qpos,
    qvel,
    policy_rng,
    skipped_rollouts: bool,
) -> None:
  """Write the last measured suite. Its failure rates are already in the CSV."""
  if environment not in LOCOMOTION_ENVS:
    print(
        f'Suite export is not implemented for {environment}; '
        'the search collection is still complete.',
        flush=True,
    )
    return
  if qpos is None or qvel is None or policy_rng is None:
    raise RuntimeError('No current suite was recorded.')
  write_feature_suite(suite_path, qpos, qvel, policy_rng)
  print(f'Suite: {suite_path}  n_cases={int(np.asarray(qpos).shape[0])}', flush=True)
  print(f'Failure rates: {rates_path}', flush=True)
  metadata['test_suite'] = suite_path
  metadata['failure_rates'] = rates_path
  _write_metadata(metadata, meta_path)
  if skipped_rollouts:
    print(
        'Failure-rate columns were left blank because the ensembles were '
        'not rolled out.',
        flush=True,
    )


if __name__ == '__main__':
  raise SystemExit(main())
