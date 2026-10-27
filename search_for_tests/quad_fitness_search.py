"""Four feature searches that share generation 0 and then diverge.

Usage (from the repository root):

  python search_for_tests/quad_fitness_search.py \\
      ENVIRONMENT ALGORITHM EXP_NUMBER ROLLOUT_LENGTH P \\
      --features +criticality_disagreement

The experiment number is the random seed (42 + experiment number). That
seed draws one generation-0 population. All four searches score it.
From generation 1 each search breeds only from its own population, using
its own fitness. The tournament key and the immigrant genomes are still
drawn once per generation from that seed, and applied to every search.

``--continue`` resumes each search from the last generation stored in
its collection. A search that already reached ``--max-generations`` is
left in place. The elite-replacement window starts again.

The four fitnesses are the requested disagreement, the search-ensemble
fail rate, the negation of the best search policy's value, and that
policy's action entropy. Every candidate that passes the health screens
is rolled out with every search policy. A candidate that fails all of
them is discarded.

Outputs, one directory per fitness:

  datasets/ga_data/{env}/{alg}/quad_search/{slug}/rollout_length={T}/experiment_{n}_ga_collection.pkl
  test_suites/quad_search/{slug}/{env}/{alg}/rollout_length={T}/experiment_{n}/suite.pkl
  test_suites/quad_search/{slug}/{env}/{alg}/rollout_length={T}/experiment_{n}/failure_rates.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)
if __name__ == '__main__':
  os.chdir(_ROOT)

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax

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
from ga.collect import load_collection
from ga.feature_search import (
    DEFAULT_ELITE_REPLACE_LIMIT,
    DEFAULT_ELITE_REPLACE_WINDOW,
    DEFAULT_FAILURE_RATE_EVERY,
    DEFAULT_MAX_GENERATIONS,
    ELITE_STOP,
    SUITE_REPORT,
    load_feature_resume,
)
from ga.quad_fitness_search import (
    LANE_KINDS,
    LANE_NAMES,
    QuadLane,
    quad_collection_path,
    quad_slugs,
    quad_suite_paths,
    run_quad_fitness_search,
)
from ga.search_cli import experiment_seed, init_metadata_path
from search_for_tests.feature_ga_search import (
    _SuiteReporter,
    _existing_output,
    _export_suite,
    _member_label,
    _prepare_output,
    _rates_ready_to_continue,
    _write_metadata,
    default_eval_batch,
    select_mixed_policy_ensembles,
    select_policy_ensembles,
    shared_difficulty_ensemble,
    validate_policy_ensembles,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Four GA searches from one generation-0 population. Fitness is '
          'the only difference after that generation.'
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog='Disagreement features: ' + feature_catalog(),
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
          'Disagreement feature for the first search. Each entry needs a '
          'direction: +name, -name, name:max, or name:min. The other three '
          'searches use search difficulty, negative best-policy value, and '
          'that policy\'s action entropy. ' + feature_catalog()
      ),
  )
  parser.add_argument(
      '--elite-replace-limit',
      type=int,
      default=DEFAULT_ELITE_REPLACE_LIMIT,
      help=(
          'Stop a search when its last --elite-replace-window generations '
          'together replaced fewer than this many elites '
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
          'Roll out each search\'s elite suite every N generations. '
          f'Default: {DEFAULT_FAILURE_RATE_EVERY}.'
      ),
  )
  parser.add_argument(
      '--max-generations',
      type=int,
      default=None,
      help=(
          'Ceiling on generations, including generation 0 '
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
      help='Rewrite each sidecar pickle every N generations (0 disables).',
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help='Replace an existing quad experiment, including all four suites.',
  )
  parser.add_argument(
      '--continue',
      dest='continue_run',
      action='store_true',
      help=(
          'Continue each search from its last saved generation. A search '
          'already at --max-generations is left in place. Existing suites '
          'and failure-rate tables are kept.'
      ),
  )
  parser.add_argument(
      '--skip-heldout-eval',
      action='store_true',
      help=(
          'Skip the 32-policy evaluation rollout. The search ensemble is '
          'still rolled out, because solvability and the difficulty '
          'fitness come from that rollout.'
      ),
  )
  parser.add_argument(
      '--allow-unequal-eval',
      action='store_true',
      help='Allow fewer held-out than search policies.',
  )
  parser.add_argument(
      '--eval-batch',
      type=int,
      default=None,
      help='Cases rolled out together by one 16-policy ensemble.',
  )
  return parser.parse_args()


def _saved_generation(path: str) -> int:
  """Last generation stored in a quad collection pickle."""
  if not os.path.isfile(path):
    raise FileNotFoundError(f'No checkpoint at {path}.')
  collection = load_collection(path, load_shards=False)
  generations = collection.search_result.history.generation
  if int(len(generations)) < 1:
    raise FileNotFoundError(f'{path} has no saved generations.')
  return int(generations[-1])


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
        'immigrant_fraction must be in [0, 1), got '
        f'{args.immigrant_fraction}'
    )
  if args.save_every < 0:
    raise ValueError(f'save_every must be >= 0, got {args.save_every}')
  if args.eval_batch < 1:
    raise ValueError(f'eval_batch must be >= 1, got {args.eval_batch}')
  if args.exp_number < 0:
    raise ValueError(f'exp_number must be >= 0, got {args.exp_number}')


def main() -> int:
  args = parse_args()
  if args.eval_batch is None:
    args.eval_batch = default_eval_batch(args.environment)
  _validate(args)
  if args.environment not in LOCOMOTION_ENVS:
    raise ValueError(
        f'{args.environment} has no locomotion state for this search.'
    )
  objectives = parse_feature_specs(args.features)
  slugs = quad_slugs(objectives)
  outputs = []
  for kind in LANE_KINDS:
    collection = quad_collection_path(
        args.environment,
        args.algorithm,
        args.exp_number,
        args.rollout_length,
        slugs[kind],
    )
    suite_path, rates_path = quad_suite_paths(
        args.environment,
        args.algorithm,
        args.exp_number,
        args.rollout_length,
        slugs[kind],
    )
    outputs.append((kind, collection, suite_path, rates_path))
  resumes = None
  if args.continue_run:
    if args.overwrite:
      print(
          '--continue keeps the existing collections and failure-rate tables.',
          flush=True,
      )
    ceiling = (
        int(args.max_generations)
        if args.max_generations is not None
        else int(DEFAULT_MAX_GENERATIONS)
    )
    resumes = {}
    for kind, collection, _suite_path, rates_path in outputs:
      last = _saved_generation(collection)
      _rates_ready_to_continue(rates_path, last)
      if last + 1 >= ceiling:
        resumes[kind] = None
        print(
            f'{slugs[kind]}: generation {last} is already at the ceiling',
            flush=True,
        )
      else:
        resumes[kind] = load_feature_resume(collection, last)
        print(
            f'{slugs[kind]}: resume from generation {last}',
            flush=True,
        )
    if all(item is None for item in resumes.values()):
      print(
          f'{args.environment}/{args.algorithm} experiment {args.exp_number}: '
          'every search is already at the generation ceiling.',
          flush=True,
      )
      return 0
  elif not args.overwrite:
    existing = []
    for _kind, collection, suite_path, rates_path in outputs:
      existing.extend(_existing_output(collection, suite_path, rates_path))
    if existing:
      print(
          f'Skipping {args.environment}/{args.algorithm} '
          f'experiment {args.exp_number}: output already exists '
          f'({", ".join(existing)}).',
          flush=True,
      )
      return 0
    for _kind, collection, suite_path, rates_path in outputs:
      _prepare_output(collection, suite_path, rates_path, args.overwrite)
  else:
    for _kind, collection, suite_path, rates_path in outputs:
      _prepare_output(collection, suite_path, rates_path, args.overwrite)

  seed = experiment_seed(args.exp_number)
  if args.continue_run:
    print(
        f'Random seed {seed} from experiment {args.exp_number}. '
        'Continued generations draw their tournament key and immigrants '
        'from this seed folded with the generation number.',
        flush=True,
    )
  else:
    print(
        f'Random seed {seed} from experiment {args.exp_number}. '
        'Generation 0 is drawn from this seed. Each later generation draws '
        'one tournament key and one immigrant population from it.',
        flush=True,
    )
  needs = requirements(
      objectives, discrete_actions=False
  ) | frozenset({'policy'})
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
  print(
      'Difficulty ensemble: '
      f'{len(ppo_eval_ids)} PPO eval + {len(sac_eval_ids)} SAC eval '
      f'= {len(ppo_eval_ids) + len(sac_eval_ids)} policies'
  )
  print(
      f'Best search policy for value and action entropy: {search_ids[0]}',
      flush=True,
  )
  policy_fields, attribution_components = measurement_plan(
      objectives, discrete_actions=False
  )
  policy_fields = frozenset(policy_fields) | frozenset({
      'actions', 'action_scale', 'values', 'q_values',
  })
  if mixed:
    bundle = make_mixed_search_bundle(
        args.environment,
        args.rollout_length,
        search_members,
        eval_members,
        with_attributions='ig' in needs,
        with_executor=False,
        with_policy_outputs=True,
        policy_output_fields=policy_fields,
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
        with_policies=True,
        with_policy_outputs=True,
        policy_ids=search_ids,
        policy_output_fields=policy_fields,
        attribution_components=attribution_components or None,
    )
  bundle.eval_policy_ids = eval_ids
  reporters = []
  lanes = []
  shared_executors = []
  for kind, collection, _suite_path, rates_path in outputs:
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
    reporter._executors = shared_executors
    if args.continue_run and os.path.isfile(rates_path):
      reporter._header = False
    reporters.append(reporter)
    lanes.append(
        QuadLane(
            name=LANE_NAMES[kind],
            slug=slugs[kind],
            kind=kind,
            save_path=collection,
            search_failures=reporter.search_failures,
            store_search_failures=reporter.store_search_failures,
            on_suite_checkpoint=reporter.record,
        )
    )
  metadata = {
      'search_kind': 'quad_fitness',
      'rng_seed': int(seed),
      'experiment_number': int(args.exp_number),
      'population': (
          'Generation 0 is shared. From generation 1 each fitness breeds '
          'its own population.'
      ),
      'elite_stop': ELITE_STOP,
      'suite_report': SUITE_REPORT,
      'rollout_length': int(args.rollout_length),
      'failure_rate_every': int(args.failure_rate_every),
      'elite_replace_limit': int(args.elite_replace_limit),
      'elite_replace_window': int(args.elite_replace_window),
      'features': [
          {'name': item.name, 'direction': item.direction_name}
          for item in objectives
      ],
      'fitnesses': [
          {'kind': kind, 'slug': slugs[kind], 'name': LANE_NAMES[kind]}
          for kind in LANE_KINDS
      ],
      'best_policy_id': int(search_ids[0]),
      'best_policy_index': 0,
      'policy_selection': selection,
      'difficulty_ensemble': {
          'n_policies': len(ppo_eval_ids) + len(sac_eval_ids),
          'ppo_eval_policy_ids': ppo_eval_ids,
          'sac_eval_policy_ids': sac_eval_ids,
      },
  }
  for _kind, collection, _suite_path, _rates_path in outputs:
    _write_metadata(metadata, init_metadata_path(collection))
  rng = jax.random.PRNGKey(seed)
  try:
    infos = run_quad_fitness_search(
        bundle,
        rng,
        objectives,
        lanes,
        seed=seed,
        best_policy_index=0,
        elite_replace_limit=args.elite_replace_limit,
        elite_replace_window=args.elite_replace_window,
        failure_rate_every=args.failure_rate_every,
        max_generations=args.max_generations,
        immigrant_fraction=args.immigrant_fraction,
        save_every=args.save_every,
        resumes=resumes,
    )
  finally:
    for reporter in reporters:
      reporter._search_failed = None
      reporter._search_index = None
    reporters[0].close()
  by_slug = {info['slug']: info for info in infos}
  environment = bundle.environment
  del bundle
  gc.collect()
  jax.clear_caches()
  for kind, collection, suite_path, rates_path in outputs:
    info = by_slug[slugs[kind]]
    if not info.get('continued', True):
      print(f'{slugs[kind]}: unchanged', flush=True)
      continue
    reporter = reporters[LANE_KINDS.index(kind)]
    meta = dict(metadata)
    meta['feature_search'] = {
        key: value for key, value in info.items() if key != 'shard_dir'
    }
    meta_path = init_metadata_path(collection)
    _write_metadata(meta, meta_path)
    print(
        f'{slugs[kind]}: elites {info["n_elites"]}/{info["elite_count"]}  '
        f'unique states {info["n_unique_states"]}  '
        f'stopped_because={info["stopped_because"]}',
        flush=True,
    )
    _export_suite(
        environment,
        suite_path,
        rates_path,
        meta,
        meta_path,
        reporter.qpos,
        reporter.qvel,
        reporter.policy_rng,
        args.skip_heldout_eval,
    )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
