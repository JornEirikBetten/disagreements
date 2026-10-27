"""Rerun the study and rebuild its figures.

This directory is the replication repository. Run every command from here,
or pass this file to Python from anywhere. Paths are relative to this
directory. ``datasets/``, ``trained_policies/``, ``test_suites/``, and
``figures/`` are the experiment archives.

  python reproduce.py check
  python reproduce.py policy-sets
  python reproduce.py quota --environment hopper --algorithm ppo --experiment 1
  python reproduce.py feature-search --environment hopper --experiment 100 --feature value_disagreement
  python reproduce.py quad-search --environment hopper --experiment 100
  python reproduce.py datasets
  python reproduce.py transfer --source ppo
  python reproduce.py associations
  python reproduce.py predictors
  python reproduce.py distributions
  python reproduce.py figures

A finished search is left in place. Pass ``--overwrite`` to run it again.
``quota --all``, ``feature-search --all``, and ``quad-search --all`` launch
the full paper grids. Each search is a long MuJoCo run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

ENVIRONMENTS = ('hopper', 'walker2d', 'humanoid', 'ant')
QUOTA_ALGORITHMS = ('ppo', 'sac', 'ppo_sac')
QUOTA_EXPERIMENTS = (1, 2, 3, 4, 5)
FITNESS_EXPERIMENTS = (100, 101, 102, 103, 104)
FITNESS_FEATURES = (
    'action_disagreement',
    'action_scale_disagreement',
    'value_disagreement',
)
HORIZON = 15
N_POLICIES = 16
BIN_CAPACITY = 2000
QUOTA_GENERATIONS = 1000
FITNESS_GENERATIONS = 101


def _run(args: list[str]) -> None:
  print('+', ' '.join(args), flush=True)
  subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def _quota_collection(environment: str, algorithm: str, experiment: int) -> Path:
  return ROOT / (
      f'datasets/ga_data/{environment}/{algorithm}/difficulty_bins/'
      f'rollout_length={HORIZON}/experiment_{experiment}_ga_collection.pkl'
  )


def _predictor(environment: str, algorithm: str) -> Path:
  return ROOT / f'datasets/{environment}/{algorithm}/predictor_data_balanced.pkl'


def _feature_algorithm(environment: str) -> str:
  return 'ppo' if environment == 'humanoid' else 'sac'


def _feature_suite(
    environment: str,
    algorithm: str,
    experiment: int,
    slug: str,
) -> Path:
  return ROOT / (
      f'test_suites/{slug}/{environment}/{algorithm}/'
      f'rollout_length={HORIZON}/experiment_{experiment}/failure_rates.csv'
  )


def _selected(values: tuple[str, ...], chosen: str | None) -> tuple[str, ...]:
  return (chosen,) if chosen else values


def command_check(_args: argparse.Namespace) -> None:
  """Report whether the released policies, searches, and tables are present."""
  missing: list[str] = []
  notes: list[str] = []

  performance = ROOT / 'datasets/policy_performances.csv'
  if not performance.is_file():
    missing.append(str(performance))

  checkpoints = 0
  for environment in ENVIRONMENTS:
    for algorithm in ('ppo', 'sac'):
      directory = ROOT / 'trained_policies' / environment / algorithm
      count = 0
      if directory.is_dir():
        count = sum(
            1 for path in directory.iterdir() if path.name.startswith('best_control_')
        )
      checkpoints += count
      if count < 32:
        missing.append(f'{directory} has {count} checkpoints, need 32')

  collections = 0
  for environment in ENVIRONMENTS:
    for algorithm in QUOTA_ALGORITHMS:
      for experiment in QUOTA_EXPERIMENTS:
        path = _quota_collection(environment, algorithm, experiment)
        if path.is_file():
          collections += 1
        else:
          missing.append(str(path.relative_to(ROOT)))

  predictors = 0
  for environment in ENVIRONMENTS:
    for algorithm in QUOTA_ALGORITHMS:
      path = _predictor(environment, algorithm)
      if path.is_file():
        predictors += 1
      else:
        missing.append(str(path.relative_to(ROOT)))

  fitness_present = 0
  fitness_expected = 0
  for environment in ENVIRONMENTS:
    algorithm = _feature_algorithm(environment)
    slugs = [
        'max_action_disagreement',
        'max_action_scale_disagreement',
        'max_value_disagreement',
        'quad_search/search_difficulty',
        'quad_search/max_criticality_disagreement',
        (
            'min_feature_attribution_disagreement'
            if environment == 'humanoid'
            else 'max_feature_attribution_disagreement'
        ),
    ]
    for slug in slugs:
      for experiment in FITNESS_EXPERIMENTS:
        fitness_expected += 1
        path = _feature_suite(environment, algorithm, experiment, slug)
        if path.is_file():
          fitness_present += 1
        elif (
            slug == 'quad_search/search_difficulty'
            and environment == 'humanoid'
            and experiment == 100
        ):
          notes.append(
              'Humanoid search-fail-rate experiment 100 has no failure-rate '
              'table. Figure 7 averages the other four repetitions.'
          )
        else:
          missing.append(str(path.relative_to(ROOT)))

  tables = (
      'figures/ensemble_transfer/quota_fail_rates.csv',
      'figures/ensemble_transfer/quota_fail_rates_sac.csv',
      'figures/ensemble_transfer/quota_fail_rates_ppo_sac.csv',
      'figures/predictors/eval_fail_correlations_p32.csv',
      'figures/predictors/difficulty_feature_curves_p32.csv',
      'figures/predictors/repeatability_band.csv',
      'figures/predictors/extra_trees_10_foldcv_experiment_scores.csv',
      'figures/predictors/extra_trees_10_foldcv_experiment_search_eval_spearman.csv',
      'figures/policy_performances/humanoid_episode_lengths.csv',
  )
  table_present = 0
  for relative in tables:
    if (ROOT / relative).is_file():
      table_present += 1
    else:
      missing.append(relative)
  caches = list((ROOT / 'figures/quota_distributions/cache').glob('*.npz'))

  print(f'checkpoints:        {checkpoints}')
  print(f'quota collections:  {collections}/60')
  print(f'predictor datasets: {predictors}/12')
  print(f'fitness traces:     {fitness_present}/{fitness_expected}')
  print(f'analysis tables:    {table_present}/{len(tables)}')
  print(f'distribution caches:{len(caches)}/12')
  for note in notes:
    print(f'note: {note}')
  if missing:
    print(f'{len(missing)} required paths are missing:', file=sys.stderr)
    for path in missing[:30]:
      print(f'  {path}', file=sys.stderr)
    if len(missing) > 30:
      print(f'  ... {len(missing) - 30} more', file=sys.stderr)
    raise SystemExit(1)
  print('archives for the paper experiments are present')


def command_policy_sets(_args: argparse.Namespace) -> None:
  """Print the alternating search and held-out split of the best 32 policies."""
  from policies.ranking import (
      split_search_eval_by_performance,
      split_search_eval_mixed,
  )

  for environment in ENVIRONMENTS:
    for algorithm in ('ppo', 'sac'):
      search_ids, eval_ids, _assigned = split_search_eval_by_performance(
          environment, algorithm, N_POLICIES,
      )
      print(
          f'{environment:9} {algorithm:3}  '
          f'search={search_ids}  eval={eval_ids}'
      )
    search_members, eval_members, _assigned = split_search_eval_mixed(
        environment, 8,
    )
    search_ids = [
        f'{member["algorithm"]}:{member["policy_id"]}' for member in search_members
    ]
    eval_ids = [
        f'{member["algorithm"]}:{member["policy_id"]}' for member in eval_members
    ]
    print(
        f'{environment:9} mix  '
        f'search={search_ids}  eval={eval_ids}'
    )


def _quota_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
  if args.all:
    environments = ENVIRONMENTS
    algorithms = QUOTA_ALGORITHMS
    experiments = QUOTA_EXPERIMENTS
  else:
    if not (args.environment and args.algorithm and args.experiment):
      raise SystemExit(
          'Name one search with --environment, --algorithm, and --experiment, '
          'or pass --all for the 60 paper searches.'
      )
    environments = (args.environment,)
    algorithms = (args.algorithm,)
    experiments = (args.experiment,)
  return [
      (environment, algorithm, experiment)
      for environment in environments
      for algorithm in algorithms
      for experiment in experiments
  ]


def command_quota(args: argparse.Namespace) -> None:
  """Fill 2,000 states in each of the 17 search-fail-rate quotas."""
  for environment, algorithm, experiment in _quota_jobs(args):
    path = _quota_collection(environment, algorithm, experiment)
    if path.is_file() and not args.overwrite:
      print(f'skip {path.relative_to(ROOT)}: already finished')
      continue
    overwrite = ['--overwrite'] if args.overwrite else []
    if algorithm == 'ppo_sac':
      _run([
          'search_for_tests/binned_ga_search_mixed.py',
          environment, str(experiment), str(HORIZON),
          '--bin-capacity', str(BIN_CAPACITY),
          '--max-generations', str(QUOTA_GENERATIONS),
          *overwrite,
      ])
    else:
      _run([
          'search_for_tests/binned_ga_search.py',
          environment, algorithm, str(experiment), str(HORIZON), str(N_POLICIES),
          '--bin-capacity', str(BIN_CAPACITY),
          '--max-generations', str(QUOTA_GENERATIONS),
          *overwrite,
      ])


def _fitness_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
  if args.all:
    return [
        (environment, experiment)
        for environment in ENVIRONMENTS
        for experiment in FITNESS_EXPERIMENTS
    ]
  if not (args.environment and args.experiment):
    raise SystemExit(
        'Name one run with --environment and --experiment, or pass --all.'
    )
  return [(args.environment, args.experiment)]


def command_feature_search(args: argparse.Namespace) -> None:
  """Maximise one initial-state disagreement for 100 generations."""
  features = (args.feature,) if args.feature else FITNESS_FEATURES
  for environment, experiment in _fitness_jobs(args):
    algorithm = _feature_algorithm(environment)
    for feature in features:
      suite = _feature_suite(
          environment, algorithm, experiment, f'max_{feature}',
      )
      if suite.is_file() and not args.overwrite:
        print(f'skip {suite.relative_to(ROOT)}: already finished')
        continue
      _run([
          'search_for_tests/feature_ga_search.py',
          environment, algorithm, str(experiment), str(HORIZON), str(N_POLICIES),
          '--features', f'{feature}:max',
          '--failure-rate-every', '1',
          '--max-generations', str(FITNESS_GENERATIONS),
      ])


def command_attribution_search(args: argparse.Namespace) -> None:
  """Maximise attribution disagreement, except Humanoid, which minimises it."""
  for environment, experiment in _fitness_jobs(args):
    algorithm = _feature_algorithm(environment)
    direction = 'min' if environment == 'humanoid' else 'max'
    slug = f'{direction}_feature_attribution_disagreement'
    suite = _feature_suite(environment, algorithm, experiment, slug)
    if suite.is_file() and not args.overwrite:
      print(f'skip {suite.relative_to(ROOT)}: already finished')
      continue
    _run([
        'search_for_tests/feature_ga_search.py',
        environment, algorithm, str(experiment), str(HORIZON), str(N_POLICIES),
        '--features', f'feature_attribution_disagreement:{direction}',
        '--failure-rate-every', '1',
        '--max-generations', str(FITNESS_GENERATIONS),
    ])


def command_quad_search(args: argparse.Namespace) -> None:
  """Search fail rate and criticality, from one shared initial population."""
  for environment, experiment in _fitness_jobs(args):
    algorithm = _feature_algorithm(environment)
    suite = _feature_suite(
        environment, algorithm, experiment,
        'quad_search/max_criticality_disagreement',
    )
    if suite.is_file() and not args.overwrite:
      print(f'skip {suite.relative_to(ROOT)}: already finished')
      continue
    _run([
        'search_for_tests/quad_fitness_search.py',
        environment, algorithm, str(experiment), str(HORIZON), str(N_POLICIES),
        '--features', '+criticality_disagreement',
        '--failure-rate-every', '1',
        '--max-generations', str(FITNESS_GENERATIONS),
    ])


def command_datasets(args: argparse.Namespace) -> None:
  """Pool the quota searches into one predictor table per policy set."""
  command = ['build_predictor_datasets.py']
  if args.environment:
    command.extend(['--environment', args.environment])
  if args.algorithm:
    command.extend(['--algorithm', args.algorithm])
  _run(command)


def command_transfer(args: argparse.Namespace) -> None:
  """Roll the other policy sets out on every quota-search state."""
  sources = (args.source,) if args.source else ('ppo', 'sac', 'ppo_sac')
  for source in sources:
    command = ['ensemble_quota_table.py', '--source', source]
    if args.environment:
      command.extend(['--environment', args.environment])
    _run(command)


def command_associations(args: argparse.Namespace) -> None:
  """Bootstrap the disagreement associations and the difficulty curves."""
  command = ['shared_eval_figures.py', '--boot', str(args.boot)]
  if args.environment:
    command.extend(['--environment', args.environment])
  if args.algorithm:
    command.extend(['--algorithm', args.algorithm])
  _run(command)


def command_predictors(args: argparse.Namespace) -> None:
  """Cross-validate the 500-tree extra-trees disagreement model."""
  command = ['cv_difficulty_predictors.py', '--folds', '5', '--n-estimators', '500']
  if args.environment:
    command.extend(['--environment', args.environment])
  if args.algorithm:
    command.extend(['--algorithm', args.algorithm])
  _run(command)


def command_distributions(args: argparse.Namespace) -> None:
  """Build the value, magnitude, and criticality distributions."""
  if args.extract:
    _run(['plotting/plot_value_magnitude_criticality.py'])
  else:
    _run(['plotting/plot_value_magnitude_criticality.py', '--plot-only'])


def command_figures(args: argparse.Namespace) -> None:
  """Draw the result figures into results/."""
  from replicate import main as draw

  draw(['--only', *args.only] if args.only else [])


def _add_job_flags(parser: argparse.ArgumentParser, *, feature: bool = False) -> None:
  parser.add_argument('--environment', choices=ENVIRONMENTS)
  parser.add_argument('--experiment', type=int, choices=FITNESS_EXPERIMENTS)
  parser.add_argument(
      '--all', action='store_true',
      help='Run the full paper grid. This is many long searches.',
  )
  parser.add_argument(
      '--overwrite', action='store_true',
      help='Rerun a search whose output is already on disk.',
  )
  if feature:
    parser.add_argument('--feature', choices=FITNESS_FEATURES)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  commands = parser.add_subparsers(dest='command', required=True)

  commands.add_parser('check', help='Verify the released experiment archives.')
  commands.add_parser('policy-sets', help='Print the search and held-out policy ids.')

  quota = commands.add_parser('quota', help='Rerun a quota-filling search.')
  quota.add_argument('--environment', choices=ENVIRONMENTS)
  quota.add_argument('--algorithm', choices=QUOTA_ALGORITHMS)
  quota.add_argument('--experiment', type=int, choices=QUOTA_EXPERIMENTS)
  quota.add_argument('--all', action='store_true')
  quota.add_argument('--overwrite', action='store_true')

  feature = commands.add_parser(
      'feature-search', help='Rerun a disagreement-maximising search.',
  )
  _add_job_flags(feature, feature=True)
  attribution = commands.add_parser(
      'attribution-search', help='Rerun an attribution-disagreement search.',
  )
  _add_job_flags(attribution)
  quad = commands.add_parser(
      'quad-search',
      help='Rerun the shared-initial-population fail-rate and criticality searches.',
  )
  _add_job_flags(quad)

  datasets = commands.add_parser(
      'datasets', help='Rebuild the balanced predictor datasets.',
  )
  datasets.add_argument('--environment', choices=ENVIRONMENTS)
  datasets.add_argument('--algorithm', choices=QUOTA_ALGORITHMS)

  transfer = commands.add_parser(
      'transfer', help='Rescore quota states with the other policy sets.',
  )
  transfer.add_argument('--source', choices=('ppo', 'sac', 'ppo_sac'))
  transfer.add_argument('--environment', choices=ENVIRONMENTS)

  associations = commands.add_parser(
      'associations', help='Recompute Figures 4 and 5 from the predictor datasets.',
  )
  associations.add_argument('--environment', choices=ENVIRONMENTS)
  associations.add_argument('--algorithm', choices=QUOTA_ALGORITHMS)
  associations.add_argument('--boot', type=int, default=1000)

  predictors = commands.add_parser(
      'predictors', help='Retrain the extra-trees models for Figure 6.',
  )
  predictors.add_argument('--environment', choices=ENVIRONMENTS)
  predictors.add_argument('--algorithm', choices=QUOTA_ALGORITHMS)

  distributions = commands.add_parser(
      'distributions', help='Redraw Figure B1, or re-extract it from the shards.',
  )
  distributions.add_argument(
      '--extract', action='store_true',
      help='Read the generation shards again before plotting.',
  )

  figures = commands.add_parser('figures', help='Draw the result figures into results/.')
  figures.add_argument('--only', nargs='+')

  args = parser.parse_args(argv)
  {
      'check': command_check,
      'policy-sets': command_policy_sets,
      'quota': command_quota,
      'feature-search': command_feature_search,
      'attribution-search': command_attribution_search,
      'quad-search': command_quad_search,
      'datasets': command_datasets,
      'transfer': command_transfer,
      'associations': command_associations,
      'predictors': command_predictors,
      'distributions': command_distributions,
      'figures': command_figures,
  }[args.command](args)


if __name__ == '__main__':
  main()
