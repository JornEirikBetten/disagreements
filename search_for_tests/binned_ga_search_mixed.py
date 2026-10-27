"""Binned GA search with a mixed PPO and SAC ensemble.

Usage (from the repository root):

  python search_for_tests/binned_ga_search_mixed.py \\
      ENVIRONMENT EXP_NUMBER ROLLOUT_LENGTH

  python search_for_tests/binned_ga_search_mixed.py ant 1 15 \\
      --bin-capacity 2000 --max-generations 1000

This is ``binned_ga_search.py`` with a different ensemble split. Everything
about the search itself -- bins, uniqueness of stored initial states,
selection, immigrants, checkpointing -- is shared and lives in
``ga/search_cli.py``.

Each of the search and eval ensembles gets ``--per-algorithm`` PPO policies
and the same number of SAC policies (default 8, so P = 16). Within each
algorithm the split is the same alternating top-performer ranking as
``binned_ga_search.py``: the highest ``best_eval_episode_reward`` goes to
search, the next to eval, and so on. Failure-matrix rows are PPO first,
then SAC.

Mixing the algorithms matters for the prediction study. A single-algorithm
ensemble shares an optimiser, an exploration scheme, and a set of failure
habits, so its disagreement can reflect PPO's idiosyncrasies rather than
anything general about the test case. Spanning two algorithm families makes
"the policies disagree here" a harder and more meaningful claim.

The GA quotas are filled from the search ensemble's fail rate. Once the
search stops, every stored unique case is replayed against the disjoint mixed
evaluation ensemble. Its exact ``eval_failed`` mask and derived
``eval_fail_rate`` are written to the shard. The evaluation outcome never
guides selection. ``--skip-heldout-eval`` defers this post-search stage to
``add_heldout_fail_rates.py``.

Collections are written to:
  datasets/ga_data/{env}/ppo_sac/difficulty_bins/rollout_length={T}/experiment_{n}_ga_collection.pkl
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
  # See binned_ga_search.py: script-only so imports stay side-effect free.
  os.chdir(_ROOT)

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from ga.env_setup import make_mixed_search_bundle
from ga.search_cli import (
    add_search_arguments,
    run_binned_search,
    score_heldout_after_search,
    validate_search_arguments,
)
from policies.ranking import split_search_eval_mixed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Binned GA search whose search and eval ensembles each contain '
          'the same number of top PPO and top SAC policies. Search fail rate '
          'fills the quotas; held-out fail rate is logged afterwards.'
      ),
  )
  parser.add_argument('environment')
  parser.add_argument('exp_number', type=int)
  parser.add_argument('rollout_length', type=int)
  parser.add_argument(
      '--per-algorithm',
      type=int,
      default=8,
      help='PPO policies and SAC policies in each ensemble (default: 8).',
  )
  add_search_arguments(parser)
  return parser.parse_args()


def _format_members(members: list[dict]) -> str:
  return ' '.join(
      f'{member["algorithm"]}:{member["policy_id"]}' for member in members
  )


def main() -> int:
  args = parse_args()
  if args.per_algorithm < 1:
    raise ValueError(f'per_algorithm must be >= 1, got {args.per_algorithm}')
  validate_search_arguments(args)

  search_members, eval_members, assigned = split_search_eval_mixed(
      args.environment, args.per_algorithm
  )
  n_policies = len(search_members)
  print(
      f'Mixed ensembles ({args.per_algorithm} PPO + {args.per_algorithm} SAC, '
      f'P={n_policies})'
  )
  print(f'  search: {_format_members(search_members)}')
  print(f'  eval:   {_format_members(eval_members)}')

  # Unlike the single-algorithm bundle, this one takes the held-out members
  # directly, so the bundle is complete as soon as it is built.
  bundle = make_mixed_search_bundle(
      args.environment,
      args.rollout_length,
      search_members,
      eval_members,
      with_attributions=True,
  )

  run_binned_search(
      bundle,
      args,
      n_policies=n_policies,
      exp_number=args.exp_number,
      selection={
          'method': 'alternate_highest_performance_by_algorithm',
          'metric': 'best_eval_episode_reward',
          'per_algorithm': int(args.per_algorithm),
          'search': search_members,
          'eval': eval_members,
          'assigned': assigned,
      },
  )
  environment = bundle.environment
  algorithm = bundle.algorithm
  del bundle
  gc.collect()
  score_heldout_after_search(
      args,
      environment=environment,
      algorithm=algorithm,
      exp_number=args.exp_number,
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
