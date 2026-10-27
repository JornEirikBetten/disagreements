"""GA search that fills a quota of test cases at each discrete difficulty.

Usage (from the repository root):

  python search_for_tests/binned_ga_search.py \\
      ENVIRONMENT ALGORITHM EXP_NUMBER ROLLOUT_LENGTH P

  python search_for_tests/binned_ga_search.py hopper ppo 1 15 16
  python search_for_tests/binned_ga_search.py hopper ppo 1 15 16 \\
      --bin-capacity 10000 --max-generations 4000

What this produces
------------------
A collection of test cases (initial states) labelled by how many of the P
*search* policies they defeat. Each discrete difficulty -- all-fail, then
k/P for k = 0, ..., P-1 -- is a bin with ``--bin-capacity`` slots, so the
finished collection is roughly uniform over difficulty rather than dominated
by the easy cases the sampler produces most often. Cases that land in a full
bin, or whose initial state has already been written, are discarded. Elites
stay in the population so the search can still mutate them; they are not
stored twice. That is what stops a bin from filling with copies of the same
ancestor.

A second, disjoint ensemble of P policies is held out and never seen by the
search. After the GA stops, this script releases the search policies, replays
every stored unique case against the held-out ensemble, and writes
the exact ``eval_failed`` policy-by-case mask, its ``eval_fail_rate`` column
mean, and its distribution to the shards and sidecar. Use
``--skip-heldout-eval`` to defer that separate scoring stage to
``add_heldout_fail_rates.py``. The held-out rate is not used to select cases
or fill bins; it remains the target of the prediction study.

How the search moves
--------------------
Selection prefers cases in unfilled bins, then higher fail counts, so
near-misses keep evolving toward all-fail. A slice of elites is re-mutated
as stepping stones, and a fraction of each generation is replaced by
immigrants drawn from the full genome box rather than reset-scale noise
(Humanoid and Ant keep reset-scale noise so the free joint is not blown up).
Generation 0 is stratified unless ``--no-stratified-init`` is set.

Kept cases are written to disk after every generation so RAM and the GPU
only hold the current population. Integrated gradients always run in
microbatches of 128 (the tail is padded, then discarded) so JAX compiles
that kernel once instead of once per kept count. The sidecar pickle stores
search metadata; generation payloads live in a sibling ``*_generations/``
folder.

Collections are written to:
  datasets/ga_data/{env}/{alg}/difficulty_bins/rollout_length={T}/experiment_{n}_ga_collection.pkl
  datasets/ga_data/{env}/{alg}/difficulty_bins/rollout_length={T}/experiment_{n}_ga_collection_generations/
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
  # Every data path in the project is relative to the repository root. Only
  # do this when run as a script, so importing the module (to reuse the
  # ensemble split, or to test it) does not move the caller's directory.
  os.chdir(_ROOT)

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from ga.env_setup import make_search_bundle
from ga.search_cli import (
    add_search_arguments,
    run_binned_search,
    score_heldout_after_search,
    validate_search_arguments,
)
from policies.ranking import split_search_eval_by_performance


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'GA search that fills a fixed number of distinct test cases per '
          'search-ensemble fail-rate bin, then scores them on held-out policies.'
      ),
  )
  parser.add_argument('environment')
  parser.add_argument('algorithm', choices=('ppo', 'sac'))
  parser.add_argument('exp_number', type=int)
  parser.add_argument('rollout_length', type=int)
  parser.add_argument('n_policies', type=int)
  add_search_arguments(parser)
  return parser.parse_args()


def select_policy_ensembles(
    environment: str, algorithm: str, n_policies: int
) -> tuple[list[int], list[int], dict]:
  """Split the trained checkpoints into a search and a held-out ensemble.

  Policies alternate down the reward ranking, which makes the two ensembles
  comparable in skill: the search set is not systematically stronger or
  weaker than the set it is meant to generalise to.
  """
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


def validate_policy_ensembles(
    search_ids: list[int],
    eval_ids: list[int],
    *,
    n_policies: int,
    allow_unequal_eval: bool,
) -> None:
  """Require distinct, disjoint ensembles and explicit size exceptions."""
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


def main() -> int:
  args = parse_args()
  if args.n_policies < 2:
    raise ValueError(f'n_policies must be >= 2, got {args.n_policies}')
  validate_search_arguments(args)

  search_ids, eval_ids, selection = select_policy_ensembles(
      args.environment, args.algorithm, args.n_policies
  )
  validate_policy_ensembles(
      search_ids,
      eval_ids,
      n_policies=args.n_policies,
      allow_unequal_eval=args.allow_unequal_eval,
  )
  print(
      f'Policy ensembles (P={args.n_policies}): '
      f'search={search_ids}  eval={eval_ids}'
  )
  if len(eval_ids) != args.n_policies:
    print(
        f'Explicit unequal-eval exception: held-out ensemble has '
        f'{len(eval_ids)} policies '
        f'(wanted {args.n_policies}); search still uses {len(search_ids)}. '
        'Held-out fail rates will be coarser than the search fail rates.'
    )

  bundle = make_search_bundle(
      args.environment,
      args.algorithm,
      args.n_policies,
      args.rollout_length,
      with_attributions=True,
      policy_ids=search_ids,
  )
  # The bundle only loads the search ensemble; the held-out ids ride along on
  # it so every generation shard records which policies it must later be
  # replayed against.
  bundle.eval_policy_ids = eval_ids

  run_binned_search(
      bundle,
      args,
      n_policies=args.n_policies,
      exp_number=args.exp_number,
      selection=selection,
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
