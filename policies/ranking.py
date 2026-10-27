"""Rank local checkpoints by W&B eval reward and split search/eval ensembles."""

from __future__ import annotations

import csv
import os
import pickle
import re

DEFAULT_CSV = os.path.join('datasets', 'policy_performances.csv')
DEFAULT_POLICY_ROOT = 'trained_policies'
PERFORMANCE_METRIC = 'best_eval_episode_reward'
_BEST_CONTROL = re.compile(r'^best_control_(\d+)$')


def sac_checkpoint_has_action_value(params_path: str) -> bool:
  """True when the SAC checkpoint includes Q-network (action-value) weights."""
  q_path = os.path.join(os.path.dirname(params_path), 'q_params.json')
  if os.path.isfile(q_path):
    return True
  with open(params_path, 'rb') as handle:
    params = pickle.load(handle)
  if not isinstance(params, (tuple, list)):
    return False
  if len(params) >= 3:
    return True
  if len(params) >= 2 and isinstance(params[1], dict) and 'q' in params[1]:
    return True
  return False


def local_policy_ids(
    environment: str,
    algorithm: str,
    policy_root: str = DEFAULT_POLICY_ROOT,
) -> list[int]:
  """Local ``best_control_{id}`` checkpoints that can be used as policies.

  SAC checkpoints without action-value (Q) parameters are omitted.
  """
  directory = os.path.join(policy_root, environment, algorithm)
  if not os.path.isdir(directory):
    return []
  ids = []
  for name in os.listdir(directory):
    match = _BEST_CONTROL.match(name)
    if not match:
      continue
    params = os.path.join(directory, name, 'params.json')
    if not os.path.isfile(params):
      continue
    if algorithm == 'sac' and not sac_checkpoint_has_action_value(params):
      continue
    ids.append(int(match.group(1)))
  return sorted(ids)


def _performance_rows(
    environment: str,
    algorithm: str,
    csv_path: str,
) -> list[dict]:
  if not os.path.isfile(csv_path):
    raise FileNotFoundError(
        f'No W&B performance CSV at {csv_path}. '
        'Run extract_wandb_policy_performances.py first.'
    )
  with open(csv_path, newline='') as handle:
    rows = list(csv.DictReader(handle))
  return [
      row
      for row in rows
      if row['environment'] == environment and row['algorithm'] == algorithm
  ]


def split_search_eval_by_performance(
    environment: str,
    algorithm: str,
    n_policies: int,
    *,
    csv_path: str = DEFAULT_CSV,
    policy_root: str = DEFAULT_POLICY_ROOT,
    metric: str = PERFORMANCE_METRIC,
) -> tuple[list[int], list[int], list[dict]]:
  """Fill search then eval by alternating down the performance ranking.

  The highest-scoring local checkpoint goes to search, the next to eval, and
  so on until each set has ``n_policies`` members. Ranking uses ``metric``
  from the W&B CSV (default: best eval episode reward, matching
  ``best_control_{id}`` checkpoints).

  Requires at least ``n_policies`` scored local checkpoints for the search
  ensemble. If fewer than ``2 * n_policies`` exist, search still gets the
  top ``n_policies`` and eval gets the remainder.
  """
  if n_policies < 1:
    raise ValueError(f'n_policies must be >= 1, got {n_policies}')
  local_ids = set(local_policy_ids(environment, algorithm, policy_root))
  ranked = []
  for row in _performance_rows(environment, algorithm, csv_path):
    policy_id = int(row['policy_id'])
    if policy_id not in local_ids:
      continue
    ranked.append(
        {
            'policy_id': policy_id,
            'best_eval_episode_reward': float(row['best_eval_episode_reward']),
            'final_eval_episode_reward': float(row['final_eval_episode_reward']),
            'rank_by_best': int(row.get('rank_by_best') or 0),
            'rank_by_final': int(row.get('rank_by_final') or 0),
            'run_name': row.get('run_name', ''),
            'wandb_url': row.get('wandb_url', ''),
        }
    )
  ranked.sort(
      key=lambda item: (-float(item[metric]), item['policy_id'])
  )
  q_note = ''
  if algorithm == 'sac':
    q_note = (
        ' SAC checkpoints without action-value (Q) parameters are excluded.'
    )
  if len(ranked) < n_policies:
    raise ValueError(
        f'Need {n_policies} local checkpoints with W&B scores for '
        f'{environment}/{algorithm} (P={n_policies} search); found '
        f'{len(ranked)} in {csv_path} with params.json under '
        f'{policy_root}.{q_note}'
    )

  search_ids: list[int] = []
  eval_ids: list[int] = []
  assigned: list[dict] = []
  if len(ranked) < 2 * n_policies:
    for item in ranked[:n_policies]:
      record = dict(item)
      record['ensemble'] = 'search'
      search_ids.append(item['policy_id'])
      assigned.append(record)
    for item in ranked[n_policies:]:
      record = dict(item)
      record['ensemble'] = 'eval'
      eval_ids.append(item['policy_id'])
      assigned.append(record)
    return search_ids, eval_ids, assigned

  for item in ranked:
    if len(search_ids) >= n_policies and len(eval_ids) >= n_policies:
      break
    record = dict(item)
    if len(search_ids) <= len(eval_ids) and len(search_ids) < n_policies:
      search_ids.append(item['policy_id'])
      record['ensemble'] = 'search'
    elif len(eval_ids) < n_policies:
      eval_ids.append(item['policy_id'])
      record['ensemble'] = 'eval'
    else:
      search_ids.append(item['policy_id'])
      record['ensemble'] = 'search'
    assigned.append(record)
  return search_ids, eval_ids, assigned


def split_search_eval_mixed(
    environment: str,
    n_per_algorithm: int,
    *,
    algorithms: tuple[str, ...] = ('ppo', 'sac'),
    csv_path: str = DEFAULT_CSV,
    policy_root: str = DEFAULT_POLICY_ROOT,
    metric: str = PERFORMANCE_METRIC,
) -> tuple[list[dict], list[dict], list[dict]]:
  """Alternate top performers separately inside each algorithm.

  Search and eval each receive ``n_per_algorithm`` policies from every
  algorithm. Members are ordered by ``algorithms`` (PPO, then SAC), which
  is also the row order of the mixed failure matrix.
  """
  if n_per_algorithm < 1:
    raise ValueError(
        f'n_per_algorithm must be >= 1, got {n_per_algorithm}'
    )
  search_members: list[dict] = []
  eval_members: list[dict] = []
  assigned: list[dict] = []
  for algorithm in algorithms:
    search_ids, eval_ids, rows = split_search_eval_by_performance(
        environment,
        algorithm,
        n_per_algorithm,
        csv_path=csv_path,
        policy_root=policy_root,
        metric=metric,
    )
    if len(search_ids) != n_per_algorithm or len(eval_ids) != n_per_algorithm:
      raise ValueError(
          f'Need {n_per_algorithm} search and {n_per_algorithm} eval '
          f'checkpoints for {environment}/{algorithm}; got search='
          f'{len(search_ids)} eval={len(eval_ids)}.'
      )
    tagged = []
    for row in rows:
      item = dict(row)
      item['algorithm'] = algorithm
      tagged.append(item)
    assigned.extend(tagged)
    lookup = {(row['ensemble'], row['policy_id']): row for row in tagged}
    for ensemble, ids in (('search', search_ids), ('eval', eval_ids)):
      members = search_members if ensemble == 'search' else eval_members
      for policy_id in ids:
        members.append(lookup[(ensemble, policy_id)])
  search_keys = [
      (str(member['algorithm']), int(member['policy_id']))
      for member in search_members
  ]
  eval_keys = [
      (str(member['algorithm']), int(member['policy_id']))
      for member in eval_members
  ]
  if len(set(search_keys)) != len(search_keys):
    raise ValueError(f'Mixed search ensemble has duplicate members: {search_keys}')
  if len(set(eval_keys)) != len(eval_keys):
    raise ValueError(f'Mixed eval ensemble has duplicate members: {eval_keys}')
  overlap = sorted(set(search_keys) & set(eval_keys))
  if overlap:
    raise ValueError(f'Mixed search and eval ensembles overlap: {overlap}')
  return search_members, eval_members, assigned
