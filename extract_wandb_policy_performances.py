"""Extract best and final eval rewards for each policy ID from W&B.

Usage:
  python extract_wandb_policy_performances.py [OUTPUT_CSV] [ENV[,ENV...]]

For every policy ID that has a local ``best_control`` or ``final_control``
checkpoint under ``trained_policies``,
keeps the **latest** W&B run (by ``created_at``) whose name encodes that ID,
then writes:

  * ``best_eval_episode_reward`` — max of ``eval/episode_reward`` over the run
  * ``final_eval_episode_reward`` — last logged ``eval/episode_reward``

Rows are grouped by environment/algorithm and ordered by best reward
(descending). ``rank_by_best`` and ``rank_by_final`` are 1-based ranks within
that group.

If ``ENV`` is given (comma-separated), only those environments are fetched
from W&B and merged into an existing CSV.

Projects:
  https://wandb.ai/jorneirikbetten/humanoid_ppo
  https://wandb.ai/jorneirikbetten/sac-humanoid
  https://wandb.ai/jorneirikbetten/hopper_ppo
  https://wandb.ai/jorneirikbetten/sac-hopper
  https://wandb.ai/jorneirikbetten/walker2d_ppo
  https://wandb.ai/jorneirikbetten/sac-walker2d
  https://wandb.ai/jorneirikbetten/ant_ppo
  https://wandb.ai/jorneirikbetten/sac-ant
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

import wandb

ENTITY = 'jorneirikbetten'
PROJECTS = (
    ('humanoid', 'ppo', 'humanoid_ppo'),
    ('humanoid', 'sac', 'sac-humanoid'),
    ('hopper', 'ppo', 'hopper_ppo'),
    ('hopper', 'sac', 'sac-hopper'),
    ('walker2d', 'ppo', 'walker2d_ppo'),
    ('walker2d', 'sac', 'sac-walker2d'),
    ('ant', 'ppo', 'ant_ppo'),
    ('ant', 'sac', 'sac-ant'),
)
# ctrl-12-..., hopper-7-..., walker2d-1-..., ant-4-...
_POLICY_NAME = re.compile(
    r'^(?:ctrl|humanoid|hopper|walker2d|ant)-(\d+)(?:-|$)'
)
_LOCAL_CHECKPOINT = re.compile(r'^(?:best|final)_control_(\d+)$')
DEFAULT_POLICY_ROOT = 'trained_policies'
_METRIC = 'eval/episode_reward'
DEFAULT_OUTPUT = os.path.join('datasets', 'policy_performances.csv')

FIELDNAMES = [
    'environment',
    'algorithm',
    'policy_id',
    'rank_by_best',
    'rank_by_final',
    'best_eval_episode_reward',
    'final_eval_episode_reward',
    'best_num_env_steps',
    'final_num_env_steps',
    'n_evals',
    'run_name',
    'run_id',
    'run_state',
    'created_at',
    'wandb_url',
]


def _parse_policy_id(run_name: str | None) -> int | None:
  if not run_name:
    return None
  match = _POLICY_NAME.match(run_name)
  if not match:
    return None
  return int(match.group(1))


def _run_url(project: str, run_id: str) -> str:
  return f'https://wandb.ai/{ENTITY}/{project}/runs/{run_id}'


def _as_datetime(value) -> datetime:
  if isinstance(value, datetime):
    if value.tzinfo is None:
      return value.replace(tzinfo=timezone.utc)
    return value
  text = str(value)
  if text.endswith('Z'):
    text = text[:-1] + '+00:00'
  return datetime.fromisoformat(text)


def _reward_series(run) -> tuple[list[float], list[float | None]]:
  """Full eval-reward history (not the 500-point sampled ``history()``)."""
  rewards: list[float] = []
  steps: list[float | None] = []
  for row in run.scan_history(keys=[_METRIC, 'num_env_steps']):
    value = row.get(_METRIC)
    if value is None:
      continue
    rewards.append(float(value))
    step = row.get('num_env_steps')
    steps.append(float(step) if step is not None else None)
  return rewards, steps


def _latest_run_per_policy(runs) -> dict[int, object]:
  """Keep the most recently created run for each parsed policy ID."""
  latest: dict[int, object] = {}
  for run in runs:
    policy_id = _parse_policy_id(run.name)
    if policy_id is None:
      continue
    created = _as_datetime(run.created_at)
    previous = latest.get(policy_id)
    if previous is None or created > _as_datetime(previous.created_at):
      latest[policy_id] = run
  return latest


def extract_project(environment: str, algorithm: str, project: str) -> list[dict]:
  api = wandb.Api()
  path = f'{ENTITY}/{project}'
  print(f'Fetching runs from {path}')
  latest = _latest_run_per_policy(api.runs(path))
  rows = []
  for policy_id in sorted(latest):
    run = latest[policy_id]
    print(f'  {environment}/{algorithm} policy {policy_id}: {run.name}')
    rewards, steps = _reward_series(run)
    if not rewards:
      summary = run.summary.get(_METRIC) if run.summary else None
      if summary is None:
        print('    skip (no eval/episode_reward)')
        continue
      rewards = [float(summary)]
      steps = [None]
    best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
    rows.append(
        {
            'environment': environment,
            'algorithm': algorithm,
            'policy_id': policy_id,
            'best_eval_episode_reward': rewards[best_idx],
            'final_eval_episode_reward': rewards[-1],
            'best_num_env_steps': steps[best_idx],
            'final_num_env_steps': steps[-1],
            'n_evals': len(rewards),
            'run_name': run.name,
            'run_id': run.id,
            'run_state': run.state,
            'created_at': str(run.created_at),
            'wandb_url': _run_url(project, run.id),
        }
    )
  return rows


def _add_ranks(rows: list[dict]) -> None:
  by_best = sorted(
      rows, key=lambda r: r['best_eval_episode_reward'], reverse=True
  )
  by_final = sorted(
      rows, key=lambda r: r['final_eval_episode_reward'], reverse=True
  )
  best_rank = {id(row): i + 1 for i, row in enumerate(by_best)}
  final_rank = {id(row): i + 1 for i, row in enumerate(by_final)}
  for row in rows:
    row['rank_by_best'] = best_rank[id(row)]
    row['rank_by_final'] = final_rank[id(row)]


def _optional_float(value) -> float | None:
  if value is None or value == '':
    return None
  return float(value)


def load_csv(path: str) -> list[dict]:
  """Load a previously written performance CSV with numeric fields restored."""
  rows: list[dict] = []
  with open(path, newline='') as handle:
    for row in csv.DictReader(handle):
      rows.append(
          {
              'environment': row['environment'],
              'algorithm': row['algorithm'],
              'policy_id': int(row['policy_id']),
              'rank_by_best': int(row['rank_by_best']) if row.get('rank_by_best') else None,
              'rank_by_final': (
                  int(row['rank_by_final']) if row.get('rank_by_final') else None
              ),
              'best_eval_episode_reward': float(row['best_eval_episode_reward']),
              'final_eval_episode_reward': float(row['final_eval_episode_reward']),
              'best_num_env_steps': _optional_float(row.get('best_num_env_steps')),
              'final_num_env_steps': _optional_float(row.get('final_num_env_steps')),
              'n_evals': int(row['n_evals']),
              'run_name': row['run_name'],
              'run_id': row['run_id'],
              'run_state': row['run_state'],
              'created_at': row['created_at'],
              'wandb_url': row['wandb_url'],
          }
      )
  return rows


def local_checkpoint_policy_ids(
    environment: str,
    algorithm: str,
    policy_root: str = DEFAULT_POLICY_ROOT,
) -> set[int]:
  """Policy IDs with a local best or final checkpoint.

  Both names are snapshots of the same trained policy.
  """
  directory = os.path.join(policy_root, environment, algorithm)
  if not os.path.isdir(directory):
    return set()
  ids = set()
  for name in os.listdir(directory):
    match = _LOCAL_CHECKPOINT.match(name)
    if match is None:
      continue
    if os.path.isfile(os.path.join(directory, name, 'params.json')):
      ids.add(int(match.group(1)))
  return ids


def restrict_to_local_checkpoints(
    rows: list[dict],
    policy_root: str = DEFAULT_POLICY_ROOT,
) -> list[dict]:
  """Keep evaluations for local checkpoints and require every checkpoint."""
  local = {
      (environment, algorithm): local_checkpoint_policy_ids(
          environment, algorithm, policy_root
      )
      for environment, algorithm, _project in PROJECTS
  }
  present: dict[tuple[str, str], set[int]] = defaultdict(set)
  for row in rows:
    present[(row['environment'], row['algorithm'])].add(int(row['policy_id']))
  missing = [
      (environment, algorithm, policy_id)
      for (environment, algorithm), ids in local.items()
      for policy_id in sorted(ids - present[(environment, algorithm)])
  ]
  if missing:
    listed = ', '.join(
        f'{environment}/{algorithm}:{policy_id}'
        for environment, algorithm, policy_id in missing
    )
    raise ValueError(
        'Local checkpoints have no W&B evaluation in the performance table: '
        f'{listed}'
    )
  return [
      row
      for row in rows
      if int(row['policy_id']) in local[(row['environment'], row['algorithm'])]
  ]


def write_csv(rows: list[dict], output_path: str) -> None:
  grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in rows:
    grouped[(row['environment'], row['algorithm'])].append(row)

  ordered: list[dict] = []
  for environment, algorithm, _project in PROJECTS:
    group = grouped[(environment, algorithm)]
    _add_ranks(group)
    group.sort(
        key=lambda r: (
            -r['best_eval_episode_reward'],
            -r['final_eval_episode_reward'],
            r['policy_id'],
        )
    )
    ordered.extend(group)

  os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
  with open(output_path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(ordered)
  print(f'Saved {output_path} ({len(ordered)} policies)')


def main():
  args = sys.argv[1:]
  output_path = DEFAULT_OUTPUT
  env_filter = None
  if args:
    if args[0] in {p[0] for p in PROJECTS} or ',' in args[0]:
      env_filter = {name.strip() for name in args[0].split(',') if name.strip()}
    else:
      output_path = args[0]
      if len(args) > 1:
        env_filter = {name.strip() for name in args[1].split(',') if name.strip()}

  wanted = PROJECTS
  if env_filter:
    unknown = env_filter - {p[0] for p in PROJECTS}
    if unknown:
      raise SystemExit(f'Unknown environment(s) {sorted(unknown)}')
    wanted = tuple(p for p in PROJECTS if p[0] in env_filter)

  rows: list[dict] = []
  fetched: set[tuple[str, str]] = set()
  for environment, algorithm, project in wanted:
    rows.extend(extract_project(environment, algorithm, project))
    fetched.add((environment, algorithm))
  if env_filter and os.path.isfile(output_path):
    for row in load_csv(output_path):
      if (row['environment'], row['algorithm']) not in fetched:
        rows.append(row)
  if not rows:
    raise SystemExit('No policy runs with eval/episode_reward were found.')
  rows = restrict_to_local_checkpoints(rows)
  write_csv(rows, output_path)


if __name__ == '__main__':
  main()
