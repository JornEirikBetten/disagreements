"""Fail rate of every other policy set against the search quota.

One panel per environment. Each curve is a policy set that did not collect
the states, scored on the 17 search-fail-rate quotas. The collecting set
lies on the diagonal.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np

import plotting.style as figs

_SOURCES = (
    ('ppo', _ROOT / 'figures/ensemble_transfer/quota_fail_rates.csv'),
    ('sac', _ROOT / 'figures/ensemble_transfer/quota_fail_rates_sac.csv'),
    ('ppo_sac', _ROOT / 'figures/ensemble_transfer/quota_fail_rates_ppo_sac.csv'),
)
_REBUILD = (
    'Rebuild a missing table with '
    '`python ensemble_quota_table.py --source ppo` '
    '(and `--source sac`, `--source ppo_sac`).'
)


def _read_csv(path: Path) -> list[dict]:
  if not path.is_file():
    raise FileNotFoundError(f'{path} is missing. {_REBUILD}')
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _quota_x(label: str) -> float:
  """Search-ensemble fail rate. The all-fail quota is 1."""
  if label == 'all-fail':
    return 1.0
  numerator, denominator = label.split('/')
  return int(numerator) / int(denominator)


def _split_ensemble(column: str) -> tuple[str, str]:
  if column.endswith('_eval'):
    return column[: -len('_eval')], 'eval'
  if column.endswith('_search'):
    return column[: -len('_search')], 'search'
  raise ValueError(f'Unrecognised ensemble column {column!r}.')


def failure_rate_transfer_figure() -> plt.Figure:
  """Mean fail rate of each other ensemble, with a one-standard-deviation band."""
  curves = {}
  for source, path in _SOURCES:
    rows = _read_csv(path)
    means = [row for row in rows if row['experiment'] == 'mean']
    spreads = {
        (row['environment'], row['quota']): row
        for row in rows if row['experiment'] == 'std'
    }
    ensemble_columns = [
        name for name in rows[0]
        if name.endswith('_eval') or name.endswith('_search')
    ]
    for environment in figs.ENVIRONMENT_ORDER:
      chosen = [row for row in means if row['environment'] == environment]
      chosen.sort(key=lambda row: _quota_x(row['quota']))
      if len(chosen) != 17:
        raise ValueError(
            f'{path.name} {environment} has {len(chosen)} quotas, expected 17.'
        )
      x = np.array([_quota_x(row['quota']) for row in chosen])
      for column in ensemble_columns:
        policy_set, role = _split_ensemble(column)
        y = np.array([float(row[column]) for row in chosen])
        half = np.array([
            float(spreads[(environment, row['quota'])][column])
            for row in chosen
        ])
        curves[(environment, source, (policy_set, role))] = figs.Curve(
            x, y, y - half, y + half,
        )
  return figs.transfer_figure(curves)


if __name__ == '__main__':
  os.chdir(_ROOT)
  path = figs.save(
      failure_rate_transfer_figure(),
      'figures/ensemble_transfer/failure_rate_transfer.pdf',
      png=True,
  )
  print(path)
