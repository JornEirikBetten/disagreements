"""Mean disagreement at each held-out fail rate.

One panel per disagreement. The horizontal axis is the evaluation fail rate
and the vertical axis is the search-set disagreement, in standard-deviation
units, averaged inside each fail-rate bin.
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

import plotting.style as figs

_CURVES = _ROOT / 'figures/predictors/difficulty_feature_curves_p32.csv'


def _read_csv(path: Path) -> list[dict]:
  if not path.is_file():
    raise FileNotFoundError(
        f'{path} is missing. Rebuild it with `python shared_eval_figures.py`.'
    )
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _curves_from_csv(rows: list[dict]) -> dict:
  grouped: dict[tuple, list[dict]] = {}
  for row in rows:
    if row['feature'] not in figs.DISAGREEMENT:
      continue
    key = (row['environment'], row['feature'], row['algorithm'])
    grouped.setdefault(key, []).append(row)
  curves = {}
  for key, group in grouped.items():
    group.sort(key=lambda row: float(row['eval_fail_rate']))
    curves[key] = figs.Curve(
        [float(row['eval_fail_rate']) for row in group],
        [float(row['mean_sd_units']) for row in group],
        [float(row['se_low']) for row in group],
        [float(row['se_high']) for row in group],
    )
  return curves


def disagreement_by_difficulty_figure() -> plt.Figure:
  """The five disagreement curves for PPO, SAC, and PPO+SAC."""
  curves = _curves_from_csv(_read_csv(_CURVES))
  expected = 4 * 5 * 3
  if len(curves) != expected:
    raise ValueError(f'Expected {expected} disagreement curves, got {len(curves)}.')
  return figs.feature_curves_figure(curves)


if __name__ == '__main__':
  os.chdir(_ROOT)
  path = figs.save(
      disagreement_by_difficulty_figure(),
      'figures/predictors/disagreement_by_difficulty.pdf',
      png=True,
  )
  print(path)
