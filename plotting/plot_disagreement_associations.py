"""Rank correlation and mutual information of each disagreement with difficulty.

Two panels. Spearman correlation and mutual information, in nats, between
each initial-state disagreement and the held-out fail rate. The band is the
same association for the search fail rate.
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

_CORRELATIONS = _ROOT / 'figures/predictors/eval_fail_correlations_p32.csv'
_BAND = _ROOT / 'figures/predictors/repeatability_band.csv'


def _read_csv(path: Path, rebuild: str) -> list[dict]:
  if not path.is_file():
    raise FileNotFoundError(f'{path} is missing. {rebuild}')
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def disagreement_associations_figure() -> plt.Figure:
  """The five disagreements against r_E, for every policy set."""
  rows = _read_csv(
      _CORRELATIONS,
      'Rebuild it with `python shared_eval_figures.py`.',
  )
  band = _read_csv(
      _BAND,
      'The r_S–r_E band is figures/predictors/repeatability_band.csv.',
  )
  estimates = {}
  for row in rows:
    if row['feature'] not in figs.DISAGREEMENT:
      continue
    key = (row['feature'], row['algorithm'], row['environment'])
    if row['measure'] == 'spearman':
      estimates[('spearman', *key)] = figs.Estimate.from_se(
          float(row['estimate']), float(row['se']),
      )
    elif row['measure'] == 'mi_classification':
      estimates[('mi', *key)] = figs.Estimate(float(row['estimate']))
  spearman_baseline = {}
  mi_baseline = {}
  for row in band:
    pair = (row['algorithm'], row['environment'])
    spearman_baseline[pair] = figs.Estimate(float(row['spearman']))
    mi_baseline[pair] = figs.Estimate(float(row['mi']))
  if len(spearman_baseline) != 12 or len(estimates) != 12 * 5 * 2:
    raise ValueError(
        f'Expected 12 policy sets and 120 associations, '
        f'got {len(spearman_baseline)} sets and {len(estimates)} associations.'
    )
  return figs.association_figure(
      estimates,
      measures=('spearman', 'mi'),
      baseline=mi_baseline,
      spearman_baseline=spearman_baseline,
  )


if __name__ == '__main__':
  os.chdir(_ROOT)
  path = figs.save(
      disagreement_associations_figure(),
      'figures/predictors/disagreement_associations.pdf',
      png=True,
  )
  print(path)
