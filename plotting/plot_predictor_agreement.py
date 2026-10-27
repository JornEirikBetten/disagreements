"""Cross-validated agreement of the extra-trees predictor with difficulty.

One point per policy set. The point is the Spearman correlation of the
stitched out-of-fold predictions with the held-out fail rate. The whisker
is the standard error of the fold-wise coefficients. A star is the same
coefficient between the search fail rate and the held-out fail rate.
"""

from __future__ import annotations

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

_DISAGREEMENT = 'disagreement'


def _estimate(value: float, se: float) -> figs.Estimate:
  if np.isfinite(se):
    return figs.Estimate.from_se(value, se)
  return figs.Estimate(value)


def predictor_agreement_figure(
    summaries: list[dict],
    search_eval: list[dict] | None = None,
) -> plt.Figure:
  """Spearman agreement of the 500-tree disagreement model with r_E."""
  estimates = {}
  for row in summaries:
    if row['features'] != _DISAGREEMENT:
      continue
    if not np.isfinite(float(row['spearman'])):
      continue
    key = (row['algorithm'], row['environment'])
    estimates[('spearman', 'disagreement', *key)] = _estimate(
        float(row['spearman']), float(row['spearman_se']),
    )
  search_correlations = None
  if search_eval:
    search_correlations = {}
    for row in search_eval:
      value = row.get('spearman')
      if value is None or not np.isfinite(float(value)):
        continue
      search_correlations[(row['algorithm'], row['environment'])] = float(value)
  return figs.model_score_figure(
      estimates,
      models=('disagreement',),
      panels=('spearman',),
      panel_titles={'spearman': 'Search-solvable states'},
      ylabels={'spearman': r'cross-validated $\rho$ with $r_E$'},
      environment_labels=True,
      search_correlations=search_correlations,
      marker_scale=1.35,
      star_markersize=5.4,
  )


def write_score_figure(
    summaries: list[dict],
    path: str,
    *,
    search_eval: list[dict] | None = None,
) -> str:
  """Draw the predictor agreement and write it to ``path``."""
  return figs.save(
      predictor_agreement_figure(summaries, search_eval),
      path,
      png=True,
  )


if __name__ == '__main__':
  import csv

  os.chdir(_ROOT)
  archived = _ROOT / 'figures/predictors/extra_trees_10_foldcv_experiment'
  retrained = _ROOT / 'figures/predictors/extra_trees_5_foldcv_experiment'
  stem = archived if (archived.parent / (archived.name + '_scores.csv')).is_file() else retrained
  scores_path = stem.parent / (stem.name + '_scores.csv')
  ceiling_path = stem.parent / (stem.name + '_search_eval_spearman.csv')
  with scores_path.open(newline='') as handle:
    scores = list(csv.DictReader(handle))
  with ceiling_path.open(newline='') as handle:
    ceiling = list(csv.DictReader(handle))
  # The score table stores numbers as strings.
  for row in scores:
    for key in ('spearman', 'spearman_se'):
      row[key] = float(row[key]) if row.get(key) not in (None, '') else float('nan')
  for row in ceiling:
    row['spearman'] = float(row['spearman']) if row.get('spearman') not in (None, '') else float('nan')
  written = write_score_figure(
      scores,
      'figures/predictors/predictor_agreement.pdf',
      search_eval=ceiling,
  )
  print(written)
