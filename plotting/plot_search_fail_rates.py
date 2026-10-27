"""Search and evaluation fail rates over 100 generations of each fitness.

Columns are Hopper, Walker2d, Ant, and Humanoid. Rows are the search-set
fail rate and the held-out fail rate. Humanoid uses the PPO search set.
The other three use SAC. Search fail-rate and criticality share one initial
population. Attribution is maximized except in Humanoid, where it is minimized.
"""

from __future__ import annotations

import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plotting.style as figs

# Experiments 100–104. Generations are 0 .. 100.
FITNESS_EXPERIMENTS = (100, 101, 102, 103, 104)
FITNESS_GENERATIONS = 101
OUTPUT = 'figures/feature_search/fitness_experiments_100_102.pdf'
FITNESS_SERIES = (
    ('search_difficulty', 'quad', r'Search fail-rate, $r_S$', '#D55E00', '-'),
    ('max_action_disagreement', 'feature', 'Action direction disagreement', '#E69F00', '--'),
    ('max_action_scale_disagreement', 'feature', 'Action magnitude disagreement', '#009E73', '-.'),
    ('max_value_disagreement', 'feature', 'Value estimate disagreement', '#56B4E9', '-'),
    ('max_criticality_disagreement', 'quad', 'Criticality disagreement', '#0072B2', '--'),
    ('feature_attribution_disagreement', 'attribution', 'Attribution disagreement', '#CC79A7', ':'),
)


def _series_location(environment: str, slug: str, source: str) -> tuple[str, str]:
  """Folder for one fitness. Attribution's direction depends on the environment."""
  if source != 'attribution':
    return slug, source
  if environment == 'humanoid':
    return 'min_feature_attribution_disagreement', 'feature'
  return 'max_feature_attribution_disagreement', 'feature'


def _rates_path(
    environment: str,
    algorithm: str,
    experiment: int,
    slug: str,
    source: str,
) -> str:
  slug, source = _series_location(environment, slug, source)
  if source == 'quad':
    from ga.quad_fitness_search import quad_suite_paths
    _suite, rates = quad_suite_paths(environment, algorithm, experiment, 15, slug)
    return rates
  return os.path.join(
      'test_suites',
      slug,
      environment,
      algorithm,
      'rollout_length=15',
      f'experiment_{int(experiment)}',
      'failure_rates.csv',
  )


def _rate_grid(path: str, column: str) -> np.ndarray:
  """One experiment's mean fail rate on generations 0..100. Missing generations are NaN."""
  grid = np.full(FITNESS_GENERATIONS, np.nan)
  if not os.path.isfile(path):
    return grid
  with open(path, newline='') as handle:
    for row in csv.DictReader(handle):
      generation = int(row['generation'])
      if 0 <= generation < FITNESS_GENERATIONS:
        grid[generation] = float(row[column])
  return grid


def _mean_rate_curve(runs: np.ndarray) -> figs.Curve:
  """Mean and one standard deviation across experiments."""
  count = np.sum(np.isfinite(runs), axis=0).astype(float)
  total = np.nansum(runs, axis=0)
  mean = np.full(runs.shape[1], np.nan)
  present = count >= 1
  mean[present] = total[present] / count[present]
  centered = np.where(np.isfinite(runs), runs - mean, 0.0)
  squared = np.sum(centered * centered, axis=0)
  spread = np.full(runs.shape[1], np.nan)
  paired = count >= 2
  spread[paired] = np.sqrt(squared[paired] / (count[paired] - 1.0))
  generations = np.arange(FITNESS_GENERATIONS, dtype=float)
  return figs.Curve(generations, mean, mean - spread, mean + spread)


def search_fail_rates_figure() -> plt.Figure:
  """Mean suite fail rate over experiments 100–104, for six fitnesses."""
  fig = figs.new_figure(2 * 1.35 + 1.45)
  axes = fig.subplots(2, 4, sharex=True, sharey=True, squeeze=False)
  rows = (
      ('search_fail_rate', r'$r_S$'),
      ('eval_fail_rate', r'$r_E$'),
  )
  for column, environment in enumerate(figs.ENVIRONMENT_ORDER):
    algorithm = 'ppo' if environment == 'humanoid' else 'sac'
    for row, (rate_name, ylabel) in enumerate(rows):
      ax = axes[row, column]
      figs.style_axis(ax, grid='y')
      ax.set_xlim(0, 100)
      # The spine sits below zero so a rate of exactly zero stays visible.
      ax.set_ylim(-0.06, 1)
      ax.set_yticks((0, 0.5, 1))
      for slug, source, _label, color, linestyle in FITNESS_SERIES:
        runs = np.vstack([
            _rate_grid(
                _rates_path(environment, algorithm, experiment, slug, source),
                rate_name,
            )
            for experiment in FITNESS_EXPERIMENTS
        ])
        curve = _mean_rate_curve(runs)
        width = figs.LINE_WIDTH + 0.8
        ax.fill_between(
            curve.x, curve.low, curve.high,
            color=color, alpha=figs.BAND_ALPHA, linewidth=0, zorder=2,
        )
        ax.plot(
            curve.x, curve.y,
            color='white', linestyle=linestyle,
            linewidth=width + 1.3,
            solid_capstyle='round', zorder=3,
        )
        ax.plot(
            curve.x, curve.y,
            color=color, linestyle=linestyle, linewidth=width,
            solid_capstyle='round', zorder=4,
        )
      if row == 0:
        ax.set_title(
            figs.ENVIRONMENTS[environment].label,
            fontsize=plt.rcParams['axes.labelsize'],
        )
      if column == 0:
        ax.set_ylabel(ylabel)
      if row == len(rows) - 1:
        ax.set_xlabel('Generation')
      ax.text(
          0.04, 0.96, 'abcdefgh'[row * 4 + column],
          transform=ax.transAxes, ha='left', va='top',
          fontweight='bold', fontsize=8.5, zorder=5,
      )
  # Matplotlib fills the legend down each column. Walk rows first so
  # the labels read left to right in FITNESS_SERIES order.
  legend_columns = 2
  legend_rows = (len(FITNESS_SERIES) + legend_columns - 1) // legend_columns
  handles = []
  for column in range(legend_columns):
    for row in range(legend_rows):
      index = row * legend_columns + column
      if index >= len(FITNESS_SERIES):
        continue
      _slug, _source, label, color, linestyle = FITNESS_SERIES[index]
      handles.append(Line2D(
          [], [], color=color, linestyle=linestyle,
          linewidth=figs.LINE_WIDTH + 0.8, label=label,
      ))
  fig.legend(
      handles=handles,
      loc='outside upper center',
      ncols=legend_columns,
      frameon=False,
      fontsize=8.0,
      columnspacing=0.9,
      handlelength=2.2,
      handletextpad=0.4,
  )
  return fig


def save() -> None:
  os.chdir(_ROOT)
  fig = search_fail_rates_figure()
  figs.save(fig, OUTPUT, png=True)
  print(OUTPUT)


if __name__ == '__main__':
  save()
