"""Appendix strips of Humanoid evaluation length and return.

Reads ``figures/policy_performances/humanoid_episode_lengths.csv``. Each row
is one of the 32 PPO or 32 SAC policies in the search and evaluation sets.
Both quantities are taken at the evaluation with the highest
``eval/episode_reward``, the checkpoint that was kept.
``best_avg_episode_length`` is ``eval/avg_episode_length``.
``best_eval_episode_reward`` is the accumulated reward, ``eval/episode_reward``.

The figure is drawn at the size it should be printed. Include it with
``\\includegraphics`` and no scaling.
"""

from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np

import plotting.style as figs

CSV_PATH = os.path.join(
    'figures', 'policy_performances', 'humanoid_episode_lengths.csv')
OUTPUT_PATH = os.path.join(
    'figures', 'policy_performances', 'humanoid_episode_lengths.pdf')

HORIZON = 1000
ALGORITHMS = ('ppo', 'sac')


def load(path: str = CSV_PATH) -> dict[str, dict[str, np.ndarray]]:
  columns = {
      alg: {'length': [], 'return': []} for alg in ALGORITHMS
  }
  with open(path, newline='') as handle:
    for row in csv.DictReader(handle):
      columns[row['algorithm']]['length'].append(
          float(row['best_avg_episode_length']))
      columns[row['algorithm']]['return'].append(
          float(row['best_eval_episode_reward']))
  return {
      alg: {key: np.asarray(values) for key, values in series.items()}
      for alg, series in columns.items()
  }


def _strip(ax: plt.Axes, series: dict[str, np.ndarray]) -> None:
  """Draw one horizontal strip. PPO is the upper row."""
  rows = {algorithm: len(ALGORITHMS) - 1 - index
          for index, algorithm in enumerate(ALGORITHMS)}
  rng = np.random.default_rng(0)
  for algorithm in ALGORITHMS:
    index = rows[algorithm]
    values = series[algorithm]
    style = figs.POLICY_SETS[algorithm]
    jitter = rng.uniform(-0.13, 0.13, size=values.size)
    ax.scatter(
        values,
        np.full(values.size, index) + jitter,
        s=14,
        color=style.color,
        edgecolor='white',
        linewidth=0.35,
        zorder=3,
    )
    mean = float(values.mean())
    ax.plot(
        [mean, mean],
        [index - 0.22, index + 0.22],
        color=figs.INK,
        linewidth=1.15,
        solid_capstyle='butt',
        zorder=4,
    )
    ax.text(
        mean, index + 0.28, f'{mean:.0f}',
        ha='center', va='bottom', color=figs.INK, fontsize=6.5, zorder=5,
    )
  ax.set_ylim(-0.55, 1.62)
  ax.set_yticks([rows['ppo'], rows['sac']], ['PPO', 'SAC'])
  ax.yaxis.grid(False)
  ax.tick_params(axis='y', length=0)


@figs.styled
def episode_length_figure(
    policies: dict[str, dict[str, np.ndarray]],
) -> plt.Figure:
  fig = figs.new_figure(height=figs.TEXT_WIDTH * 0.78, width=figs.TEXT_WIDTH)
  length_axis, return_axis = fig.subplots(2, 1)

  _strip(length_axis, {alg: policies[alg]['length'] for alg in ALGORITHMS})
  length_axis.axvline(
      HORIZON, color=figs.REFERENCE, linewidth=0.6, linestyle='--', zorder=1)
  length_axis.text(
      HORIZON - 8, 1.55, 'horizon',
      ha='right', va='top', color=figs.MUTED, fontsize=6.5,
  )
  length_axis.set_xlim(0, HORIZON + 40)
  length_axis.set_xticks([0, 200, 400, 600, 800, 1000])
  length_axis.set_xlabel('Average episode lengths (control steps)')
  for tick in (200, 400, 600, 800):
    length_axis.axvline(tick, color=figs.GRID, linewidth=0.5, zorder=0)

  _strip(return_axis, {alg: policies[alg]['return'] for alg in ALGORITHMS})
  return_axis.set_xlim(0, 17000)
  return_axis.set_xticks([0, 5000, 10000, 15000])
  return_axis.set_xlabel('Average episode return')
  for tick in (5000, 10000, 15000):
    return_axis.axvline(tick, color=figs.GRID, linewidth=0.5, zorder=0)
  figs.label_panels((length_axis, return_axis))
  return fig


def main() -> None:
  policies = load()
  fig = episode_length_figure(policies)
  figs.save(fig, OUTPUT_PATH, png=True)
  for algorithm in ALGORITHMS:
    length = policies[algorithm]['length']
    episode_return = policies[algorithm]['return']
    print(
        f'{algorithm}: n={length.size} '
        f'length {length.mean():.1f} ({length.min():.0f}–{length.max():.0f}) '
        f'return {episode_return.mean():.0f} '
        f'({episode_return.min():.0f}–{episode_return.max():.0f})')
  print(OUTPUT_PATH)


if __name__ == '__main__':
  main()
