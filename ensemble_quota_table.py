"""Fail rate of each ensemble on every search quota.

The cases are the initial states from the finished difficulty-bin searches,
in every environment and every experiment. A quota is one search-ensemble
fail count, from 0/16 through 15/16, plus the all-fail quota. The evaluation
rate of the algorithm that collected the tests is read from the shards. The
other ensembles are rolled out on the same states. Cached rollouts are reused.

``--source ppo`` is the PPO-search table. ``--source sac`` and
``--source ppo_sac`` repeat the measurement on the tests collected by the
SAC and PPO+SAC searches. The ensemble that filled the quota is the row,
so it is not also a column.

The table is the mean fail rate inside each quota. The parenthetical value
is the standard deviation of the five experiment means.

Usage:
  python ensemble_quota_table.py
  python ensemble_quota_table.py --source sac
  python ensemble_quota_table.py --source ppo_sac
  python ensemble_quota_table.py --figures-only
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from cross_ensemble_humanoid import (
    ENSEMBLES,
    ENVIRONMENTS,
    N_POLICIES,
    OUTPUT_DIR,
    load_ensemble_members,
    load_search_cases,
    score_ensemble,
)
from ga.binned import ALL_FAIL_SENTINEL, difficulty_levels, snap_difficulty_bins
from ga.env_setup import MIXED_ALGORITHM

_ENV_LABEL = {
    'ant': 'Ant',
    'humanoid': 'Humanoid',
    'walker2d': 'Walker2d',
    'hopper': 'Hopper',
}
# On SAC-search tests the SAC search fail count is the quota. The stored
# SAC eval rate is the column whose algorithm is None.
SAC_ENSEMBLES = (
    ('ppo_search', 'PPO search', 'ppo', 'search'),
    ('ppo_eval', 'PPO eval', 'ppo', 'eval'),
    ('sac_eval', 'SAC eval', None, None),
    ('ppo_sac_search', 'PPO+SAC search', MIXED_ALGORITHM, 'search'),
    ('ppo_sac_eval', 'PPO+SAC eval', MIXED_ALGORITHM, 'eval'),
)
MIXED_ENSEMBLES = (
    ('ppo_search', 'PPO search', 'ppo', 'search'),
    ('ppo_eval', 'PPO eval', 'ppo', 'eval'),
    ('sac_search', 'SAC search', 'sac', 'search'),
    ('sac_eval', 'SAC eval', 'sac', 'eval'),
    ('ppo_sac_eval', 'PPO+SAC eval', None, None),
)
_SOURCE_ENSEMBLES = {
    'ppo': ENSEMBLES,
    'sac': SAC_ENSEMBLES,
    'ppo_sac': MIXED_ENSEMBLES,
}
_SOURCE_LABEL = {'ppo': 'PPO', 'sac': 'SAC', 'ppo_sac': 'PPO+SAC'}
_SOURCE_CURVES = (
    ('ppo', 'PPO', 'quota_fail_rates.csv'),
    ('sac', 'SAC', 'quota_fail_rates_sac.csv'),
    ('ppo_sac', 'PPO+SAC', 'quota_fail_rates_ppo_sac.csv'),
)
# Search and evaluation of one policy set share a color. Evaluation is solid.
_CURVE_SERIES = (
    ('ppo_eval', 'PPO eval', '#0072B2', '-'),
    ('ppo_search', 'PPO search', '#0072B2', '--'),
    ('sac_eval', 'SAC eval', '#D55E00', '-'),
    ('sac_search', 'SAC search', '#D55E00', '--'),
    ('ppo_sac_eval', 'PPO+SAC eval', '#009E73', '-'),
    ('ppo_sac_search', 'PPO+SAC search', '#009E73', '--'),
)


def _quota_order() -> list[tuple[int, str]]:
  """Bin id and label, from the easiest quota to all-fail.

  ``snap_difficulty_bins`` stores all-fail at id 0 and fail count ``k`` at
  id ``k + 1``.
  """
  order = [(k + 1, f'{k}/{N_POLICIES}') for k in range(N_POLICIES)]
  order.append((0, 'all-fail'))
  return order


def collect_rates(
    environment: str,
    batch: int,
    source: str,
    ensembles: tuple,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Score every requested ensemble on one environment's search tests."""
  tag = 'all'
  cases = load_search_cases(environment, source, None, None)
  rates: dict[str, np.ndarray] = {}
  for key, _label, algorithm, role in ensembles:
    if algorithm is None:
      values = np.asarray(cases['source_eval'], dtype=np.float64)
      print(
          f'{environment} {key}: stored mean={float(np.mean(values)):.4f}',
          flush=True,
      )
    else:
      members = load_ensemble_members(environment, algorithm, role)
      values = score_ensemble(
          environment, key, members, cases, tag, batch, source
      )
    if values.shape != cases['search_difficulty'].shape:
      raise ValueError(
          f'{environment} {key} has shape {values.shape}, '
          f'expected {cases["search_difficulty"].shape}'
      )
    rates[key] = values
  return cases, rates


def quota_means(
    cases: dict[str, np.ndarray],
    rates: dict[str, np.ndarray],
    columns: tuple[str, ...],
) -> list[dict]:
  """One record per experiment and quota, plus the five-experiment summary."""
  difficulty = np.asarray(cases['search_difficulty'], dtype=np.float64)
  experiment = np.asarray(cases['experiment'], dtype=np.int32)
  bin_ids = snap_difficulty_bins(difficulty, N_POLICIES)
  levels = difficulty_levels(N_POLICIES)
  experiments = [int(value) for value in np.unique(experiment)]
  rows = []
  for bin_id, label in _quota_order():
    level = float(levels[bin_id])
    if bin_id == 0 and not (
        level < 0.0 or np.isclose(level, ALL_FAIL_SENTINEL)
    ):
      raise ValueError(f'Bin 0 is {level}, expected the all-fail sentinel.')
    mask = bin_ids == bin_id
    per_experiment = []
    for number in experiments:
      selected = mask & (experiment == number)
      n_cases = int(np.count_nonzero(selected))
      means = {
          key: float(np.mean(rates[key][selected])) if n_cases else float('nan')
          for key in columns
      }
      per_experiment.append(means)
      rows.append({
          'experiment': number,
          'quota': label,
          'bin_id': bin_id,
          'n': n_cases,
          **means,
      })
    summary = {'experiment': 'mean', 'quota': label, 'bin_id': bin_id}
    summary_std = {'experiment': 'std', 'quota': label, 'bin_id': bin_id}
    summary['n'] = int(np.count_nonzero(mask))
    summary_std['n'] = summary['n']
    for key in columns:
      values = np.asarray(
          [item[key] for item in per_experiment], dtype=np.float64
      )
      summary[key] = float(np.mean(values))
      summary_std[key] = (
          float(np.std(values, ddof=1)) if values.size > 1 else float('nan')
      )
    rows.append(summary)
    rows.append(summary_std)
  return rows


def _write_csv(
    path: str,
    tables: dict[str, list[dict]],
    columns: tuple[str, ...],
) -> None:
  header = ['environment', 'experiment', 'quota', 'n', *columns]
  lines = [','.join(header)]
  for environment in ENVIRONMENTS:
    if environment not in tables:
      continue
    for row in tables[environment]:
      cells = [
          environment,
          str(row['experiment']),
          row['quota'],
          str(row['n']),
      ]
      cells.extend(f'{row[key]:.6f}' for key in columns)
      lines.append(','.join(cells))
  temporary = path + '.tmp'
  with open(temporary, 'w') as handle:
    handle.write('\n'.join(lines) + '\n')
  os.replace(temporary, path)


def _fmt(mean: float, std: float) -> str:
  return f'${mean:.3f}\\,({std:.3f})$'


def _write_tex(
    path: str,
    tables: dict[str, list[dict]],
    source: str,
    columns: tuple[str, ...],
    labels: dict[str, str],
) -> None:
  lines = [
      '% Generated by ensemble_quota_table.py',
      '\\documentclass{article}',
      '\\usepackage[margin=0.75in]{geometry}',
      '\\usepackage{booktabs}',
      '\\begin{document}',
      '',
  ]
  for environment in ENVIRONMENTS:
    rows = tables.get(environment)
    if not rows:
      continue
    means = [row for row in rows if row['experiment'] == 'mean']
    stds = [row for row in rows if row['experiment'] == 'std']
    if len(means) != len(stds):
      raise ValueError(f'{environment} mean and std rows differ.')
    label = _ENV_LABEL[environment]
    source_label = _SOURCE_LABEL[source]
    header = ' & '.join(labels[key] for key in columns)
    column_spec = 'l' + 'r' * len(columns)
    lines.extend([
        '\\begin{table}[t]',
        '\\centering',
        '\\small',
        '\\setlength{\\tabcolsep}{5pt}',
        (
            f'\\caption{{{label}. Mean fail rate of each ensemble on the '
            f'tests in one {source_label}-search quota, averaged over the five '
            'experiments. The parenthetical value is the standard deviation '
            'of those five experiment means. A quota is the number of search '
            'policies that failed the test. All-fail is the quota in which '
            'every search policy failed.}'
        ),
        f'\\label{{tab:quota-fail-{source}-{environment}}}',
        f'\\begin{{tabular}}{{{column_spec}}}',
        '\\toprule',
        f'Quota & {header} \\\\',
        '\\midrule',
    ])
    for mean_row, std_row in zip(means, stds):
      cells = ' & '.join(
          _fmt(mean_row[key], std_row[key]) for key in columns
      )
      lines.append(f'{mean_row["quota"]} & {cells} \\\\')
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
        '',
    ])
  lines.append('\\end{document}')
  lines.append('')
  temporary = path + '.tmp'
  with open(temporary, 'w') as handle:
    handle.write('\n'.join(lines))
  os.replace(temporary, path)


def _quota_x(label: str) -> float:
  """Search-ensemble fail rate of a quota label. All-fail is 1."""
  if label == 'all-fail':
    return 1.0
  numerator, denominator = label.split('/')
  return int(numerator) / int(denominator)


def _read_quota_csv(path: str) -> list[dict]:
  import csv
  with open(path, newline='') as handle:
    return list(csv.DictReader(handle))


def plot_quota_curves(path: str | None = None) -> str:
  """Fail rate of every other ensemble against the search-ensemble fail rate.

  Columns are the three search sets. Rows are the environments. A gray line
  is the search quota itself.
  """
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D

  tables = []
  for _source, title, filename in _SOURCE_CURVES:
    csv_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(csv_path):
      raise SystemExit(f'Missing {csv_path}')
    tables.append((title, _read_quota_csv(csv_path)))
  environments = [
      name for name in ENVIRONMENTS
      if any(row['environment'] == name for _title, rows in tables for row in rows)
  ]
  if path is None:
    path = os.path.join(OUTPUT_DIR, 'search_quota_fail_rates.pdf')
  figure, axes = plt.subplots(
      len(environments),
      len(tables),
      figsize=(11.4, 2.05 * len(environments) + 0.7),
      sharex=True,
      sharey=True,
      squeeze=False,
  )
  for column, (title, rows) in enumerate(tables):
    for row_index, environment in enumerate(environments):
      axis = axes[row_index, column]
      env_rows = [row for row in rows if row['environment'] == environment]
      means = [row for row in env_rows if row['experiment'] == 'mean']
      stds = {
          row['quota']: row
          for row in env_rows if row['experiment'] == 'std'
      }
      means.sort(key=lambda row: _quota_x(row['quota']))
      x = np.array([_quota_x(row['quota']) for row in means])
      axis.plot(
          [0.0, 1.0], [0.0, 1.0], color='#c8c8c8', linewidth=0.9, zorder=0,
      )
      for key, _label, color, style in _CURVE_SERIES:
        if key not in means[0]:
          continue
        y = np.array([float(row[key]) for row in means])
        se = np.array([
            float(stds[row['quota']][key]) if row['quota'] in stds else 0.0
            for row in means
        ])
        axis.fill_between(
            x, y - se, y + se, color=color, alpha=0.13, linewidth=0, zorder=1,
        )
        axis.plot(x, y, color=color, linestyle=style, linewidth=1.55, zorder=3)
      if row_index == 0:
        axis.set_title(title, fontsize=11, pad=4)
      axis.set_xlim(-0.02, 1.02)
      axis.set_ylim(-0.04, 1.04)
      axis.set_xticks((0.0, 0.5, 1.0))
      axis.set_yticks((0.0, 0.5, 1.0))
      axis.grid(True, color='#ececec', linewidth=0.6, zorder=0)
      axis.spines['top'].set_visible(False)
      axis.spines['right'].set_visible(False)
      axis.tick_params(length=2.5, width=0.6, labelsize=8)
      for spine in ('left', 'bottom'):
        axis.spines[spine].set_linewidth(0.6)
  for row_index, environment in enumerate(environments):
    axes[row_index, 0].set_ylabel(_ENV_LABEL[environment], fontsize=10)
  for axis in axes[-1]:
    axis.set_xlabel('Search-ensemble fail rate', fontsize=8)
  handles = [
      Line2D([0], [0], color='#b0b0b0', linewidth=1.2, label='Search quota'),
  ]
  handles.extend(
      Line2D(
          [0], [0], color=color, linestyle=style, linewidth=1.7, label=label,
      )
      for _key, label, color, style in _CURVE_SERIES
  )
  figure.legend(
      handles=handles,
      loc='upper center',
      bbox_to_anchor=(0.52, 0.995),
      ncol=4,
      frameon=False,
      fontsize=8.5,
      handlelength=2.5,
      columnspacing=1.15,
  )
  figure.supylabel('Fail rate', fontsize=9, x=0.012)
  figure.subplots_adjust(
      left=0.085, right=0.985, top=0.90, bottom=0.055, hspace=0.22, wspace=0.10,
  )
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  figure.savefig(path)
  plt.close(figure)
  print(f'wrote {path}', flush=True)
  return path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--environment', choices=ENVIRONMENTS)
  parser.add_argument('--source', choices=('ppo', 'sac', 'ppo_sac'), default='ppo')
  parser.add_argument('--batch', type=int, default=6000)
  parser.add_argument(
      '--figures-only',
      action='store_true',
      help='Draw the quota figure from the finished tables.',
  )
  args = parser.parse_args()
  if args.figures_only:
    plot_quota_curves()
    return
  if args.batch < 1:
    raise SystemExit(f'--batch must be >= 1, got {args.batch}')
  ensembles = _SOURCE_ENSEMBLES[args.source]
  columns = tuple(key for key, _label, _algorithm, _role in ensembles)
  labels = {key: label for key, label, _algorithm, _role in ensembles}
  # Humanoid first, then the smaller environments, then Ant.
  environments = (
      (args.environment,)
      if args.environment
      else ('humanoid', 'hopper', 'walker2d', 'ant')
  )
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  suffix = '' if args.source == 'ppo' else f'_{args.source}'
  csv_path = os.path.join(OUTPUT_DIR, f'quota_fail_rates{suffix}.csv')
  tex_path = os.path.join(OUTPUT_DIR, f'quota_fail_rates{suffix}.tex')
  tables: dict[str, list[dict]] = {}
  for environment in environments:
    print(f'=== {environment} {args.source} ===', flush=True)
    cases, rates = collect_rates(environment, args.batch, args.source, ensembles)
    tables[environment] = quota_means(cases, rates, columns)
    _write_csv(csv_path, tables, columns)
    _write_tex(tex_path, tables, args.source, columns, labels)
    print(f'wrote {tex_path}', flush=True)
    for row in tables[environment]:
      if row['experiment'] != 'mean':
        continue
      parts = '  '.join(
          f'{labels[key]}={row[key]:.3f}' for key in columns
      )
      print(f'  {row["quota"]:>8}  n={row["n"]}  {parts}', flush=True)


if __name__ == '__main__':
  main()
