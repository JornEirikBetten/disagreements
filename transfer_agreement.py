"""Spearman correlation and Lin's CCC for the transfer pairs in Table A1.

Table A1 reports Spearman correlation between the collecting ensemble's
search fail rate ``r_S`` and every other ensemble's fail rate. Spearman
measures whether the two rates order the cases the same way. Lin's
concordance correlation coefficient measures absolute agreement: it is 1
when the two rates take the same value.

This script reads the cached case-level rates under
``figures/ensemble_transfer`` and writes one table. Each cell stacks the
published Spearman correlation above Lin's CCC, with the same dashes and
the same parenthetical bootstrap uncertainty as Table A1. The mean absolute
difference is written to the CSV beside those two coefficients.

The sample is the 160,000 solvable cases (the all-fail quota is excluded).
The script does not roll out policies. A missing cache is an error.

Usage:
  python transfer_agreement.py
  python transfer_agreement.py --boot 1000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ga.binned import ALL_FAIL_SENTINEL

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'figures' / 'ensemble_transfer'
TEX_PATH = ROOT / 'transfer_agreement.tex'
CSV_PATH = CACHE_DIR / 'transfer_agreement.csv'

ENVIRONMENTS = ('hopper', 'walker2d', 'humanoid', 'ant')
ENV_LABEL = {
    'hopper': 'Hopper',
    'walker2d': 'Walker2d',
    'humanoid': 'Humanoid',
    'ant': 'Ant',
}
SOURCES = ('ppo', 'sac', 'ppo_sac')
SOURCE_LABEL = {'ppo': 'PPO', 'sac': 'SAC', 'ppo_sac': 'PPO+SAC'}

# Column order matches transfer_value_correlations.tex.
COLUMNS = (
    ('ppo_eval', 'Evaluation'),
    ('sac_eval', 'Evaluation'),
    ('ppo_sac_eval', 'Evaluation'),
    ('ppo_search', 'Search'),
    ('sac_search', 'Search'),
)

# Table A1 cells, including the parenthetical bootstrap digit. ``None`` is a dash.
# The leading number is checked against the recomputed Spearman correlation.
PUBLISHED_SPEARMAN = {
    ('hopper', 'ppo'): ('0.827(5)', '0.718(8)', '0.79(1)', None, '0.833(5)'),
    ('hopper', 'sac'): ('0.828(5)', '0.802(5)', '0.845(7)', '0.808(9)', None),
    ('hopper', 'ppo_sac'): ('0.842(4)', '0.805(6)', '0.844(7)', None, None),
    ('walker2d', 'ppo'): ('0.914(1)', '0.851(2)', '0.8942(7)', None, '0.856(2)'),
    ('walker2d', 'sac'): ('0.841(1)', '0.897(1)', '0.8851(9)', '0.858(2)', None),
    ('walker2d', 'ppo_sac'): ('0.889(3)', '0.893(1)', '0.902(1)', None, None),
    ('humanoid', 'ppo'): ('0.895(1)', '0.882(2)', '0.9046(4)', None, '0.878(2)'),
    ('humanoid', 'sac'): ('0.843(2)', '0.8842(3)', '0.8790(3)', '0.8691(8)', None),
    ('humanoid', 'ppo_sac'): ('0.876(1)', '0.8786(7)', '0.888(2)', None, None),
    ('ant', 'ppo'): ('0.9032(5)', '0.697(1)', '0.8383(8)', None, '0.687(2)'),
    ('ant', 'sac'): ('0.775(4)', '0.9076(9)', '0.871(2)', '0.773(7)', None),
    ('ant', 'ppo_sac'): ('0.879(2)', '0.8497(9)', '0.9161(9)', None, None),
}


def omitted(source: str, column: str) -> bool:
  """Dash in Table A1: the collecting search, or a search that shares its policies."""
  if source == 'ppo' and column == 'ppo_search':
    return True
  if source == 'sac' and column == 'sac_search':
    return True
  if source == 'ppo_sac' and column in ('ppo_search', 'sac_search'):
    return True
  return False


def _cases_path(environment: str, source: str) -> Path:
  return CACHE_DIR / f'{environment}_{source}_cases_all.npz'


def _rates_path(environment: str, source: str, column: str) -> Path:
  return CACHE_DIR / f'{environment}_{source}_{column}_all.npy'


def load_pair(environment: str, source: str, column: str):
  """Solvable ``(r_S, other fail rate, experiment id)`` for one table cell."""
  path = _cases_path(environment, source)
  if not path.is_file():
    raise FileNotFoundError(f'missing case cache {path}')
  with np.load(path) as archive:
    keys = set(archive.files)
    if 'ppo_search' in keys:
      search = np.asarray(archive['ppo_search'], dtype=np.float64)
    elif 'search_difficulty' in keys:
      search = np.asarray(archive['search_difficulty'], dtype=np.float64)
    else:
      raise KeyError(f'{path} has no search fail rate')
    experiment = np.asarray(archive['experiment'])
    own_eval = _own_eval(source, column, keys)
    if own_eval is not None:
      other = np.asarray(archive[own_eval], dtype=np.float64)
    else:
      rate_path = _rates_path(environment, source, column)
      if not rate_path.is_file():
        raise FileNotFoundError(f'missing rate cache {rate_path}')
      other = np.asarray(np.load(rate_path), dtype=np.float64)
  if other.shape != search.shape:
    raise ValueError(
        f'{environment} {source} {column}: rate length {other.shape[0]} '
        f'!= {search.shape[0]} cases'
    )
  solvable = search >= 0.0
  if np.any(np.isclose(search[solvable], ALL_FAIL_SENTINEL)):
    raise ValueError(f'{environment} {source}: all-fail sentinel among solvable cases')
  return search[solvable], other[solvable], experiment[solvable]


def _own_eval(source: str, column: str, keys: set[str]) -> str | None:
  """Archive field that already stores this source's own evaluation rate."""
  if source == 'ppo' and column == 'ppo_eval' and 'ppo_eval' in keys:
    return 'ppo_eval'
  if source == 'sac' and column == 'sac_eval' and 'source_eval' in keys:
    return 'source_eval'
  if source == 'ppo_sac' and column == 'ppo_sac_eval' and 'source_eval' in keys:
    return 'source_eval'
  return None


def cluster_moments(search: np.ndarray, other: np.ndarray, experiment: np.ndarray):
  """Per-experiment sums. Agreement of a resample is a weighted sum of these."""
  order = []
  moments = []
  for experiment_id in np.unique(experiment):
    mask = experiment == experiment_id
    x = search[mask]
    y = other[mask]
    order.append(int(experiment_id))
    moments.append((
        float(x.size),
        float(x.sum()),
        float(y.sum()),
        float(np.dot(x, x)),
        float(np.dot(y, y)),
        float(np.dot(x, y)),
        float(np.abs(x - y).sum()),
    ))
  return order, moments


def agreement_from_counts(counts: np.ndarray, moments: list[tuple]) -> tuple[float, float]:
  """Lin's CCC and MAE on a multiset of experiments.

  CCC uses Lin's divisor-n moments:
  ``2 cov / (var(x) + var(y) + (mean(x) - mean(y))^2)``.
  """
  n = sx = sy = sx2 = sy2 = sxy = sab = 0.0
  for count, (ni, sxi, syi, sx2i, sy2i, sxyi, sabi) in zip(counts, moments):
    if count == 0:
      continue
    weight = float(count)
    n += weight * ni
    sx += weight * sxi
    sy += weight * syi
    sx2 += weight * sx2i
    sy2 += weight * sy2i
    sxy += weight * sxyi
    sab += weight * sabi
  if n <= 0.0:
    return float('nan'), float('nan')
  mean_x = sx / n
  mean_y = sy / n
  var_x = sx2 / n - mean_x * mean_x
  var_y = sy2 / n - mean_y * mean_y
  cov = sxy / n - mean_x * mean_y
  denom = var_x + var_y + (mean_x - mean_y) ** 2
  if denom <= 0.0:
    ccc = 1.0 if var_x <= 0.0 and var_y <= 0.0 else 0.0
  else:
    ccc = 2.0 * cov / denom
  return float(ccc), float(sab / n)


def bootstrap(moments: list[tuple], n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  """CCC and MAE on ``n_boot`` resamples of the five searches."""
  n_clusters = len(moments)
  rng = np.random.default_rng(seed)
  draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
  ccc = np.empty(n_boot, dtype=np.float64)
  mae = np.empty(n_boot, dtype=np.float64)
  for index, draw in enumerate(draws):
    counts = np.bincount(draw, minlength=n_clusters)
    ccc[index], mae[index] = agreement_from_counts(counts, moments)
  return ccc, mae


def compact_cell(estimate: float, se: float) -> str:
  """``0.827(5)`` means ``0.827 ± 0.005``, matching Table A1."""
  if not np.isfinite(estimate):
    return '---'
  if not np.isfinite(se) or se <= 0.0:
    return f'${float(estimate):.3f}$'
  exponent = int(np.floor(np.log10(se)))
  digit = int(round(se / 10.0 ** exponent))
  if digit >= 10:
    exponent += 1
    digit = 1
  if digit <= 0:
    digit = 1
  decimals = max(0, -exponent)
  shown = round(float(estimate), decimals)
  if abs(shown) < 0.5 * 10.0 ** (-decimals):
    shown = 0.0
  return f'${shown:.{decimals}f}({digit})$'


def _published_point(text: str) -> float:
  return float(text.split('(', 1)[0])


def _published_tolerance(text: str) -> float:
  point = text.split('(', 1)[0]
  decimals = len(point.split('.')[1]) if '.' in point else 0
  return 0.5 * 10.0 ** (-decimals)


def measure_cell(environment: str, source: str, column: str, n_boot: int, seed: int) -> dict:
  search, other, experiment = load_pair(environment, source, column)
  experiment_ids, moments = cluster_moments(search, other, experiment)
  counts = np.ones(len(moments), dtype=np.int64)
  ccc, mae = agreement_from_counts(counts, moments)
  ccc_draws, mae_draws = bootstrap(moments, n_boot, seed)
  spearman = float(spearmanr(search, other).statistic)
  return {
      'environment': environment,
      'source': source,
      'column': column,
      'n': int(search.size),
      'n_experiments': len(experiment_ids),
      'experiments': experiment_ids,
      'spearman': spearman,
      'ccc': ccc,
      'ccc_sd': float(np.std(ccc_draws, ddof=1)),
      'mae': mae,
      'mae_sd': float(np.std(mae_draws, ddof=1)),
  }


def check_spearman(rows: list[dict]) -> None:
  """The recomputed rank correlation has to match Table A1's printed estimates."""
  by_key = {(row['environment'], row['source'], row['column']): row for row in rows}
  mismatches = []
  for (environment, source), published in PUBLISHED_SPEARMAN.items():
    for (column, _group), text in zip(COLUMNS, published):
      if text is None:
        continue
      row = by_key[(environment, source, column)]
      gap = abs(row['spearman'] - _published_point(text))
      if gap > _published_tolerance(text):
        mismatches.append(
            f'{environment} {source} {column}: spearman {row["spearman"]:.6f} '
            f'vs Table A1 {text}'
        )
  if mismatches:
    raise SystemExit(
        'Spearman does not match Table A1, so CCC and MAE are not on that sample:\n'
        + '\n'.join(mismatches)
    )


def write_csv(rows: list[dict], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fields = (
      'environment', 'source', 'column', 'n', 'n_experiments',
      'spearman', 'ccc', 'ccc_sd', 'mae', 'mae_sd',
  )
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row[field] for field in fields})


def _lookup(rows: list[dict], environment: str, source: str, column: str) -> dict | None:
  for row in rows:
    if (
        row['environment'] == environment
        and row['source'] == source
        and row['column'] == column
    ):
      return row
  return None


def _pair(spearman: str, ccc: str) -> str:
  """One cell: Spearman $\\rho$ above Lin's $\\rho_c$."""
  return rf'\pair{{${spearman}$}}{{${ccc}$}}'


def write_tex(rows: list[dict], path: Path, n_boot: int) -> None:
  n_cases = f'{rows[0]["n"]:,}' if rows else '0'
  caption = (
      "Spearman rank correlation $\\rho$ (upper) and Lin's concordance "
      "correlation coefficient $\\rho_c$ (lower) between the collecting "
      "ensemble's fail rate $r_S$ and each other ensemble's fail rate. "
      'The row names the ensemble that collected the cases. '
      f'The sample is the {n_cases} solvable test cases; the all-fail quota is excluded. '
      '$\\rho$ measures agreement of the ordering. '
      '$\\rho_c$ equals 1 when the two rates take the same value, so it measures '
      'absolute agreement. '
      'The upper entries are the correlations in '
      'Table~\\ref{tab:transfer-value-correlations}. '
      'A parenthetical digit is the standard deviation of the experiment-clustered '
      'bootstrap over resamples of the five searches, in the last decimal place'
      f' ($B={n_boot}$ for $\\rho_c$): '
      '$0.827(5)$ means $0.827 \\pm 0.005$. '
      'A dash marks the collecting search, or a search that shares policies with it.'
  )
  lines = [
      '% Generated by transfer_agreement.py',
      '\\documentclass{article}',
      '\\usepackage[margin=0.75in]{geometry}',
      '\\usepackage{booktabs}',
      '\\usepackage{amsmath}',
      '\\newcommand{\\pair}[2]{%',
      '  \\begin{tabular}[c]{@{}c@{}}#1\\\\[0.65ex]#2\\end{tabular}%',
      '}',
      '\\begin{document}',
      '',
      '\\begin{table}[t]',
      '\\centering',
      '\\small',
      '\\setlength{\\tabcolsep}{6pt}',
      '\\setlength{\\belowcaptionskip}{9pt}',
      '\\renewcommand{\\arraystretch}{1.05}',
      '\\caption{' + caption + '}',
      '\\label{tab:transfer-agreement}',
      '\\begin{tabular}{ll ccc cc}',
      '\\toprule',
      '& & \\multicolumn{3}{c}{Evaluation} & \\multicolumn{2}{c}{Search} \\\\',
      '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}',
      'Environment & $r_S$ from & PPO & SAC & PPO+SAC & PPO & SAC \\\\',
      '\\midrule',
  ]
  for environment in ENVIRONMENTS:
    for source_index, source in enumerate(SOURCES):
      published = PUBLISHED_SPEARMAN[(environment, source)]
      cells = []
      for (column, _group), spearman in zip(COLUMNS, published):
        if spearman is None:
          cells.append('---')
          continue
        row = _lookup(rows, environment, source, column)
        ccc = compact_cell(row['ccc'], row['ccc_sd']).strip('$')
        cells.append(_pair(spearman, ccc))
      env_cell = ENV_LABEL[environment] if source_index == 1 else ''
      lines.append(
          f'{env_cell} & {SOURCE_LABEL[source]} & ' + ' & '.join(cells) + r' \\'
      )
      if source_index < len(SOURCES) - 1:
        lines.append('\\addlinespace[0.45em]')
    if environment != ENVIRONMENTS[-1]:
      lines.append('\\addlinespace[1.05em]')
  lines.extend([
      '\\bottomrule',
      '\\end{tabular}',
      '\\end{table}',
      '',
      '\\end{document}',
      '',
  ])
  path.write_text('\n'.join(lines))


def collect(n_boot: int, seed: int) -> list[dict]:
  rows = []
  for environment in ENVIRONMENTS:
    for source in SOURCES:
      for column_index, (column, _group) in enumerate(COLUMNS):
        if omitted(source, column):
          continue
        row = measure_cell(environment, source, column, n_boot, seed + column_index)
        rows.append(row)
        print(
            f'{ENV_LABEL[environment]:9} {SOURCE_LABEL[source]:8} {column:14} '
            f'n={row["n"]}  spearman={row["spearman"]:.4f}  '
            f'CCC={compact_cell(row["ccc"], row["ccc_sd"])}  '
            f'MAE={compact_cell(row["mae"], row["mae_sd"])}',
            flush=True,
        )
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--boot', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  rows = collect(args.boot, args.seed)
  check_spearman(rows)
  write_csv(rows, CSV_PATH)
  write_tex(rows, TEX_PATH, args.boot)
  print(f'wrote {CSV_PATH}')
  print(f'wrote {TEX_PATH}')


if __name__ == '__main__':
  main()
