"""Association of search-set disagreement with held-out fail rate.

The research question is how disagreement among the *search* policies at an
initial state relates to that state's general difficulty --- the fail rate
of a disjoint evaluation ensemble drawn from the same policy population.
The search fail rate is how the cases were sampled; it is not a predictor
and it is not partialled out of these coefficients.

Each distinct initial state is kept once per experiment. Exact repeats of
an elite (the genetic search used to re-collect them every generation) are
dropped, including on collections that were gathered before the search
started refusing duplicates. Uncertainty is an experiment-clustered
bootstrap: whole independent GA runs are resampled, because lineages cross
generations and a generation cluster would treat a parent and its child as
independent.

Measures
--------
Pearson, Spearman, and Kendall describe monotone association. Mutual
information is included because some disagreement features are not monotone
in difficulty (they peak in the middle of the fail-rate range). Two
estimators are reported, both in nats with the same neighbor count. The
regression estimate treats the fail rate as continuous (Kraskov). The
classification estimate treats the fail count as discrete (Ross). The
semi-partial Spearman of each feature, residualised on the others, is
the joint-structure summary: it asks what that feature adds once the rest
are known, without putting the search fail rate in the model.

Usage:
  python eval_fail_correlations.py
  python eval_fail_correlations.py --environment hopper --algorithm ppo

After a new unique search:

  python add_heldout_fail_rates.py --environment ENV --algorithm ALG
  python build_predictor_datasets.py --environment ENV --algorithm ALG
  python eval_fail_correlations.py --environment ENV --algorithm ALG
  python cv_difficulty_predictors.py --environment ENV --algorithm ALG
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, rankdata
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from analysis.predictor_dataset import (
    collapse_duplicate_states,
    load_predictor_dataset,
    predictor_dataset_path,
)
from train_difficulty_predictors import (
    ALGORITHMS,
    ENVIRONMENTS,
    FEATURE_COLUMNS_USED,
    FEATURE_LABELS_USED,
    _load_eval_fail_rates,
    _search_solvable,
)

FIGURE_DIR = os.path.join('figures', 'predictors')
CSV_PATH = os.path.join(FIGURE_DIR, 'eval_fail_correlations.csv')
TEX_PATH = os.path.join(FIGURE_DIR, 'eval_fail_correlations.tex')
N_BOOT = 1000
MEASURES = (
    ('pearson', "Pearson $r$"),
    ('spearman', "Spearman $\\rho$"),
    ('kendall', "Kendall $\\tau_b$"),
    ('mi_regression', 'MI, continuous'),
    ('mi_classification', 'MI, discrete'),
    ('semipartial_spearman', r'Semi-partial $\rho$'),
)
_BOOTSTRAP_MEASURES = (
    'pearson',
    'spearman',
    'kendall',
    'semipartial_spearman',
)
_ENV_DISPLAY = {
    'ant': 'Ant',
    'humanoid': 'Humanoid',
    'walker2d': 'Walker2d',
    'hopper': 'Hopper',
}
_ALG_DISPLAY = {'ppo': 'PPO', 'sac': 'SAC', 'ppo_sac': 'PPO+SAC'}
_BOOT = {}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Correlate distinct-state search-set disagreement with the '
          'held-out fail rate, with experiment-clustered standard errors.'
      ),
  )
  parser.add_argument('--environment', choices=ENVIRONMENTS)
  parser.add_argument('--algorithm', choices=ALGORITHMS)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--boot', type=int, default=N_BOOT)
  parser.add_argument(
      '--workers',
      type=int,
      default=max(1, min(8, os.cpu_count() or 1)),
  )
  parser.add_argument(
      '--mi-neighbors',
      type=int,
      default=3,
      help=(
          'Neighbors for both mutual-information estimators. '
          'Default 3 is the usual Kraskov/Ross choice.'
      ),
  )
  return parser.parse_args()


def _pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  xc = x - x.mean(axis=0)
  yc = y - y.mean()
  numer = xc.T @ yc
  denom = np.sqrt((xc * xc).sum(axis=0)) * np.sqrt(yc @ yc)
  out = np.full(x.shape[1], np.nan, dtype=np.float64)
  ok = denom > 0.0
  out[ok] = numer[ok] / denom[ok]
  return out


def _rank_columns(x: np.ndarray) -> np.ndarray:
  return np.column_stack([rankdata(x[:, i]) for i in range(x.shape[1])])


def _semipartial_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Spearman of each column of ``x`` after residualising on the others."""
  x_rank = _rank_columns(x)
  y_rank = rankdata(y).astype(np.float64)
  n_features = x.shape[1]
  out = np.full(n_features, np.nan, dtype=np.float64)
  intercept = np.ones((x.shape[0], 1), dtype=np.float64)
  for index in range(n_features):
    others = np.column_stack(
        [intercept, np.delete(x_rank, index, axis=1).astype(np.float64)]
    )
    coef, *_ = np.linalg.lstsq(others, x_rank[:, index], rcond=None)
    residual = x_rank[:, index] - others @ coef
    out[index] = _pearson_columns(residual[:, None], y_rank)[0]
  return out


def _fail_counts(y: np.ndarray) -> np.ndarray:
  """Integer number of held-out failures.

  The stored label is a rate k/P. The classification estimator needs the
  count k. P is the smallest ensemble size that makes every rate an integer
  count; 16 is that size when both ensembles have sixteen policies.
  """
  rates = np.asarray(y, dtype=np.float64)
  for n_policies in range(1, 65):
    scaled = rates * n_policies
    if np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-5):
      return np.rint(scaled).astype(np.int64)
  raise ValueError(
      'Held-out fail rate is not an integer count over a fixed ensemble size.'
  )


def _mutual_information(
    x: np.ndarray, y: np.ndarray, *, n_neighbors: int, seed: int
) -> dict[str, np.ndarray]:
  """Continuous-target and discrete-target mutual information, in nats."""
  neighbors = int(n_neighbors)
  regression = mutual_info_regression(
      x,
      y,
      discrete_features=False,
      n_neighbors=neighbors,
      copy=False,
      random_state=int(seed),
      n_jobs=-1,
  )
  classification = mutual_info_classif(
      x,
      _fail_counts(y),
      discrete_features=False,
      n_neighbors=neighbors,
      copy=False,
      random_state=int(seed),
      n_jobs=-1,
  )
  return {
      'mi_regression': np.asarray(regression, dtype=np.float64),
      'mi_classification': np.asarray(classification, dtype=np.float64),
  }


def _associations(
    x: np.ndarray, y: np.ndarray, *, with_mi: bool, mi_neighbors: int, seed: int
) -> dict[str, np.ndarray]:
  pearson = _pearson_columns(x, y)
  spearman = _pearson_columns(_rank_columns(x), rankdata(y).astype(np.float64))
  kendall = np.array(
      [
          float(kendalltau(x[:, i], y, variant='b').statistic)
          for i in range(x.shape[1])
      ],
      dtype=np.float64,
  )
  result = {
      'pearson': pearson,
      'spearman': spearman,
      'kendall': kendall,
      'semipartial_spearman': _semipartial_spearman(x, y),
  }
  if with_mi:
    result.update(
        _mutual_information(x, y, n_neighbors=mi_neighbors, seed=seed)
    )
  return result


def _boot_init(x: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> None:
  _BOOT['x'] = x
  _BOOT['y'] = y
  _BOOT['groups'] = groups


def _boot_replicate(seed: int) -> np.ndarray:
  rng = np.random.default_rng(int(seed))
  groups = _BOOT['groups']
  chosen = rng.integers(0, len(groups), size=len(groups))
  index = np.concatenate([groups[i] for i in chosen])
  values = _associations(
      _BOOT['x'][index],
      _BOOT['y'][index],
      with_mi=False,
      mi_neighbors=3,
      seed=0,
  )
  return np.stack([values[name] for name in _BOOTSTRAP_MEASURES], axis=0)


def _clusters(groups: np.ndarray) -> list[np.ndarray]:
  _unique, inverse = np.unique(groups, return_inverse=True)
  return [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]


def _feature_matrix(dataset) -> np.ndarray:
  return np.column_stack(
      [
          np.asarray(getattr(dataset, name), dtype=np.float64)
          for name in FEATURE_COLUMNS_USED
      ]
  )


class CaseBundle:
  """Distinct states, with both labels aligned."""

  def __init__(self, environment: str, algorithm: str):
    path = predictor_dataset_path(environment, algorithm, balanced=True)
    print(f'Correlations {environment}/{algorithm} from {path}', flush=True)
    raw = load_predictor_dataset(path, load_topk=False)
    eval_rate = _load_eval_fail_rates(raw)
    unique, eval_rate = collapse_duplicate_states(raw, eval_rate)
    search_difficulty = np.asarray(unique.difficulty, dtype=np.float64)
    solvable = _search_solvable(search_difficulty)
    features = _feature_matrix(unique)
    finite = np.isfinite(eval_rate) & np.all(np.isfinite(features), axis=1)
    self.environment = environment
    self.algorithm = algorithm
    self.features_all = features[finite]
    self.eval_all = eval_rate[finite]
    self.solvable = solvable[finite]
    self.experiment_all = np.asarray(unique.experiment, dtype=np.int64)[finite]
    self.search_all = search_difficulty[finite]
    print(
        f'  search-solvable distinct: {int(self.solvable.sum())}/'
        f'{int(self.solvable.size)}',
        flush=True,
    )
    del raw, unique
    gc.collect()

  @property
  def x(self) -> np.ndarray:
    return self.features_all[self.solvable]

  @property
  def y(self) -> np.ndarray:
    return self.eval_all[self.solvable]

  @property
  def experiment(self) -> np.ndarray:
    return self.experiment_all[self.solvable]


def analyse_bundle(
    bundle: CaseBundle, *, n_boot: int, seed: int, workers: int, mi_neighbors: int
) -> dict:
  x = np.ascontiguousarray(bundle.x)
  y = np.ascontiguousarray(bundle.y)
  groups = _clusters(bundle.experiment)
  if len(groups) < 2:
    raise ValueError(
        f'{bundle.environment}/{bundle.algorithm} has fewer than 2 '
        'experiments to resample.'
    )
  point = _associations(
      x, y, with_mi=True, mi_neighbors=mi_neighbors, seed=seed
  )
  spearman_all = _pearson_columns(
      _rank_columns(bundle.features_all),
      rankdata(bundle.eval_all).astype(np.float64),
  )
  seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_boot)
  if workers <= 1:
    _boot_init(x, y, groups)
    replicates = np.stack([_boot_replicate(int(value)) for value in seeds], axis=0)
  else:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_boot_init,
        initargs=(x, y, groups),
    ) as pool:
      replicates = np.stack(
          list(pool.map(_boot_replicate, seeds, chunksize=4)), axis=0
      )
  boot_se = np.nanstd(replicates, axis=0, ddof=1)
  boot_low = np.nanpercentile(replicates, 2.5, axis=0)
  boot_high = np.nanpercentile(replicates, 97.5, axis=0)
  se = {name: np.full(len(FEATURE_COLUMNS_USED), np.nan) for name, _ in MEASURES}
  ci_low = {name: np.full(len(FEATURE_COLUMNS_USED), np.nan) for name, _ in MEASURES}
  ci_high = {name: np.full(len(FEATURE_COLUMNS_USED), np.nan) for name, _ in MEASURES}
  for boot_index, name in enumerate(_BOOTSTRAP_MEASURES):
    se[name] = boot_se[boot_index]
    ci_low[name] = boot_low[boot_index]
    ci_high[name] = boot_high[boot_index]
  return {
      'environment': bundle.environment,
      'algorithm': bundle.algorithm,
      'n': int(y.shape[0]),
      'n_all': int(bundle.eval_all.shape[0]),
      'n_clusters': len(groups),
      'n_boot': int(n_boot),
      'estimate': point,
      'se': se,
      'ci_low': ci_low,
      'ci_high': ci_high,
      'spearman_all_rows': spearman_all,
  }


def _print_result(result: dict) -> None:
  label = (
      f"{_ENV_DISPLAY[result['environment']]} "
      f"{_ALG_DISPLAY[result['algorithm']]}"
  )
  print(
      f"  {label}: n={result['n']} distinct search-solvable, "
      f"{result['n_all']} including all-fail, "
      f"experiments={result['n_clusters']}",
      flush=True,
  )
  for measure, _heading in MEASURES:
    parts = []
    for feature_index, phrase in enumerate(FEATURE_LABELS_USED):
      estimate = result['estimate'][measure][feature_index]
      se = result['se'][measure][feature_index]
      if np.isfinite(se):
        parts.append(f'{phrase}={estimate:.3f}±{se:.3f}')
      else:
        parts.append(f'{phrase}={estimate:.3f}')
    print(f'    {measure}: ' + '; '.join(parts), flush=True)
  shifts = []
  for feature_index, phrase in enumerate(FEATURE_LABELS_USED):
    filtered = result['estimate']['spearman'][feature_index]
    unfiltered = result['spearman_all_rows'][feature_index]
    shifts.append(f'{phrase} {filtered:+.3f}->{unfiltered:+.3f}')
  print('    spearman all-rows (filter sensitivity): ' + '; '.join(shifts), flush=True)


def _rows(results: list[dict]) -> list[dict]:
  rows = []
  for result in results:
    for measure, _heading in MEASURES:
      for feature_index, (name, phrase) in enumerate(
          zip(FEATURE_COLUMNS_USED, FEATURE_LABELS_USED)
      ):
        rows.append({
            'environment': result['environment'],
            'algorithm': result['algorithm'],
            'n': result['n'],
            'n_all': result['n_all'],
            'n_clusters': result['n_clusters'],
            'n_boot': result['n_boot'],
            'cluster': 'experiment',
            'unique_within': 'experiment',
            'measure': measure,
            'feature': name,
            'feature_name': phrase,
            'estimate': float(result['estimate'][measure][feature_index]),
            'se': float(result['se'][measure][feature_index]),
            'ci_low': float(result['ci_low'][measure][feature_index]),
            'ci_high': float(result['ci_high'][measure][feature_index]),
            'spearman_all_rows': float(
                result['spearman_all_rows'][feature_index]
            ),
        })
  return rows


def _write_csv(rows: list[dict], path: str) -> None:
  if not rows:
    return
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _cell(estimate: float, se: float) -> str:
  if not np.isfinite(estimate):
    return '---'
  if not np.isfinite(se):
    return f'${estimate:.3f}$'
  return f'${estimate:.3f} \\pm {se:.3f}$'


def _compact_cell(estimate: float, se: float) -> str:
  """Coefficient with a one-digit standard error in the last decimal place.

  ``0.104(5)`` means ``0.104 ± 0.005``. The error is rounded to one
  significant digit, and the coefficient is rounded to that same place.
  """
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


# Display order groups the two action columns and the two scale-vector columns
# so the header can name each group once.
_TABLE_COLUMNS = (
    'action_scale_disagreement',
    'action_disagreement',
    'value_disagreement',
    'feature_attribution_disagreement',
    'criticality_disagreement',
    'std_magnitude_disagreement',
    'std_direction_disagreement',
)


# Rows shown in the paper table. Kendall duplicates the Spearman ranking.
# The continuous mutual-information estimate matches the discrete one here,
# so the table keeps the fail-count estimate and names that row plainly.
_TABLE_MEASURES = (
    ('pearson', "Pearson $r$"),
    ('spearman', "Spearman $\\rho$"),
    ('mi_classification', 'Mutual information'),
    ('semipartial_spearman', r'Semi-partial $\rho$'),
)
_TABLE_GROUPS = (
    (
        'ppo',
        'tab:eval-fail-ppo',
        'PPO and PPO.',
    ),
    (
        'sac',
        'tab:eval-fail-sac',
        'SAC and SAC.',
    ),
    (
        'ppo_sac',
        'tab:eval-fail-ppo-sac',
        'PPO+SAC and PPO+SAC.',
    ),
)


def _table_column_index() -> tuple[int, ...]:
  position = {name: index for index, name in enumerate(FEATURE_COLUMNS_USED)}
  missing = [name for name in _TABLE_COLUMNS if name not in position]
  if missing:
    raise ValueError(f'Table columns missing from the feature list: {missing}')
  return tuple(position[name] for name in _TABLE_COLUMNS)


def _write_tex(results: list[dict], path: str) -> None:
  """Three tables, one per search/evaluation pairing."""
  columns = _table_column_index()
  by_algorithm = {}
  for result in results:
    by_algorithm.setdefault(result['algorithm'], []).append(result)
  lines = [
      '% Generated by eval_fail_correlations.py',
      '\\documentclass{article}',
      '\\usepackage[margin=0.75in]{geometry}',
      '\\usepackage{booktabs}',
      '\\usepackage{multirow}',
      '\\usepackage{amsmath}',
      '\\begin{document}',
      '',
  ]
  header = [
      '\\toprule',
      (
          'Environment & Measure'
          ' & \\multicolumn{2}{c}{Action disagreement}'
          ' & \\multirow{2}{*}{Value}'
          ' & \\multirow{2}{*}{Attribution}'
          ' & \\multirow{2}{*}{Criticality}'
          ' & \\multicolumn{2}{c}{Std.\\ deviation} \\\\'
      ),
      '\\cmidrule(lr){3-4}\\cmidrule(lr){8-9}',
      '& & Scale & Vector & & & & Magnitude & Direction \\\\',
      '\\midrule',
  ]
  for algorithm, label, pairing in _TABLE_GROUPS:
    group = by_algorithm.get(algorithm, [])
    if not group:
      continue
    order = {name: index for index, name in enumerate(ENVIRONMENTS)}
    group.sort(key=lambda result: order[result['environment']])
    n_boot = int(group[0]['n_boot'])
    n_cases = ', '.join(
        f"{_ENV_DISPLAY[result['environment']]} {result['n']}"
        f" ({result['n_clusters']} experiments)"
        for result in group
    )
    lines.extend([
        '\\begin{table}[t]',
        '\\centering',
        '\\small',
        '\\setlength{\\tabcolsep}{4pt}',
        (
            '\\caption{' + pairing + ' Association between disagreement '
            'within the search ensemble and the failure rate of the '
            'disjoint evaluation ensemble. Each distinct initial state is '
            'kept once per experiment. '
            'A parenthetical digit is the experiment-clustered bootstrap '
            'standard error in the last decimal place, rounded to one '
            f'digit ($B={n_boot}$): $0.104(5)$ means $0.104 \\pm 0.005$. '
            'Mutual information treats the fail count as discrete and uses '
            'three neighbors; it has no standard error. '
            'Semi-partial Spearman residualises each feature on the '
            'others, never on the search fail rate.}'
        ),
        f'\\label{{{label}}}',
        '\\begin{tabular}{llccccccc}',
        *header,
    ])
    for result_index, result in enumerate(group):
      dataset = _ENV_DISPLAY[result['environment']]
      for measure_index, (measure, measure_heading) in enumerate(_TABLE_MEASURES):
        cells = ' & '.join(
            _compact_cell(
                float(result['estimate'][measure][feature_index]),
                float(result['se'][measure][feature_index]),
            )
            for feature_index in columns
        )
        env_cell = dataset if measure_index == 0 else ''
        lines.append(f'{env_cell} & {measure_heading} & {cells} \\\\')
      if result_index != len(group) - 1:
        lines.append('\\addlinespace')
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\vspace{0.4em}',
        '{\\footnotesize',
        f'Sample sizes (distinct search-solvable states): {n_cases}.',
        '}',
        '\\end{table}',
        '',
    ])
  lines.extend(['\\end{document}', ''])
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w') as handle:
    handle.write('\n'.join(lines))
  write_correlation_figures(path)


_CELL = re.compile(r'^(-?\d+(?:\.\d+)?)(?:\((\d)\))?$')
# Column order in the table. Magnitude and direction are parsed, then omitted
# from the figure so the remaining measures can sit further apart.
_TABLE_FEATURE_COUNT = 7
_FIGURE_FEATURES = (
    (0, 'Action scale'),
    (1, 'Action vector'),
    (2, 'Value'),
    (3, 'Attribution'),
    (4, 'Criticality'),
)
_FIGURE_MEASURES = (
    ('pearson', 'Pearson $r$', 'Pearson correlation', True),
    ('spearman', 'Spearman $\\rho$', 'Spearman correlation', True),
    ('mi', 'Mutual information', 'Mutual information', False),
    ('semipartial', 'Semi-partial $\\rho$', 'Semi-partial Spearman', True),
)
_FIGURE_ENVS = (
    ('Ant', 'o'),
    ('Humanoid', 's'),
    ('Walker2d', '^'),
    ('Hopper', 'D'),
)
_FIGURE_ENSEMBLES = (
    ('ppo', 'PPO', '#0072B2'),
    ('sac', 'SAC', '#D55E00'),
    ('ppo_sac', 'PPO+SAC', '#009E73'),
)
_MEASURE_IN_ROW = (
    ('Pearson', 'pearson'),
    ('Spearman', 'spearman'),
    ('Mutual information', 'mi'),
    ('Semi-partial', 'semipartial'),
)


def _parse_table_cell(token: str) -> tuple[float, float]:
  """Read one compact cell. ``0.104(5)`` is ``0.104`` with standard error ``0.005``."""
  token = token.strip().strip('$').replace('\\,', '')
  match = _CELL.fullmatch(token)
  if match is None:
    raise ValueError(f'Cannot read table cell {token!r}')
  estimate = float(match.group(1))
  digit = match.group(2)
  if digit is None:
    return estimate, float('nan')
  decimals = len(match.group(1).split('.')[1]) if '.' in match.group(1) else 0
  return estimate, int(digit) * 10.0 ** (-decimals)


def _rows_from_correlation_table(path: str) -> list[dict]:
  """Points shown in ``eval_fail_correlations.tex``, one per cell."""
  with open(path) as handle:
    text = handle.read()
  rows = []
  for algorithm, _label, _color in _FIGURE_ENSEMBLES:
    label = {
        'ppo': 'tab:eval-fail-ppo',
        'sac': 'tab:eval-fail-sac',
        'ppo_sac': 'tab:eval-fail-ppo-sac',
    }[algorithm]
    start = text.index(f'\\label{{{label}}}')
    stop = text.index('\\bottomrule', start)
    environment = None
    for line in text[start:stop].splitlines():
      if '&' not in line or '\\midrule' in line:
        continue
      cells = [cell.strip() for cell in line.split('&')]
      if len(cells) != 9:
        continue
      if cells[0]:
        environment = cells[0]
      measure = None
      for needle, key in _MEASURE_IN_ROW:
        if needle in cells[1]:
          measure = key
          break
      if environment is None or measure is None:
        continue
      for feature_index, token in enumerate(cells[2:]):
        token = token.replace('\\\\', '').strip()
        estimate, se = _parse_table_cell(token)
        rows.append({
            'environment': environment,
            'algorithm': algorithm,
            'measure': measure,
            'feature': feature_index,
            'estimate': estimate,
            'se': se,
        })
  expected = (
      len(_FIGURE_ENSEMBLES)
      * len(_FIGURE_ENVS)
      * len(_FIGURE_MEASURES)
      * _TABLE_FEATURE_COUNT
  )
  if len(rows) != expected:
    raise ValueError(f'Parsed {len(rows)} table cells, expected {expected}')
  return rows


def write_correlation_figures(tex_path: str) -> list[str]:
  """One figure per association measure, using the numbers printed in the table."""
  rows = [
      row for row in _rows_from_correlation_table(tex_path)
      if row['feature'] in {index for index, _label in _FIGURE_FEATURES}
  ]
  output_dir = os.path.dirname(tex_path) or '.'
  # Each measure owns a wide slot. Ensembles sit apart inside it, and the
  # four environments sit apart inside each ensemble.
  feature_step = 1.7
  feature_x = {
      index: position * feature_step
      for position, (index, _label) in enumerate(_FIGURE_FEATURES)
  }
  env_pitch = 0.055
  ensemble_pitch = 0.28
  env_shift = {
      name: (index - (len(_FIGURE_ENVS) - 1) / 2) * env_pitch
      for index, (name, _marker) in enumerate(_FIGURE_ENVS)
  }
  ensemble_shift = {
      algorithm: (index - (len(_FIGURE_ENSEMBLES) - 1) / 2) * ensemble_pitch
      for index, (algorithm, _label, _color) in enumerate(_FIGURE_ENSEMBLES)
  }
  correlation_rows = [
      row for row in rows if row['measure'] != 'mi' and np.isfinite(row['se'])
  ]
  y_span = [row['estimate'] + sign * row['se'] for row in correlation_rows for sign in (-1, 1)]
  correlation_limits = (min(y_span) - 0.06, max(y_span) + 0.06)
  written = []
  for measure, ylabel, title, with_error in _FIGURE_MEASURES:
    figure, axis = plt.subplots(figsize=(8.4, 4.15))
    selected = [row for row in rows if row['measure'] == measure]
    for row in selected:
      x = (
          feature_x[row['feature']]
          + ensemble_shift[row['algorithm']]
          + env_shift[row['environment']]
      )
      color = dict((alg, col) for alg, _lab, col in _FIGURE_ENSEMBLES)[row['algorithm']]
      marker = dict(_FIGURE_ENVS)[row['environment']]
      yerr = row['se'] if with_error and np.isfinite(row['se']) else None
      axis.errorbar(
          x,
          row['estimate'],
          yerr=None if yerr is None else [[yerr], [yerr]],
          fmt=marker,
          color=color,
          ecolor=color,
          elinewidth=0.9,
          capsize=1.7,
          capthick=0.75,
          markersize=5.4,
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
          linestyle='none',
          zorder=3,
      )
    if with_error:
      axis.axhline(0.0, color='#c8c8c8', linewidth=0.7, zorder=0)
      axis.set_ylim(*correlation_limits)
    else:
      finite = [row['estimate'] for row in selected]
      axis.set_ylim(-0.02, max(finite) * 1.12)
    half_slot = ensemble_pitch + env_pitch * (len(_FIGURE_ENVS) - 1) / 2 + 0.16
    axis.set_xlim(-half_slot, feature_x[_FIGURE_FEATURES[-1][0]] + half_slot)
    axis.set_xticks([feature_x[index] for index, _label in _FIGURE_FEATURES])
    axis.set_xticklabels([label for _index, label in _FIGURE_FEATURES])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.yaxis.grid(True, color='#ececec', linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(length=3, width=0.6)
    for spine in ('left', 'bottom'):
      axis.spines[spine].set_linewidth(0.6)
    environment_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            color='black',
            markerfacecolor='black',
            markeredgecolor='black',
            linestyle='None',
            markersize=5.6,
            label=name,
        )
        for name, marker in _FIGURE_ENVS
    ]
    ensemble_handles = [
        Line2D(
            [0], [0],
            color=color,
            marker='|',
            markersize=11,
            markeredgewidth=1.6,
            linestyle='None',
            label=label,
        )
        for _algorithm, label, color in _FIGURE_ENSEMBLES
    ]
    environment_legend = axis.legend(
        handles=environment_handles,
        loc='upper center',
        bbox_to_anchor=(0.30, -0.24),
        ncol=4,
        frameon=False,
        title='Environment',
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    environment_legend.get_title().set_fontsize(8)
    axis.add_artist(environment_legend)
    ensemble_legend = axis.legend(
        handles=ensemble_handles,
        loc='upper center',
        bbox_to_anchor=(0.80, -0.24),
        ncol=3,
        frameon=False,
        title='Ensemble',
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    ensemble_legend.get_title().set_fontsize(8)
    figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.30)
    filename = os.path.join(output_dir, f'eval_fail_{measure}.pdf')
    figure.savefig(filename)
    plt.close(figure)
    written.append(filename)
    print(f'  wrote {filename}', flush=True)
  return written


def main() -> int:
  args = parse_args()
  environments = (args.environment,) if args.environment else ENVIRONMENTS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  results = []
  for environment in environments:
    for algorithm in algorithms:
      try:
        bundle = CaseBundle(environment, algorithm)
        result = analyse_bundle(
            bundle,
            n_boot=args.boot,
            seed=args.seed,
            workers=args.workers,
            mi_neighbors=args.mi_neighbors,
        )
      except (FileNotFoundError, ValueError) as exc:
        print(f'  skipped: {exc}', flush=True)
        continue
      _print_result(result)
      results.append(result)
      rows = _rows(results)
      _write_csv(rows, CSV_PATH)
      _write_tex(results, TEX_PATH)
      print(f'  wrote {TEX_PATH}', flush=True)
      del bundle
      gc.collect()
  if not results:
    raise SystemExit('No datasets had held-out evaluation fail rates.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
