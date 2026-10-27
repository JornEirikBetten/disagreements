"""Disagreement associations and feature curves against the shared evaluation set.

The target is the fail rate of the 32 policies in the PPO evaluation ensemble
and the SAC evaluation ensemble, each policy once. The same 32 score every
search. Semi-partial Spearman is not computed.

Usage:
  python shared_eval_figures.py
  python shared_eval_figures.py --environment hopper --algorithm ppo
  python shared_eval_figures.py --min-value
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use('Agg')

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import plotting.style as figs
from analysis.disagreement import (
    criticality_log_volume_np,
    minimum_value_zscore_np,
    standardized_criticality_disagreement_np,
    value_estimates_np,
)
from analysis.predictor_dataset import (
    collapse_duplicate_states,
    load_predictor_dataset,
    predictor_dataset_path,
)
from cross_ensemble_humanoid import _rates_path, load_search_cases
from eval_fail_correlations import _pearson_columns, _rank_columns
from train_difficulty_predictors import (
    ALGORITHMS,
    ENVIRONMENTS,
    FEATURE_COLUMNS_USED,
    FEATURE_LABELS_USED,
    _collection_path,
    _search_solvable,
)

FIGURE_DIR = os.path.join('figures', 'predictors')
N_EVAL = 32
N_BOOT = 1000
MI_NEIGHBORS = 3
CURVE_FEATURES = tuple(figs.DISAGREEMENT)
MIN_VALUE_FEATURE = 'minimum_value_zscore'
MIN_VALUE_LABEL = 'Min. value z'
MIN_VALUE_FEATURES = CURVE_FEATURES + (MIN_VALUE_FEATURE,)
_BOOT = {}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--environment', choices=ENVIRONMENTS)
  parser.add_argument('--algorithm', choices=ALGORITHMS)
  parser.add_argument('--boot', type=int, default=N_BOOT)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--workers', type=int, default=max(1, min(8, os.cpu_count() or 1)),
  )
  parser.add_argument(
      '--min-value',
      action='store_true',
      help=(
          'Add the minimum per-policy value z-score and write '
          'associations_minV.pdf and feature_curves_minV.pdf.'
      ),
  )
  return parser.parse_args()


def _state_keys(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float32).reshape(qpos.shape[0], -1)
  qvel = np.asarray(qvel, dtype=np.float32).reshape(qvel.shape[0], -1)
  rows = np.ascontiguousarray(np.column_stack([qpos, qvel]))
  width = int(rows.dtype.itemsize * rows.shape[1])
  return rows.view(np.dtype((np.void, width))).reshape(rows.shape[0])


def _load_pair(environment: str, source: str):
  """PPO-eval rate, SAC-eval rate, and the search quota, one row per case."""
  cases = load_search_cases(environment, source, None, None)
  if source == 'ppo':
    ppo = np.asarray(cases['ppo_eval'], dtype=np.float64)
    sac = np.load(_rates_path(environment, 'all', 'sac_eval', 'ppo'))
    search = np.asarray(cases['ppo_search'], dtype=np.float64)
  elif source == 'sac':
    sac = np.asarray(cases['source_eval'], dtype=np.float64)
    ppo = np.load(_rates_path(environment, 'all', 'ppo_eval', 'sac'))
    search = np.asarray(cases['search_difficulty'], dtype=np.float64)
  else:
    ppo = np.load(_rates_path(environment, 'all', 'ppo_eval', 'ppo_sac'))
    sac = np.load(_rates_path(environment, 'all', 'sac_eval', 'ppo_sac'))
    search = np.asarray(cases['search_difficulty'], dtype=np.float64)
  if ppo.shape != sac.shape or ppo.shape != search.shape:
    raise ValueError(
        f'{environment}/{source}: PPO eval {ppo.shape}, SAC eval {sac.shape}, '
        f'search {search.shape}'
    )
  counts = np.rint(ppo * 16.0) + np.rint(sac * 16.0)
  rate = counts / float(N_EVAL)
  return cases, rate, search


def _align(dataset, cases, *values: np.ndarray) -> tuple[np.ndarray, ...]:
  """Copy cached rows onto ``dataset`` by experiment and initial state."""
  source_keys = _state_keys(cases['qpos'], cases['qvel'])
  source_experiment = np.asarray(cases['experiment'], dtype=np.int64)
  index = {
      (int(experiment), key.tobytes()): row
      for row, (experiment, key) in enumerate(zip(source_experiment, source_keys))
  }
  dest_keys = _state_keys(dataset.qpos, dataset.qvel)
  dest_experiment = np.asarray(dataset.experiment, dtype=np.int64)
  located = np.full(dataset.n_cases, -1, dtype=np.int64)
  for row, (experiment, key) in enumerate(zip(dest_experiment, dest_keys)):
    located[row] = index.get((int(experiment), key.tobytes()), -1)
  missing = int(np.count_nonzero(located < 0))
  if missing:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm}: {missing} dataset rows '
        'have no 32-policy evaluation rate.'
    )
  return tuple(np.asarray(value, dtype=np.float64)[located] for value in values)


def _associations(x: np.ndarray, y: np.ndarray, *, with_mi: bool, seed: int) -> dict:
  y_rank = rankdata(y).astype(np.float64)
  result = {
      'pearson': _pearson_columns(x, y),
      'spearman': _pearson_columns(_rank_columns(x), y_rank),
      'kendall': np.array(
          [
              float(kendalltau(x[:, i], y, variant='b').statistic)
              for i in range(x.shape[1])
          ],
          dtype=np.float64,
      ),
  }
  if not with_mi:
    return result
  counts = np.rint(y * N_EVAL).astype(np.int64)
  result['mi_regression'] = np.asarray(
      mutual_info_regression(
          x, y, n_neighbors=MI_NEIGHBORS, random_state=seed, n_jobs=4,
      ),
      dtype=np.float64,
  )
  result['mi_classification'] = np.asarray(
      mutual_info_classif(
          x, counts, n_neighbors=MI_NEIGHBORS, random_state=seed, n_jobs=4,
      ),
      dtype=np.float64,
  )
  return result


def _boot_init(x, y, groups) -> None:
  _BOOT['x'] = x
  _BOOT['y'] = y
  _BOOT['groups'] = groups


def _boot_replicate(seed: int) -> np.ndarray:
  rng = np.random.default_rng(int(seed))
  groups = _BOOT['groups']
  chosen = rng.integers(0, len(groups), size=len(groups))
  index = np.concatenate([groups[int(i)] for i in chosen])
  x = _BOOT['x'][index]
  y = _BOOT['y'][index]
  y_rank = rankdata(y).astype(np.float64)
  return np.stack(
      [
          _pearson_columns(x, y),
          _pearson_columns(_rank_columns(x), y_rank),
      ],
      axis=0,
  )


def _clusters(experiment: np.ndarray) -> list[np.ndarray]:
  _unique, inverse = np.unique(experiment, return_inverse=True)
  return [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]


def _criticality_experiment(job: tuple[str, str, int]):
  """Z-scored criticality disagreement for one experiment's shards."""
  from ga.collect import iter_generation_shards

  environment, algorithm, experiment = job
  path = _collection_path(environment, algorithm, experiment)
  generations = []
  case_indices = []
  pieces = []
  for record in iter_generation_shards(path):
    volume = criticality_log_volume_np(record.action_scale)
    n_cases = int(volume.shape[1])
    pieces.append(volume)
    generations.append(np.full(n_cases, int(record.generation), dtype=np.int32))
    case_indices.append(np.arange(n_cases, dtype=np.int32))
    del record
  if not pieces:
    raise ValueError(f'{path} has no generation shards.')
  disagreement = standardized_criticality_disagreement_np(
      np.concatenate(pieces, axis=1)
  )
  return (
      int(experiment),
      np.concatenate(generations),
      np.concatenate(case_indices),
      disagreement,
  )


def _standardized_criticality_column(dataset, *, workers: int) -> np.ndarray:
  """Per-policy z-scored criticality disagreement, one scale per experiment.

  The stored predictor column is the raw mean pairwise gap in ``sum log σ``.
  Here each policy is standardized across every case in its experiment, and
  disagreement is the mean pairwise absolute difference of those z-scores.
  """
  scores = np.full(dataset.n_cases, np.nan, dtype=np.float64)
  jobs = [
      (dataset.environment, dataset.algorithm, int(value))
      for value in np.unique(dataset.experiment)
  ]
  if workers <= 1 or len(jobs) == 1:
    packed = [_criticality_experiment(job) for job in jobs]
  else:
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
      packed = list(pool.map(_criticality_experiment, jobs))
  for experiment, generation, case_index, disagreement in packed:
    _assign_experiment_scores(
        scores, dataset, experiment, generation, case_index, disagreement,
    )
    print(
        f'  experiment {experiment}: standardized criticality '
        f'[{float(np.min(disagreement)):.3f}, {float(np.max(disagreement)):.3f}]',
        flush=True,
    )
  missing = int((~np.isfinite(scores)).sum())
  if missing:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm}: {missing} cases '
        'have no standardized criticality.'
    )
  return scores


def _prepare(environment: str, algorithm: str, *, workers: int):
  path = predictor_dataset_path(environment, algorithm, balanced=True)
  print(f'{environment}/{algorithm} from {path}', flush=True)
  dataset = load_predictor_dataset(path, load_topk=False)
  dataset.criticality_disagreement = _standardized_criticality_column(
      dataset, workers=workers,
  )
  cases, rate, search = _load_pair(environment, algorithm)
  rate, search = _align(dataset, cases, rate, search)
  del cases
  dataset, rate, search = collapse_duplicate_states(dataset, rate, search)
  solvable = _search_solvable(np.asarray(dataset.difficulty, dtype=np.float64))
  features = np.column_stack(
      [
          np.asarray(getattr(dataset, name), dtype=np.float64)
          for name in FEATURE_COLUMNS_USED
      ]
  )
  experiment = np.asarray(dataset.experiment, dtype=np.int64)
  finite = np.isfinite(rate) & np.isfinite(search) & np.all(np.isfinite(features), axis=1)
  if int((~finite).sum()):
    raise ValueError(
        f'{environment}/{algorithm}: {int((~finite).sum())} non-finite rows.'
    )
  print(
      f'  distinct {int(solvable.sum())}/{int(solvable.size)} search-solvable, '
      f'32-policy rate mean {float(rate[solvable].mean()):.3f}',
      flush=True,
  )
  kept = {
      'features': features[solvable],
      'rate': rate[solvable],
      'search': search[solvable],
      'experiment': experiment[solvable],
      'features_all': features,
      'rate_all': rate,
  }
  del dataset
  gc.collect()
  return kept


def _curve_runs(features: np.ndarray, rate: np.ndarray, experiment: np.ndarray):
  """Per-search mean of each feature, in SD units, at each 32-policy fail rate."""
  centre = features.mean(axis=0)
  scale = features.std(axis=0)
  scale = np.where(scale > 0.0, scale, 1.0)
  standardised = (features - centre) / scale
  levels = np.arange(N_EVAL + 1, dtype=np.float64) / N_EVAL
  counts = np.rint(rate * N_EVAL).astype(np.int64)
  experiments = np.unique(experiment)
  names = [
      name for name in CURVE_FEATURES if name in FEATURE_COLUMNS_USED
  ]
  columns = [FEATURE_COLUMNS_USED.index(name) for name in names]
  runs = {
      name: np.full((experiments.size, levels.size), np.nan) for name in names
  }
  occupancy = np.zeros(levels.size, dtype=np.int64)
  for level_index, level in enumerate(levels):
    at_level = counts == level_index
    occupancy[level_index] = int(at_level.sum())
    if not np.any(at_level):
      continue
    for experiment_index, value in enumerate(experiments):
      chosen = at_level & (experiment == value)
      if not np.any(chosen):
        continue
      for name, column in zip(names, columns):
        runs[name][experiment_index, level_index] = float(
            np.mean(standardised[chosen, column])
        )
  present = occupancy > 0
  curves = {}
  for name in names:
    curves[name] = figs.Curve.from_runs(levels[present], runs[name][:, present])
  return curves, levels, occupancy, standardised, counts


def analyse(environment: str, algorithm: str, *, n_boot: int, seed: int, workers: int):
  kept = _prepare(environment, algorithm, workers=workers)
  x = np.ascontiguousarray(kept['features'])
  y = np.ascontiguousarray(kept['rate'])
  groups = _clusters(kept['experiment'])
  point = _associations(x, y, with_mi=True, seed=seed)
  seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_boot)
  print(f'  bootstrap B={n_boot} on {len(groups)} searches', flush=True)
  if workers <= 1:
    _boot_init(x, y, groups)
    replicates = np.stack([_boot_replicate(int(value)) for value in seeds])
  else:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_boot_init,
        initargs=(x, y, groups),
    ) as pool:
      replicates = np.stack(list(pool.map(_boot_replicate, seeds, chunksize=8)))
  se = np.nanstd(replicates, axis=0, ddof=1)
  low = np.nanpercentile(replicates, 2.5, axis=0)
  high = np.nanpercentile(replicates, 97.5, axis=0)
  spearman_all = _pearson_columns(
      _rank_columns(kept['features_all']),
      rankdata(kept['rate_all']).astype(np.float64),
  )
  curves, levels, occupancy, standardised, counts = _curve_runs(
      kept['features'], kept['rate'], kept['experiment'],
  )
  search_rho = float(
      _pearson_columns(
          rankdata(kept['search'])[:, None],
          rankdata(kept['rate']).astype(np.float64),
      )[0]
  )
  search_mi = float(
      mutual_info_classif(
          kept['search'][:, None],
          np.rint(kept['rate'] * N_EVAL).astype(np.int64),
          n_neighbors=MI_NEIGHBORS,
          random_state=seed,
          n_jobs=1,
      )[0]
  )
  rows = []
  for measure in ('pearson', 'spearman', 'kendall', 'mi_regression', 'mi_classification'):
    boot_index = {'pearson': 0, 'spearman': 1}.get(measure)
    for feature_index, (name, phrase) in enumerate(
        zip(FEATURE_COLUMNS_USED, FEATURE_LABELS_USED)
    ):
      rows.append({
          'environment': environment,
          'algorithm': algorithm,
          'n': int(y.shape[0]),
          'n_clusters': len(groups),
          'n_boot': int(n_boot) if boot_index is not None else 0,
          'eval_policies': N_EVAL,
          'measure': measure,
          'feature': name,
          'feature_name': phrase,
          'estimate': float(point[measure][feature_index]),
          'se': float(se[boot_index, feature_index]) if boot_index is not None else '',
          'ci_low': float(low[boot_index, feature_index]) if boot_index is not None else '',
          'ci_high': float(high[boot_index, feature_index]) if boot_index is not None else '',
          'spearman_all_rows': float(spearman_all[feature_index]),
      })
  curve_rows = []
  for name, curve in curves.items():
    phrase = figs.DISAGREEMENT[name]
    for level, mean, band_low, band_high in zip(curve.x, curve.y, curve.low, curve.high):
      count = int(occupancy[int(round(level * N_EVAL))])
      curve_rows.append({
          'environment': environment,
          'algorithm': algorithm,
          'feature': name,
          'feature_name': phrase,
          'eval_fail_rate': float(level),
          'n': count,
          'mean_sd_units': float(mean),
          'se_low': float(band_low),
          'se_high': float(band_high),
      })
  summary = {
      'environment': environment,
      'algorithm': algorithm,
      'rows': rows,
      'curves': curves,
      'curve_rows': curve_rows,
      'search_rho': search_rho,
      'search_mi': search_mi,
      'point': point,
      'se': se,
  }
  del kept, standardised, counts
  gc.collect()
  return summary


def _write_csv(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f'wrote {path}', flush=True)


def _write_tex(path: str, rows: list[dict]) -> None:
  measures = (
      ('pearson', 'Pearson $r$'),
      ('spearman', 'Spearman $\\rho$'),
      ('mi_classification', 'Mutual information'),
  )
  columns = tuple(figs.DISAGREEMENT)
  lines = [
      '% Generated by shared_eval_figures.py',
      '% Target: fail rate of the 32 PPO and SAC evaluation policies.',
      '\\begin{tabular}{llccccc}',
      '\\toprule',
      'Environment & Measure & '
      + ' & '.join(figs.DISAGREEMENT[name].replace('\n', ' ') for name in columns)
      + ' \\\\',
      '\\midrule',
  ]
  by_key = {
      (row['algorithm'], row['environment'], row['measure'], row['feature']): row
      for row in rows
  }
  algorithms = list(dict.fromkeys(row['algorithm'] for row in rows))
  environments = list(dict.fromkeys(row['environment'] for row in rows))
  for algorithm in algorithms:
    lines.append(f'\\multicolumn{{7}}{{l}}{{{algorithm}}} \\\\')
    for environment in environments:
      for index, (measure, heading) in enumerate(measures):
        cells = []
        for name in columns:
          row = by_key[(algorithm, environment, measure, name)]
          if row['se'] == '':
            cells.append(f"${row['estimate']:.3f}$")
          else:
            cells.append(f"${row['estimate']:.3f}\\pm{float(row['se']):.3f}$")
        env = environment if index == 0 else ''
        lines.append(f'{env} & {heading} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\addlinespace')
  lines.extend(['\\bottomrule', '\\end{tabular}', ''])
  with open(path, 'w') as handle:
    handle.write('\n'.join(lines))
  print(f'wrote {path}', flush=True)


def _figures(results: list[dict]) -> None:
  estimates = {}
  spearman_baseline = {}
  mi_baseline = {}
  curves = {}
  for result in results:
    environment = result['environment']
    algorithm = result['algorithm']
    for row in result['rows']:
      if row['feature'] not in CURVE_FEATURES:
        continue
      if row['measure'] == 'spearman':
        estimates[('spearman', row['feature'], algorithm, environment)] = (
            figs.Estimate.from_se(row['estimate'], float(row['se']))
        )
      elif row['measure'] == 'mi_classification':
        estimates[('mi', row['feature'], algorithm, environment)] = figs.Estimate(
            row['estimate']
        )
    spearman_baseline[(algorithm, environment)] = figs.Estimate(result['search_rho'])
    mi_baseline[(algorithm, environment)] = figs.Estimate(result['search_mi'])
    for feature, curve in result['curves'].items():
      curves[(environment, feature, algorithm)] = curve
  association = figs.association_figure(
      estimates,
      measures=('spearman', 'mi'),
      baseline=mi_baseline,
      spearman_baseline=spearman_baseline,
  )
  curves_figure = figs.feature_curves_figure(curves)
  os.makedirs(FIGURE_DIR, exist_ok=True)
  figs.save(association, os.path.join(FIGURE_DIR, 'associations.pdf'))
  figs.save(curves_figure, os.path.join(FIGURE_DIR, 'feature_curves.pdf'))
  print(f'wrote {FIGURE_DIR}/associations.pdf', flush=True)
  print(f'wrote {FIGURE_DIR}/feature_curves.pdf', flush=True)


def _assign_experiment_scores(
    scores: np.ndarray,
    dataset,
    experiment: int,
    generation: np.ndarray,
    case_index: np.ndarray,
    zmin: np.ndarray,
) -> None:
  """Copy per-case scores onto dataset rows of one experiment."""
  mask = np.asarray(dataset.experiment) == int(experiment)
  if not np.any(mask):
    raise ValueError(f'{dataset.environment}/{dataset.algorithm}: experiment {experiment} is missing.')
  generation = np.asarray(generation, dtype=np.int64)
  case_index = np.asarray(case_index, dtype=np.int64)
  dest_generation = np.asarray(dataset.generation, dtype=np.int64)[mask]
  dest_case = np.asarray(dataset.case_index, dtype=np.int64)[mask]
  stride = int(max(case_index.max(), dest_case.max())) + 1
  keys = generation * stride + case_index
  if np.unique(keys).size != keys.size:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm} experiment {experiment} '
        'has duplicate (generation, case) rows.'
    )
  order = np.argsort(keys)
  sorted_keys = keys[order]
  dest_keys = dest_generation * stride + dest_case
  pos = np.searchsorted(sorted_keys, dest_keys)
  matched = (pos < sorted_keys.size) & (sorted_keys[np.minimum(pos, sorted_keys.size - 1)] == dest_keys)
  if not np.all(matched):
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm} experiment {experiment}: '
        f'{int((~matched).sum())} dataset rows have no value estimates.'
    )
  scores[np.flatnonzero(mask)] = np.asarray(zmin, dtype=np.float64)[order[pos]]


def _minimum_value_column(dataset) -> np.ndarray:
  """Min per-policy value z-score, standardized within each experiment."""
  from ga.collect import iter_generation_shards

  scores = np.full(dataset.n_cases, np.nan, dtype=np.float64)
  experiments = [int(value) for value in np.unique(dataset.experiment)]
  for experiment in experiments:
    path = _collection_path(dataset.environment, dataset.algorithm, experiment)
    generations = []
    case_indices = []
    pieces = []
    for record in iter_generation_shards(path):
      estimates = value_estimates_np(
          record.values,
          getattr(record, 'q_values', None),
          algorithm=dataset.algorithm,
      )
      n_cases = int(estimates.shape[1])
      pieces.append(estimates)
      generations.append(np.full(n_cases, int(record.generation), dtype=np.int32))
      case_indices.append(np.arange(n_cases, dtype=np.int32))
      del record
    if not pieces:
      raise ValueError(f'{path} has no generation shards.')
    zmin = minimum_value_zscore_np(np.concatenate(pieces, axis=1))
    del pieces
    _assign_experiment_scores(
        scores,
        dataset,
        experiment,
        np.concatenate(generations),
        np.concatenate(case_indices),
        zmin,
    )
    print(
        f'  experiment {experiment}: min value z '
        f'[{float(np.min(zmin)):.2f}, {float(np.max(zmin)):.2f}]',
        flush=True,
    )
    del zmin
    gc.collect()
  missing = int((~np.isfinite(scores)).sum())
  if missing:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm}: {missing} rows have no minimum value z-score.'
    )
  return scores


def _curve_column(
    values: np.ndarray,
    rate: np.ndarray,
    experiment: np.ndarray,
) -> tuple[figs.Curve, np.ndarray]:
  """One feature curve in the same SD units as the other columns."""
  centre = float(values.mean())
  scale = float(values.std())
  if scale <= 0.0:
    scale = 1.0
  standardised = (values - centre) / scale
  levels = np.arange(N_EVAL + 1, dtype=np.float64) / N_EVAL
  counts = np.rint(rate * N_EVAL).astype(np.int64)
  experiments = np.unique(experiment)
  runs = np.full((experiments.size, levels.size), np.nan)
  occupancy = np.zeros(levels.size, dtype=np.int64)
  for level_index in range(levels.size):
    at_level = counts == level_index
    occupancy[level_index] = int(at_level.sum())
    if not np.any(at_level):
      continue
    for experiment_index, value in enumerate(experiments):
      chosen = at_level & (experiment == value)
      if np.any(chosen):
        runs[experiment_index, level_index] = float(np.mean(standardised[chosen]))
  present = occupancy > 0
  return figs.Curve.from_runs(levels[present], runs[:, present]), occupancy


def analyse_minimum_value(
    environment: str,
    algorithm: str,
    *,
    n_boot: int,
    seed: int,
    workers: int,
) -> dict:
  """Spearman, mutual information, and the feature curve of the min value z-score."""
  path = predictor_dataset_path(environment, algorithm, balanced=True)
  print(f'{environment}/{algorithm} minimum value z from {path}', flush=True)
  dataset = load_predictor_dataset(path, load_topk=False)
  column = _minimum_value_column(dataset)
  _cases, rate, search = _load_pair(environment, algorithm)
  rate, search = _align(dataset, _cases, rate, search)
  del _cases
  dataset, rate, search, column = collapse_duplicate_states(
      dataset, rate, search, column,
  )
  solvable = _search_solvable(np.asarray(dataset.difficulty, dtype=np.float64))
  features = np.column_stack(
      [
          np.asarray(getattr(dataset, name), dtype=np.float64)
          for name in FEATURE_COLUMNS_USED
      ]
  )
  experiment_ids = np.asarray(dataset.experiment, dtype=np.int64)
  finite = (
      np.isfinite(rate)
      & np.isfinite(search)
      & np.isfinite(column)
      & np.all(np.isfinite(features), axis=1)
  )
  del features, dataset
  gc.collect()
  if int((~finite).sum()):
    raise ValueError(
        f'{environment}/{algorithm}: {int((~finite).sum())} non-finite rows.'
    )
  kept_column = column[solvable]
  kept_rate = rate[solvable]
  kept_search = search[solvable]
  kept_experiment = experiment_ids[solvable]
  print(
      f'  distinct {int(solvable.sum())}/{int(solvable.size)} search-solvable, '
      f'min value z mean {float(kept_column.mean()):.3f}',
      flush=True,
  )
  x = np.ascontiguousarray(kept_column[:, None])
  y = np.ascontiguousarray(kept_rate)
  groups = _clusters(kept_experiment)
  point = _associations(x, y, with_mi=True, seed=seed)
  seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_boot)
  print(f'  bootstrap B={n_boot} on {len(groups)} searches', flush=True)
  if workers <= 1:
    _boot_init(x, y, groups)
    replicates = np.stack([_boot_replicate(int(value)) for value in seeds])
  else:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_boot_init,
        initargs=(x, y, groups),
    ) as pool:
      replicates = np.stack(list(pool.map(_boot_replicate, seeds, chunksize=8)))
  se = np.nanstd(replicates, axis=0, ddof=1)
  spearman_all = float(
      _pearson_columns(
          _rank_columns(column[:, None]),
          rankdata(rate).astype(np.float64),
      )[0]
  )
  curve, occupancy = _curve_column(kept_column, kept_rate, kept_experiment)
  search_rho = float(
      _pearson_columns(
          rankdata(kept_search)[:, None],
          rankdata(kept_rate).astype(np.float64),
      )[0]
  )
  search_mi = float(
      mutual_info_classif(
          kept_search[:, None],
          np.rint(kept_rate * N_EVAL).astype(np.int64),
          n_neighbors=MI_NEIGHBORS,
          random_state=seed,
          n_jobs=1,
      )[0]
  )
  rows = []
  for measure in ('pearson', 'spearman', 'kendall', 'mi_regression', 'mi_classification'):
    boot_index = {'pearson': 0, 'spearman': 1}.get(measure)
    rows.append({
        'environment': environment,
        'algorithm': algorithm,
        'n': int(y.shape[0]),
        'n_clusters': len(groups),
        'n_boot': int(n_boot) if boot_index is not None else 0,
        'eval_policies': N_EVAL,
        'measure': measure,
        'feature': MIN_VALUE_FEATURE,
        'feature_name': MIN_VALUE_LABEL,
        'estimate': float(point[measure][0]),
        'se': float(se[boot_index, 0]) if boot_index is not None else '',
        'ci_low': '',
        'ci_high': '',
        'spearman_all_rows': spearman_all,
    })
  curve_rows = []
  for level, mean, band_low, band_high in zip(curve.x, curve.y, curve.low, curve.high):
    curve_rows.append({
        'environment': environment,
        'algorithm': algorithm,
        'feature': MIN_VALUE_FEATURE,
        'feature_name': MIN_VALUE_LABEL,
        'eval_fail_rate': float(level),
        'n': int(occupancy[int(round(level * N_EVAL))]),
        'mean_sd_units': float(mean),
        'se_low': float(band_low),
        'se_high': float(band_high),
    })
  print(
      f'  spearman {point["spearman"][0]:.3f} ± {float(se[1, 0]):.3f}  '
      f'mi {point["mi_classification"][0]:.3f}',
      flush=True,
  )
  return {
      'environment': environment,
      'algorithm': algorithm,
      'rows': rows,
      'curve': curve,
      'curve_rows': curve_rows,
      'search_rho': search_rho,
      'search_mi': search_mi,
  }


def _read_csv(path: str) -> list[dict]:
  with open(path, newline='') as handle:
    return list(csv.DictReader(handle))


def _curves_from_csv(rows: list[dict]) -> dict:
  grouped: dict[tuple, list[dict]] = {}
  for row in rows:
    if row['feature'] not in CURVE_FEATURES:
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


def _slot_labels(features: tuple[str, ...]) -> list[str]:
  """Two-line slot names. Six columns collide when the names stay on one line."""
  return [
      figs.DISAGREEMENT_LINES.get(key, figs.label(key))
      for key in features
  ]


def _search_baselines(environment: str, algorithm: str, *, seed: int) -> tuple[float, float]:
  """ρ(r_S, r_E) and I(r_S; r_E) on the same rows as the association figure."""
  path = predictor_dataset_path(environment, algorithm, balanced=True)
  dataset = load_predictor_dataset(path, load_topk=False)
  _cases, rate, search = _load_pair(environment, algorithm)
  rate, search = _align(dataset, _cases, rate, search)
  del _cases
  dataset, rate, search = collapse_duplicate_states(dataset, rate, search)
  solvable = _search_solvable(np.asarray(dataset.difficulty, dtype=np.float64))
  del dataset
  gc.collect()
  rate = rate[solvable]
  search = search[solvable]
  rho = float(
      _pearson_columns(
          rankdata(search)[:, None],
          rankdata(rate).astype(np.float64),
      )[0]
  )
  mi = float(
      mutual_info_classif(
          search[:, None],
          np.rint(rate * N_EVAL).astype(np.int64),
          n_neighbors=MI_NEIGHBORS,
          random_state=seed,
          n_jobs=1,
      )[0]
  )
  return rho, mi


def _draw_min_value_figures(
    estimates: dict,
    curves: dict,
    spearman_baseline: dict,
    mi_baseline: dict,
) -> None:
  association = figs.association_figure(
      estimates,
      measures=('spearman', 'mi'),
      features=MIN_VALUE_FEATURES,
      slot_labels=_slot_labels(MIN_VALUE_FEATURES),
      baseline=mi_baseline,
      spearman_baseline=spearman_baseline,
  )
  curves_figure = figs.feature_curves_figure(
      curves, measures=MIN_VALUE_FEATURES, markers=8,
  )
  os.makedirs(FIGURE_DIR, exist_ok=True)
  association_path = os.path.join(FIGURE_DIR, 'associations_minV.pdf')
  curves_path = os.path.join(FIGURE_DIR, 'feature_curves_minV.pdf')
  figs.save(association, association_path, png=True)
  figs.save(curves_figure, curves_path, png=True)
  print(f'wrote {association_path}', flush=True)
  print(f'wrote {curves_path}', flush=True)


def write_min_value_figures(
    *,
    n_boot: int,
    seed: int,
    workers: int,
    environments: tuple[str, ...] = ENVIRONMENTS,
    algorithms: tuple[str, ...] = ALGORITHMS,
) -> None:
  """Copy the association and feature-curve figures and add the min value z-score."""
  if tuple(environments) != tuple(ENVIRONMENTS) or tuple(algorithms) != tuple(ALGORITHMS):
    raise ValueError(
        'The min-value figures need every environment and policy set, '
        'so the existing columns stay complete.'
    )
  correlation_rows = _read_csv(os.path.join(FIGURE_DIR, 'eval_fail_correlations_p32.csv'))
  curve_rows = _read_csv(os.path.join(FIGURE_DIR, 'difficulty_feature_curves_p32.csv'))
  estimates = {}
  for row in correlation_rows:
    if row['feature'] not in CURVE_FEATURES:
      continue
    if row['measure'] == 'spearman':
      estimates[('spearman', row['feature'], row['algorithm'], row['environment'])] = (
          figs.Estimate.from_se(float(row['estimate']), float(row['se']))
      )
    elif row['measure'] == 'mi_classification':
      estimates[('mi', row['feature'], row['algorithm'], row['environment'])] = (
          figs.Estimate(float(row['estimate']))
      )
  curves = _curves_from_csv(curve_rows)
  analyses = []
  for environment in environments:
    for algorithm in algorithms:
      analyses.append(
          analyse_minimum_value(
              environment,
              algorithm,
              n_boot=n_boot,
              seed=seed,
              workers=workers,
          )
      )
  spearman_baseline = {}
  mi_baseline = {}
  for result in analyses:
    environment = result['environment']
    algorithm = result['algorithm']
    for row in result['rows']:
      if row['measure'] == 'spearman':
        estimates[('spearman', MIN_VALUE_FEATURE, algorithm, environment)] = (
            figs.Estimate.from_se(row['estimate'], float(row['se']))
        )
      elif row['measure'] == 'mi_classification':
        estimates[('mi', MIN_VALUE_FEATURE, algorithm, environment)] = figs.Estimate(
            row['estimate']
        )
    spearman_baseline[(algorithm, environment)] = figs.Estimate(result['search_rho'])
    mi_baseline[(algorithm, environment)] = figs.Estimate(result['search_mi'])
    curves[(environment, MIN_VALUE_FEATURE, algorithm)] = result['curve']
  _write_csv(
      os.path.join(FIGURE_DIR, 'eval_fail_correlations_minV.csv'),
      [row for result in analyses for row in result['rows']],
  )
  _write_csv(
      os.path.join(FIGURE_DIR, 'difficulty_feature_curves_minV.csv'),
      [row for result in analyses for row in result['curve_rows']],
  )
  _draw_min_value_figures(estimates, curves, spearman_baseline, mi_baseline)


def _load_min_value_results() -> tuple[dict, dict]:
  """Estimates and curves already computed for the six-column figures."""
  correlation_rows = _read_csv(os.path.join(FIGURE_DIR, 'eval_fail_correlations_p32.csv'))
  correlation_rows.extend(
      _read_csv(os.path.join(FIGURE_DIR, 'eval_fail_correlations_minV.csv'))
  )
  estimates = {}
  for row in correlation_rows:
    if row['feature'] not in MIN_VALUE_FEATURES:
      continue
    if row['measure'] == 'spearman':
      estimates[('spearman', row['feature'], row['algorithm'], row['environment'])] = (
          figs.Estimate.from_se(float(row['estimate']), float(row['se']))
      )
    elif row['measure'] == 'mi_classification':
      estimates[('mi', row['feature'], row['algorithm'], row['environment'])] = (
          figs.Estimate(float(row['estimate']))
      )
  curves = _curves_from_csv(
      _read_csv(os.path.join(FIGURE_DIR, 'difficulty_feature_curves_p32.csv'))
  )
  grouped: dict[tuple, list[dict]] = {}
  for row in _read_csv(os.path.join(FIGURE_DIR, 'difficulty_feature_curves_minV.csv')):
    key = (row['environment'], row['feature'], row['algorithm'])
    grouped.setdefault(key, []).append(row)
  for key, group in grouped.items():
    group.sort(key=lambda row: float(row['eval_fail_rate']))
    curves[key] = figs.Curve(
        [float(row['eval_fail_rate']) for row in group],
        [float(row['mean_sd_units']) for row in group],
        [float(row['se_low']) for row in group],
        [float(row['se_high']) for row in group],
    )
  return estimates, curves


def redraw_min_value_figures(*, seed: int) -> None:
  """Redraw the min-value figures from the saved estimates."""
  estimates, curves = _load_min_value_results()
  expected = (
      len(MIN_VALUE_FEATURES) * 2 * len(ENVIRONMENTS) * len(ALGORITHMS)
  )
  if len(estimates) != expected or len(curves) != expected // 2:
    raise ValueError(
        f'Expected {expected} estimates and {expected // 2} curves, '
        f'found {len(estimates)} and {len(curves)}.'
    )
  spearman_baseline = {}
  mi_baseline = {}
  for environment in ENVIRONMENTS:
    for algorithm in ALGORITHMS:
      rho, mi = _search_baselines(environment, algorithm, seed=seed)
      spearman_baseline[(algorithm, environment)] = figs.Estimate(rho)
      mi_baseline[(algorithm, environment)] = figs.Estimate(mi)
      print(f'{environment}/{algorithm} baseline ρ={rho:.3f}  I={mi:.3f}', flush=True)
  _draw_min_value_figures(estimates, curves, spearman_baseline, mi_baseline)


def main() -> None:
  args = parse_args()
  if args.min_value:
    write_min_value_figures(
        n_boot=args.boot,
        seed=args.seed,
        workers=args.workers,
        environments=(args.environment,) if args.environment else ENVIRONMENTS,
        algorithms=(args.algorithm,) if args.algorithm else ALGORITHMS,
    )
    return
  environments = (args.environment,) if args.environment else ENVIRONMENTS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  results = []
  for environment in environments:
    for algorithm in algorithms:
      results.append(
          analyse(
              environment,
              algorithm,
              n_boot=args.boot,
              seed=args.seed,
              workers=args.workers,
          )
      )
  rows = [row for result in results for row in result['rows']]
  curve_rows = [row for result in results for row in result['curve_rows']]
  _write_csv(os.path.join(FIGURE_DIR, 'eval_fail_correlations_p32.csv'), rows)
  _write_tex(os.path.join(FIGURE_DIR, 'eval_fail_correlations_p32.tex'), rows)
  _write_csv(os.path.join(FIGURE_DIR, 'difficulty_feature_curves_p32.csv'), curve_rows)
  if len(results) == len(ENVIRONMENTS) * len(ALGORITHMS):
    _figures(results)
  else:
    print('partial run: figures need every environment and policy set', flush=True)


if __name__ == '__main__':
  main()
