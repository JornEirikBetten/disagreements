"""Cross-validate the held-out fail-rate regressors.

The study question is whether disagreement inside the *search* ensemble
predicts how many *held-out* policies a test case defeats. The search fail
rate is how the cases were sampled, not a feature that would be available
when the test is used: computing it means rolling every search policy, which
is the cost the features exist to avoid. This script therefore scores five search-ensemble disagreements against a
mean baseline: action magnitude, action direction, value estimate,
attributions, and criticality. Criticality is the mean pairwise absolute
difference of per-policy z-scores of ``sum log σ``, standardized across
the cases in each experiment. The stored predictor column is the raw gap
and is not used.

An optional ``--include-search-rate`` arm reports how well the search fail
rate itself predicts the held-out rate. That number is a measurement
ceiling -- two 16-policy estimates of the same latent difficulty -- not a
competitor the features have to beat.

Folds are grouped by experiment (leave-one-search-run-out) and cases are
collapsed to distinct initial states first, so a re-collected elite cannot
sit in both the training and test fold. The target and the case filter match
``train_difficulty_predictors.py``.

Usage:
  python cv_difficulty_predictors.py
  python cv_difficulty_predictors.py --environment hopper --algorithm ppo
  python cv_difficulty_predictors.py --folds 5
  python cv_difficulty_predictors.py --split random   # leaky, for comparison
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, StratifiedKFold

from analysis.difficulty_predictor import (
    difficulty_bins,
    fail_rate_target,
    feature_matrix,
    lin_ccc,
    mean_baseline_metrics,
    regression_metrics,
)
from analysis.predictor_dataset import (
    collapse_duplicate_states,
    load_predictor_dataset,
    predictor_dataset_path,
)
from plotting.plot_predictor_agreement import write_score_figure
from train_difficulty_predictors import (
    ALGORITHMS,
    ENVIRONMENTS,
    _search_solvable,
    _load_eval_fail_rates,
    prepare_eval_target,
)

# The five disagreements in the association figure, in that order.
# Criticality is replaced with the z-scored measure before the matrix is built.
CV_COLUMNS = (
    'action_scale_disagreement',
    'action_disagreement',
    'value_disagreement',
    'feature_attribution_disagreement',
    'criticality_disagreement',
)
CV_LABELS = (
    'action magnitude disagreement',
    'action direction disagreement',
    'value disagreement',
    'feature attribution disagreement',
    'criticality disagreement',
)
FEATURE_SET = 'five_zcriticality'

N_SPLITS = 10
FIGURE_DIR = os.path.join('figures', 'predictors')
METRIC_KEYS = ('mae', 'rmse', 'r2', 'spearman', 'ccc')

DISAGREEMENT_ARM = 'disagreement'
SEARCH_RATE_ARM = 'search_fail_rate'


def _result_tag(n_splits: int, split: str, *, pooled: bool = False) -> str:
  tag = f'extra_trees_{int(n_splits)}_foldcv'
  if split != 'random':
    tag += f'_{split}'
  if pooled:
    tag += '_pooled'
  return tag


def metrics_path(n_splits: int, split: str, *, pooled: bool = False) -> str:
  """Result file for a given fold count and split, so runs cannot collide."""
  return os.path.join(
      FIGURE_DIR, f'{_result_tag(n_splits, split, pooled=pooled)}_metrics.csv'
  )


def scores_path(n_splits: int, split: str, *, pooled: bool = False) -> str:
  """One out-of-fold score per dataset, the number drawn in the figure."""
  return os.path.join(
      FIGURE_DIR, f'{_result_tag(n_splits, split, pooled=pooled)}_scores.csv'
  )


def figure_path(n_splits: int, split: str, *, pooled: bool = False) -> str:
  return os.path.join(
      FIGURE_DIR, f'{_result_tag(n_splits, split, pooled=pooled)}_scores.pdf'
  )


def search_eval_path(n_splits: int, split: str, *, pooled: bool = False) -> str:
  return os.path.join(
      FIGURE_DIR,
      f'{_result_tag(n_splits, split, pooled=pooled)}_search_eval_spearman.csv',
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Cross-validate extra-trees regressors of the held-out fail rate '
          'from search-ensemble disagreement features.'
      ),
  )
  parser.add_argument('--environment', choices=ENVIRONMENTS)
  parser.add_argument('--algorithm', choices=ALGORITHMS)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--folds',
      type=int,
      default=N_SPLITS,
      help=(
          f'Number of folds (default: {N_SPLITS}). Each fold trains on '
          '1 - 1/folds of the cases and tests on the rest.'
      ),
  )
  parser.add_argument('--n-estimators', type=int, default=500)
  parser.add_argument(
      '--n-jobs',
      type=int,
      default=-1,
      help='Workers used to grow trees. -1 uses every core.',
  )
  parser.add_argument(
      '--forest-batch',
      type=int,
      default=0,
      help=(
          'Trees held in memory at once. 0 grows the whole forest in one '
          'fit. A positive value grows that many, records their predictions, '
          'and discards them, which is the same average with a smaller peak.'
      ),
  )
  parser.add_argument(
      '--permutation-repeats',
      type=int,
      default=5,
      help='Shuffles per feature for held-out importance (0 disables).',
  )
  parser.add_argument(
      '--include-search-rate',
      action='store_true',
      help=(
          'Also score the search fail rate as a measurement-ceiling '
          'diagnostic. It is not a feature the study would have at test time.'
      ),
  )
  parser.add_argument(
      '--pool-algorithms',
      action='store_true',
      help=(
          'Pool PPO, SAC, and PPO+SAC within each environment. Each of the '
          'fifteen searches is one cross-validation fold.'
      ),
  )
  parser.add_argument(
      '--split',
      choices=('experiment', 'random'),
      default='experiment',
      help=(
          'How to form folds. ``experiment`` (default) holds out a whole GA '
          'run. ``random`` is the leaky split that put elite copies in both '
          'sides.'
      ),
  )
  return parser.parse_args()


@dataclass
class CaseTable:
  """One dataset's cases, with both difficulty labels kept side by side."""

  environment: str
  algorithm: str
  features: np.ndarray  # (n_cases, n_features) search-ensemble disagreement
  search_fail_rate: np.ndarray  # (n_cases,) share of search policies that failed
  eval_fail_rate: np.ndarray  # (n_cases,) share of held-out policies: the target
  experiment: np.ndarray  # (n_cases,) independent GA run id

  @property
  def n_cases(self) -> int:
    return int(self.eval_fail_rate.size)

  def design_matrix(self, arm: str) -> np.ndarray:
    if arm == DISAGREEMENT_ARM:
      return self.features
    if arm == SEARCH_RATE_ARM:
      return self.search_fail_rate[:, None]
    raise ValueError(f'Unknown arm {arm!r}')


def load_case_table(environment: str, algorithm: str) -> CaseTable:
  """Load one dataset, keeping the search fail rate that the target swap drops.

  ``prepare_eval_target`` discards cases no search policy solved and then
  overwrites ``difficulty`` with the held-out fail rate. The search fail
  rate is kept only as a measurement-ceiling diagnostic.
  """
  path = predictor_dataset_path(environment, algorithm, balanced=True)
  print(
      f'Cross-validating extra trees {environment}/{algorithm} from {path}',
      flush=True,
  )
  dataset = load_predictor_dataset(path, load_topk=False)
  from shared_eval_figures import _standardized_criticality_column

  dataset.criticality_disagreement = _standardized_criticality_column(
      dataset, workers=max(1, min(8, os.cpu_count() or 1)),
  )
  eval_rates = _load_eval_fail_rates(dataset)
  dataset, eval_rates = collapse_duplicate_states(dataset, eval_rates)
  solvable = _search_solvable(dataset.difficulty)
  search_fail_rate = np.asarray(dataset.difficulty, dtype=np.float64)[solvable]
  experiment = np.asarray(dataset.experiment, dtype=np.int64)[solvable]
  kept = prepare_eval_target(dataset, eval_rates=eval_rates)
  table = CaseTable(
      environment=kept.environment,
      algorithm=kept.algorithm,
      features=feature_matrix(kept, columns=CV_COLUMNS),
      search_fail_rate=search_fail_rate,
      eval_fail_rate=fail_rate_target(kept.difficulty),
      experiment=experiment,
  )
  del dataset, kept
  gc.collect()
  return table


def _estimator(
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> ExtraTreesRegressor:
  return ExtraTreesRegressor(
      n_estimators=int(n_estimators),
      max_depth=None,
      min_samples_leaf=1,
      max_features=1.0,
      n_jobs=int(n_jobs),
      random_state=random_state,
  )


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    batch_size: int,
) -> tuple[np.ndarray, ExtraTreesRegressor | None]:
  """Mean prediction of an extra-trees forest.

  ``batch_size`` of 0 fits every tree in one model. A smaller batch fits
  that many trees, adds their predictions, and frees them. The result is
  still the mean of ``n_estimators`` trees; only the resident set is smaller.
  """
  if batch_size <= 0 or batch_size >= n_estimators:
    model = _estimator(n_estimators, random_state, n_jobs)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64), model
  total = np.zeros(x_test.shape[0], dtype=np.float64)
  done = 0
  batch_index = 0
  while done < n_estimators:
    size = min(int(batch_size), int(n_estimators) - done)
    print(
        f'    trees {done}:{done + size} of {n_estimators}',
        flush=True,
    )
    model = _estimator(size, random_state + batch_index, n_jobs)
    model.fit(x_train, y_train)
    total += np.asarray(model.predict(x_test), dtype=np.float64) * size
    done += size
    batch_index += 1
    del model
    gc.collect()
  return total / float(n_estimators), None


def _held_out_importance(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    n_repeats: int,
) -> np.ndarray:
  """Drop in held-out R2 when one feature column is shuffled.

  ``feature_importances_`` is impurity based: it is measured on the training
  rows and inflates features with many distinct values, so it reports which
  feature the forest split on, not which feature carries signal about cases
  the forest has not seen. Shuffling a column of the test fold answers the
  second question, which is the one this study asks.
  """
  result = permutation_importance(
      model,
      x_test,
      y_test,
      n_repeats=int(n_repeats),
      random_state=random_state,
      scoring='r2',
      n_jobs=1,
  )
  return np.asarray(result.importances_mean, dtype=np.float64)


def _metric_row(
    table: CaseTable,
    arm: str,
    fold: int,
    n_train: int,
    n_test: int,
    scores: dict,
    baseline: dict,
    split: str,
) -> dict:
  row = {
      'environment': table.environment,
      'algorithm': table.algorithm,
      'features': arm,
      'feature_set': FEATURE_SET,
      'split': split,
      'unique_within': 'experiment',
      'fold': fold,
      'n_train': n_train,
      'n_test': n_test,
      'train_fraction': float(n_train) / float(table.n_cases),
  }
  for key in METRIC_KEYS:
    row[key] = scores[key]
    row[f'baseline_{key}'] = baseline[key]
  return row


def _summarize_arm(
    table: CaseTable,
    arm: str,
    rows: list[dict],
    oof_prediction: np.ndarray,
    split: str,
) -> dict:
  """One score per dataset: Spearman of the stitched out-of-fold predictions.

  The interval is the standard error of the fold-wise Spearman coefficients.
  With leave-one-experiment-out, that is the spread across search runs.
  """
  if not np.all(np.isfinite(oof_prediction)):
    missing = int(np.isnan(oof_prediction).sum())
    raise ValueError(
        f'{table.environment}/{table.algorithm} {arm} left {missing} cases '
        'without an out-of-fold prediction.'
    )
  pooled = regression_metrics(table.eval_fail_rate, oof_prediction)
  fold_rho = np.asarray([row['spearman'] for row in rows], dtype=np.float64)
  fold_ccc = np.asarray([row['ccc'] for row in rows], dtype=np.float64)
  n_folds = int(fold_rho.size)
  fold_sd = float(np.std(fold_rho, ddof=1)) if n_folds > 1 else float('nan')
  fold_se = fold_sd / np.sqrt(n_folds) if n_folds > 1 else float('nan')
  ccc_sd = float(np.std(fold_ccc, ddof=1)) if n_folds > 1 else float('nan')
  ccc_se = ccc_sd / np.sqrt(n_folds) if n_folds > 1 else float('nan')
  return {
      'environment': table.environment,
      'algorithm': table.algorithm,
      'features': arm,
      'feature_set': FEATURE_SET,
      'split': split,
      'n_cases': table.n_cases,
      'n_folds': n_folds,
      'spearman': pooled['spearman'],
      'r2': pooled['r2'],
      'mae': pooled['mae'],
      'rmse': pooled['rmse'],
      'ccc': pooled['ccc'],
      'spearman_fold_mean': float(np.mean(fold_rho)),
      'spearman_fold_sd': fold_sd,
      'spearman_se': fold_se,
      'ccc_fold_mean': float(np.mean(fold_ccc)),
      'ccc_fold_sd': ccc_sd,
      'ccc_se': ccc_se,
  }


def cross_validate_dataset(
    table: CaseTable,
    *,
    n_splits: int,
    n_estimators: int,
    seed: int,
    permutation_repeats: int = 0,
    arms: tuple[str, ...] = (DISAGREEMENT_ARM,),
    split: str = 'experiment',
    n_jobs: int = -1,
    forest_batch: int = 0,
) -> tuple[list[dict], list[dict]]:
  """Fit every requested arm on identical folds."""
  if n_splits < 2:
    raise ValueError(f'Need at least 2 folds, got {n_splits}')
  y = table.eval_fail_rate
  if split == 'experiment':
    n_groups = int(np.unique(table.experiment).size)
    n_used = min(int(n_splits), n_groups)
    if n_used < 2:
      raise ValueError(
          f'{table.environment}/{table.algorithm} has {n_groups} experiments; '
          'need at least 2 for grouped cross-validation.'
      )
    if n_used < n_splits:
      print(
          f'  {n_groups} experiments: using {n_used}-fold leave-one-run-out '
          f'(requested {n_splits}).',
          flush=True,
      )
    splitter = GroupKFold(n_splits=n_used)
    fold_iter = splitter.split(table.features, y, groups=table.experiment)
  elif split == 'random':
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_source = np.zeros((table.n_cases, 1), dtype=np.int8)
    fold_iter = splitter.split(fold_source, difficulty_bins(y))
  else:
    raise ValueError(f'Unknown split {split!r}')
  rows = []
  importances = []
  oof = {arm: np.full(y.shape, np.nan, dtype=np.float64) for arm in arms}
  for fold, (train_index, test_index) in enumerate(fold_iter):
    baseline = mean_baseline_metrics(y[train_index], y[test_index])
    for arm in arms:
      design = table.design_matrix(arm)
      prediction, model = _fit_predict(
          design[train_index],
          y[train_index],
          design[test_index],
          n_estimators=n_estimators,
          random_state=seed + fold,
          n_jobs=n_jobs,
          batch_size=forest_batch,
      )
      oof[arm][test_index] = prediction
      scores = regression_metrics(y[test_index], prediction)
      if arm == DISAGREEMENT_ARM and permutation_repeats > 0:
        if model is None:
          raise ValueError(
              'Held-out importance needs the whole forest resident; '
              'leave --forest-batch at 0.'
          )
        importances.append(
            _held_out_importance(
                model,
                design[test_index],
                y[test_index],
                seed + fold,
                permutation_repeats,
            )
        )
      row = _metric_row(
          table,
          arm,
          fold,
          int(train_index.size),
          int(test_index.size),
          scores,
          baseline,
          split,
      )
      rows.append(row)
      print(
          f"  fold {fold} {arm:>16}: "
          f"n_train={row['n_train']} n_test={row['n_test']} "
          f"({row['train_fraction']:.0%}/{1.0 - row['train_fraction']:.0%})  "
          f"MAE={scores['mae']:.4f}  RMSE={scores['rmse']:.4f}  "
          f"R2={scores['r2']:.4f}  spearman={scores['spearman']:.4f}  "
          f"ccc={scores['ccc']:.4f}",
          flush=True,
      )
      del model, prediction
      gc.collect()
  _print_summary(rows, importances, arms)
  summaries = [
      _summarize_arm(
          table,
          arm,
          [row for row in rows if row['features'] == arm],
          oof[arm],
          split,
      )
      for arm in arms
  ]
  for summary in summaries:
    print(
        f"  out-of-fold {summary['features']:>16}: "
        f"spearman={summary['spearman']:.4f}  "
        f"fold mean={summary['spearman_fold_mean']:.4f} "
        f"+- se {summary['spearman_se']:.4f}  "
        f"ccc={summary['ccc']:.4f}  "
        f"fold mean={summary['ccc_fold_mean']:.4f} "
        f"+- se {summary['ccc_se']:.4f}  "
        f"R2={summary['r2']:.4f}",
        flush=True,
    )
  return rows, summaries


def _fold_mean(rows: list[dict], key: str) -> tuple[float, float]:
  values = np.asarray([row[key] for row in rows], dtype=np.float64)
  return float(np.mean(values)), float(np.std(values))


def _print_summary(
    rows: list[dict], importances: list[np.ndarray], arms: tuple[str, ...]
) -> None:
  print(
      '  mean over folds (+- is the spread across folds, '
      'not a confidence interval):',
      flush=True,
  )
  arm_rows = {arm: [row for row in rows if row['features'] == arm] for arm in arms}
  for arm in arms:
    parts = []
    for key in METRIC_KEYS:
      mean, spread = _fold_mean(arm_rows[arm], key)
      parts.append(f'{key}={mean:.4f} +- {spread:.4f}')
    print(f'    {arm:>16}: ' + '  '.join(parts), flush=True)
  parts = []
  for key in METRIC_KEYS:
    mean, spread = _fold_mean(arm_rows[arms[0]], f'baseline_{key}')
    parts.append(f'{key}={mean:.4f} +- {spread:.4f}')
  print(f'    {"mean":>16}: ' + '  '.join(parts), flush=True)

  if SEARCH_RATE_ARM in arm_rows:
    ceiling, _ = _fold_mean(arm_rows[SEARCH_RATE_ARM], 'r2')
    treatment, _ = _fold_mean(arm_rows[DISAGREEMENT_ARM], 'r2')
    print(
        f'    search-fail-rate measurement ceiling: R2={ceiling:.4f}; '
        f'disagreement R2={treatment:.4f}',
        flush=True,
    )

  if not importances:
    return
  mean_importance = np.mean(importances, axis=0)
  print('  held-out permutation importance (R2 lost when shuffled):', flush=True)
  for index in np.argsort(mean_importance)[::-1]:
    print(
        f'    {CV_LABELS[index]}: {mean_importance[index]:.4f}',
        flush=True,
    )


def _read_metrics(path: str) -> list[dict]:
  if not os.path.isfile(path):
    return []
  with open(path, newline='') as handle:
    return list(csv.DictReader(handle))


def _coerce_metric(row: dict) -> dict:
  """CSV reload turns numbers into strings. Restore the ones the figure uses."""
  coerced = dict(row)
  for key, value in coerced.items():
    if key in (
        'environment', 'algorithm', 'features', 'feature_set',
        'split', 'unique_within',
    ):
      continue
    if value is None or value == '':
      continue
    try:
      number = float(value)
    except ValueError:
      continue
    integer_field = key == 'fold' or key.startswith('n_')
    coerced[key] = int(number) if integer_field and number.is_integer() else number
  return coerced


def _has_ccc(row: dict) -> bool:
  value = row.get('ccc')
  if value is None or value == '':
    return False
  return bool(np.isfinite(float(value)))


def _drop_dataset(rows: list[dict], environment: str, algorithm: str) -> list[dict]:
  return [
      row for row in rows
      if not (
          row.get('environment') == environment
          and row.get('algorithm') == algorithm
      )
  ]


def _write_metrics(rows: list[dict], path: str) -> None:
  if not rows:
    return
  fieldnames = list(rows[0])
  for row in rows[1:]:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
  from scipy.stats import spearmanr

  if np.unique(x).size < 2 or np.unique(y).size < 2:
    return float('nan')
  return float(spearmanr(x, y).statistic)


def search_eval_spearman(table: CaseTable) -> float:
  """Spearman ρ(r_S, r_E) on the rows the extra-trees model is scored on."""
  return _spearman(table.search_fail_rate, table.eval_fail_rate)


def search_eval_agreement(table: CaseTable) -> dict:
  """Rank correlation and Lin's CCC of r_S with r_E on the scored rows."""
  return {
      'environment': table.environment,
      'algorithm': table.algorithm,
      'n_cases': table.n_cases,
      'spearman': _spearman(table.search_fail_rate, table.eval_fail_rate),
      'ccc': lin_ccc(table.search_fail_rate, table.eval_fail_rate),
  }


def load_pooled_environment_table(
    environment: str,
) -> tuple[CaseTable, list[dict]]:
  """Stack PPO, SAC, and PPO+SAC, with one group id per search.

  Experiment numbers restart at 1 inside each policy set, so the group id
  keeps the policy set and the search run apart. Fifteen searches is five
  runs of each of the three policy sets.
  """
  features = []
  search_rates = []
  eval_rates = []
  groups = []
  per_search = []
  for index, algorithm in enumerate(ALGORITHMS):
    table = load_case_table(environment, algorithm)
    features.append(table.features)
    search_rates.append(table.search_fail_rate)
    eval_rates.append(table.eval_fail_rate)
    groups.append(index * 10 + np.asarray(table.experiment, dtype=np.int64))
    for experiment in np.unique(table.experiment):
      mask = np.asarray(table.experiment) == experiment
      per_search.append({
          'environment': environment,
          'algorithm': algorithm,
          'experiment': int(experiment),
          'n_cases': int(np.sum(mask)),
          'spearman': _spearman(
              table.search_fail_rate[mask], table.eval_fail_rate[mask]
          ),
      })
    del table
    gc.collect()
  pooled = CaseTable(
      environment=environment,
      algorithm='all',
      features=np.concatenate(features, axis=0),
      search_fail_rate=np.concatenate(search_rates),
      eval_fail_rate=np.concatenate(eval_rates),
      experiment=np.concatenate(groups),
  )
  n_searches = int(np.unique(pooled.experiment).size)
  print(
      f'  pooled {environment}: {pooled.n_cases} cases from {n_searches} searches',
      flush=True,
  )
  return pooled, per_search


def write_pooled_score_figure(
    summaries: list[dict],
    path: str,
    *,
    per_search: list[dict],
) -> str:
  """Pooled-environment scores, with one star per search."""
  import matplotlib

  matplotlib.use('Agg', force=False)
  import plotting.style as figs

  estimates = {}
  for row in summaries:
    if row['features'] != DISAGREEMENT_ARM:
      continue
    se = row['spearman_se']
    if np.isfinite(se):
      estimate = figs.Estimate.from_se(row['spearman'], se)
    else:
      estimate = figs.Estimate(row['spearman'])
    estimates[row['environment']] = estimate
  correlations = {
      (row['environment'], row['algorithm'], row['experiment']): row['spearman']
      for row in per_search
  }
  figure = figs.pooled_environment_score_figure(
      estimates, correlations,
  )
  return figs.save(figure, path, png=True)


def main() -> int:
  args = parse_args()
  if args.pool_algorithms and args.algorithm:
    raise ValueError('--pool-algorithms combines every policy set; drop --algorithm.')
  if args.folds < 2:
    raise ValueError(f'Need at least 2 folds, got {args.folds}')
  if args.n_jobs == 0 or args.n_estimators < 1 or args.forest_batch < 0:
    raise ValueError(
        f'Need n_estimators >= 1, forest_batch >= 0, and n_jobs != 0; '
        f'got n_estimators={args.n_estimators}, '
        f'forest_batch={args.forest_batch}, n_jobs={args.n_jobs}.'
    )
  if args.permutation_repeats < 0:
    raise ValueError(
        f'permutation_repeats must be >= 0, got {args.permutation_repeats}'
    )
  test_fraction = 1.0 / args.folds
  pooled = bool(args.pool_algorithms)
  output_path = metrics_path(args.folds, args.split, pooled=pooled)
  score_output = scores_path(args.folds, args.split, pooled=pooled)
  score_figure = figure_path(args.folds, args.split, pooled=pooled)
  search_output = search_eval_path(args.folds, args.split, pooled=pooled)
  print(
      f'Using {args.folds}-fold cross-validation, split={args.split} '
      f'({1.0 - test_fraction:.0%}/{test_fraction:.0%} per fold), '
      f'n_jobs={args.n_jobs}, forest_batch={args.forest_batch}. '
      f'Features: {", ".join(CV_LABELS)}. Criticality is z-scored per policy.',
      flush=True,
  )
  environments = (args.environment,) if args.environment else ENVIRONMENTS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  # A scores file means this sweep already finished some datasets. The older
  # metrics table has no paired scores file, so it is not resumed.
  def _current(row: dict) -> bool:
    return row.get('feature_set') == FEATURE_SET

  all_summaries = [
      row for row in
      (_coerce_metric(row) for row in _read_metrics(score_output))
      if _current(row)
  ]
  all_rows = [
      row for row in
      (_coerce_metric(row) for row in _read_metrics(output_path))
      if _current(row)
  ] if all_summaries else []
  all_searches = (
      [_coerce_metric(row) for row in _read_metrics(search_output)]
      if all_summaries else []
  )
  finished = {
      (row['environment'], row['algorithm']) for row in all_summaries
      if row.get('features') == DISAGREEMENT_ARM and _has_ccc(row)
  }
  if finished:
    print(
        f'Resuming. Already scored: '
        + ', '.join(f'{env}/{alg}' for env, alg in sorted(finished)),
        flush=True,
    )
  for environment in environments:
    if pooled:
      if (environment, 'all') in finished:
        print(f'  {environment}: already scored', flush=True)
        continue
      try:
        table, per_search = load_pooled_environment_table(environment)
      except (FileNotFoundError, ValueError) as exc:
        print(f'  skipped: {exc}', flush=True)
        continue
      n_searches = int(np.unique(table.experiment).size)
      if n_searches != args.folds:
        raise ValueError(
            f'{environment} has {n_searches} searches, so leave-one-search-out '
            f'is {n_searches}-fold, not {args.folds}-fold.'
        )
    else:
      per_search = []
    algorithm_loop = (None,) if pooled else algorithms
    for algorithm in algorithm_loop:
      if not pooled:
        if (environment, algorithm) in finished:
          print(f'  {environment}/{algorithm}: already scored', flush=True)
          continue
        try:
          table = load_case_table(environment, algorithm)
        except (FileNotFoundError, ValueError) as exc:
          print(f'  skipped: {exc}', flush=True)
          continue
      arms = (DISAGREEMENT_ARM,)
      if args.include_search_rate:
        arms = (DISAGREEMENT_ARM, SEARCH_RATE_ARM)
      if not pooled:
        agreement = search_eval_agreement(table)
        all_searches = _drop_dataset(
            all_searches, table.environment, table.algorithm,
        )
        all_searches.append(agreement)
      scored_algorithm = 'all' if pooled else algorithm
      all_rows = _drop_dataset(all_rows, environment, scored_algorithm)
      all_summaries = _drop_dataset(all_summaries, environment, scored_algorithm)
      rows, summaries = cross_validate_dataset(
          table,
          n_splits=args.folds,
          n_estimators=args.n_estimators,
          seed=args.seed,
          permutation_repeats=args.permutation_repeats,
          arms=arms,
          split=args.split,
          n_jobs=args.n_jobs,
          forest_batch=args.forest_batch,
      )
      all_rows.extend(rows)
      all_summaries.extend(summaries)
      if pooled:
        all_searches.extend(per_search)
      # Rewritten after every dataset so a long sweep that dies part way
      # through still leaves the datasets it finished.
      _write_metrics(all_rows, output_path)
      _write_metrics(all_summaries, score_output)
      if pooled:
        _write_metrics(all_searches, search_output)
        written = write_pooled_score_figure(
            all_summaries, score_figure, per_search=all_searches,
        )
      else:
        _write_metrics(all_searches, search_output)
        expected = len(ENVIRONMENTS) * len(ALGORITHMS)
        ready = [
            row for row in all_summaries
            if row.get('features') == DISAGREEMENT_ARM and _has_ccc(row)
        ]
        # The published figure stays in place until every dataset in this
        # feature set has concordance.
        if len(ready) == expected:
          written = write_score_figure(
              all_summaries, score_figure, search_eval=all_searches,
          )
        else:
          written = (
              f'{score_figure} waits ({len(ready)}/{expected} '
              'datasets have concordance)'
          )
      print(f'  wrote {output_path}', flush=True)
      print(f'  wrote {score_output}', flush=True)
      print(f'  wrote {written}', flush=True)
      del table
      gc.collect()
  if not all_rows:
    raise SystemExit('No datasets had held-out evaluation fail rates.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
