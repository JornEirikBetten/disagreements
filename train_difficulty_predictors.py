"""Train forest difficulty predictors from difficulty-balanced GA data.

Reads ``datasets/{env}/{alg}/predictor_data_balanced.pkl`` without the
top-k attribution matrices. The regression target is the held-out
evaluation fail rate, and only cases the search ensemble solved (not the
all-fail bin) are kept. Each regressor dataset is released before the
next one is loaded.

Usage:
  python train_difficulty_predictors.py
  python train_difficulty_predictors.py --model random-forest
  python train_difficulty_predictors.py --task classification-3 --balance
  python train_difficulty_predictors.py --task classification-binary

Regression and classification both use ``INCLUDED_FEATURES``. Four-class
classification is easy / fault-revealing / discriminating / impossible.
Three-class classification merges the middle two bands
(easy / fault-revealing / impossible).
Binary classification is fault-revealing vs not
(fail rate in (0, 1) vs always-succeed or always-fail).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np

from analysis.difficulty_predictor import (
    ALL_FAIL_SENTINEL,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    classifier_model_path,
    fit_classifier,
    fit_forest,
    predictor_model_path,
    save_fitted_classifier,
    save_fitted_predictor,
)
from analysis.predictor_dataset import load_predictor_dataset, predictor_dataset_path
from ga.collect import generation_shard_dir, iter_generation_shards

ENVIRONMENTS = ('ant', 'humanoid', 'walker2d', 'hopper')
ALGORITHMS = ('ppo', 'sac', 'ppo_sac')
ROLLOUT_LENGTHS = {
    'ant': 15,
    'humanoid': 15,
    'walker2d': 15,
    'hopper': 15,
}
_COMBOS = tuple(
    (environment, algorithm)
    for environment in ENVIRONMENTS
    for algorithm in ALGORITHMS
)
_ENV_DISPLAY = {'ant': 'Ant', 'humanoid': 'Humanoid', 'walker2d': 'Walker2d', 'hopper': 'Hopper'}
_ALG_DISPLAY = {'ppo': 'PPO', 'sac': 'SAC', 'ppo_sac': 'PPO+SAC'}
_NAVY = '#1f4e79'
_RED = '#c0392b'
FIGURE_DIR = os.path.join('figures', 'predictors')

# Dataset fields, or the labels in FEATURE_LABELS. Order is the model order.
INCLUDED_FEATURES = (
    'action scale disagreement',
    'value disagreement',
    'action disagreement',
    'feature attribution disagreement',
    'criticality disagreement',
    'std magnitude disagreement',
    'std direction disagreement',
)
_COLUMN_BY_LABEL = dict(zip(FEATURE_LABELS, FEATURE_COLUMNS))
_LABEL_BY_COLUMN = dict(zip(FEATURE_COLUMNS, FEATURE_LABELS))


def _resolve_included_features(
    names: tuple[str, ...] = INCLUDED_FEATURES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Map ``INCLUDED_FEATURES`` to dataset columns and plot labels."""
  columns = []
  labels = []
  for name in names:
    if name in _LABEL_BY_COLUMN:
      column = name
      label = _LABEL_BY_COLUMN[name]
    elif name in _COLUMN_BY_LABEL:
      column = _COLUMN_BY_LABEL[name]
      label = name
    else:
      known = ', '.join(FEATURE_LABELS)
      raise ValueError(
          f'Unknown feature {name!r}. Use a dataset field or one of: {known}'
      )
    columns.append(column)
    labels.append(label)
  if not columns:
    raise ValueError('INCLUDED_FEATURES is empty.')
  return tuple(columns), tuple(labels)


FEATURE_COLUMNS_USED, FEATURE_LABELS_USED = _resolve_included_features()
_CLI_MODELS = {
    'extra-trees': 'extra_trees',
    'random-forest': 'random_forest',
    'both': 'both',
}
_FIGURE_PREFIX = {
    'extra_trees': 'extra_trees',
    'random_forest': 'random_forest',
}
_CLASS_TICKS = {
    'easy': 'easy',
    'fault-revealing': 'fault-rev.',
    'discriminating': 'discrim.',
    'impossible': 'impossible',
    'not-fault-revealing': 'not fault-rev.',
}
_TASK_SCHEME = {
    'classification': 'four',
    'classification-3': 'three',
    'classification-binary': 'binary',
}
_SCHEME_TASK = {scheme: task for task, scheme in _TASK_SCHEME.items()}
_SCHEME_LABEL = {
    'four': '4-class',
    'three': '3-class',
    'binary': 'binary',
}
_TASK_FIGURE_SUFFIX = {
    'classification-3': '_3class',
    'classification-binary': '_binary',
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Train forest difficulty predictors on disagreement features.',
  )
  parser.add_argument(
      '--environment',
      choices=ENVIRONMENTS,
      help='If omitted, train all locomotion environments.',
  )
  parser.add_argument(
      '--algorithm',
      choices=ALGORITHMS,
      help='If omitted, train PPO and SAC.',
  )
  parser.add_argument(
      '--model',
      choices=tuple(_CLI_MODELS),
      default='extra-trees',
      help='Forest family to train. Default: extra-trees.',
  )
  parser.add_argument(
      '--task',
      choices=(
          'regression',
          'classification',
          'classification-3',
          'classification-binary',
      ),
      default='regression',
      help=(
          'Regression on fail-rate, 4-class labels, 3-class labels '
          '(easy / fault-revealing / impossible), or binary '
          'fault-revealing vs not.'
      ),
  )
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--balance',
      action='store_true',
      help=(
          'Undersample at random so each difficulty class has the same '
          'number of cases (the size of the smallest class) before the split.'
      ),
  )
  return parser.parse_args()


def _search_solvable(difficulty: np.ndarray) -> np.ndarray:
  """True where the search ensemble did not all fail."""
  y = np.asarray(difficulty, dtype=np.float64)
  all_fail = (y < 0.0) | np.isclose(y, ALL_FAIL_SENTINEL)
  return ~all_fail


def _collection_path(environment: str, algorithm: str, experiment: int) -> str:
  rollout_length = ROLLOUT_LENGTHS[environment]
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'difficulty_bins',
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(experiment)}_ga_collection.pkl',
  )


def _load_eval_fail_rates(dataset) -> np.ndarray:
  """Held-out fail rate for each predictor row, aligned to the dataset order."""
  rates = np.full(dataset.n_cases, np.nan, dtype=np.float64)
  experiments = [int(value) for value in np.unique(dataset.experiment)]
  for experiment in experiments:
    path = _collection_path(dataset.environment, dataset.algorithm, experiment)
    if not os.path.isdir(generation_shard_dir(path)):
      raise FileNotFoundError(
          f'No generation shards for {path}. '
          'Run add_heldout_fail_rates.py before training on eval rates.'
      )
    for record in iter_generation_shards(path):
      rate = getattr(record, 'eval_fail_rate', None)
      generation = int(record.generation)
      if rate is None:
        raise ValueError(
            f'{path} generation {generation} has no eval_fail_rate. '
            'Run add_heldout_fail_rates.py first.'
        )
      rate = np.asarray(rate, dtype=np.float64)
      positions = np.flatnonzero(
          (np.asarray(dataset.experiment) == experiment)
          & (np.asarray(dataset.generation) == generation)
      )
      case_index = np.asarray(dataset.case_index[positions], dtype=np.int64)
      if case_index.shape[0] != rate.shape[0] or not np.array_equal(
          np.sort(case_index), np.arange(rate.shape[0])
      ):
        raise ValueError(
            f'{path} generation {generation}: dataset has '
            f'{case_index.shape[0]} rows, shard has {rate.shape[0]} '
            'eval fail rates.'
        )
      rates[positions] = rate[case_index]
      del record
  missing = int(np.isnan(rates).sum())
  if missing:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm} is missing eval fail '
        f'rates for {missing} rows.'
    )
  return rates


def prepare_eval_target(dataset, eval_rates=None):
  """Keep search-solvable cases and use the held-out fail rate as difficulty.

  Pass ``eval_rates`` when the caller has already loaded them from the
  generation shards. Those shards are aligned with the *uncollapsed* dataset,
  so duplicates must be dropped from the rates array in the same step as
  the rows, then handed in here.
  """
  if eval_rates is None:
    rates = _load_eval_fail_rates(dataset)
  else:
    rates = np.asarray(eval_rates, dtype=np.float64)
    if rates.shape[0] != dataset.n_cases:
      raise ValueError(
          f'eval_rates has {rates.shape[0]} rows, dataset has {dataset.n_cases}'
      )
  solvable = _search_solvable(dataset.difficulty)
  n_drop = int(np.size(solvable) - int(np.sum(solvable)))
  print(
      f'  search-solvable {int(np.sum(solvable))}/{dataset.n_cases} '
      f'(dropped {n_drop} all-fail)',
      flush=True,
  )
  if int(np.sum(solvable)) < 3:
    raise ValueError(
        f'{dataset.environment}/{dataset.algorithm} has fewer than 3 '
        'search-solvable cases.'
    )
  kept = dataset.subset(np.flatnonzero(solvable))
  kept.difficulty = np.asarray(rates[solvable], dtype=np.float64)
  return kept


def test_fail_rate_ratios(y: np.ndarray) -> list[dict]:
  """Share of a split at each discrete held-out fail rate."""
  values = np.round(np.asarray(y, dtype=np.float64), decimals=6)
  levels, counts = np.unique(values, return_counts=True)
  n = int(values.size)
  rows = []
  for level, count in zip(levels, counts):
    rows.append({
        'fail_rate': float(level),
        'n': int(count),
        'ratio': float(count) / n if n else 0.0,
    })
  return rows


def _print_test_fail_rate_ratios(y: np.ndarray) -> None:
  rows = test_fail_rate_ratios(y)
  n = int(np.asarray(y).shape[0])
  print(f'  test fail-rate ratios (n={n}):', flush=True)
  for row in rows:
    print(
        f"    {row['fail_rate']:.4f}  {row['ratio']:.3f}  ({row['n']})",
        flush=True,
    )


def _style_ax(ax) -> None:
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_axisbelow(True)


def _combo_title(environment: str, algorithm: str) -> str:
  return f'{_ENV_DISPLAY[environment]} {_ALG_DISPLAY[algorithm]}'


def _combo_figure(cell: tuple[float, float], **kwargs):
  """One panel per environment (rows) and algorithm (columns)."""
  n_rows = len(ENVIRONMENTS)
  n_cols = len(ALGORITHMS)
  fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(cell[0] * n_cols, cell[1] * n_rows),
      **kwargs,
  )
  return fig, np.atleast_2d(axes)


def plot_pred_vs_true(fitted_by_combo: dict, path: str) -> None:
  fig, axes = _combo_figure(
      (4.3, 3.9), sharex=True, sharey=True, layout='constrained'
  )
  hexbin = None
  for ax, (environment, algorithm) in zip(axes.ravel(), _COMBOS, strict=True):
    fitted = fitted_by_combo[(environment, algorithm)]
    hexbin = ax.hexbin(
        fitted.y_test,
        fitted.y_test_pred,
        gridsize=35,
        mincnt=1,
        cmap='Blues',
        linewidths=0.0,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color=_RED, linewidth=1.1, zorder=3)
    test = fitted.metrics['test']
    ax.set_title(_combo_title(environment, algorithm))
    ax.text(
        0.04,
        0.96,
        f"R$^2$={test['r2']:.3f}\n"
        f"MAE={test['mae']:.3f}\n"
        f"$\\rho$={test['spearman']:.3f}",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8.5,
        color=_NAVY,
    )
    _style_ax(ax)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
  fig.colorbar(hexbin, ax=axes, fraction=0.03, pad=0.02, label='test cases')
  for ax in axes[-1]:
    ax.set_xlabel('held-out fail rate')
  for row in range(axes.shape[0]):
    axes[row, 0].set_ylabel('predicted fail rate')
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path)
  plt.close(fig)


def plot_feature_importances(
    fitted_by_combo: dict,
    path: str,
    feature_labels: tuple[str, ...] = FEATURE_LABELS,
) -> None:
  fig, axes = _combo_figure((4.8, 3.8), sharex=True)
  y_pos = np.arange(len(feature_labels))
  xmax = max(
      0.45,
      max(
          float(np.max(fitted.feature_importances))
          for fitted in fitted_by_combo.values()
      )
      * 1.15,
  )
  for ax, (environment, algorithm) in zip(axes.ravel(), _COMBOS, strict=True):
    fitted = fitted_by_combo[(environment, algorithm)]
    ax.barh(y_pos, fitted.feature_importances, color=_NAVY, height=0.7)
    ax.set_yticks(y_pos, feature_labels)
    ax.set_title(_combo_title(environment, algorithm))
    ax.set_xlim(0.0, xmax)
    _style_ax(ax)
    ax.grid(True, axis='x', alpha=0.28, linewidth=0.6)
  for ax in axes[-1]:
    ax.set_xlabel('impurity importance')
  fig.tight_layout()
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path)
  plt.close(fig)


def plot_confusion_matrices(fitted_by_combo: dict, path: str) -> None:
  fig, axes = _combo_figure((4.7, 4.1), layout='constrained')
  for ax, (environment, algorithm) in zip(axes.ravel(), _COMBOS, strict=True):
    fitted = fitted_by_combo[(environment, algorithm)]
    matrix = np.asarray(fitted.confusion_test, dtype=np.float64)
    ticks = [_CLASS_TICKS.get(name, name) for name in fitted.class_names]
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    rates = matrix / row_sums
    image = ax.imshow(rates, cmap='Blues', vmin=0.0, vmax=1.0)
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        count = int(matrix[i, j])
        color = 'white' if rates[i, j] > 0.55 else _NAVY
        ax.text(j, i, str(count), ha='center', va='center', color=color, fontsize=8)
    ax.set_xticks(np.arange(len(ticks)), ticks, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(ticks)), ticks)
    test = fitted.metrics['test']
    ax.set_title(
        f"{_combo_title(environment, algorithm)}\n"
        f"acc={test['accuracy']:.3f}  bal-acc={test['balanced_accuracy']:.3f}  "
        f"macro-F1={test['macro_f1']:.3f}"
    )
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
  fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label='row-normalized')
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path)
  plt.close(fig)


def write_classifier_metrics_csv(fitted_by_combo: dict, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  class_names = next(iter(fitted_by_combo.values())).class_names
  fieldnames = [
      'environment',
      'algorithm',
      'model',
      'scheme',
      'balanced',
      'n_per_class',
      'n_train',
      'n_val',
      'n_test',
      'n_estimators',
      'max_depth',
      'min_samples_leaf',
      'max_features',
      'val_macro_f1',
      'test_accuracy',
      'test_balanced_accuracy',
      'test_macro_f1',
      'baseline_test_accuracy',
  ]
  for name in class_names:
    fieldnames.append(f'test_f1_{name}')
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for environment in ENVIRONMENTS:
      for algorithm in ALGORITHMS:
        fitted = fitted_by_combo[(environment, algorithm)]
        row = {
            'environment': environment,
            'algorithm': algorithm,
            'model': fitted.model_kind,
            'scheme': getattr(fitted, 'scheme', 'four'),
            'balanced': int(bool(getattr(fitted, 'balanced', False))),
            'n_per_class': getattr(fitted, 'n_per_class', '') or '',
            'n_train': fitted.split.sizes['train'],
            'n_val': fitted.split.sizes['val'],
            'n_test': fitted.split.sizes['test'],
            'n_estimators': fitted.params['n_estimators'],
            'max_depth': (
                'None' if fitted.params['max_depth'] is None
                else fitted.params['max_depth']
            ),
            'min_samples_leaf': fitted.params['min_samples_leaf'],
            'max_features': fitted.params['max_features'],
            'val_macro_f1': f"{fitted.metrics['val']['macro_f1']:.6f}",
            'test_accuracy': f"{fitted.metrics['test']['accuracy']:.6f}",
            'test_balanced_accuracy': (
                f"{fitted.metrics['test']['balanced_accuracy']:.6f}"
            ),
            'test_macro_f1': f"{fitted.metrics['test']['macro_f1']:.6f}",
            'baseline_test_accuracy': (
                f"{fitted.baseline['test']['accuracy']:.6f}"
            ),
        }
        for name in class_names:
          row[f'test_f1_{name}'] = f"{fitted.metrics['test'][f'f1_{name}']:.6f}"
        writer.writerow(row)


def write_metrics_csv(fitted_by_combo: dict, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fieldnames = [
      'environment',
      'algorithm',
      'model',
      'balanced',
      'n_per_class',
      'n_train',
      'n_val',
      'n_test',
      'n_estimators',
      'max_depth',
      'min_samples_leaf',
      'max_features',
      'val_mae',
      'test_mae',
      'test_rmse',
      'test_r2',
      'test_spearman',
      'baseline_test_mae',
  ]
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for environment in ENVIRONMENTS:
      for algorithm in ALGORITHMS:
        fitted = fitted_by_combo[(environment, algorithm)]
        writer.writerow({
            'environment': environment,
            'algorithm': algorithm,
            'model': fitted.model_kind,
            'balanced': int(bool(getattr(fitted, 'balanced', False))),
            'n_per_class': getattr(fitted, 'n_per_class', '') or '',
            'n_train': fitted.split.sizes['train'],
            'n_val': fitted.split.sizes['val'],
            'n_test': fitted.split.sizes['test'],
            'n_estimators': fitted.params['n_estimators'],
            'max_depth': (
                'None' if fitted.params['max_depth'] is None
                else fitted.params['max_depth']
            ),
            'min_samples_leaf': fitted.params['min_samples_leaf'],
            'max_features': fitted.params['max_features'],
            'val_mae': f"{fitted.metrics['val']['mae']:.6f}",
            'test_mae': f"{fitted.metrics['test']['mae']:.6f}",
            'test_rmse': f"{fitted.metrics['test']['rmse']:.6f}",
            'test_r2': f"{fitted.metrics['test']['r2']:.6f}",
            'test_spearman': f"{fitted.metrics['test']['spearman']:.6f}",
            'baseline_test_mae': f"{fitted.baseline['test']['mae']:.6f}",
        })


def _print_summary(fitted, feature_labels: tuple[str, ...] = FEATURE_LABELS) -> None:
  sizes = fitted.split.sizes
  balanced = bool(getattr(fitted, 'balanced', False))
  n_per_class = getattr(fitted, 'n_per_class', None)
  extra = ''
  if balanced:
    extra = f'  balanced n_per_class={n_per_class}'
  print(
      f'  split train/val/test = {sizes["train"]}/{sizes["val"]}/{sizes["test"]}  '
      f'params={fitted.params}{extra}'
  )
  for split_name in ('train', 'val', 'test'):
    m = fitted.metrics[split_name]
    b = fitted.baseline[split_name]
    if 'mae' in m:
      print(
          f'  {split_name:5s}  MAE={m["mae"]:.4f} (mean baseline {b["mae"]:.4f})  '
          f'RMSE={m["rmse"]:.4f}  R2={m["r2"]:.4f}  Spearman={m["spearman"]:.4f}'
      )
    else:
      print(
          f'  {split_name:5s}  acc={m["accuracy"]:.4f} '
          f'(majority {b["accuracy"]:.4f})  '
          f'bal-acc={m["balanced_accuracy"]:.4f}  '
          f'macro-F1={m["macro_f1"]:.4f}'
      )
      if split_name == 'test':
        class_names = getattr(fitted, 'class_names', CLASS_NAMES)
        f1s = '  '.join(
            f'{name}={m[f"f1_{name}"]:.3f}' for name in class_names
        )
        print(f'         F1  {f1s}')
  ranked = sorted(
      zip(feature_labels, fitted.feature_importances),
      key=lambda item: item[1],
      reverse=True,
  )
  print('  importances:')
  for label, value in ranked:
    print(f'    {label:36s} {value:.4f}')


def _model_kinds(cli_model: str) -> tuple[str, ...]:
  if cli_model == 'both':
    return ('extra_trees', 'random_forest')
  return (_CLI_MODELS[cli_model],)


def _figure_prefix(model_kind: str, task: str, balanced: bool) -> str:
  prefix = _FIGURE_PREFIX[model_kind]
  if task in _TASK_SCHEME:
    prefix = f'{prefix}_classifier'
  prefix = f'{prefix}{_TASK_FIGURE_SUFFIX.get(task, "")}'
  if balanced:
    prefix = f'{prefix}_balanced'
  return prefix


def _train_regression(
    environments, algorithms, model_kind: str, seed: int, balance: bool
) -> dict:
  fitted_by_combo = {}
  for environment in environments:
    for algorithm in algorithms:
      data_path = predictor_dataset_path(environment, algorithm, balanced=True)
      print(
          f'Training {model_kind} regressor {environment}/{algorithm} from {data_path}'
          f'{"  [balanced]" if balance else ""}',
          flush=True,
      )
      dataset = prepare_eval_target(
          load_predictor_dataset(data_path, load_topk=False)
      )
      fitted = fit_forest(
          dataset,
          model_kind=model_kind,
          random_state=seed,
          balance=balance,
          columns=FEATURE_COLUMNS_USED,
      )
      _print_test_fail_rate_ratios(dataset.difficulty[fitted.split.test])
      del dataset
      dataset = None
      gc.collect()
      model_path = predictor_model_path(
          environment, algorithm, model_kind, balanced=balance
      )
      save_fitted_predictor(fitted, model_path)
      fitted.model = None
      gc.collect()
      print(f'  saved {model_path}', flush=True)
      _print_summary(fitted, FEATURE_LABELS_USED)
      fitted_by_combo[(environment, algorithm)] = fitted
  if set(fitted_by_combo) == {
      (env, alg) for env in ENVIRONMENTS for alg in ALGORITHMS
  }:
    prefix = _figure_prefix(model_kind, 'regression', balance)
    pred_path = os.path.join(FIGURE_DIR, f'{prefix}_pred_vs_true.pdf')
    imp_path = os.path.join(FIGURE_DIR, f'{prefix}_feature_importances.pdf')
    metrics_path = os.path.join(FIGURE_DIR, f'{prefix}_metrics.csv')
    plot_pred_vs_true(fitted_by_combo, pred_path)
    plot_feature_importances(fitted_by_combo, imp_path, FEATURE_LABELS_USED)
    write_metrics_csv(fitted_by_combo, metrics_path)
    print(f'Wrote {pred_path}')
    print(f'Wrote {imp_path}')
    print(f'Wrote {metrics_path}')
  return fitted_by_combo


def _train_classification(
    environments,
    algorithms,
    model_kind: str,
    seed: int,
    balance: bool,
    scheme: str = 'four',
) -> dict:
  task = _SCHEME_TASK[scheme]
  fitted_by_combo = {}
  for environment in environments:
    for algorithm in algorithms:
      data_path = predictor_dataset_path(environment, algorithm, balanced=True)
      print(
          f'Training {model_kind} {_SCHEME_LABEL[scheme]} classifier '
          f'{environment}/{algorithm} from {data_path}'
          f'{"  [balanced]" if balance else ""}',
          flush=True,
      )
      dataset = prepare_eval_target(
          load_predictor_dataset(data_path, load_topk=False)
      )
      fitted = fit_classifier(
          dataset,
          model_kind=model_kind,
          random_state=seed,
          balance=balance,
          scheme=scheme,
          columns=FEATURE_COLUMNS_USED,
      )
      model_path = classifier_model_path(
          environment,
          algorithm,
          model_kind,
          balanced=balance,
          scheme=scheme,
      )
      save_fitted_classifier(fitted, model_path)
      fitted.model = None
      gc.collect()
      print(f'  saved {model_path}', flush=True)
      _print_test_fail_rate_ratios(dataset.difficulty[fitted.split.test])
      _print_summary(fitted, FEATURE_LABELS_USED)
      fitted_by_combo[(environment, algorithm)] = fitted
  if set(fitted_by_combo) == {
      (env, alg) for env in ENVIRONMENTS for alg in ALGORITHMS
  }:
    prefix = _figure_prefix(model_kind, task, balance)
    conf_path = os.path.join(FIGURE_DIR, f'{prefix}_confusion.pdf')
    imp_path = os.path.join(FIGURE_DIR, f'{prefix}_feature_importances.pdf')
    metrics_path = os.path.join(FIGURE_DIR, f'{prefix}_metrics.csv')
    plot_confusion_matrices(fitted_by_combo, conf_path)
    plot_feature_importances(fitted_by_combo, imp_path, FEATURE_LABELS_USED)
    write_classifier_metrics_csv(fitted_by_combo, metrics_path)
    print(f'Wrote {conf_path}')
    print(f'Wrote {imp_path}')
    print(f'Wrote {metrics_path}')
  return fitted_by_combo


def main() -> int:
  args = parse_args()
  environments = (args.environment,) if args.environment else ENVIRONMENTS
  algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
  for model_kind in _model_kinds(args.model):
    if args.task == 'regression':
      _train_regression(
          environments, algorithms, model_kind, args.seed, args.balance
      )
    else:
      _train_classification(
          environments,
          algorithms,
          model_kind,
          args.seed,
          args.balance,
          scheme=_TASK_SCHEME[args.task],
      )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
