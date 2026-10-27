"""Forest regressors from ensemble disagreement to empirical difficulty."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import os
import pickle
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from analysis.predictor_dataset import PredictorDataset

ALL_FAIL_SENTINEL = -0.1

FEATURE_COLUMNS = (
    'action_disagreement',
    'gait_action_disagreement',
    'nongait_action_disagreement',
    'feature_attribution_disagreement',
    'value_disagreement',
    'criticality_disagreement',
    'std_magnitude_disagreement',
    'std_direction_disagreement',
    'ensemble_action_scale',
    'action_scale_disagreement',
    'mechanical_power_proxy',
)

FEATURE_LABELS = (
    'action disagreement',
    'gait action disagreement',
    'non-gait action disagreement',
    'feature attribution disagreement',
    'value disagreement',
    'criticality disagreement',
    'std magnitude disagreement',
    'std direction disagreement',
    'ensemble action scale',
    'action scale disagreement',
    'mechanical power proxy',
)

CLASS_FEATURE_COLUMNS = (
    'action_disagreement',
    'value_disagreement',
    'feature_attribution_disagreement',
    'ensemble_action_scale',
    'action_scale_disagreement',
    'mechanical_power_proxy',
)

CLASS_FEATURE_LABELS = (
    'action disagreement',
    'value disagreement',
    'feature attribution disagreement',
    'ensemble action scale',
    'action scale disagreement',
    'mechanical power proxy',
)

CLASS_NAMES = (
    'easy',
    'fault-revealing',
    'discriminating',
    'impossible',
)
THREE_CLASS_NAMES = (
    'easy',
    'fault-revealing',
    'impossible',
)
BINARY_CLASS_NAMES = (
    'not-fault-revealing',
    'fault-revealing',
)
CLASS_SCHEMES = {
    'four': CLASS_NAMES,
    'three': THREE_CLASS_NAMES,
    'binary': BINARY_CLASS_NAMES,
}
_SCHEME_CLASSIFIER_INFIX = {
    'four': '_difficulty_classifier',
    'three': '_3class_difficulty_classifier',
    'binary': '_binary_difficulty_classifier',
}

TRAIN_FRACTION = 0.50
VAL_FRACTION = 0.10
TEST_FRACTION = 0.40

PARAM_GRID: tuple[dict[str, Any], ...] = (
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 1.0},
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 8, 'max_features': 1.0},
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 8, 'max_features': 'sqrt'},
    {'n_estimators': 500, 'max_depth': 16, 'min_samples_leaf': 1, 'max_features': 1.0},
    {'n_estimators': 500, 'max_depth': 16, 'min_samples_leaf': 8, 'max_features': 'sqrt'},
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 8, 'max_features': 1.0},
)

MODEL_KINDS = ('extra_trees', 'random_forest')
_ESTIMATORS = {
    'extra_trees': ExtraTreesRegressor,
    'random_forest': RandomForestRegressor,
}
_MODEL_FILENAMES = {
    'extra_trees': 'extra_trees_difficulty_predictor.pkl',
    'random_forest': 'random_forest_difficulty_predictor.pkl',
}
_CLASSIFIERS = {
    'extra_trees': ExtraTreesClassifier,
    'random_forest': RandomForestClassifier,
}
_CLASSIFIER_FILENAMES = {
    'extra_trees': 'extra_trees_difficulty_classifier.pkl',
    'random_forest': 'random_forest_difficulty_classifier.pkl',
}


def fail_rate_target(difficulty: np.ndarray) -> np.ndarray:
  """Map stored difficulty to a fail-rate in [0, 1].

  All-fail encounters are stored as ``-0.1``; they correspond to every
  ensemble member failing, so they become ``1.0`` here.
  """
  y = np.asarray(difficulty, dtype=np.float64)
  all_fail = (y < 0.0) | np.isclose(y, ALL_FAIL_SENTINEL)
  return np.where(all_fail, 1.0, y)


def feature_matrix(
    dataset: PredictorDataset,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
  return np.column_stack(
      [np.asarray(getattr(dataset, name), dtype=np.float64) for name in columns]
  )


def difficulty_class_ids(
    difficulty: np.ndarray,
    scheme: str = 'four',
) -> np.ndarray:
  """Map empirical fail-rate to difficulty class ids.

  After mapping the all-fail sentinel to ``1.0``:

  ``four``
    * easy: ``0.0``
    * fault-revealing: ``(0, 0.5]``
    * discriminating: ``(0.5, 1.0)``
    * impossible: ``1.0``

  ``three``
    * easy: ``0.0``
    * fault-revealing: ``(0, 1.0)``
    * impossible: ``1.0``

  ``binary``
    * not-fault-revealing: ``0.0`` or ``1.0``
    * fault-revealing: ``(0, 1.0)``
  """
  if scheme not in CLASS_SCHEMES:
    raise ValueError(f'Unknown class scheme {scheme!r}')
  rate = fail_rate_target(difficulty)
  labels = np.empty(rate.shape, dtype=np.int32)
  easy = np.isclose(rate, 0.0)
  impossible = np.isclose(rate, 1.0)
  middle = (~easy) & (~impossible)
  if scheme == 'binary':
    labels[easy | impossible] = 0
    labels[middle] = 1
    return labels
  labels[easy] = 0
  labels[impossible] = len(CLASS_SCHEMES[scheme]) - 1
  if scheme == 'three':
    labels[middle] = 1
    return labels
  labels[middle & (rate <= 0.5)] = 1
  labels[middle & (rate > 0.5)] = 2
  return labels


def class_counts(labels: np.ndarray, n_classes: int) -> np.ndarray:
  return np.bincount(
      np.asarray(labels, dtype=np.int32), minlength=n_classes
  )


def balanced_class_indices(
    labels: np.ndarray,
    class_names: tuple[str, ...],
    random_state: int = 0,
    n_per_class: int | None = None,
) -> np.ndarray:
  """Undersample so every class has the same number of rows.

  Draws ``n_per_class`` indices without replacement from each class
  (default: size of the smallest class).
  """
  labels = np.asarray(labels, dtype=np.int32)
  n_classes = len(class_names)
  counts = class_counts(labels, n_classes)
  if np.any(counts == 0):
    missing = [
        class_names[i] for i, count in enumerate(counts) if count == 0
    ]
    raise ValueError(f'Cannot balance: missing classes {missing}')
  if n_per_class is None:
    n_per_class = int(counts.min())
  if n_per_class < 1:
    raise ValueError('n_per_class must be >= 1')
  if n_per_class > int(counts.min()):
    raise ValueError(
        f'n_per_class={n_per_class} exceeds smallest class size {int(counts.min())}'
    )
  rng = np.random.RandomState(random_state)
  chosen = []
  for class_id in range(n_classes):
    idx = np.flatnonzero(labels == class_id)
    chosen.append(rng.choice(idx, size=n_per_class, replace=False))
  return np.sort(np.concatenate(chosen))


def balance_predictor_dataset(
    dataset: PredictorDataset,
    random_state: int = 0,
    n_per_class: int | None = None,
    scheme: str = 'four',
) -> tuple[PredictorDataset, int]:
  """Randomly subsample so difficulty classes are equal-sized."""
  class_names = CLASS_SCHEMES[scheme]
  labels = difficulty_class_ids(dataset.difficulty, scheme=scheme)
  indices = balanced_class_indices(
      labels,
      class_names,
      random_state=random_state,
      n_per_class=n_per_class,
  )
  n_kept = int(indices.size // len(class_names))
  return dataset.subset(indices), n_kept


def difficulty_bins(difficulty: np.ndarray) -> np.ndarray:
  """Integer codes for the discrete empirical difficulty levels."""
  rounded = np.round(np.asarray(difficulty, dtype=np.float64), decimals=6)
  _values, inverse = np.unique(rounded, return_inverse=True)
  return inverse


@dataclass
class SplitIndices:
  train: np.ndarray
  val: np.ndarray
  test: np.ndarray

  @property
  def sizes(self) -> dict[str, int]:
    return {
        'train': int(self.train.size),
        'val': int(self.val.size),
        'test': int(self.test.size),
    }


def split_indices(
    n_samples: int,
    bins: np.ndarray,
    random_state: int = 0,
) -> SplitIndices:
  """50/10/40 train/validation/test split, stratified on difficulty bins."""
  if n_samples < 3:
    raise ValueError(f'Need at least 3 samples, got {n_samples}')
  if abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
    raise ValueError('Split fractions must sum to 1.')
  indices = np.arange(n_samples)
  val_of_tv = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
  try:
    tv, test = train_test_split(
        indices,
        test_size=TEST_FRACTION,
        random_state=random_state,
        stratify=bins,
    )
    train, val = train_test_split(
        tv,
        test_size=val_of_tv,
        random_state=random_state,
        stratify=bins[tv],
    )
  except ValueError:
    tv, test = train_test_split(
        indices, test_size=TEST_FRACTION, random_state=random_state
    )
    train, val = train_test_split(
        tv, test_size=val_of_tv, random_state=random_state
    )
  return SplitIndices(
      train=np.sort(train),
      val=np.sort(val),
      test=np.sort(test),
  )


def lin_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Lin's concordance correlation, using divisor-n moments.

  The coefficient is 1 when the two series take the same value. Rank
  correlation can stay high while this falls, because a shift in location
  or scale lowers concordance.
  """
  y_true = np.asarray(y_true, dtype=np.float64)
  y_pred = np.asarray(y_pred, dtype=np.float64)
  if y_true.size < 2:
    return float('nan')
  mean_true = float(y_true.mean())
  mean_pred = float(y_pred.mean())
  var_true = float(np.mean((y_true - mean_true) ** 2))
  var_pred = float(np.mean((y_pred - mean_pred) ** 2))
  cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
  denom = var_true + var_pred + (mean_true - mean_pred) ** 2
  if denom <= 0.0:
    return 1.0 if var_true <= 0.0 and var_pred <= 0.0 else 0.0
  return 2.0 * cov / denom


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
  y_true = np.asarray(y_true, dtype=np.float64)
  y_pred = np.asarray(y_pred, dtype=np.float64)
  mae = float(mean_absolute_error(y_true, y_pred))
  rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
  r2 = float(r2_score(y_true, y_pred))
  if (
      y_true.size < 2
      or np.unique(y_true).size < 2
      or np.unique(y_pred).size < 2
  ):
    spearman = float('nan')
  else:
    spearman = float(spearmanr(y_true, y_pred).statistic)
  return {
      'mae': mae,
      'rmse': rmse,
      'r2': r2,
      'spearman': spearman,
      'ccc': lin_ccc(y_true, y_pred),
  }


def mean_baseline_metrics(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
  pred = np.full_like(y_eval, np.mean(y_train), dtype=np.float64)
  return regression_metrics(y_eval, pred)


def _make_estimator(
    model_kind: str,
    params: dict[str, Any],
    random_state: int,
):
  try:
    cls = _ESTIMATORS[model_kind]
  except KeyError as exc:
    raise ValueError(
        f'Unknown model_kind {model_kind!r}; expected one of {MODEL_KINDS}'
    ) from exc
  return cls(
      n_estimators=int(params['n_estimators']),
      max_depth=params['max_depth'],
      min_samples_leaf=int(params['min_samples_leaf']),
      max_features=params['max_features'],
      n_jobs=-1,
      random_state=random_state,
  )


@dataclass
class FittedPredictor:
  environment: str
  algorithm: str
  model: ExtraTreesRegressor | RandomForestRegressor
  feature_names: tuple[str, ...]
  params: dict[str, Any]
  split: SplitIndices
  random_state: int
  metrics: dict[str, dict[str, float]]
  baseline: dict[str, dict[str, float]]
  feature_importances: np.ndarray
  y_test: np.ndarray
  y_test_pred: np.ndarray
  search_scores: list[dict[str, Any]] = field(default_factory=list)
  model_kind: str = 'extra_trees'
  balanced: bool = False
  n_per_class: int | None = None

  @property
  def n_features(self) -> int:
    return len(self.feature_names)


def fit_forest(
    dataset: PredictorDataset,
    model_kind: str = 'extra_trees',
    random_state: int = 0,
    param_grid: tuple[dict[str, Any], ...] = PARAM_GRID,
    balance: bool = False,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> FittedPredictor:
  """Train a forest on disagreement features; select hyperparameters on val."""
  n_per_class = None
  if balance:
    dataset, n_per_class = balance_predictor_dataset(
        dataset, random_state=random_state
    )
  X = feature_matrix(dataset, columns=columns)
  y = fail_rate_target(dataset.difficulty)
  split = split_indices(
      n_samples=y.shape[0],
      bins=difficulty_bins(dataset.difficulty),
      random_state=random_state,
  )
  X_train, y_train = X[split.train], y[split.train]
  X_val, y_val = X[split.val], y[split.val]
  X_test, y_test = X[split.test], y[split.test]

  search_scores = []
  best_params = None
  best_val_mae = np.inf
  for params in param_grid:
    model = _make_estimator(model_kind, params, random_state)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    scores = regression_metrics(y_val, val_pred)
    search_scores.append({'params': dict(params), **scores})
    if scores['mae'] < best_val_mae:
      best_val_mae = scores['mae']
      best_params = dict(params)
    del model
    gc.collect()

  if best_params is None:
    raise RuntimeError('Hyperparameter search produced no candidate.')

  selected = _make_estimator(model_kind, best_params, random_state)
  selected.fit(X_train, y_train)
  train_pred = selected.predict(X_train)
  val_pred = selected.predict(X_val)
  del selected
  gc.collect()

  X_trainval = np.concatenate([X_train, X_val], axis=0)
  y_trainval = np.concatenate([y_train, y_val], axis=0)
  model = _make_estimator(model_kind, best_params, random_state)
  model.fit(X_trainval, y_trainval)
  test_pred = model.predict(X_test)

  metrics = {
      'train': regression_metrics(y_train, train_pred),
      'val': regression_metrics(y_val, val_pred),
      'test': regression_metrics(y_test, test_pred),
  }
  baseline = {
      'train': mean_baseline_metrics(y_train, y_train),
      'val': mean_baseline_metrics(y_train, y_val),
      'test': mean_baseline_metrics(y_train, y_test),
  }
  return FittedPredictor(
      environment=dataset.environment,
      algorithm=dataset.algorithm,
      model=model,
      feature_names=columns,
      params=best_params,
      split=split,
      random_state=random_state,
      metrics=metrics,
      baseline=baseline,
      feature_importances=np.asarray(model.feature_importances_, dtype=np.float64),
      y_test=y_test,
      y_test_pred=test_pred,
      search_scores=search_scores,
      model_kind=model_kind,
      balanced=balance,
      n_per_class=n_per_class,
  )


def fit_extra_trees(
    dataset: PredictorDataset,
    random_state: int = 0,
    param_grid: tuple[dict[str, Any], ...] = PARAM_GRID,
) -> FittedPredictor:
  return fit_forest(
      dataset,
      model_kind='extra_trees',
      random_state=random_state,
      param_grid=param_grid,
  )


def fit_random_forest(
    dataset: PredictorDataset,
    random_state: int = 0,
    param_grid: tuple[dict[str, Any], ...] = PARAM_GRID,
) -> FittedPredictor:
  return fit_forest(
      dataset,
      model_kind='random_forest',
      random_state=random_state,
      param_grid=param_grid,
  )


def _with_balanced_suffix(filename: str, balanced: bool) -> str:
  if not balanced:
    return filename
  stem, ext = os.path.splitext(filename)
  return f'{stem}_balanced{ext}'


def predictor_model_path(
    environment: str,
    algorithm: str,
    model_kind: str = 'extra_trees',
    root: str = 'datasets',
    balanced: bool = False,
) -> str:
  try:
    filename = _MODEL_FILENAMES[model_kind]
  except KeyError as exc:
    raise ValueError(
        f'Unknown model_kind {model_kind!r}; expected one of {MODEL_KINDS}'
    ) from exc
  return os.path.join(
      root, environment, algorithm, _with_balanced_suffix(filename, balanced)
  )


def save_fitted_predictor(fitted: FittedPredictor, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'wb') as handle:
    pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_fitted_predictor(path: str) -> FittedPredictor:
  with open(path, 'rb') as handle:
    fitted = pickle.load(handle)
  if not isinstance(fitted, FittedPredictor):
    raise TypeError(f'{path} is not a FittedPredictor ({type(fitted)!r})')
  return fitted


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, float]:
  y_true = np.asarray(y_true, dtype=np.int32)
  y_pred = np.asarray(y_pred, dtype=np.int32)
  n_classes = len(class_names)
  per_class = f1_score(
      y_true,
      y_pred,
      labels=np.arange(n_classes),
      average=None,
      zero_division=0,
  )
  metrics = {
      'accuracy': float(accuracy_score(y_true, y_pred)),
      'balanced_accuracy': float(
          balanced_accuracy_score(y_true, y_pred)
      ),
      'macro_f1': float(
          f1_score(y_true, y_pred, average='macro', zero_division=0)
      ),
  }
  for name, value in zip(class_names, per_class):
    metrics[f'f1_{name}'] = float(value)
  return metrics


def majority_baseline_metrics(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, float]:
  counts = np.bincount(
      np.asarray(y_train, dtype=np.int32), minlength=len(class_names)
  )
  majority = int(np.argmax(counts))
  pred = np.full(np.asarray(y_eval).shape, majority, dtype=np.int32)
  return classification_metrics(y_eval, pred, class_names=class_names)


def _make_classifier(
    model_kind: str,
    params: dict[str, Any],
    random_state: int,
    class_weight: str | None = 'balanced',
):
  try:
    cls = _CLASSIFIERS[model_kind]
  except KeyError as exc:
    raise ValueError(
        f'Unknown model_kind {model_kind!r}; expected one of {MODEL_KINDS}'
    ) from exc
  return cls(
      n_estimators=int(params['n_estimators']),
      max_depth=params['max_depth'],
      min_samples_leaf=int(params['min_samples_leaf']),
      max_features=params['max_features'],
      class_weight=class_weight,
      n_jobs=-1,
      random_state=random_state,
  )


@dataclass
class FittedClassifier:
  environment: str
  algorithm: str
  model: ExtraTreesClassifier | RandomForestClassifier
  feature_names: tuple[str, ...]
  class_names: tuple[str, ...]
  params: dict[str, Any]
  split: SplitIndices
  random_state: int
  metrics: dict[str, dict[str, float]]
  baseline: dict[str, dict[str, float]]
  feature_importances: np.ndarray
  y_test: np.ndarray
  y_test_pred: np.ndarray
  confusion_test: np.ndarray
  search_scores: list[dict[str, Any]] = field(default_factory=list)
  model_kind: str = 'extra_trees'
  balanced: bool = False
  n_per_class: int | None = None
  scheme: str = 'four'


def fit_classifier(
    dataset: PredictorDataset,
    model_kind: str = 'extra_trees',
    random_state: int = 0,
    param_grid: tuple[dict[str, Any], ...] = PARAM_GRID,
    balance: bool = False,
    scheme: str = 'four',
    columns: tuple[str, ...] = CLASS_FEATURE_COLUMNS,
) -> FittedClassifier:
  """Train a forest classifier on action/value/attribution disagreement."""
  class_names = CLASS_SCHEMES[scheme]
  n_per_class = None
  if balance:
    dataset, n_per_class = balance_predictor_dataset(
        dataset, random_state=random_state, scheme=scheme
    )
  X = feature_matrix(dataset, columns=columns)
  y = difficulty_class_ids(dataset.difficulty, scheme=scheme)
  split = split_indices(
      n_samples=y.shape[0],
      bins=y,
      random_state=random_state,
  )
  X_train, y_train = X[split.train], y[split.train]
  X_val, y_val = X[split.val], y[split.val]
  X_test, y_test = X[split.test], y[split.test]
  class_weight = None if balance else 'balanced'

  search_scores = []
  best_params = None
  best_val_score = -np.inf
  for params in param_grid:
    model = _make_classifier(
        model_kind, params, random_state, class_weight=class_weight
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    scores = classification_metrics(y_val, val_pred, class_names=class_names)
    search_scores.append({'params': dict(params), **scores})
    if scores['macro_f1'] > best_val_score:
      best_val_score = scores['macro_f1']
      best_params = dict(params)
    del model
    gc.collect()

  if best_params is None:
    raise RuntimeError('Hyperparameter search produced no candidate.')

  selected = _make_classifier(
      model_kind, best_params, random_state, class_weight=class_weight
  )
  selected.fit(X_train, y_train)
  train_pred = selected.predict(X_train)
  val_pred = selected.predict(X_val)
  del selected
  gc.collect()

  X_trainval = np.concatenate([X_train, X_val], axis=0)
  y_trainval = np.concatenate([y_train, y_val], axis=0)
  model = _make_classifier(
      model_kind, best_params, random_state, class_weight=class_weight
  )
  model.fit(X_trainval, y_trainval)
  test_pred = model.predict(X_test)

  metrics = {
      'train': classification_metrics(y_train, train_pred, class_names=class_names),
      'val': classification_metrics(y_val, val_pred, class_names=class_names),
      'test': classification_metrics(y_test, test_pred, class_names=class_names),
  }
  baseline = {
      'train': majority_baseline_metrics(y_train, y_train, class_names=class_names),
      'val': majority_baseline_metrics(y_train, y_val, class_names=class_names),
      'test': majority_baseline_metrics(y_train, y_test, class_names=class_names),
  }
  return FittedClassifier(
      environment=dataset.environment,
      algorithm=dataset.algorithm,
      model=model,
      feature_names=columns,
      class_names=class_names,
      params=best_params,
      split=split,
      random_state=random_state,
      metrics=metrics,
      baseline=baseline,
      feature_importances=np.asarray(model.feature_importances_, dtype=np.float64),
      y_test=y_test,
      y_test_pred=np.asarray(test_pred, dtype=np.int32),
      confusion_test=confusion_matrix(
          y_test, test_pred, labels=np.arange(len(class_names))
      ),
      search_scores=search_scores,
      model_kind=model_kind,
      balanced=balance,
      n_per_class=n_per_class,
      scheme=scheme,
  )


def classifier_model_path(
    environment: str,
    algorithm: str,
    model_kind: str = 'extra_trees',
    root: str = 'datasets',
    balanced: bool = False,
    scheme: str = 'four',
) -> str:
  try:
    filename = _CLASSIFIER_FILENAMES[model_kind]
  except KeyError as exc:
    raise ValueError(
        f'Unknown model_kind {model_kind!r}; expected one of {MODEL_KINDS}'
    ) from exc
  try:
    infix = _SCHEME_CLASSIFIER_INFIX[scheme]
  except KeyError as exc:
    raise ValueError(f'Unknown class scheme {scheme!r}') from exc
  stem, ext = os.path.splitext(filename)
  filename = stem.replace('_difficulty_classifier', infix) + ext
  return os.path.join(
      root, environment, algorithm, _with_balanced_suffix(filename, balanced)
  )


def save_fitted_classifier(fitted: FittedClassifier, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'wb') as handle:
    pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_fitted_classifier(path: str) -> FittedClassifier:
  with open(path, 'rb') as handle:
    fitted = pickle.load(handle)
  if not isinstance(fitted, FittedClassifier):
    raise TypeError(f'{path} is not a FittedClassifier ({type(fitted)!r})')
  return fitted
