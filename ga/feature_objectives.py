"""Fitness features that do not require rolling out a test.

Disagreement features are read from the search policies at the initial
state: actions, scales, values, and, when requested, integrated gradients.
Each feature can be maximized or minimized. Disagreement features are
scored at their absolute values, and several of them sum in those units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analysis.disagreement import (
    criticality_disagreement_np,
    pairwise_disagreement_np,
    standardized_value_disagreement_np,
    std_direction_disagreement_np,
    std_magnitude_disagreement_np,
    value_estimates_np,
)
from analysis.energy import (
    action_scale_disagreement,
    ensemble_action_scale,
    mechanical_power_proxy,
)

DISAGREEMENT_FEATURES = (
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
    'state_importance_disagreement',
    'obs_ig_disagreement',
)

# What has to be measured before a feature exists. ``policy`` is an
# initial-state forward pass. ``ig`` is integrated gradients. ``qvel``
# never touches a policy. ``gait`` projects actions with a subspace fit
# earlier, not by rolling out the candidate.
FEATURE_NEEDS = {
    'action_disagreement': frozenset({'policy'}),
    'gait_action_disagreement': frozenset({'policy', 'gait'}),
    'nongait_action_disagreement': frozenset({'policy', 'gait'}),
    'feature_attribution_disagreement': frozenset({'ig'}),
    'value_disagreement': frozenset({'policy'}),
    'criticality_disagreement': frozenset({'policy'}),
    'std_magnitude_disagreement': frozenset({'policy'}),
    'std_direction_disagreement': frozenset({'policy'}),
    'ensemble_action_scale': frozenset({'policy'}),
    'action_scale_disagreement': frozenset({'policy'}),
    'mechanical_power_proxy': frozenset({'policy', 'qvel'}),
    'state_importance_disagreement': frozenset({'ig'}),
    'obs_ig_disagreement': frozenset({'ig'}),
}
_GAUSSIAN_ONLY = frozenset({
    'std_magnitude_disagreement',
    'std_direction_disagreement',
})
_ALIASES = {
    'state_ig_disagreement': 'feature_attribution_disagreement',
    'attribution_disagreement': 'feature_attribution_disagreement',
    'observation_ig_disagreement': 'obs_ig_disagreement',
    'non_gait_action_disagreement': 'nongait_action_disagreement',
}
_LABELS = {
    'action disagreement': 'action_disagreement',
    'gait action disagreement': 'gait_action_disagreement',
    'non-gait action disagreement': 'nongait_action_disagreement',
    'nongait action disagreement': 'nongait_action_disagreement',
    'feature attribution disagreement': 'feature_attribution_disagreement',
    'value disagreement': 'value_disagreement',
    'criticality disagreement': 'criticality_disagreement',
    'std magnitude disagreement': 'std_magnitude_disagreement',
    'std direction disagreement': 'std_direction_disagreement',
    'ensemble action scale': 'ensemble_action_scale',
    'action scale disagreement': 'action_scale_disagreement',
    'mechanical power proxy': 'mechanical_power_proxy',
    'state importance disagreement': 'state_importance_disagreement',
    'observation ig disagreement': 'obs_ig_disagreement',
}
_DIRECTION_WORDS = {
    'max': 1,
    'maximize': 1,
    'min': -1,
    'minimize': -1,
}
ABSOLUTE_DISAGREEMENT_FITNESS = (
    'Absolute disagreement. Fitness is the feature value, or the sum of '
    'those values when several disagreements are named. A minimized '
    'feature is negated. The score is not recentered within the '
    'generation, so it tracks the disagreement.'
)


@dataclass(frozen=True)
class FeatureObjective:
  """One feature and whether the search should push it up or down."""

  name: str
  direction: int  # +1 maximize, -1 minimize

  @property
  def direction_name(self) -> str:
    return 'max' if self.direction > 0 else 'min'


@dataclass
class PopulationSignals:
  """Measurements for one population, in population order."""

  qvel: np.ndarray | None = None
  actions: np.ndarray | None = None
  action_loc: np.ndarray | None = None
  action_scale: np.ndarray | None = None
  values: np.ndarray | None = None
  q_values: np.ndarray | None = None
  state_ig_qpos: np.ndarray | None = None
  state_ig_qvel: np.ndarray | None = None
  obs_ig: np.ndarray | None = None
  gait_actions: np.ndarray | None = None
  nongait_actions: np.ndarray | None = None
  discrete: bool = False


def feature_catalog() -> str:
  """Help text listing every legal feature name."""
  return 'Disagreement and ensemble features: ' + ', '.join(DISAGREEMENT_FEATURES) + '.'


_ACTION_OBJECTIVES = frozenset({
    'action_disagreement',
    'gait_action_disagreement',
    'nongait_action_disagreement',
    'ensemble_action_scale',
    'action_scale_disagreement',
    'mechanical_power_proxy',
})
_SCALE_OBJECTIVES = frozenset({
    'criticality_disagreement',
    'std_magnitude_disagreement',
    'std_direction_disagreement',
})


def measurement_plan(
    objectives: Sequence[FeatureObjective],
    *,
    discrete_actions: bool = False,
) -> tuple[frozenset[str], frozenset[str]]:
  """Policy outputs and attribution tensors these objectives actually read.

  The first set is a subset of ``actions``, ``action_scale``, ``values``,
  and ``q_values``. The second is a subset of ``state_ig`` and ``obs_ig``.
  Anything else is not measured.
  """
  policy: set[str] = set()
  attribution: set[str] = set()
  for objective in objectives:
    name = objective.name
    if name in _ACTION_OBJECTIVES:
      policy.add('actions')
    elif name in _SCALE_OBJECTIVES:
      policy.add('action_scale')
    elif name == 'value_disagreement':
      policy.update(('values', 'q_values'))
    elif name == 'feature_attribution_disagreement':
      attribution.add('state_ig')
    elif name == 'obs_ig_disagreement':
      attribution.add('obs_ig')
    elif name == 'state_importance_disagreement':
      if discrete_actions:
        policy.add('action_scale')
      else:
        attribution.add('state_ig')
    else:
      raise ValueError(f'Unknown feature {name!r}.')
  return frozenset(policy), frozenset(attribution)


def requirements(
    objectives: Sequence[FeatureObjective],
    *,
    discrete_actions: bool = False,
) -> frozenset[str]:
  """Union of measurement tags for ``objectives``.

  On a discrete action space, state-importance disagreement is the gap in
  action-distribution entropy, which is already in the policy outputs.
  """
  needed: set[str] = set()
  for objective in objectives:
    tags = FEATURE_NEEDS[objective.name]
    if (
        discrete_actions
        and objective.name == 'state_importance_disagreement'
    ):
      tags = frozenset({'policy'})
    needed.update(tags)
  return frozenset(needed)


def parse_feature_specs(tokens: Sequence[str]) -> tuple[FeatureObjective, ...]:
  """Parse ``--features`` into objectives.

  Each entry needs a direction: ``+name``, ``-name``, ``name:max``,
  ``name:min``, ``max:name``, or ``min:name``. Commas split one token into
  several entries. A bare name is rejected so maximize and minimize cannot
  be confused.
  """
  pieces: list[str] = []
  for token in tokens:
    pieces.extend(
        part.strip() for part in str(token).split(',') if part.strip()
    )
  if not pieces:
    raise ValueError(
        'Provide at least one --features entry. ' + feature_catalog()
    )
  objectives = []
  seen = set()
  for piece in pieces:
    name, direction = _split_direction(piece)
    canonical = _canonicalize(name)
    if canonical in seen:
      raise ValueError(f'Feature {canonical} was given more than once.')
    seen.add(canonical)
    objectives.append(FeatureObjective(canonical, direction))
  return tuple(objectives)


def feature_slug(objectives: Sequence[FeatureObjective]) -> str:
  """Filesystem token for one objective list, in the order given."""
  return '__'.join(
      f'{objective.direction_name}_{objective.name}' for objective in objectives
  )


def fitness_definition(objectives: Sequence[FeatureObjective]) -> str:
  """How ``combine_feature_fitness`` turns these objectives into a score."""
  text = ABSOLUTE_DISAGREEMENT_FITNESS
  if any(item.name == 'value_disagreement' for item in objectives):
    text += (
        ' Value disagreement is the standardized value disagreement, '
        'not a second z-score of that column.'
    )
  return text


def _absolute_fitness(
    columns: dict[str, np.ndarray],
    objectives: Sequence[FeatureObjective],
) -> np.ndarray:
  """Signed sum of the raw feature columns. Higher is better."""
  total = None
  width = None
  for objective in objectives:
    if objective.name not in columns:
      raise KeyError(f'No column for {objective.name}.')
    values = np.asarray(columns[objective.name], dtype=np.float64).reshape(-1)
    if width is None:
      width = int(values.shape[0])
      total = np.zeros(width, dtype=np.float64)
    elif int(values.shape[0]) != width:
      raise ValueError(
          f'{objective.name} has length {values.shape[0]}, expected {width}.'
      )
    total = total + float(objective.direction) * values
  return np.asarray(total, dtype=np.float32)


def combine_feature_fitness(
    columns: dict[str, np.ndarray],
    objectives: Sequence[FeatureObjective],
) -> np.ndarray:
  """Scalar fitness, higher better, one value per case.

  The score is the feature value, or the sum of those values when several
  disagreements are named. It rises as the population disagrees more,
  instead of being recentered at zero every generation. A minimized
  feature is negated.
  """
  if not objectives:
    raise ValueError('Need at least one feature to score fitness.')
  return _absolute_fitness(columns, objectives)


def feature_columns(
    signals: PopulationSignals,
    objectives: Sequence[FeatureObjective],
    *,
    algorithm: str,
    environment: str,
) -> dict[str, np.ndarray]:
  """Raw feature values for each objective. Shape ``(N,)`` per name."""
  columns = {}
  for objective in objectives:
    values = _column(signals, objective.name, algorithm)
    _require_finite(values, objective.name, environment)
    columns[objective.name] = values
  return columns


def _split_direction(token: str) -> tuple[str, int]:
  text = token.strip()
  if not text:
    raise ValueError('Empty feature spec.')
  if text[0] in '+-':
    name = text[1:].strip()
    if not name:
      raise ValueError(f'{token!r} has a direction and no feature name.')
    return name, 1 if text[0] == '+' else -1
  if ':' in text:
    left, right = (part.strip() for part in text.split(':', 1))
    if right.lower() in _DIRECTION_WORDS and left:
      return left, _DIRECTION_WORDS[right.lower()]
    if left.lower() in _DIRECTION_WORDS and right:
      return right, _DIRECTION_WORDS[left.lower()]
  raise ValueError(
      f'{token!r} needs a direction: +name, -name, name:max, or name:min. '
      + feature_catalog()
  )


def _canonicalize(name: str) -> str:
  raw = ' '.join(name.strip().lower().split())
  spaced = ' '.join(raw.replace('-', ' ').replace('_', ' ').split())
  underscored = spaced.replace(' ', '_')
  for candidate in (raw, spaced, underscored, raw.replace(' ', '_')):
    mapped = _ALIASES.get(candidate, _LABELS.get(candidate, candidate))
    mapped = _ALIASES.get(mapped, mapped)
    if mapped in FEATURE_NEEDS:
      return mapped
  raise ValueError(f'Unknown feature {name!r}. {feature_catalog()}')


def _require_finite(values: np.ndarray, name: str, environment: str) -> None:
  if values.ndim != 1:
    raise ValueError(f'{name} must be one value per case, got {values.shape}.')
  if values.size and np.all(np.isfinite(values)):
    return
  n_bad = int(np.size(values) - np.isfinite(values).sum())
  raise ValueError(
      f'{name} is undefined for {environment}: {n_bad} of {values.size} '
      'cases are non-finite.'
  )


def _flatten_trailing(vectors: np.ndarray) -> np.ndarray:
  if vectors.ndim == 3:
    return vectors
  p, n = vectors.shape[:2]
  return np.asarray(vectors).reshape(p, n, -1)


def _pairwise_abs(values: np.ndarray) -> np.ndarray:
  """Mean pairwise absolute difference. ``(P, N)`` or ``(P, N, 1)`` to ``(N,)``."""
  array = np.asarray(values, dtype=np.float64)
  if array.ndim == 3:
    array = np.mean(array, axis=-1)
  if array.ndim != 2:
    raise ValueError(f'Expected (P, N) values, got {array.shape}.')
  n_policies = int(array.shape[0])
  if n_policies < 2:
    return np.zeros(array.shape[1], dtype=np.float64)
  idx_i, idx_j = np.triu_indices(n_policies, k=1)
  return np.mean(np.abs(array[idx_i] - array[idx_j]), axis=0)


def _require_gaussian(signals: PopulationSignals, name: str) -> None:
  if signals.discrete:
    raise ValueError(
        f'{name} is defined for Gaussian action scales, not discrete actions.'
    )


def _state_ig(signals: PopulationSignals) -> np.ndarray:
  if signals.state_ig_qpos is None or signals.state_ig_qvel is None:
    raise RuntimeError('State integrated gradients were not measured.')
  return np.concatenate(
      [
          _flatten_trailing(np.asarray(signals.state_ig_qpos)),
          _flatten_trailing(np.asarray(signals.state_ig_qvel)),
      ],
      axis=-1,
  )


def _column(
    signals: PopulationSignals,
    name: str,
    algorithm: str,
) -> np.ndarray:
  if name in _GAUSSIAN_ONLY:
    _require_gaussian(signals, name)
  if name == 'action_disagreement':
    return pairwise_disagreement_np(_flatten_trailing(signals.actions))
  if name == 'gait_action_disagreement':
    return pairwise_disagreement_np(_flatten_trailing(signals.gait_actions))
  if name == 'nongait_action_disagreement':
    return pairwise_disagreement_np(_flatten_trailing(signals.nongait_actions))
  if name == 'feature_attribution_disagreement':
    return pairwise_disagreement_np(_state_ig(signals))
  if name == 'value_disagreement':
    values = value_estimates_np(
        signals.values,
        signals.q_values,
        algorithm=algorithm,
    )
    return standardized_value_disagreement_np(values)
  if name == 'criticality_disagreement':
    if signals.discrete:
      return _pairwise_abs(signals.action_scale)
    return criticality_disagreement_np(signals.action_scale)
  if name == 'std_magnitude_disagreement':
    return std_magnitude_disagreement_np(signals.action_scale)
  if name == 'std_direction_disagreement':
    return std_direction_disagreement_np(signals.action_scale)
  if name == 'ensemble_action_scale':
    return ensemble_action_scale(signals.actions)
  if name == 'action_scale_disagreement':
    return action_scale_disagreement(signals.actions)
  if name == 'mechanical_power_proxy':
    if signals.qvel is None:
      raise RuntimeError('Mechanical power needs joint velocities.')
    return mechanical_power_proxy(signals.actions, signals.qvel)
  if name == 'state_importance_disagreement':
    if signals.discrete:
      return _pairwise_abs(signals.action_scale)
    return pairwise_disagreement_np(np.abs(_state_ig(signals)))
  if name == 'obs_ig_disagreement':
    if signals.obs_ig is None:
      raise RuntimeError('Observation integrated gradients were not measured.')
    return pairwise_disagreement_np(_flatten_trailing(signals.obs_ig))
  raise ValueError(f'Unknown feature {name!r}.')
