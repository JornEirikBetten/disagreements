"""Pooled per-test-case disagreement features for difficulty prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickle
import struct

import numpy as np

from analysis.disagreement import (
    criticality_disagreement_np,
    pairwise_disagreement_np,
    pairwise_topk_attribution_cosine_all_k,
    standardized_value_disagreement_np,
    std_direction_disagreement_np,
    std_magnitude_disagreement_np,
    value_estimates_np,
)
from analysis.energy import energy_observables_from_generation, energy_metrics_from_collection
from analysis.ga_collection import (
    _flatten_trailing,
    disagreement_metrics_from_collection,
    gait_action_disagreement_metrics_from_collection,
)
from ga.collect import GASearchCollection


@dataclass
class PredictorDataset:
  """One row per test-case encounter, pooled across GA experiments."""

  environment: str
  algorithm: str
  policy_ids: list[int]
  names: np.ndarray
  experiment: np.ndarray
  generation: np.ndarray
  case_index: np.ndarray
  difficulty: np.ndarray
  action_disagreement: np.ndarray
  gait_action_disagreement: np.ndarray
  nongait_action_disagreement: np.ndarray
  feature_attribution_disagreement: np.ndarray
  value_disagreement: np.ndarray
  criticality_disagreement: np.ndarray
  qpos: np.ndarray
  qvel: np.ndarray
  std_magnitude_disagreement: np.ndarray = field(
      default_factory=lambda: np.zeros(0, dtype=np.float64)
  )
  std_direction_disagreement: np.ndarray = field(
      default_factory=lambda: np.zeros(0, dtype=np.float64)
  )
  ensemble_action_scale: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
  action_scale_disagreement: np.ndarray = field(
      default_factory=lambda: np.zeros(0, dtype=np.float64)
  )
  mechanical_power_proxy: np.ndarray = field(
      default_factory=lambda: np.zeros(0, dtype=np.float64)
  )
  feature_attribution_topk_disagreement: np.ndarray = field(
      default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
  )
  observation_ig_topk_disagreement: np.ndarray = field(
      default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
  )

  def __setstate__(self, state):
    """Accept pickles written before energy or top-k attribution fields existed."""
    object.__setattr__(self, '__dict__', dict(state))
    n = int(np.asarray(self.difficulty).shape[0])
    for name in (
        'ensemble_action_scale',
        'action_scale_disagreement',
        'mechanical_power_proxy',
        'std_magnitude_disagreement',
        'std_direction_disagreement',
    ):
      value = getattr(self, name, None)
      if value is None or np.asarray(value).shape[0] != n:
        object.__setattr__(self, name, np.full(n, np.nan, dtype=np.float64))
    topk = getattr(self, 'feature_attribution_topk_disagreement', None)
    if topk is None or np.asarray(topk).shape[0] != n:
      object.__setattr__(
          self,
          'feature_attribution_topk_disagreement',
          np.full((n, 0), np.nan, dtype=np.float64),
      )
    obs_topk = getattr(self, 'observation_ig_topk_disagreement', None)
    if obs_topk is None or np.asarray(obs_topk).shape[0] != n:
      object.__setattr__(
          self,
          'observation_ig_topk_disagreement',
          np.full((n, 0), np.nan, dtype=np.float64),
      )

  @property
  def n_cases(self) -> int:
    return int(self.difficulty.shape[0])

  @property
  def n_policies(self) -> int:
    return len(self.policy_ids)

  def subset(self, indices: np.ndarray) -> 'PredictorDataset':
    """Return a dataset containing only the rows at ``indices``."""
    indices = np.asarray(indices)
    return PredictorDataset(
        environment=self.environment,
        algorithm=self.algorithm,
        policy_ids=list(self.policy_ids),
        names=self.names[indices],
        experiment=self.experiment[indices],
        generation=self.generation[indices],
        case_index=self.case_index[indices],
        difficulty=self.difficulty[indices],
        action_disagreement=self.action_disagreement[indices],
        gait_action_disagreement=self.gait_action_disagreement[indices],
        nongait_action_disagreement=self.nongait_action_disagreement[indices],
        feature_attribution_disagreement=self.feature_attribution_disagreement[
            indices
        ],
        feature_attribution_topk_disagreement=(
            self.feature_attribution_topk_disagreement[indices]
        ),
        observation_ig_topk_disagreement=(
            self.observation_ig_topk_disagreement[indices]
        ),
        value_disagreement=self.value_disagreement[indices],
        criticality_disagreement=self.criticality_disagreement[indices],
        std_magnitude_disagreement=self.std_magnitude_disagreement[indices],
        std_direction_disagreement=self.std_direction_disagreement[indices],
        qpos=self.qpos[indices],
        qvel=self.qvel[indices],
        ensemble_action_scale=self.ensemble_action_scale[indices],
        action_scale_disagreement=self.action_scale_disagreement[indices],
        mechanical_power_proxy=self.mechanical_power_proxy[indices],
    )


def initial_state_matrix(dataset: PredictorDataset) -> np.ndarray:
  """Concatenate ``qpos`` and ``qvel`` into one row per case.

  Float32 matches the identity the binned search uses when it refuses to
  store a state twice, so analysis and collection agree on what a duplicate
  is.
  """
  qpos = np.asarray(dataset.qpos, dtype=np.float32)
  qvel = np.asarray(dataset.qvel, dtype=np.float32)
  return np.column_stack(
      [qpos.reshape(qpos.shape[0], -1), qvel.reshape(qvel.shape[0], -1)]
  )


def _packed_state_keys(dataset: PredictorDataset) -> np.ndarray:
  rows = np.ascontiguousarray(initial_state_matrix(dataset))
  width = int(rows.dtype.itemsize * rows.shape[1])
  return rows.view(np.dtype((np.void, width))).reshape(rows.shape[0])


def distinct_state_indices(
    dataset: PredictorDataset, *, within: str = 'experiment'
) -> np.ndarray:
  """Index of the first occurrence of each distinct initial state.

  ``within='experiment'`` (the default) keeps a state that two independent
  searches both found: those are two draws from the search, not copies of
  one elite. ``within='all'`` collapses across runs as well.
  """
  if dataset.n_cases == 0:
    return np.zeros(0, dtype=np.int64)
  keys = _packed_state_keys(dataset)
  if within == 'all':
    _unique, first = np.unique(keys, return_index=True)
    return np.sort(first.astype(np.int64))
  if within != 'experiment':
    raise ValueError(f'within must be "experiment" or "all", got {within!r}')
  experiment = np.asarray(dataset.experiment)
  kept = []
  for value in np.unique(experiment):
    index = np.flatnonzero(experiment == value)
    _unique, first = np.unique(keys[index], return_index=True)
    kept.append(index[first])
  return np.sort(np.concatenate(kept).astype(np.int64))


def collapse_duplicate_states(
    dataset: PredictorDataset,
    *aligned: np.ndarray,
    within: str = 'experiment',
):
  """One row per distinct initial state, keeping the first encounter.

  Extra arrays passed in ``aligned`` are sliced with the same index, so
  held-out fail rates (which are looked up from generation shards *before*
  collapsing) stay matched to the surviving rows.
  """
  index = distinct_state_indices(dataset, within=within)
  if int(index.size) == int(dataset.n_cases):
    if aligned:
      return (dataset, *(np.asarray(arr) for arr in aligned))
    return dataset
  print(
      f'  distinct states: {int(index.size)}/{dataset.n_cases} '
      f'(dropped {dataset.n_cases - int(index.size)} exact repeats, '
      f'within={within})',
      flush=True,
  )
  dataset = dataset.subset(index)
  if aligned:
    return (dataset, *(np.asarray(arr)[index] for arr in aligned))
  return dataset


def _case_name(experiment: int, generation: int, case_index: int) -> str:
  return f'exp{int(experiment)}_gen{int(generation):03d}_case{int(case_index):03d}'


def _encounter_index(
    collection: GASearchCollection,
    experiment: int,
) -> dict[str, np.ndarray]:
  names: list[str] = []
  experiment_ids: list[np.ndarray] = []
  generations: list[np.ndarray] = []
  case_indices: list[np.ndarray] = []
  qpos: list[np.ndarray] = []
  qvel: list[np.ndarray] = []
  for record in collection.generations:
    n_cases = int(record.difficulty.shape[0])
    generation = int(record.generation)
    names.extend(_case_name(experiment, generation, i) for i in range(n_cases))
    experiment_ids.append(np.full(n_cases, experiment, dtype=np.int32))
    generations.append(np.full(n_cases, generation, dtype=np.int32))
    case_indices.append(np.arange(n_cases, dtype=np.int32))
    qpos.append(np.asarray(record.suite.qpos, dtype=np.float32))
    qvel.append(np.asarray(record.suite.qvel, dtype=np.float32))
  return {
      'names': np.asarray(names, dtype=object),
      'experiment': np.concatenate(experiment_ids),
      'generation': np.concatenate(generations),
      'case_index': np.concatenate(case_indices),
      'qpos': np.concatenate(qpos),
      'qvel': np.concatenate(qvel),
  }


def _state_ig_from_record(record) -> np.ndarray:
  """Concatenate signed state IG on qpos and qvel. Shape (P, N, nq + nv)."""
  return np.concatenate(
      [
          _flatten_trailing(record.state_ig_qpos),
          _flatten_trailing(record.state_ig_qvel),
      ],
      axis=-1,
  )


def _obs_ig_from_record(record) -> np.ndarray:
  """Observation IG. Shape (P, N, D) with D the observation dimension."""
  return _flatten_trailing(np.asarray(record.obs_ig))


def _concat_generation_metric(records, metric_fn) -> np.ndarray:
  """Apply ``metric_fn`` to each generation's action scale and stack cases."""
  parts = [metric_fn(record.action_scale) for record in records]
  if not parts:
    return np.zeros(0, dtype=np.float64)
  return np.concatenate(parts)


def _topk_disagreement_from_records(records, vectors_fn) -> np.ndarray:
  """Per-encounter top-k cosine distance. Shape (N, D)."""
  parts = [pairwise_topk_attribution_cosine_all_k(vectors_fn(record)) for record in records]
  if not parts:
    return np.zeros((0, 0), dtype=np.float64)
  return np.concatenate(parts, axis=0)


def predictor_dataset_from_collection(
    collection: GASearchCollection,
    environment: str,
    algorithm: str,
    experiment: int,
) -> PredictorDataset:
  """Build predictor rows from one GA collection."""
  policy_ids = list(collection.policy_ids)
  if not policy_ids:
    raise ValueError(f'Collection experiment {experiment} has no policy_ids.')
  metrics = disagreement_metrics_from_collection(
      collection, algorithm=algorithm
  )
  gait = gait_action_disagreement_metrics_from_collection(
      collection, environment, algorithm
  )
  if 'value_disagreement' not in metrics:
    raise ValueError(
        f'Collection experiment {experiment} has no value estimates.'
    )
  if not np.allclose(metrics['difficulty'], gait['difficulty']):
    raise ValueError(
        f'Collection experiment {experiment}: gait and IG difficulty arrays differ.'
    )
  energy = energy_metrics_from_collection(collection)
  index = _encounter_index(collection, experiment)
  n = int(metrics['difficulty'].shape[0])
  topk = _topk_disagreement_from_records(
      collection.generations, _state_ig_from_record
  )
  obs_topk = _topk_disagreement_from_records(
      collection.generations, _obs_ig_from_record
  )
  if index['names'].shape[0] != n:
    raise ValueError(
        f'Collection experiment {experiment}: {index["names"].shape[0]} names '
        f'but {n} metric rows.'
    )
  if energy['ensemble_action_scale'].shape[0] != n:
    raise ValueError(
        f'Collection experiment {experiment}: {n} metric rows but '
        f'{energy["ensemble_action_scale"].shape[0]} energy rows.'
    )
  if topk.shape[0] != n or obs_topk.shape[0] != n:
    raise ValueError(
        f'Collection experiment {experiment}: {n} metric rows but '
        f'{topk.shape[0]} state top-k rows and {obs_topk.shape[0]} '
        'observation top-k rows.'
    )
  return PredictorDataset(
      environment=environment,
      algorithm=algorithm,
      policy_ids=policy_ids,
      names=index['names'],
      experiment=index['experiment'],
      generation=index['generation'],
      case_index=index['case_index'],
      difficulty=np.asarray(metrics['difficulty'], dtype=np.float64),
      action_disagreement=np.asarray(
          metrics['action_disagreement'], dtype=np.float64
      ),
      gait_action_disagreement=np.asarray(
          gait['gait_action_disagreement'], dtype=np.float64
      ),
      nongait_action_disagreement=np.asarray(
          gait['nongait_action_disagreement'], dtype=np.float64
      ),
      feature_attribution_disagreement=np.asarray(
          metrics['state_ig_disagreement'], dtype=np.float64
      ),
      value_disagreement=np.asarray(
          metrics['value_disagreement'], dtype=np.float64
      ),
      criticality_disagreement=np.asarray(
          metrics['criticality_disagreement'], dtype=np.float64
      ),
      std_magnitude_disagreement=_concat_generation_metric(
          collection.generations, std_magnitude_disagreement_np
      ),
      std_direction_disagreement=_concat_generation_metric(
          collection.generations, std_direction_disagreement_np
      ),
      qpos=index['qpos'],
      qvel=index['qvel'],
      ensemble_action_scale=np.asarray(
          energy['ensemble_action_scale'], dtype=np.float64
      ),
      action_scale_disagreement=np.asarray(
          energy['action_scale_disagreement'], dtype=np.float64
      ),
      mechanical_power_proxy=np.asarray(
          energy['mechanical_power_proxy'], dtype=np.float64
      ),
      feature_attribution_topk_disagreement=np.asarray(
          topk, dtype=np.float64
      ),
      observation_ig_topk_disagreement=np.asarray(
          obs_topk, dtype=np.float64
      ),
  )


def predictor_dataset_from_generation(
    record,
    environment: str,
    algorithm: str,
    policy_ids: list[int],
    *,
    experiment: int = -1,
    name_prefix: str = 'aug_easy',
    name_offset: int = 0,
    subspaces=None,
) -> PredictorDataset:
  """Build predictor rows from one evaluated generation/batch of test cases."""
  from analysis.gait_subspace import (
      decompose_ensemble_actions,
      load_gait_subspaces_for_policies,
  )

  n = int(record.difficulty.shape[0])
  if n == 0:
    raise ValueError('Generation record has no test cases.')
  if subspaces is None:
    subspaces = load_gait_subspaces_for_policies(
        environment, algorithm, policy_ids
    )
  names = np.asarray(
      [f'{name_prefix}_{name_offset + i:06d}' for i in range(n)],
      dtype=object,
  )
  actions = _flatten_trailing(np.asarray(record.actions))
  gait, residual = decompose_ensemble_actions(record.actions, subspaces)
  state_ig = _state_ig_from_record(record)
  if not hasattr(record, 'action_scale'):
    raise ValueError(
        'Criticality disagreement requires the Gaussian action scale σ.'
    )
  energy = energy_observables_from_generation(record)
  return PredictorDataset(
      environment=environment,
      algorithm=algorithm,
      policy_ids=list(policy_ids),
      names=names,
      experiment=np.full(n, experiment, dtype=np.int32),
      generation=np.full(n, int(record.generation), dtype=np.int32),
      case_index=np.arange(n, dtype=np.int32) + int(name_offset),
      difficulty=np.asarray(record.difficulty, dtype=np.float64),
      action_disagreement=pairwise_disagreement_np(actions),
      gait_action_disagreement=pairwise_disagreement_np(gait),
      nongait_action_disagreement=pairwise_disagreement_np(residual),
      feature_attribution_disagreement=pairwise_disagreement_np(state_ig),
      value_disagreement=standardized_value_disagreement_np(
          value_estimates_np(
              record.values,
              getattr(record, 'q_values', None),
              algorithm=algorithm,
          )
      ),
      criticality_disagreement=criticality_disagreement_np(
          record.action_scale
      ),
      std_magnitude_disagreement=std_magnitude_disagreement_np(
          record.action_scale
      ),
      std_direction_disagreement=std_direction_disagreement_np(
          record.action_scale
      ),
      qpos=np.asarray(record.suite.qpos, dtype=np.float32),
      qvel=np.asarray(record.suite.qvel, dtype=np.float32),
      ensemble_action_scale=np.asarray(
          energy['ensemble_action_scale'], dtype=np.float64
      ),
      action_scale_disagreement=np.asarray(
          energy['action_scale_disagreement'], dtype=np.float64
      ),
      mechanical_power_proxy=np.asarray(
          energy['mechanical_power_proxy'], dtype=np.float64
      ),
      feature_attribution_topk_disagreement=pairwise_topk_attribution_cosine_all_k(
          state_ig
      ),
      observation_ig_topk_disagreement=pairwise_topk_attribution_cosine_all_k(
          _obs_ig_from_record(record)
      ),
  )


_ARRAY_FIELDS = (
    'names',
    'experiment',
    'generation',
    'case_index',
    'difficulty',
    'action_disagreement',
    'gait_action_disagreement',
    'nongait_action_disagreement',
    'feature_attribution_disagreement',
    'value_disagreement',
    'criticality_disagreement',
    'std_magnitude_disagreement',
    'std_direction_disagreement',
    'qpos',
    'qvel',
    'ensemble_action_scale',
    'action_scale_disagreement',
    'mechanical_power_proxy',
    'feature_attribution_topk_disagreement',
    'observation_ig_topk_disagreement',
)


def concat_predictor_datasets(
    parts: list[PredictorDataset],
) -> PredictorDataset:
  if not parts:
    raise ValueError('No predictor datasets to concatenate.')
  first = parts[0]
  for part in parts[1:]:
    if part.environment != first.environment or part.algorithm != first.algorithm:
      raise ValueError('Cannot concatenate datasets from different env/alg combos.')
    if part.policy_ids != first.policy_ids:
      raise ValueError(
          f'policy_ids {part.policy_ids} do not match {first.policy_ids}'
      )
  kwargs = {
      'environment': first.environment,
      'algorithm': first.algorithm,
      'policy_ids': list(first.policy_ids),
  }
  for field in _ARRAY_FIELDS:
    kwargs[field] = np.concatenate([getattr(part, field) for part in parts])
  return PredictorDataset(**kwargs)


def predictor_dataset_path(
    environment: str,
    algorithm: str,
    root: str = 'datasets',
    *,
    balanced: bool = False,
) -> str:
  name = 'predictor_data_balanced.pkl' if balanced else 'predictor_data.pkl'
  return os.path.join(root, environment, algorithm, name)


def save_predictor_dataset(dataset: PredictorDataset, path: str) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


_TOPK_FIELDS = frozenset((
    'feature_attribution_topk_disagreement',
    'observation_ig_topk_disagreement',
))


class _PredictorUnpickler(pickle._Unpickler):
  """Drop top-k matrices instead of materializing them.

  Those fields are numeric arrays stored as a bytearray plus
  ``numpy._frombuffer``. After a top-k key, the bytearray is read and
  discarded, and the reconstructed value is an empty array.
  ``PredictorDataset.__setstate__`` then installs the ``(n, 0)`` placeholder.
  """

  dispatch = pickle._Unpickler.dispatch.copy()

  def __init__(self, file):
    super().__init__(file)
    self._skip_array = False

  def find_class(self, module, name):
    if name == '_frombuffer' and module.startswith('numpy'):
      real = super().find_class(module, name)

      def frombuffer(buf, dtype, shape, order, _real=real):
        if self._skip_array:
          self._skip_array = False
          return np.zeros((0, 0), dtype=np.float64)
        return _real(buf, dtype, shape, order)

      return frombuffer
    return super().find_class(module, name)

  def _remember_string(self, text: str) -> None:
    self.append(text)
    if text in _TOPK_FIELDS:
      self._skip_array = True

  def load_short_binunicode(self):
    length = self.read(1)[0]
    self._remember_string(str(self.read(length), 'utf-8', 'surrogatepass'))
  dispatch[pickle.SHORT_BINUNICODE[0]] = load_short_binunicode

  def load_binunicode(self):
    length, = struct.unpack('<I', self.read(4))
    self._remember_string(str(self.read(length), 'utf-8', 'surrogatepass'))
  dispatch[pickle.BINUNICODE[0]] = load_binunicode

  def _discard(self, length: int) -> None:
    remaining = length
    while remaining:
      chunk = self.read(min(remaining, 1 << 20))
      if not chunk:
        raise pickle.UnpicklingError(
            'unexpected end of pickle while skipping an array'
        )
      remaining -= len(chunk)

  def load_bytearray8(self):
    length, = struct.unpack('<Q', self.read(8))
    if self._skip_array:
      self._discard(length)
      self.append(bytearray())
      return
    payload = bytearray(length)
    self.readinto(payload)
    self.append(payload)
  dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def load_predictor_dataset(
    path: str,
    *,
    load_topk: bool = True,
) -> PredictorDataset:
  """Load a predictor dataset.

  ``load_topk=False`` skips the state and observation top-k matrices. The
  file is still scanned, but those arrays are not allocated.
  """
  with open(path, 'rb') as handle:
    if load_topk:
      dataset = pickle.load(handle)
    else:
      dataset = _PredictorUnpickler(handle).load()
  if not isinstance(dataset, PredictorDataset):
    raise TypeError(f'{path} is not a PredictorDataset ({type(dataset)!r})')
  return dataset
