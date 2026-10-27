"""Collect per-test-case metrics during GA search."""

from dataclasses import dataclass, replace
import json
import os
import pickle
from typing import Callable, Iterator

import jax
import numpy as np
from jax import numpy as jp

from attributions.types import AttributionConfig, SuiteAttributionResult
from ga.fitness import difficulty_per_case, evaluate_population
from ga.ga import (
    BestIndividual,
    GAConfig,
    GenerationRecord,
    GenerationStats,
    Population,
    SearchResult,
)
from ga.search import (
    _compute_generation_record,
    _log_generation_record,
    _stack_generation_history,
    _update_best,
    init_population,
    next_generation,
)
from mutators.humanoid import genomes_to_test_suite as _default_genomes_to_test_suite
from mutators.humanoid import (
    crossover_genomes as _default_crossover_genomes,
    mutate_genome as _default_mutate_genome,
    random_genome as _default_random_genome,
    repair_genome as _default_repair_genome,
)
from test_case_execution.types import TestSuite


@dataclass
class GenerationCollection:
  """Metrics collected for every test case in one GA generation."""

  generation: int
  suite: TestSuite
  difficulty: np.ndarray
  failed: np.ndarray
  fitness: np.ndarray
  actions: np.ndarray
  action_loc: np.ndarray
  action_scale: np.ndarray
  values: np.ndarray
  action_mean: np.ndarray
  action_std: np.ndarray
  state_ig_qpos: np.ndarray
  state_ig_qvel: np.ndarray
  obs_ig: np.ndarray
  state_importance_qpos: np.ndarray
  state_importance_qvel: np.ndarray
  eval_fail_rate: np.ndarray | None = None
  eval_failed: np.ndarray | None = None
  # Twin SAC critics at the initial state, shape (P, N, 2), in search-policy
  # order. PPO rows are NaN. Axis 2 is the Q-network's critic order.
  q_values: np.ndarray | None = None

  @property
  def n_cases(self) -> int:
    return int(self.difficulty.shape[0])


@dataclass
class GASearchCollection:
  """All test cases encountered during a GA search."""

  generations: list[GenerationCollection]
  policy_ids: list[int]
  ga_config: GAConfig
  attribution_config: AttributionConfig
  search_result: SearchResult
  shard_dir: str | None = None

  @property
  def n_generations(self) -> int:
    if self.generations:
      return len(self.generations)
    manifest = _read_shard_manifest(getattr(self, 'shard_dir', None))
    if manifest is not None:
      return int(len(manifest.get('shards', [])))
    return 0

  @property
  def n_encounters(self) -> int:
    if self.generations:
      return int(sum(record.n_cases for record in self.generations))
    manifest = _read_shard_manifest(getattr(self, 'shard_dir', None))
    if manifest is not None:
      return int(manifest.get('n_collected', 0))
    return 0

  def _concat_case_arrays(self, field: str, policy_batched: bool) -> np.ndarray:
    """Stack one per-generation array along the test-case axis.

    Policy-batched fields have shape ``(P, N, ...)`` and concatenate on
    axis 1 so generations may have different ``N``. Other fields are
    ``(N, ...)`` and concatenate on axis 0.
    """
    arrays = [getattr(record, field) for record in self.generations]
    if not arrays:
      raise ValueError(f'No generations to flatten for {field}')
    axis = 1 if policy_batched else 0
    return np.concatenate(arrays, axis=axis)

  def flattened_difficulty(self) -> np.ndarray:
    return self._concat_case_arrays('difficulty', False)

  def flattened_action_mean(self) -> np.ndarray:
    return self._concat_case_arrays('action_mean', False)

  def flattened_action_std(self) -> np.ndarray:
    return self._concat_case_arrays('action_std', False)

  def flattened_action_loc(self) -> np.ndarray:
    return self._concat_case_arrays('action_loc', True)

  def flattened_action_scale(self) -> np.ndarray:
    return self._concat_case_arrays('action_scale', True)

  def flattened_values(self) -> np.ndarray:
    return self._concat_case_arrays('values', True)

  def flattened_state_ig_qpos(self) -> np.ndarray:
    return self._concat_case_arrays('state_ig_qpos', True)

  def flattened_state_ig_qvel(self) -> np.ndarray:
    return self._concat_case_arrays('state_ig_qvel', True)

  def flattened_obs_ig(self) -> np.ndarray:
    return self._concat_case_arrays('obs_ig', True)

  def flattened_state_importance_qpos(self) -> np.ndarray:
    return self._concat_case_arrays('state_importance_qpos', True)

  def flattened_state_importance_qvel(self) -> np.ndarray:
    return self._concat_case_arrays('state_importance_qvel', True)


def ga_data_dir(
    environment: str,
    algorithm: str,
    rollout_length: int | None = None,
) -> str:
  """Directory for GA collections of one environment and algorithm.

  If ``rollout_length`` is set, collections live under
  ``rollout_length={length}/`` so runs with different horizons do not
  overwrite each other.
  """
  path = os.path.join('datasets', 'ga_data', environment, algorithm)
  if rollout_length is not None:
    path = os.path.join(path, f'rollout_length={int(rollout_length)}')
  return path


def collection_output_path(
    environment: str,
    algorithm: str,
    experiment_number: int,
    rollout_length: int | None = None,
) -> str:
  """Path to a single-experiment collection pickle.

  With ``rollout_length``, this is
  ``datasets/ga_data/{env}/{alg}/rollout_length={length}/experiment_{n}_ga_collection.pkl``.
  """
  return os.path.join(
      ga_data_dir(environment, algorithm, rollout_length=rollout_length),
      f'experiment_{experiment_number}_ga_collection.pkl',
  )


def generation_shard_dir(collection_path: str) -> str:
  """Directory of per-generation pickles next to ``collection_path``."""
  stem, _ = os.path.splitext(collection_path)
  return stem + '_generations'


def generation_shard_path(collection_path: str, generation: int) -> str:
  return os.path.join(
      generation_shard_dir(collection_path),
      f'generation_{int(generation):05d}.pkl',
  )


def _manifest_path(shard_dir: str) -> str:
  return os.path.join(shard_dir, 'manifest.json')


def _read_shard_manifest(shard_dir: str | None) -> dict | None:
  if not shard_dir:
    return None
  path = _manifest_path(shard_dir)
  if not os.path.isfile(path):
    return None
  with open(path) as handle:
    return json.load(handle)


def _numpy_tree(value):
  """Copy a pytree of device arrays onto the host as NumPy."""
  if isinstance(value, TestSuite):
    return TestSuite(
        qpos=np.asarray(value.qpos),
        qvel=np.asarray(value.qvel),
        policy_rng=np.asarray(value.policy_rng),
    )
  if hasattr(value, 'generation') and hasattr(value, 'difficulty'):
    copied = replace(
        value,
        suite=_numpy_tree(value.suite),
        difficulty=np.asarray(value.difficulty),
        failed=np.asarray(value.failed),
        fitness=np.asarray(value.fitness),
        actions=np.asarray(value.actions),
        action_loc=np.asarray(value.action_loc),
        action_scale=np.asarray(value.action_scale),
        values=np.asarray(value.values),
        action_mean=np.asarray(value.action_mean),
        action_std=np.asarray(value.action_std),
        state_ig_qpos=np.asarray(value.state_ig_qpos),
        state_ig_qvel=np.asarray(value.state_ig_qvel),
        obs_ig=np.asarray(value.obs_ig),
        state_importance_qpos=np.asarray(value.state_importance_qpos),
        state_importance_qvel=np.asarray(value.state_importance_qvel),
        eval_fail_rate=(
            None
            if getattr(value, 'eval_fail_rate', None) is None
            else np.asarray(value.eval_fail_rate)
        ),
        eval_failed=(
            None
            if getattr(value, 'eval_failed', None) is None
            else np.asarray(value.eval_failed)
        ),
        q_values=(
            None
            if getattr(value, 'q_values', None) is None
            else np.asarray(value.q_values)
        ),
    )
    features = getattr(value, 'search_features', None)
    if features is not None:
      # Extra attribute, not a dataclass field, so older collections still
      # unpickle and still pass through replace().
      copied.search_features = {
          str(name): np.array(np.asarray(column), dtype=np.float64, copy=True)
          for name, column in features.items()
      }
    return copied
  return value


def save_case_states(
    qpos: np.ndarray,
    qvel: np.ndarray,
    collection_path: str,
    generation: int,
    *,
    counts: np.ndarray | None = None,
) -> str:
  """Write the initial states kept in one generation.

  ``qpos`` and ``qvel`` are the pair the suite membership check hashes.
  Policy outputs are not stored.
  """
  qpos = np.asarray(qpos, dtype=np.float32)
  qvel = np.asarray(qvel, dtype=np.float32)
  if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
    raise ValueError(
        f'qpos {qpos.shape} and qvel {qvel.shape} are not aligned states.'
    )
  if int(qpos.shape[0]) < 1:
    raise ValueError('Cannot store an empty generation.')
  path = generation_shard_path(collection_path, generation)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as handle:
    pickle.dump(
        {
            'generation': int(generation),
            'qpos': qpos,
            'qvel': qvel,
        },
        handle,
    )
  _record_shard(
      collection_path,
      generation,
      n_cases=int(qpos.shape[0]),
      filename=os.path.basename(path),
      counts=counts,
  )
  return path


def _record_shard(
    collection_path: str,
    generation: int,
    *,
    n_cases: int,
    filename: str,
    counts: np.ndarray | None,
) -> None:
  shard_dir = generation_shard_dir(collection_path)
  manifest = _read_shard_manifest(shard_dir) or {'shards': []}
  shards = [
      entry
      for entry in manifest.get('shards', [])
      if int(entry['generation']) != int(generation)
  ]
  shards.append(
      {
          'generation': int(generation),
          'file': filename,
          'n_cases': int(n_cases),
      }
  )
  shards.sort(key=lambda entry: int(entry['generation']))
  manifest['shards'] = shards
  manifest['n_collected'] = int(sum(int(item['n_cases']) for item in shards))
  if counts is not None:
    manifest['counts'] = [int(x) for x in np.asarray(counts).reshape(-1)]
  with open(_manifest_path(shard_dir), 'w') as handle:
    json.dump(manifest, handle, indent=2)


def save_generation_shard(
    record: GenerationCollection,
    collection_path: str,
    *,
    counts: np.ndarray | None = None,
) -> str:
  """Write one generation's kept cases to disk and update the shard manifest."""
  host = _numpy_tree(record)
  path = generation_shard_path(collection_path, host.generation)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as handle:
    pickle.dump(host, handle)
  _record_shard(
      collection_path,
      host.generation,
      n_cases=int(host.n_cases),
      filename=os.path.basename(path),
      counts=counts,
  )
  return path


def iter_generation_shards(
    collection_path: str,
) -> Iterator[GenerationCollection]:
  """Yield kept-case records from disk, one generation at a time."""
  shard_dir = generation_shard_dir(collection_path)
  manifest = _read_shard_manifest(shard_dir)
  if manifest is None:
    return
  for entry in manifest.get('shards', []):
    path = os.path.join(shard_dir, entry['file'])
    with open(path, 'rb') as handle:
      yield pickle.load(handle)


def load_generation_shards(collection_path: str) -> list[GenerationCollection]:
  return list(iter_generation_shards(collection_path))


def load_collection(path: str, *, load_shards: bool = True) -> GASearchCollection:
  """Load a collection saved by :func:`save_collection`.

  Sharded binned searches store per-generation pickles beside ``path``.
  ``load_shards=True`` (default) assembles them into ``generations``.
  """
  with open(path, 'rb') as handle:
    collection = pickle.load(handle)
  inferred = generation_shard_dir(path)
  if os.path.isdir(inferred):
    collection.shard_dir = inferred
  elif not hasattr(collection, 'shard_dir'):
    collection.shard_dir = None
  shard_dir = getattr(collection, 'shard_dir', None)
  if load_shards and shard_dir and not collection.generations:
    collection.generations = load_generation_shards(path)
  return collection


def save_collection(collection: GASearchCollection, path: str) -> None:
  """Persist a GA search collection to disk.

  When ``shard_dir`` is set, generation payloads stay in shards; the pickle
  only stores search metadata so it stays small.
  """
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  to_save = collection
  if collection.shard_dir:
    to_save = replace(collection, generations=[])
  with open(path, 'wb') as handle:
    pickle.dump(to_save, handle)


def _print_search_best(collection: GASearchCollection) -> None:
  difficulty = float(collection.search_result.best.difficulty)
  fitness = float(collection.search_result.best.fitness)
  if np.isfinite(difficulty):
    print(f'Best difficulty found: {difficulty:.4f}')
  else:
    print(f'Best fitness found: {fitness:.4f}')


def summarize_collection(collection: GASearchCollection) -> None:
  """Print summary statistics for a search collection.

  Streams generation shards when ``generations`` is empty so Humanoid
  archives are not pulled entirely into RAM.
  """
  shard_dir = getattr(collection, 'shard_dir', None)
  if collection.generations:
    difficulty = collection.flattened_difficulty()
    action_std = collection.flattened_action_std()
    action_loc = collection.flattened_action_loc()
    action_scale = collection.flattened_action_scale()
    n_enc = collection.n_encounters
    n_gen = collection.n_generations
    values_shape = (
        collection.flattened_values().shape
        if collection.generations and hasattr(collection.generations[0], 'values')
        else None
    )
    ig_qpos_shape = collection.flattened_state_ig_qpos().shape
    obs_ig_shape = collection.flattened_obs_ig().shape
  elif shard_dir:
    n_sum = 0
    d_sum = 0.0
    d_sq = 0.0
    d_min = None
    d_max = None
    n_gen = 0
    std_sum = 0.0
    loc_shape = None
    scale_shape = None
    values_shape = None
    ig_qpos_shape = None
    obs_ig_shape = None
    shard_root = shard_dir
    manifest = _read_shard_manifest(shard_root) or {'shards': []}
    for entry in manifest.get('shards', []):
      path = os.path.join(shard_root, entry['file'])
      with open(path, 'rb') as handle:
        record = pickle.load(handle)
      diff = np.asarray(record.difficulty).reshape(-1)
      n = int(diff.size)
      n_sum += n
      d_sum += float(diff.sum())
      d_sq += float(np.square(diff).sum())
      d_min = float(diff.min()) if d_min is None else min(d_min, float(diff.min()))
      d_max = float(diff.max()) if d_max is None else max(d_max, float(diff.max()))
      std_sum += float(np.asarray(record.action_std).mean()) * n
      loc_shape = tuple(np.asarray(record.action_loc).shape)
      scale_shape = tuple(np.asarray(record.action_scale).shape)
      values_shape = tuple(np.asarray(record.values).shape)
      ig_qpos_shape = tuple(np.asarray(record.state_ig_qpos).shape)
      obs_ig_shape = tuple(np.asarray(record.obs_ig).shape)
      n_gen += 1
      del record
    n_enc = n_sum
    mean = d_sum / max(n_sum, 1)
    var = d_sq / max(n_sum, 1) - mean * mean
    difficulty_mean = mean
    difficulty_std = float(np.sqrt(max(var, 0.0)))
    difficulty_min = 0.0 if d_min is None else d_min
    difficulty_max = 0.0 if d_max is None else d_max
    action_std_mean = std_sum / max(n_sum, 1)
    print(
        f'Collected {n_enc} test-case encounters across {n_gen} generations'
    )
    if d_min is None or not np.isfinite(d_sum):
      print(
          'Difficulty was not estimated; the search did not roll out '
          'an ensemble.'
      )
    else:
      print(
          f'Difficulty: mean={difficulty_mean:.4f}  '
          f'std={difficulty_std:.4f}  '
          f'range=[{difficulty_min:.4f}, {difficulty_max:.4f}]'
      )
    print(
        f'Policy action std (mean over cases/actions): {action_std_mean:.4f}'
    )
    if loc_shape is not None:
      print(f'Policy action loc shape (last shard): {loc_shape}')
    if scale_shape is not None:
      print(f'Policy action scale shape (last shard): {scale_shape}')
    if values_shape is not None:
      print(f'Policy values shape (last shard): {values_shape}')
    if ig_qpos_shape is not None:
      print(f'State IG qpos shape (last shard): {ig_qpos_shape}')
    if obs_ig_shape is not None:
      print(f'Observation IG shape (last shard): {obs_ig_shape}')
    _print_search_best(collection)
    return
  else:
    raise ValueError('Collection has no generations and no shard_dir.')

  print(
      f'Collected {n_enc} test-case encounters '
      f'across {n_gen} generations'
  )
  if not np.any(np.isfinite(difficulty)):
    print(
        'Difficulty was not estimated; the search did not roll out '
        'an ensemble.'
    )
  else:
    print(
        f'Difficulty: mean={difficulty.mean():.4f}  '
        f'std={difficulty.std():.4f}  '
        f'range=[{difficulty.min():.4f}, {difficulty.max():.4f}]'
    )
  print(f'Policy action std (mean over cases/actions): {action_std.mean():.4f}')
  print(f'Policy action loc shape (all encounters): {action_loc.shape}')
  print(f'Policy action scale shape (all encounters): {action_scale.shape}')
  if values_shape is not None:
    print(f'Policy values shape (all encounters): {values_shape}')
  print(f'State IG qpos shape (all encounters): {ig_qpos_shape}')
  print(f'Observation IG shape (all encounters): {obs_ig_shape}')
  _print_search_best(collection)


def _append_generation_history(
    history: GenerationStats,
    records: list[GenerationRecord],
) -> GenerationStats:
  """Append generation summary records to an existing history."""
  if not records:
    return history
  new = _stack_generation_history(records)
  return GenerationStats(
      generation=jp.concatenate([history.generation, new.generation]),
      mean_difficulty=jp.concatenate([history.mean_difficulty, new.mean_difficulty]),
      max_difficulty=jp.concatenate([history.max_difficulty, new.max_difficulty]),
      min_difficulty=jp.concatenate([history.min_difficulty, new.min_difficulty]),
      std_difficulty=jp.concatenate([history.std_difficulty, new.std_difficulty]),
      mean_fitness=jp.concatenate([history.mean_fitness, new.mean_fitness]),
      max_fitness=jp.concatenate([history.max_fitness, new.max_fitness]),
      min_fitness=jp.concatenate([history.min_fitness, new.min_fitness]),
      std_fitness=jp.concatenate([history.std_fitness, new.std_fitness]),
      best_fitness=jp.concatenate([history.best_fitness, new.best_fitness]),
      best_difficulty=jp.concatenate([history.best_difficulty, new.best_difficulty]),
  )


def continue_ga_search_collection(
    collection: GASearchCollection,
    qpos0: jax.Array,
    execute_fn: Callable,
    extract_attributions_fn: Callable[[TestSuite], SuiteAttributionResult],
    additional_generations: int,
    rng: jax.Array,
    genomes_to_suite_fn: Callable = _default_genomes_to_test_suite,
    crossover_genomes_fn=_default_crossover_genomes,
    mutate_genome_fn=_default_mutate_genome,
) -> GASearchCollection:
  """Continue a GA search from a saved collection for more generations."""
  config = collection.ga_config
  attribution_config = collection.attribution_config

  population = collection.search_result.final_population
  last_record = collection.generations[-1]
  fitness = jp.array(last_record.fitness)
  best = collection.search_result.best
  start_generation = last_record.generation

  new_generation_records: list[GenerationCollection] = []
  new_history_records: list[GenerationRecord] = []

  for step in range(additional_generations):
    rng, step_rng = jax.random.split(rng)
    population = next_generation(
        step_rng,
        population,
        fitness,
        config,
        crossover_genomes_fn=crossover_genomes_fn,
        mutate_genome_fn=mutate_genome_fn,
    )
    fitness, difficulty, failed = evaluate_population(
        execute_fn,
        population.genomes,
        population.policy_rng,
        qpos0,
        config,
        genomes_to_suite_fn=genomes_to_suite_fn,
    )
    suite = genomes_to_suite_fn(
        population.genomes, qpos0, population.policy_rng
    )
    generation = start_generation + step + 1
    new_generation_records.append(
        collect_population_metrics(
            suite,
            failed,
            fitness,
            generation=generation,
            extract_attributions_fn=extract_attributions_fn,
        )
    )
    best = _update_best(best, population, fitness, difficulty)
    record = _compute_generation_record(generation, fitness, difficulty, best)
    new_history_records.append(record)
    _log_generation_record(record, config)

  search_result = SearchResult(
      best=best,
      final_population=population,
      history=_append_generation_history(
          collection.search_result.history, new_history_records
      ),
      final_difficulties=difficulty,
  )
  return GASearchCollection(
      generations=collection.generations + new_generation_records,
      policy_ids=collection.policy_ids,
      ga_config=config,
      attribution_config=attribution_config,
      search_result=search_result,
  )


def collect_population_metrics(
    suite: TestSuite,
    failed: jax.Array,
    fitness: jax.Array,
    generation: int,
    extract_attributions_fn: Callable[[TestSuite], SuiteAttributionResult],
) -> GenerationCollection:
  """Collect difficulty, policy outputs, and IG attributions for a population."""
  attributions = extract_attributions_fn(suite)
  actions = attributions.actions
  return GenerationCollection(
      generation=generation,
      suite=TestSuite(
          qpos=np.asarray(suite.qpos),
          qvel=np.asarray(suite.qvel),
          policy_rng=np.asarray(suite.policy_rng),
      ),
      difficulty=np.asarray(difficulty_per_case(failed)),
      failed=np.asarray(failed),
      fitness=np.asarray(fitness),
      actions=np.asarray(actions),
      action_loc=np.asarray(attributions.action_loc),
      action_scale=np.asarray(attributions.action_scale),
      values=np.asarray(attributions.values),
      action_mean=np.asarray(jp.mean(actions, axis=0)),
      action_std=np.asarray(jp.std(actions, axis=0)),
      state_ig_qpos=np.asarray(attributions.state_integrated_gradient_qpos),
      state_ig_qvel=np.asarray(attributions.state_integrated_gradient_qvel),
      obs_ig=np.asarray(attributions.observation_integrated_gradient),
      state_importance_qpos=np.asarray(attributions.state_importance_qpos),
      state_importance_qvel=np.asarray(attributions.state_importance_qvel),
  )


def build_ga_search_collector(
    env,
    qpos0: jax.Array,
    nq: int,
    nv: int,
    execute_fn: Callable,
    extract_attributions_fn: Callable[[TestSuite], SuiteAttributionResult],
    attribution_config: AttributionConfig | None = None,
    config: GAConfig | None = None,
    genomes_to_suite_fn: Callable = _default_genomes_to_test_suite,
    random_genome_fn=_default_random_genome,
    repair_genome_fn=_default_repair_genome,
    crossover_genomes_fn=_default_crossover_genomes,
    mutate_genome_fn=_default_mutate_genome,
    initial_population: Population | None = None,
):
  """Build a GA runner that records metrics for every evaluated test case.

  If ``initial_population`` is set, generation 0 starts from it instead of
  sampling a fresh random population.

  Returns:
    Callable ``run(rng) -> GASearchCollection``.
  """
  del env  # closed over by execute_fn / extract_attributions_fn
  if config is None:
    config = GAConfig()
  if attribution_config is None:
    attribution_config = AttributionConfig()

  def run(rng: jax.Array) -> GASearchCollection:
    rng, init_rng = jax.random.split(rng)
    if initial_population is None:
      population = init_population(
          init_rng,
          nq,
          nv,
          config,
          random_genome_fn=random_genome_fn,
          repair_genome_fn=repair_genome_fn,
      )
    else:
      population = initial_population
    generation_records: list[GenerationCollection] = []

    fitness, difficulty, failed = evaluate_population(
        execute_fn,
        population.genomes,
        population.policy_rng,
        qpos0,
        config,
        genomes_to_suite_fn=genomes_to_suite_fn,
    )
    suite = genomes_to_suite_fn(
        population.genomes, qpos0, population.policy_rng
    )
    generation_records.append(
        collect_population_metrics(
            suite,
            failed,
            fitness,
            generation=0,
            extract_attributions_fn=extract_attributions_fn,
        )
    )
    best = BestIndividual(
        genome=jax.tree.map(
            lambda x: x[jp.argmax(fitness)], population.genomes
        ),
        policy_rng=population.policy_rng[jp.argmax(fitness)],
        difficulty=difficulty[jp.argmax(fitness)],
        fitness=fitness[jp.argmax(fitness)],
    )

    history_records: list[GenerationRecord] = []
    init_record = _compute_generation_record(0, fitness, difficulty, best)
    history_records.append(init_record)
    _log_generation_record(init_record, config)

    for gen in range(config.n_generations):
      rng, step_rng = jax.random.split(rng)
      population = next_generation(
          step_rng,
          population,
          fitness,
          config,
          crossover_genomes_fn=crossover_genomes_fn,
          mutate_genome_fn=mutate_genome_fn,
      )
      fitness, difficulty, failed = evaluate_population(
          execute_fn,
          population.genomes,
          population.policy_rng,
          qpos0,
          config,
          genomes_to_suite_fn=genomes_to_suite_fn,
      )
      suite = genomes_to_suite_fn(
          population.genomes, qpos0, population.policy_rng
      )
      generation_records.append(
          collect_population_metrics(
              suite,
              failed,
              fitness,
              generation=gen + 1,
              extract_attributions_fn=extract_attributions_fn,
          )
      )
      best = _update_best(best, population, fitness, difficulty)
      record = _compute_generation_record(gen + 1, fitness, difficulty, best)
      history_records.append(record)
      _log_generation_record(record, config)

    search_result = SearchResult(
        best=best,
        final_population=population,
        history=_stack_generation_history(history_records),
        final_difficulties=difficulty,
    )
    return GASearchCollection(
        generations=generation_records,
        policy_ids=[],
        ga_config=config,
        attribution_config=attribution_config,
        search_result=search_result,
    )

  return run
