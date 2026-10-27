"""GA search whose fitness is a disagreement feature.

Difficulty is not estimated and is not used to select parents. Only the
objective feature is measured. When a search screen is supplied, every
candidate that passes the health screens is rolled out with the search
ensemble, and a candidate that fails every search policy is discarded.
The suite report reuses that rollout for the elites. Every
new initial state is stored once, as ``qpos`` and ``qvel``, which is the
pair used to tell whether that state is already in the test suite. The
run keeps 10000 elites and stops when fewer than 1000 of them are
replaced across the last 5 generations. A candidate outside the
environment's healthy set is discarded before selection. Every candidate
also takes a zero-action rollout of 3 steps, in batches of 10000, and is
discarded if any step leaves the healthy set. Hopper and Walker2d also
discard a candidate whose limited joints start more than 0.18 rad past
their stops. The elite set is the highest-fitness distinct states, up
to 10000.
"""

from __future__ import annotations

import csv
import os
import pickle
from types import SimpleNamespace
from typing import Sequence

import jax
import numpy as np
from jax import numpy as jp

from attributions.types import AttributionConfig
from ga.binned import (
    _checkpoint_sidecar,
    _immigrant_sampler,
    _make_collection,
    _operator_kwargs,
    _replace_tail,
    state_keys,
)
from ga.collect import (
    _read_shard_manifest,
    generation_shard_dir,
    load_collection,
    save_case_states,
)
from ga.feature_objectives import (
    FEATURE_NEEDS,
    FeatureObjective,
    PopulationSignals,
    combine_feature_fitness,
    feature_columns,
    feature_slug,
    fitness_definition,
    requirements,
)
from ga.ga import BestIndividual, GAConfig, GenerationRecord, Population
from ga.healthy import (
    ZERO_ACTION_BATCH,
    ZERO_ACTION_HORIZON,
    healthy_initial_mask,
    zero_action_fall_mask,
)
from ga.search import (
    _compute_generation_record,
    _update_best,
    init_population,
    next_generation,
)
from ga.stratified_init import sample_population, sampler_from_bundle
from mutators.humanoid import genomes_to_test_suite as _default_genomes_to_test_suite
from test_case_execution.types import TestSuite

DEFAULT_FAILURE_RATE_EVERY = 5
DEFAULT_MAX_GENERATIONS = 500
ELITE_COUNT = 10_000
CANDIDATES_PER_GENERATION = 10_000
DEFAULT_ELITE_REPLACE_LIMIT = 1000
DEFAULT_ELITE_REPLACE_WINDOW = 5
_LOCOMOTION_TAGS = frozenset({'qvel', 'gait'})
ELITE_STOP = (
    'Stop once 10000 distinct elites have been filled and the last '
    'elite_replace_window generations together replaced fewer than '
    'elite_replace_limit of them. Generation 0 fills the elite set and '
    'does not count as a replacement. A replaced elite is a previous '
    'elite state that is absent from the new elite set.'
)
SUITE_REPORT_COLUMNS = (
    'generation',
    'search_fail_rate',
    'eval_fail_rate',
    'candidates_discarded',
    'tests_replaced',
    'tests_added',
    'elites_replaced',
    'n_cases',
)
SUITE_REPORT = (
    'Every failure_rate_every generations, including generation 0 and the '
    'final generation, the mean failure rate of the current elite suite is '
    'recorded. The search rate is taken from this generation\'s '
    'search-ensemble screen of those elites when that screen ran. The '
    'checkpoint rolls the evaluation ensemble. The '
    'same rows record the fraction of elites that failed exactly k policies '
    'for each k from 0 through P, separately for the search ensemble and '
    'the evaluation ensemble. The suite is the distinct elites, at most '
    '10000, not the candidates drawn to challenge them and not the history '
    'of every unique state stored so far. Those rates do not enter '
    'selection. candidates_discarded, tests_replaced, tests_added, and '
    'elites_replaced are totals since the previous row. A discarded '
    'candidate is an offspring removed to make room for an immigrant, or '
    'an evaluated non-elite whose initial state was already stored. A '
    'replaced test is a non-elite population slot whose previous occupant '
    'left. A new test is a unique initial state stored for the first time. '
    'An elite replacement is a previous elite state that is no longer in '
    'the elite set.'
)


def feature_search_config(
    config: GAConfig,
    *,
    elite_count: int = ELITE_COUNT,
    candidates_per_generation: int = CANDIDATES_PER_GENERATION,
) -> GAConfig:
  """Keep ``elite_count`` elites and score that many new candidates each generation."""
  elite_count = int(elite_count)
  candidates_per_generation = int(candidates_per_generation)
  if elite_count < 1:
    raise ValueError(f'elite_count must be >= 1, got {elite_count}')
  if candidates_per_generation < 1:
    raise ValueError(
        'candidates_per_generation must be >= 1, got '
        f'{candidates_per_generation}'
    )
  return config.replace(
      population_size=elite_count + candidates_per_generation,
      elite_count=elite_count,
  )


def feature_collection_path(
    environment: str,
    algorithm: str,
    exp_number: int,
    rollout_length: int,
    objectives: Sequence[FeatureObjective],
) -> str:
  """Where one feature-search experiment is written.

  The objective list is part of the path so two searches that share an
  experiment number and horizon do not overwrite each other.
  """
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'feature_search',
      feature_slug(objectives),
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(exp_number)}_ga_collection.pkl',
  )


def feature_suite_paths(
    environment: str,
    algorithm: str,
    exp_number: int,
    rollout_length: int,
    objectives: Sequence[FeatureObjective],
) -> tuple[str, str]:
  """Suite pickle and failure-rate table for one finished feature search.

  The fitness objective names the folder. Environment, algorithm, horizon,
  and experiment stay in the path so two runs that optimize the same
  feature do not overwrite each other.
  """
  directory = os.path.join(
      'test_suites',
      feature_slug(objectives),
      environment,
      algorithm,
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(exp_number)}',
  )
  return (
      os.path.join(directory, 'suite.pkl'),
      os.path.join(directory, 'failure_rates.csv'),
  )


def write_feature_suite(
    path: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    policy_rng: np.ndarray,
) -> TestSuite:
  """Write the current elite suite, best elite first, at most 10000 cases."""
  qpos = np.array(np.asarray(qpos), copy=True)
  qvel = np.array(np.asarray(qvel), copy=True)
  policy_rng = np.array(np.asarray(policy_rng), copy=True)
  if qpos.ndim < 2 or qvel.ndim < 2 or policy_rng.ndim != 2:
    raise ValueError(
        f'Expected qpos (N, nq), qvel (N, nv), and policy_rng (N, 2); '
        f'got {qpos.shape}, {qvel.shape}, {policy_rng.shape}.'
    )
  n_cases = int(qpos.shape[0])
  if n_cases > ELITE_COUNT:
    raise ValueError(
        f'A saved suite has {n_cases} cases; the limit is {ELITE_COUNT}.'
    )
  if (
      n_cases < 1
      or int(qvel.shape[0]) != n_cases
      or int(policy_rng.shape[0]) != n_cases
  ):
    raise ValueError(
        f'Suite arrays disagree or are empty: qpos {qpos.shape}, '
        f'qvel {qvel.shape}, policy_rng {policy_rng.shape}.'
    )
  suite = TestSuite(qpos=qpos, qvel=qvel, policy_rng=policy_rng)
  _atomic_pickle(path, suite)
  return suite


def suite_turnover(
    generation: int,
    kept: np.ndarray,
    population_size: int,
    elite_count: int,
    n_immigrants: int,
) -> tuple[int, int, int]:
  """Discarded candidates, replaced tests, and new tests for one generation.

  Generation 0 has no previous occupant, so nothing is replaced. Later
  generations keep the elites and replace every other population slot.
  Immigrants are counted as discarded candidates because the offspring
  they displace are never evaluated. An evaluated non-elite is discarded
  when its initial state is already in the archive.
  """
  kept = np.asarray(kept, dtype=np.int64).reshape(-1)
  population_size = int(population_size)
  n_immigrants = int(n_immigrants)
  if population_size < 1:
    raise ValueError(f'population_size must be >= 1, got {population_size}')
  if n_immigrants < 0:
    raise ValueError(f'n_immigrants must be >= 0, got {n_immigrants}')
  n_added = int(kept.size)
  if np.any(kept < 0) or np.any(kept >= population_size):
    raise ValueError('Kept indices fall outside the population.')
  if int(generation) == 0:
    discarded = population_size - n_added
    if discarded < 0:
      raise ValueError(
          f'Generation 0 kept {n_added} cases from a population of '
          f'{population_size}.'
      )
    return discarded, 0, n_added
  n_elite = min(max(int(elite_count), 0), population_size)
  n_replaced = population_size - n_elite
  if n_immigrants > n_replaced:
    raise ValueError(
        f'Cannot discard {n_immigrants} offspring for immigrants when only '
        f'{n_replaced} non-elite slots exist.'
    )
  n_added_candidates = int(np.sum(kept >= n_elite))
  if n_added_candidates > n_replaced:
    raise ValueError(
        f'{n_added_candidates} new tests sit in non-elite slots, but only '
        f'{n_replaced} such slots exist.'
    )
  discarded = n_immigrants + (n_replaced - n_added_candidates)
  return discarded, n_replaced, n_added


def failure_count_fractions(failed: np.ndarray) -> np.ndarray:
  """Fraction of elites that failed exactly ``k`` policies, for ``k = 0..P``.

  ``failed`` has shape ``(P, N)``, one row per policy and one column per
  elite. The returned vector has length ``P + 1`` and sums to 1.
  """
  failed = np.asarray(failed, dtype=bool)
  if failed.ndim != 2 or int(failed.shape[1]) < 1:
    raise ValueError(
        f'Failure matrix must have shape (P, N) with N >= 1, got {failed.shape}.'
    )
  counts = np.sum(failed, axis=0)
  n_policies = int(failed.shape[0])
  histogram = np.bincount(counts, minlength=n_policies + 1)
  if int(histogram.shape[0]) != n_policies + 1:
    raise ValueError(
        f'Failure counts exceed P={n_policies}: max {int(np.max(counts))}.'
    )
  return histogram.astype(np.float64) / float(failed.shape[1])


def record_failure_count_fractions(
    report: dict,
    search_failed: np.ndarray,
    eval_failed: np.ndarray,
) -> None:
  """Store search and evaluation failure-count fractions on a suite row."""
  _store_failure_fractions(report, 'search_failed_', search_failed)
  _store_failure_fractions(report, 'eval_failed_', eval_failed)


def suite_report_fieldnames(report: dict) -> list[str]:
  """CSV columns for one suite row, including any failure-count fractions."""
  columns = list(SUITE_REPORT_COLUMNS)
  for prefix in ('search_failed_', 'eval_failed_'):
    keys = [key for key in report if key.startswith(prefix)]
    keys.sort(key=lambda key: int(key[len(prefix):]))
    columns.extend(keys)
  return columns


def _store_failure_fractions(report: dict, prefix: str, failed: np.ndarray) -> None:
  fractions = failure_count_fractions(failed)
  for count, fraction in enumerate(fractions):
    report[f'{prefix}{count}'] = float(fraction)


def append_suite_report(
    path: str,
    report: dict,
    *,
    header: bool,
    columns: Sequence[str] | None = None,
) -> None:
  """Append one suite checkpoint. The first row writes the column names."""
  fieldnames = list(columns) if columns is not None else suite_report_fieldnames(report)
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w' if header else 'a', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
    if header:
      writer.writeheader()
    writer.writerow({
        column: _report_cell(report.get(column))
        for column in fieldnames
    })


def _report_cell(value) -> str:
  if value is None:
    return ''
  if isinstance(value, (float, np.floating)):
    number = float(value)
    if not np.isfinite(number):
      return ''
    return format(number, '.16g')
  return str(int(value))


def _format_rate(value) -> str:
  if value is None:
    return 'n/a'
  number = float(value)
  if not np.isfinite(number):
    return 'n/a'
  return f'{number:.4g}'


def _current_suite_index(suite, limit: int) -> np.ndarray:
  """First occurrence of each distinct case, at most ``limit`` of them."""
  if int(limit) < 1:
    raise ValueError(f'limit must be >= 1, got {limit}')
  chosen = []
  seen: set[bytes] = set()
  for index, key in enumerate(state_keys(suite)):
    if key in seen:
      continue
    seen.add(key)
    chosen.append(index)
    if len(chosen) >= int(limit):
      break
  if not chosen:
    raise RuntimeError('The current suite has no cases to evaluate.')
  return np.asarray(chosen, dtype=np.int64)


def current_suite_states(
    suite,
    limit: int = CANDIDATES_PER_GENERATION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Distinct locomotion cases in the current generation, at most ``limit``.

  Copied elites are one test, not two. Order follows the population, so
  the retained elites come before the new candidates.
  """
  if not hasattr(suite, 'qpos'):
    raise TypeError(
        'Suite failure rates are recorded for locomotion states only.'
    )
  index = _current_suite_index(suite, limit)
  return (
      np.asarray(suite.qpos)[index],
      np.asarray(suite.qvel)[index],
      np.asarray(suite.policy_rng)[index],
  )


def _atomic_pickle(path: str, payload) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  temporary = path + '.tmp'
  with open(temporary, 'wb') as handle:
    pickle.dump(payload, handle)
  os.replace(temporary, path)


def select_unique(
    keys: Sequence[bytes],
    fitness: np.ndarray,
    seen: set[bytes],
) -> tuple[np.ndarray, int]:
  """Indices of unseen cases, highest fitness first.

  ``seen`` gains the kept keys. A case already in ``seen``, including a
  second copy inside this population, is counted as a duplicate and is
  not stored. A non-finite score is discarded and is not stored. Every
  other case is kept.
  """
  fitness = np.asarray(fitness, dtype=np.float64).reshape(-1)
  if len(keys) != int(fitness.shape[0]):
    raise ValueError(
        f'keys has length {len(keys)}, fitness has length {fitness.shape[0]}.'
    )
  order = np.argsort(-fitness, kind='mergesort')
  kept = []
  n_duplicate = 0
  for index in order:
    index = int(index)
    if not np.isfinite(fitness[index]):
      continue
    key = keys[index]
    if key in seen:
      n_duplicate += 1
      continue
    seen.add(key)
    kept.append(index)
  return np.asarray(kept, dtype=np.int32), n_duplicate


def mask_duplicate_fitness(
    fitness: np.ndarray,
    keys: Sequence[bytes],
) -> np.ndarray:
  """Keep the best copy of each state. Later copies cannot become elites."""
  masked = np.array(np.asarray(fitness, dtype=np.float64), copy=True).reshape(-1)
  if len(keys) != int(masked.shape[0]):
    raise ValueError(
        f'keys has length {len(keys)}, fitness has length {masked.shape[0]}.'
    )
  seen: set[bytes] = set()
  for index in np.argsort(-masked, kind='mergesort'):
    key = keys[int(index)]
    if key in seen:
      masked[int(index)] = -np.inf
    else:
      seen.add(key)
  return masked


def elite_member_indices(
    masked_fitness: np.ndarray,
    elite_count: int,
) -> np.ndarray:
  """Best distinct elites, highest fitness first.

  Duplicate states were already sent to ``-inf`` and are not counted.
  """
  masked = np.asarray(masked_fitness, dtype=np.float64).reshape(-1)
  elite_count = int(elite_count)
  if elite_count < 1:
    raise ValueError(f'elite_count must be >= 1, got {elite_count}')
  if int(masked.shape[0]) < elite_count:
    raise ValueError(
        f'Cannot choose {elite_count} elites from {masked.shape[0]} individuals.'
    )
  order = np.argsort(-masked, kind='mergesort')
  chosen: list[int] = []
  for index in order:
    index = int(index)
    if not np.isfinite(masked[index]):
      continue
    chosen.append(index)
    if len(chosen) == elite_count:
      break
  if not chosen:
    raise RuntimeError('The current elite set is empty.')
  return np.asarray(chosen, dtype=np.int64)


class EliteReplacementStop:
  """Stop once a full elite set barely changes.

  The first time the set reaches ``elite_count`` distinct states, it is
  full and nothing has been replaced yet. Each later generation that is
  still full records how many of those states dropped out. The run stops
  when the last ``window`` of those generations replaced fewer than
  ``limit`` elites in total.
  """

  def __init__(
      self,
      elite_count: int,
      *,
      limit: int = DEFAULT_ELITE_REPLACE_LIMIT,
      window: int = DEFAULT_ELITE_REPLACE_WINDOW,
  ):
    if int(elite_count) < 1:
      raise ValueError(f'elite_count must be >= 1, got {elite_count}')
    if int(limit) < 1:
      raise ValueError(f'elite_replace_limit must be >= 1, got {limit}')
    if int(window) < 1:
      raise ValueError(f'elite_replace_window must be >= 1, got {window}')
    self.elite_count = int(elite_count)
    self.limit = int(limit)
    self.window = int(window)
    self._previous: list[bytes] | None = None
    self._recent: list[int] = []
    self.replaced = 0
    self.n_elites = 0

  @property
  def window_total(self) -> int:
    return int(sum(self._recent))

  def observe(self, elite_keys: Sequence[bytes]) -> bool:
    """Record this elite set. Return whether the replacement window has settled."""
    current = list(dict.fromkeys(elite_keys))
    self.n_elites = len(current)
    if self._previous is None:
      self.replaced = 0
      self._previous = current
      return False
    was_full = len(self._previous) >= self.elite_count
    self.replaced = len(set(self._previous) - set(current))
    self._previous = current
    full = self.n_elites >= self.elite_count
    if was_full and full:
      self._recent.append(self.replaced)
      if len(self._recent) > self.window:
        del self._recent[:-self.window]
    else:
      self._recent.clear()
    if len(self._recent) < self.window:
      return False
    return self.window_total < self.limit

  def format_progress(self) -> str:
    """Elite fill and how many replacements the current window holds."""
    if not self._recent:
      window = 'n/a'
    else:
      window = f'{self.window_total}/{len(self._recent)}'
    return (
        f'elites={self.n_elites}/{self.elite_count}  '
        f'elites_replaced={self.replaced}  '
        f'replace_window={window}'
    )


class FeatureResume:
  """Population and stored states from one completed generation."""

  def __init__(
      self,
      generation: int,
      population: Population,
      best: BestIndividual,
      history_records: list[GenerationRecord],
      seen: set[bytes],
      n_generations_stored: int,
  ):
    self.generation = int(generation)
    self.population = population
    self.best = best
    self.history_records = history_records
    self.seen = seen
    self.n_generations_stored = int(n_generations_stored)


def _restore_population(population: Population) -> Population:
  genomes = jax.tree.map(
      lambda leaf: jp.asarray(np.asarray(leaf)), population.genomes
  )
  return Population(
      genomes=genomes,
      policy_rng=jp.asarray(np.asarray(population.policy_rng)),
  )


def _restore_best(best: BestIndividual) -> BestIndividual:
  return BestIndividual(
      genome=jax.tree.map(
          lambda leaf: jp.asarray(np.asarray(leaf)), best.genome
      ),
      policy_rng=jp.asarray(np.asarray(best.policy_rng)),
      difficulty=jp.asarray(np.asarray(best.difficulty)),
      fitness=jp.asarray(np.asarray(best.fitness)),
  )


def _records_from_history(history) -> list[GenerationRecord]:
  generations = np.asarray(history.generation).reshape(-1)
  records = []
  for index in range(int(generations.shape[0])):
    records.append(
        GenerationRecord(
            generation=int(generations[index]),
            mean_difficulty=jp.asarray(
                np.asarray(history.mean_difficulty)[index]
            ),
            max_difficulty=jp.asarray(np.asarray(history.max_difficulty)[index]),
            min_difficulty=jp.asarray(np.asarray(history.min_difficulty)[index]),
            std_difficulty=jp.asarray(np.asarray(history.std_difficulty)[index]),
            mean_fitness=jp.asarray(np.asarray(history.mean_fitness)[index]),
            max_fitness=jp.asarray(np.asarray(history.max_fitness)[index]),
            min_fitness=jp.asarray(np.asarray(history.min_fitness)[index]),
            std_fitness=jp.asarray(np.asarray(history.std_fitness)[index]),
            best_fitness=jp.asarray(np.asarray(history.best_fitness)[index]),
            best_difficulty=jp.asarray(
                np.asarray(history.best_difficulty)[index]
            ),
        )
    )
  return records


def load_feature_resume(save_path: str, generation: int) -> FeatureResume:
  """Load the population saved at ``generation``.

  The collection pickle keeps only the latest population. ``generation``
  has to be that population's generation, and the matching state shard
  has to be on disk. Earlier shards rebuild the set of stored states.
  """
  generation = int(generation)
  if generation < 0:
    raise ValueError(f'start generation must be >= 0, got {generation}')
  if not os.path.isfile(save_path):
    raise FileNotFoundError(
        f'No checkpoint at {save_path} for generation {generation}.'
    )
  collection = load_collection(save_path, load_shards=False)
  history = collection.search_result.history
  generations = np.asarray(history.generation).reshape(-1).astype(int)
  if int(generations.size) < 1:
    raise FileNotFoundError(f'{save_path} has no saved generations.')
  saved = {int(item) for item in generations}
  last = int(generations[-1])
  if generation not in saved:
    raise FileNotFoundError(
        f'Generation {generation} is not in {save_path}. '
        f'Saved generations run from {int(generations[0])} through {last}.'
    )
  if last != generation:
    raise FileNotFoundError(
        f'Generation {generation} is in the history, but the saved '
        f'population is generation {last}. The checkpoint keeps only the '
        f'latest population, so the start generation has to be {last}.'
    )
  shard_dir = generation_shard_dir(save_path)
  manifest = _read_shard_manifest(shard_dir)
  if manifest is None:
    raise FileNotFoundError(f'No generation shards beside {save_path}.')
  entries = [
      entry
      for entry in manifest.get('shards', [])
      if int(entry['generation']) <= generation
  ]
  if not any(int(entry['generation']) == generation for entry in entries):
    raise FileNotFoundError(
        f'Generation {generation} has no shard under {shard_dir}.'
    )
  seen: set[bytes] = set()
  for entry in entries:
    path = os.path.join(shard_dir, entry['file'])
    with open(path, 'rb') as handle:
      payload = pickle.load(handle)
    if not isinstance(payload, dict) or 'qpos' not in payload:
      raise TypeError(f'{path} is not a feature-search state shard.')
    seen.update(
        state_keys(
            SimpleNamespace(qpos=payload['qpos'], qvel=payload['qvel'])
        )
    )
  return FeatureResume(
      generation=generation,
      population=_restore_population(
          collection.search_result.final_population
      ),
      best=_restore_best(collection.search_result.best),
      history_records=_records_from_history(history),
      seen=seen,
      n_generations_stored=len(entries),
  )


def run_feature_ga_search(
    bundle,
    rng: jax.Array,
    objectives: Sequence[FeatureObjective],
    *,
    elite_count: int = ELITE_COUNT,
    candidates_per_generation: int = CANDIDATES_PER_GENERATION,
    elite_replace_limit: int = DEFAULT_ELITE_REPLACE_LIMIT,
    elite_replace_window: int = DEFAULT_ELITE_REPLACE_WINDOW,
    failure_rate_every: int = DEFAULT_FAILURE_RATE_EVERY,
    max_generations: int | None = None,
    immigrant_fraction: float = 0.25,
    save_path: str | None = None,
    save_every: int = 1,
    on_suite_checkpoint=None,
    screen_search_all_fail=None,
    resume: FeatureResume | None = None,
) -> tuple:
  """Fill ``elite_count`` elites, then stop when they almost stop changing.

  Generation 0 draws the elites. Every later generation scores those elites
  plus ``candidates_per_generation`` new candidates and keeps the best
  distinct elites. The run stops when the last ``elite_replace_window``
  generations together replaced fewer than ``elite_replace_limit`` elites.
  ``max_generations`` is only a ceiling. The failure rate is not the
  fitness. When ``screen_search_all_fail`` is set, each candidate that
  passes the health screens is rolled out with the search ensemble, and
  one that fails every search policy is discarded. When
  ``on_suite_checkpoint`` is set, it receives the current elite suite every
  ``failure_rate_every`` generations, including generation 0 and the final
  generation. Stored difficulty stays NaN. ``resume`` continues from a
  completed generation: that population is scored again so the next
  generation can be bred, and the stored states are not written a second
  time.
  """
  objectives = tuple(objectives)
  if not objectives:
    raise ValueError('Feature search needs at least one objective.')
  if not 0.0 <= float(immigrant_fraction) < 1.0:
    raise ValueError(
        f'immigrant_fraction must be in [0, 1), got {immigrant_fraction}'
    )
  if max_generations is None:
    max_generations = DEFAULT_MAX_GENERATIONS
  if int(max_generations) < 1:
    raise ValueError(f'max_generations must be >= 1, got {max_generations}')
  if int(failure_rate_every) < 1:
    raise ValueError(
        f'failure_rate_every must be >= 1, got {failure_rate_every}'
    )
  failure_rate_every = int(failure_rate_every)

  discrete = False
  needs = requirements(objectives, discrete_actions=discrete)
  _reject_unsupported(bundle, objectives, needs)
  max_generations = int(max_generations)
  elites = EliteReplacementStop(
      elite_count,
      limit=elite_replace_limit,
      window=elite_replace_window,
  )

  config: GAConfig = feature_search_config(
      bundle.ga_config,
      elite_count=elite_count,
      candidates_per_generation=candidates_per_generation,
  ).replace(n_generations=max_generations)
  operators = _operator_kwargs(bundle)
  suite_fn = operators.get('genomes_to_suite_fn', _default_genomes_to_test_suite)
  random_genome_fn = operators.get('random_genome_fn')
  repair_genome_fn = operators.get('repair_genome_fn')
  crossover_fn = operators.get('crossover_genomes_fn')
  mutate_fn = operators.get('mutate_genome_fn')
  next_kwargs = {}
  if crossover_fn is not None:
    next_kwargs['crossover_genomes_fn'] = crossover_fn
  if mutate_fn is not None:
    next_kwargs['mutate_genome_fn'] = mutate_fn

  n_immigrants = int(round(float(immigrant_fraction) * config.population_size))
  n_immigrants = min(
      n_immigrants, max(0, int(config.population_size - config.elite_count))
  )
  sampler = sampler_from_bundle(bundle)
  immigrant_sampler = _immigrant_sampler(sampler, bundle.environment)
  subspaces = _load_subspaces(bundle) if 'gait' in needs else None
  attribution_config = bundle.attribution_config or AttributionConfig()

  seen: set[bytes] = set()
  n_duplicate_skipped = 0
  n_zero_action_fails = 0
  n_search_all_fails = 0
  suite_reports: list[dict] = []
  turnover_window = {
      'discarded': 0,
      'replaced': 0,
      'added': 0,
      'elites_replaced': 0,
  }
  last_reported_generation: int | None = None
  current_states = None
  current_n_cases = 0
  generation_records: list = []
  history_records: list[GenerationRecord] = []
  best: BestIndividual | None = None
  n_generations_stored = 0
  n_generations_run = 0
  shard_dir = generation_shard_dir(save_path) if save_path else None
  if shard_dir:
    os.makedirs(shard_dir, exist_ok=True)

  objective_text = ', '.join(
      f'{item.direction_name} {item.name}' for item in objectives
  )
  n_candidates = int(config.population_size - config.elite_count)
  print(
      f'Feature GA {bundle.environment}/{bundle.algorithm}  '
      f'stop when fewer than {elites.limit} elites are replaced '
      f'across {elites.window} generations  '
      f'max_generations={max_generations}  '
      f'elites={config.elite_count}  '
      f'candidates={n_candidates} assessed each later generation  '
      f'objectives: {objective_text}',
      flush=True,
  )
  print(
      'Selection fitness is ' + fitness_definition(objectives),
      flush=True,
  )
  if screen_search_all_fail is not None:
    print(
        'Search screen: candidates that pass the health screens are rolled '
        'out with the search ensemble. A candidate that fails every search '
        'policy is discarded. The suite report reuses those search '
        'failures and rolls only the evaluation ensemble.',
        flush=True,
    )
  print(ELITE_STOP, flush=True)
  print(
      f'Zero-action screen: horizon {ZERO_ACTION_HORIZON}, '
      f'batch {ZERO_ACTION_BATCH}. '
      'A candidate that leaves the healthy set is discarded. '
      'Hopper and Walker2d also discard a candidate whose limited joints '
      'start more than 0.18 rad past their stops.',
      flush=True,
  )

  if resume is None:
    rng, init_rng = jax.random.split(rng)
    init_kwargs = {}
    if random_genome_fn is not None:
      init_kwargs['random_genome_fn'] = random_genome_fn
    if repair_genome_fn is not None:
      init_kwargs['repair_genome_fn'] = repair_genome_fn
    population = init_population(
        init_rng,
        bundle.nq,
        bundle.nv,
        config.replace(population_size=int(config.elite_count)),
        **init_kwargs,
    )
  else:
    if int(resume.generation) + 1 >= max_generations:
      raise ValueError(
          f'start generation {int(resume.generation)} is already at the '
          f'max_generations ceiling {max_generations}.'
      )
    population = resume.population
    best = resume.best
    history_records = list(resume.history_records)
    seen = set(resume.seen)
    n_generations_stored = int(resume.n_generations_stored)
    rng = jax.random.fold_in(rng, int(resume.generation) + 1)
    print(
        f'Resuming from generation {int(resume.generation)} '
        f'({len(seen)} stored states, '
        f'{n_generations_stored} shards). Scoring that population, then '
        f'continuing through generation {max_generations - 1}. The '
        f'elite-replacement window starts again from this population.',
        flush=True,
    )

  def report_suite(generation: int, *, force: bool) -> None:
    """Measure the current suite on schedule. Rates come back through the callback."""
    nonlocal last_reported_generation
    generation = int(generation)
    if last_reported_generation == generation:
      return
    if not force and generation % failure_rate_every != 0:
      return
    if current_n_cases > ELITE_COUNT:
      raise RuntimeError(
          f'The current elite suite has {current_n_cases} cases; evaluation '
          f'is limited to {ELITE_COUNT}.'
      )
    if on_suite_checkpoint is None:
      qpos = qvel = policy_rng = None
    else:
      if current_states is None:
        raise TypeError(
            'Suite failure rates are recorded for locomotion states only.'
        )
      qpos, qvel, policy_rng = current_states
      if int(np.asarray(qpos).shape[0]) != int(current_n_cases):
        raise RuntimeError(
            f'The current suite has {current_n_cases} distinct tests, but '
            f'{np.asarray(qpos).shape[0]} states were prepared for rollout.'
        )
    n_cases = int(current_n_cases)
    report = {
        'generation': generation,
        'candidates_discarded': int(turnover_window['discarded']),
        'tests_replaced': int(turnover_window['replaced']),
        'tests_added': int(turnover_window['added']),
        'elites_replaced': int(turnover_window['elites_replaced']),
        'n_cases': int(n_cases),
    }
    if on_suite_checkpoint is not None:
      on_suite_checkpoint(report, qpos, qvel, policy_rng)
    suite_reports.append(report)
    turnover_window['discarded'] = 0
    turnover_window['replaced'] = 0
    turnover_window['added'] = 0
    turnover_window['elites_replaced'] = 0
    last_reported_generation = generation
    print(
        f'suite gen {generation:4d}  n_cases={report["n_cases"]}  '
        f'search_fail_rate={_format_rate(report.get("search_fail_rate"))}  '
        f'eval_fail_rate={_format_rate(report.get("eval_fail_rate"))}  '
        f'discarded={report["candidates_discarded"]}  '
        f'replaced={report["tests_replaced"]}  '
        f'added={report["tests_added"]}  '
        f'elites_replaced={report["elites_replaced"]}',
        flush=True,
    )

  def evaluate(population: Population, generation: int, *, archive: bool = True):
    nonlocal best, n_duplicate_skipped, n_generations_stored
    nonlocal n_zero_action_fails, n_search_all_fails
    nonlocal current_states, current_n_cases
    suite = suite_fn(population.genomes, bundle.qpos0, population.policy_rng)
    n_population = int(population.policy_rng.shape[0])
    if hasattr(suite, 'qpos'):
      step_fail = zero_action_fall_mask(bundle.env, suite.qpos, suite.qvel)
    else:
      step_fail = np.zeros(n_population, dtype=bool)
    n_step_fail = int(np.sum(step_fail))
    if archive:
      n_zero_action_fails += n_step_fail
    if hasattr(suite, 'qpos'):
      passed = healthy_initial_mask(bundle.env, suite.qpos, suite.qvel)
    else:
      passed = np.ones(n_population, dtype=bool)
    passed = passed & ~step_fail
    all_fail = np.zeros(n_population, dtype=bool)
    if (
        screen_search_all_fail is not None
        and hasattr(suite, 'qpos')
        and np.any(passed)
    ):
      survivor = np.flatnonzero(passed)
      failed_all = np.asarray(
          screen_search_all_fail(
              np.asarray(suite.qpos)[survivor],
              np.asarray(suite.qvel)[survivor],
              np.asarray(suite.policy_rng)[survivor],
          ),
          dtype=bool,
      ).reshape(-1)
      if int(failed_all.shape[0]) != int(survivor.shape[0]):
        raise RuntimeError(
            f'Search screen returned {failed_all.shape[0]} rows for '
            f'{survivor.shape[0]} candidates.'
        )
      all_fail[survivor] = failed_all
    n_all_fail = int(np.sum(all_fail))
    if archive:
      n_search_all_fails += n_all_fail
    passed = passed & ~all_fail
    signals = _gather_signals(
        suite,
        bundle,
        needs,
        subspaces=subspaces,
        discrete=discrete,
    )
    columns = feature_columns(
        signals,
        objectives,
        algorithm=bundle.algorithm,
        environment=bundle.environment,
    )
    fitness_np = np.asarray(
        combine_feature_fitness(columns, objectives), dtype=np.float32
    )
    healthy = passed
    n_unhealthy = int(n_population - int(np.sum(passed)) - n_all_fail)
    fitness_np = np.where(healthy, fitness_np, -np.inf).astype(np.float32)
    fitness = jp.asarray(fitness_np)
    nan_difficulty = jp.full(fitness.shape, jp.nan)
    keys = state_keys(suite)
    masked = mask_duplicate_fitness(fitness_np, keys)
    elite_index = elite_member_indices(masked, config.elite_count)
    healthy_elite = elite_index[healthy[elite_index]]
    if healthy_elite.size:
      elite_mean = _feature_value_mean(
          {
              name: np.asarray(values, dtype=np.float64)[healthy_elite]
              for name, values in columns.items()
          },
          objectives,
      )
    else:
      elite_mean = float('nan')
    settled = elites.observe([keys[int(index)] for index in elite_index])
    current_n_cases = int(elite_index.shape[0])
    if hasattr(suite, 'qpos'):
      current_states = (
          np.array(np.asarray(suite.qpos)[elite_index], copy=True),
          np.array(np.asarray(suite.qvel)[elite_index], copy=True),
          np.array(np.asarray(suite.policy_rng)[elite_index], copy=True),
      )
    else:
      current_states = None
    n_scored = int(fitness_np.shape[0])
    if archive:
      kept, n_duplicate = select_unique(keys, fitness_np, seen)
      n_duplicate_skipped += int(n_duplicate)
      n_kept = int(kept.size)
      discarded, replaced, added = suite_turnover(
          generation,
          kept,
          n_scored,
          config.elite_count,
          0 if int(generation) == 0 else n_immigrants,
      )
      turnover_window['discarded'] += int(discarded)
      turnover_window['replaced'] += int(replaced)
      turnover_window['added'] += int(added)
      turnover_window['elites_replaced'] += int(elites.replaced)
    else:
      n_duplicate = 0
      n_kept = 0
      discarded = replaced = added = 0
    if archive and n_kept > 0 and save_path:
      if not hasattr(suite, 'qpos'):
        raise TypeError(
            'Kept states are stored as qpos and qvel. This suite has neither.'
        )
      kept_index = np.asarray(kept, dtype=np.int64)
      shard = save_case_states(
          np.asarray(suite.qpos)[kept_index],
          np.asarray(suite.qvel)[kept_index],
          save_path,
          generation,
          counts=np.asarray([len(seen)], dtype=np.int32),
      )
      print(
          f'      wrote {shard}  n_cases={int(kept_index.shape[0])}',
          flush=True,
      )
      n_generations_stored += 1
    if best is None:
      index = int(np.argmax(fitness_np))
      best = BestIndividual(
          genome=jax.tree.map(lambda x: x[index], population.genomes),
          policy_rng=population.policy_rng[index],
          difficulty=jp.nan,
          fitness=fitness[index],
      )
    else:
      best = _update_best(best, population, fitness, nan_difficulty)
    scored = jp.asarray(fitness_np[healthy]) if np.any(healthy) else fitness
    if archive:
      history_records.append(
          _compute_generation_record(
              generation,
              scored,
              jp.full(scored.shape, jp.nan),
              best,
          )
      )
    healthy_columns = {
        name: np.asarray(values, dtype=np.float64)[healthy]
        for name, values in columns.items()
    }
    if np.any(healthy) and healthy_columns:
      fitness_mean = _feature_value_mean(healthy_columns, objectives)
    else:
      fitness_mean = float('nan')
    means = '  '.join(
        _format_feature(item, healthy_columns[item.name])
        for item in objectives
        if healthy_columns[item.name].size
    )
    elite_mean_text = (
        f'{elite_mean:.4f}' if np.isfinite(elite_mean) else 'n/a'
    )
    if archive:
      kept_text = (
          f'kept {n_kept}/{n_scored}  duplicates={int(n_duplicate)}'
      )
    else:
      kept_text = f'rescore {n_scored} (not stored)'
    print(
        f'gen {generation:4d}  fitness mean={fitness_mean:.4f}  '
        f'elite_mean={elite_mean_text}  '
        f'max={float(np.max(fitness_np)):.4f}  {means}  '
        f'{kept_text}  '
        f'unique={len(seen)}  '
        f'unhealthy={n_unhealthy}  '
        f'zero_action_fails={n_step_fail}  '
        f'all_fail={n_all_fail}  '
        f'discarded={int(discarded)}  replaced={int(replaced)}  '
        f'added={int(added)}  '
        f'{elites.format_progress()}',
        flush=True,
    )
    # The suite rollout is the memory peak. Drop this generation's policy
    # outputs before it starts.
    del suite, signals, columns
    if archive:
      report_suite(generation, force=False)
    return jp.asarray(masked), settled

  if resume is None:
    breed_fitness, settled = evaluate(population, 0)
    n_generations_run = 1
    if save_path:
      _checkpoint(
          best,
          population,
          history_records,
          config,
          attribution_config,
          bundle.policy_ids,
          save_path,
          len(seen),
          n_generations_stored,
      )
  else:
    breed_fitness, settled = evaluate(
        population, int(resume.generation), archive=False
    )
    n_generations_run = int(resume.generation) + 1
    last_reported_generation = int(resume.generation)

  for step in range(n_generations_run, max_generations):
    if settled:
      break
    rng, step_rng = jax.random.split(rng)
    population = next_generation(
        step_rng, population, breed_fitness, config, **next_kwargs
    )
    if n_immigrants > 0:
      rng, immigrants = sample_population(
          rng, bundle, immigrant_sampler, n_immigrants
      )
      population = _replace_tail(population, immigrants)
    breed_fitness, settled = evaluate(population, step)
    n_generations_run += 1
    if save_path and save_every > 0 and step % int(save_every) == 0:
      _checkpoint(
          best,
          population,
          history_records,
          config,
          attribution_config,
          bundle.policy_ids,
          save_path,
          len(seen),
          n_generations_stored,
      )
  report_suite(n_generations_run - 1, force=True)
  if settled:
    print(
        f'Fewer than {elites.limit} elites replaced across the last '
        f'{elites.window} generations ({elites.window_total} replaced, '
        f'{elites.n_elites} elites, {len(seen)} unique states, '
        f'{n_zero_action_fails} zero-action fails, '
        f'{n_search_all_fails} search all-fails); stopping.',
        flush=True,
    )
  else:
    print(
        f'Stopped at the generation ceiling ({n_generations_run} generations, '
        f'{elites.n_elites} elites, {len(seen)} unique states, '
        f'{n_zero_action_fails} zero-action fails, '
        f'{n_search_all_fails} search all-fails).',
        flush=True,
    )

  placeholder = jp.full((population.policy_rng.shape[0],), jp.nan)
  collection = _make_collection(
      generation_records,
      best,
      population,
      placeholder,
      history_records,
      config,
      attribution_config,
      bundle.policy_ids,
      shard_dir=shard_dir,
  )
  if save_path:
    _checkpoint(
        best,
        population,
        history_records,
        config,
        attribution_config,
        bundle.policy_ids,
        save_path,
        len(seen),
        n_generations_stored,
    )
  info = {
      'search_kind': 'feature',
      'executed_search_tests': screen_search_all_fail is not None,
      'stopped_because': (
          'elite_replacements' if settled else 'max_generations'
      ),
      'elite_stop': ELITE_STOP,
      'elite_replace_limit': int(elites.limit),
      'elite_replace_window': int(elites.window),
      'elite_replace_window_total': int(elites.window_total),
      'elites_replaced': int(elites.replaced),
      'n_elites': int(elites.n_elites),
      'n_unique_states': int(len(seen)),
      'n_duplicate_skipped': int(n_duplicate_skipped),
      'zero_action_fails': int(n_zero_action_fails),
      'search_all_fails': int(n_search_all_fails),
      'zero_action_horizon': int(ZERO_ACTION_HORIZON),
      'zero_action_batch': int(ZERO_ACTION_BATCH),
      'unique_states': True,
      'failure_rate_every': int(failure_rate_every),
      'suite_report': SUITE_REPORT,
      'suite_reports': suite_reports,
      'features': [
          {'name': item.name, 'direction': item.direction_name}
          for item in objectives
      ],
      'fitness': fitness_definition(objectives),
      'max_generations': int(max_generations),
      'population_size': int(config.population_size),
      'elite_count': int(config.elite_count),
      'candidates_per_generation': int(
          config.population_size - config.elite_count
      ),
      'n_generations_run': int(n_generations_run),
      'n_generations_stored': int(n_generations_stored),
      'immigrant_fraction': float(immigrant_fraction),
      'n_immigrants': int(n_immigrants),
      'shard_dir': shard_dir,
  }
  return collection, info


def _feature_value_mean(
    columns: dict[str, np.ndarray],
    objectives: Sequence[FeatureObjective],
) -> float:
  """Mean raw feature value. Several features are averaged one mean at a time."""
  means = [
      float(np.mean(np.asarray(columns[item.name], dtype=np.float64)))
      for item in objectives
  ]
  return float(np.mean(means))


def _format_feature(objective: FeatureObjective, values: np.ndarray) -> str:
  """Mean, and the extreme the objective is pushing toward."""
  values = np.asarray(values, dtype=np.float64)
  extreme = np.max(values) if objective.direction > 0 else np.min(values)
  return (
      f'{objective.direction_name} {objective.name} '
      f'mean={float(np.mean(values)):.4g} '
      f'extreme={float(extreme):.4g}'
  )


def _reject_unsupported(bundle, objectives, needs: frozenset[str]) -> None:
  if bundle.kind != 'locomotion' and needs & _LOCOMOTION_TAGS:
    names = [
        item.name
        for item in objectives
        if FEATURE_NEEDS[item.name] & _LOCOMOTION_TAGS
    ]
    raise ValueError(
        f'{bundle.environment} has no locomotion state to measure. '
        f'Cannot search on {names}.'
    )
  if ('policy' in needs or 'gait' in needs) and bundle.policy_outputs_fn is None:
    raise RuntimeError(
        'This objective needs initial-state policy outputs, and the '
        'bundle was built without them.'
    )
  if 'ig' in needs and bundle.extract_attributions_fn is None:
    raise RuntimeError(
        'This objective needs integrated gradients, and the bundle was '
        'built without them.'
    )


def _load_subspaces(bundle):
  from analysis.gait_subspace import load_gait_subspaces_for_policies

  return load_gait_subspaces_for_policies(
      bundle.environment,
      bundle.algorithm,
      bundle.policy_ids,
      algorithms=bundle.policy_algorithms,
  )


def _as_numpy(value):
  if value is None:
    return None
  return np.asarray(value)


def _is_discrete(actions: np.ndarray | None) -> bool:
  if actions is None or actions.ndim != 3:
    return False
  totals = np.sum(np.asarray(actions), axis=-1)
  return bool(np.allclose(totals, 1.0, atol=1e-3))


def _gather_signals(
    suite,
    bundle,
    needs: frozenset[str],
    *,
    subspaces,
    discrete: bool,
) -> PopulationSignals:
  signals = PopulationSignals()
  if 'qvel' in needs:
    signals.qvel = np.asarray(suite.qvel)
  if 'ig' in needs:
    result = bundle.extract_attributions_fn(suite)
    if result is None:
      raise RuntimeError(
          'Integrated gradients were not built for this bundle.'
      )
    signals.state_ig_qpos = _as_numpy(result.state_integrated_gradient_qpos)
    signals.state_ig_qvel = _as_numpy(result.state_integrated_gradient_qvel)
    signals.obs_ig = _as_numpy(result.observation_integrated_gradient)
  if 'policy' in needs or 'gait' in needs:
    outputs = bundle.policy_outputs_fn(suite)
    if 'actions' in outputs:
      signals.actions = _as_numpy(outputs['actions'])
    if 'action_loc' in outputs:
      signals.action_loc = _as_numpy(outputs['action_loc'])
    if 'action_scale' in outputs:
      signals.action_scale = _as_numpy(outputs['action_scale'])
    if 'values' in outputs:
      signals.values = _as_numpy(outputs['values'])
    if 'q_values' in outputs:
      signals.q_values = _as_numpy(outputs['q_values'])
  signals.discrete = bool(discrete) or _is_discrete(signals.actions)
  if 'gait' in needs:
    from analysis.gait_subspace import decompose_ensemble_actions

    gait, residual = decompose_ensemble_actions(signals.actions, subspaces)
    signals.gait_actions = gait
    signals.nongait_actions = residual
  return signals


def _checkpoint(
    best,
    population,
    history_records,
    config,
    attribution_config,
    policy_ids,
    save_path: str,
    n_unique: int,
    n_generations_stored: int,
) -> None:
  difficulty = jp.full((population.policy_rng.shape[0],), jp.nan)
  _checkpoint_sidecar(
      best,
      population,
      difficulty,
      history_records,
      config,
      attribution_config,
      policy_ids,
      np.asarray([int(n_unique)], dtype=np.int32),
      save_path,
      n_generations_stored,
  )
