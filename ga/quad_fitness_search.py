"""Four copies of the feature search, differing only in fitness.

Generation 0 is one population, drawn from the experiment seed. Every
search scores that population. From generation 1, each search breeds
only from its own population, so the four populations diverge. The
tournament key and the immigrant draw still come from that same seed,
and the same draw is used for every search: fitness is what separates
them. ``experiment_seed`` in ``ga.search_cli`` builds the key from the
experiment number.

Each candidate that passes the health screens is rolled out with every
search policy. A candidate that fails all of them is discarded, so every
stored test is solved by at least one search policy. The same rollout is
the empirical difficulty of the search ensemble.

The four fitnesses, each maximized, are:

- the requested disagreement feature, at its absolute value
- the fraction of search policies that fail the candidate
- the negation of the best search policy's state value
- the action-distribution entropy of that same policy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import jax
import numpy as np
from jax import numpy as jp

from attributions.types import AttributionConfig
from ga.binned import (
    _immigrant_sampler,
    _operator_kwargs,
    _replace_tail,
    state_keys,
)
from ga.collect import generation_shard_dir, save_case_states
from ga.feature_objectives import (
    DISAGREEMENT_FEATURES,
    FeatureObjective,
    combine_feature_fitness,
    feature_columns,
    feature_slug,
    requirements,
)
from ga.feature_search import (
    FeatureResume,
    DEFAULT_ELITE_REPLACE_LIMIT,
    DEFAULT_ELITE_REPLACE_WINDOW,
    DEFAULT_FAILURE_RATE_EVERY,
    DEFAULT_MAX_GENERATIONS,
    ELITE_COUNT,
    ELITE_STOP,
    SUITE_REPORT,
    CANDIDATES_PER_GENERATION,
    EliteReplacementStop,
    _checkpoint,
    _compute_generation_record,
    _format_feature,
    _format_rate,
    _gather_signals,
    _load_subspaces,
    _make_collection,
    _update_best,
    elite_member_indices,
    feature_search_config,
    mask_duplicate_fitness,
    select_unique,
    suite_turnover,
)
from ga.ga import BestIndividual, GAConfig, GenerationRecord, Population
from ga.healthy import healthy_initial_mask, zero_action_fall_mask
from ga.search import init_population, next_generation
from ga.stratified_init import sample_population, sampler_from_bundle
from analysis.disagreement import value_estimates_np
from mutators.humanoid import genomes_to_test_suite as _default_genomes_to_test_suite

FEATURE = 'feature'
DIFFICULTY = 'difficulty'
VALUE = 'value'
ENTROPY = 'entropy'
LANE_KINDS = (FEATURE, DIFFICULTY, VALUE, ENTROPY)
LANE_NAMES = {
    FEATURE: 'feature disagreement',
    DIFFICULTY: 'search difficulty',
    VALUE: 'negative best value',
    ENTROPY: 'action entropy',
}
FIXED_SLUGS = {
    DIFFICULTY: 'search_difficulty',
    VALUE: 'negative_best_value',
    ENTROPY: 'action_entropy',
}


def quad_slugs(objectives: Sequence[FeatureObjective]) -> dict[str, str]:
  """Filesystem token for each of the four searches."""
  if not objectives or any(item.name not in DISAGREEMENT_FEATURES for item in objectives):
    raise ValueError(
        'The disagreement search takes disagreement features. '
        f'Got {[item.name for item in objectives]}.'
    )
  slugs = {FEATURE: feature_slug(objectives)}
  slugs.update(FIXED_SLUGS)
  if slugs[FEATURE] in FIXED_SLUGS.values():
    raise ValueError(
        f'Feature slug {slugs[FEATURE]} collides with another quad search.'
    )
  return slugs


def quad_collection_path(
    environment: str,
    algorithm: str,
    exp_number: int,
    rollout_length: int,
    slug: str,
) -> str:
  """Collection pickle for one fitness inside a quad experiment."""
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'quad_search',
      slug,
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(exp_number)}_ga_collection.pkl',
  )


def quad_suite_paths(
    environment: str,
    algorithm: str,
    exp_number: int,
    rollout_length: int,
    slug: str,
) -> tuple[str, str]:
  """Suite pickle and failure-rate table for one quad fitness."""
  directory = os.path.join(
      'test_suites',
      'quad_search',
      slug,
      environment,
      algorithm,
      f'rollout_length={int(rollout_length)}',
      f'experiment_{int(exp_number)}',
  )
  return (
      os.path.join(directory, 'suite.pkl'),
      os.path.join(directory, 'failure_rates.csv'),
  )


def gaussian_action_entropy(scale: np.ndarray) -> np.ndarray:
  """Differential entropy, in nats, of a diagonal Gaussian with scale σ.

  ``scale`` has shape ``(N, n_actuators)``. The per-actuator term is
  ``log σ``, plus the constant ``0.5 * log(2 π e)`` on each actuator.
  """
  sigma = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
  if sigma.ndim != 2:
    raise ValueError(
        f'Gaussian scale must have shape (N, n_actuators), got {sigma.shape}.'
    )
  n_actuators = int(sigma.shape[-1])
  return (
      0.5 * n_actuators * np.log(2.0 * np.pi * np.e)
      + np.sum(np.log(sigma), axis=-1)
  )


def action_entropy_np(
    action_scale: np.ndarray,
    policy_index: int,
    *,
    discrete: bool,
) -> np.ndarray:
  """Action-distribution entropy of one policy. Shape ``(N,)``."""
  scale = np.asarray(action_scale, dtype=np.float64)
  if scale.ndim < 2:
    raise ValueError(
        f'action_scale must be (P, N) or (P, N, n_actuators), got {scale.shape}.'
    )
  policy_index = int(policy_index)
  if policy_index < 0 or policy_index >= int(scale.shape[0]):
    raise IndexError(
        f'Policy index {policy_index} is outside the ensemble of '
        f'{scale.shape[0]}.'
    )
  row = scale[policy_index]
  if discrete:
    flat = row.reshape(int(row.shape[0]), -1)
    if int(flat.shape[-1]) != 1:
      raise ValueError(
          f'Discrete action entropy must be one value per case, got {row.shape}.'
      )
    return flat[:, 0]
  if row.ndim == 1:
    row = row.reshape(-1, 1)
  return gaussian_action_entropy(row)


def negative_best_value_np(
    values: np.ndarray,
    q_values: np.ndarray | None,
    *,
    algorithm: str,
    policy_index: int,
) -> np.ndarray:
  """Negation of one policy's state value. Shape ``(N,)``. Higher is worse for the policy."""
  estimates = value_estimates_np(values, q_values, algorithm=algorithm)
  policy_index = int(policy_index)
  if policy_index < 0 or policy_index >= int(estimates.shape[0]):
    raise IndexError(
        f'Policy index {policy_index} is outside the ensemble of '
        f'{estimates.shape[0]}.'
    )
  return -estimates[policy_index]


def search_difficulty_np(fail_rate: np.ndarray) -> np.ndarray:
  """Fraction of search policies that fail each candidate. Shape ``(N,)``."""
  return np.asarray(fail_rate, dtype=np.float64).reshape(-1)


@dataclass
class QuadLane:
  """One of the four searches. Rollout hooks belong to its suite reporter."""

  name: str
  slug: str
  kind: str
  save_path: str | None
  search_failures: Callable
  store_search_failures: Callable
  on_suite_checkpoint: Callable | None


def _require_passed_finite(values: np.ndarray, passed: np.ndarray, name: str) -> None:
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  if int(values.shape[0]) != int(passed.shape[0]):
    raise ValueError(
        f'{name} has length {values.shape[0]}, expected {passed.shape[0]}.'
    )
  bad = passed & ~np.isfinite(values)
  if np.any(bad):
    raise ValueError(
        f'{name} is non-finite for {int(np.sum(bad))} candidates that '
        f'passed the screens.'
    )


def _raw_fitness(lane: QuadLane, assessment, objectives, algorithm: str, environment: str, policy_index: int) -> np.ndarray:
  if lane.kind == FEATURE:
    columns = feature_columns(
        assessment.signals,
        objectives,
        algorithm=algorithm,
        environment=environment,
    )
    assessment.feature_columns = columns
    return np.asarray(
        combine_feature_fitness(columns, objectives), dtype=np.float64
    ).reshape(-1)
  if lane.kind == DIFFICULTY:
    return search_difficulty_np(assessment.fail_rate)
  if lane.kind == VALUE:
    values = negative_best_value_np(
        assessment.signals.values,
        assessment.signals.q_values,
        algorithm=algorithm,
        policy_index=policy_index,
    )
    _require_passed_finite(values, assessment.passed, lane.name)
    return values
  if lane.kind == ENTROPY:
    values = action_entropy_np(
        assessment.signals.action_scale,
        policy_index,
        discrete=bool(assessment.signals.discrete),
    )
    _require_passed_finite(values, assessment.passed, lane.name)
    return values
  raise ValueError(f'Unknown fitness {lane.kind!r}.')


class _Assessment:
  def __init__(self):
    self.suite = None
    self.passed = None
    self.fail_rate = None
    self.signals = None
    self.keys = None
    self.n_step_fail = 0
    self.n_all_fail = 0
    self.n_unhealthy = 0
    self.survivor_qpos = None
    self.survivor_qvel = None
    self.survivor_policy_rng = None
    self.failure_matrix = None
    self.feature_columns = None


def _assess(population: Population, suite_fn, bundle, needs, subspaces, discrete: bool, search_failures) -> _Assessment:
  """Health screens, then a full search-ensemble rollout of the survivors."""
  assessment = _Assessment()
  suite = suite_fn(population.genomes, bundle.qpos0, population.policy_rng)
  if not hasattr(suite, 'qpos'):
    raise TypeError(
        'Quad fitness search rolls out locomotion states with qpos and qvel.'
    )
  n_population = int(population.policy_rng.shape[0])
  step_fail = zero_action_fall_mask(bundle.env, suite.qpos, suite.qvel)
  assessment.n_step_fail = int(np.sum(step_fail))
  passed = healthy_initial_mask(bundle.env, suite.qpos, suite.qvel)
  passed = np.asarray(passed, dtype=bool) & ~np.asarray(step_fail, dtype=bool)
  fail_rate = np.full(n_population, np.nan, dtype=np.float64)
  survivor = np.flatnonzero(passed)
  if survivor.size:
    qpos = np.asarray(suite.qpos)[survivor]
    qvel = np.asarray(suite.qvel)[survivor]
    policy_rng = np.asarray(suite.policy_rng)[survivor]
    failed = np.asarray(
        search_failures(qpos, qvel, policy_rng), dtype=bool
    )
    if failed.ndim != 2 or int(failed.shape[1]) != int(survivor.shape[0]):
      raise RuntimeError(
          f'Search screen returned shape {failed.shape} for '
          f'{survivor.shape[0]} candidates.'
      )
    fail_rate[survivor] = np.mean(failed, axis=0)
    all_fail = np.all(failed, axis=0)
    assessment.n_all_fail = int(np.sum(all_fail))
    passed = passed.copy()
    passed[survivor] = ~all_fail
    assessment.survivor_qpos = qpos
    assessment.survivor_qvel = qvel
    assessment.survivor_policy_rng = policy_rng
    assessment.failure_matrix = failed
  assessment.n_unhealthy = int(
      n_population - int(np.sum(passed)) - assessment.n_all_fail
  )
  assessment.suite = suite
  assessment.passed = passed
  assessment.fail_rate = fail_rate
  assessment.signals = _gather_signals(
      suite,
      bundle,
      needs,
      subspaces=subspaces,
      discrete=discrete,
  )
  assessment.keys = state_keys(suite)
  return assessment


class _LaneState:
  def __init__(self, spec: QuadLane, elite_count: int, limit: int, window: int):
    self.spec = spec
    self.seen: set[bytes] = set()
    self.history_records: list[GenerationRecord] = []
    self.best: BestIndividual | None = None
    self.stop = EliteReplacementStop(elite_count, limit=limit, window=window)
    self.population: Population | None = None
    self.breed_fitness = None
    self.settled = False
    self.last_generation = None
    self.n_generations_stored = 0
    self.n_duplicate_skipped = 0
    self.n_zero_action_fails = 0
    self.n_search_all_fails = 0
    self.suite_reports: list[dict] = []
    self.turnover_window = {
        'discarded': 0,
        'replaced': 0,
        'added': 0,
        'elites_replaced': 0,
    }
    self.last_reported_generation: int | None = None
    self.current_states = None
    self.current_n_cases = 0
    self.shard_dir = (
        generation_shard_dir(spec.save_path) if spec.save_path else None
    )
    self.carry_on = True


def _prepare_lane(
    state: _LaneState,
    assessment: _Assessment,
    generation: int,
    *,
    objectives,
    algorithm: str,
    environment: str,
    policy_index: int,
    elite_count: int,
    n_immigrants: int,
    archive: bool = True,
) -> dict:
  """Score this lane and choose its elites. The suite rollout happens later."""
  raw = _raw_fitness(
      state.spec, assessment, objectives, algorithm, environment, policy_index
  )
  passed = assessment.passed
  fitness_np = np.where(passed, raw, -np.inf).astype(np.float32)
  masked = mask_duplicate_fitness(fitness_np, assessment.keys)
  elite_index = elite_member_indices(masked, elite_count)
  settled = state.stop.observe(
      [assessment.keys[int(index)] for index in elite_index]
  )
  suite = assessment.suite
  state.current_n_cases = int(elite_index.shape[0])
  state.current_states = (
      np.array(np.asarray(suite.qpos)[elite_index], copy=True),
      np.array(np.asarray(suite.qvel)[elite_index], copy=True),
      np.array(np.asarray(suite.policy_rng)[elite_index], copy=True),
  )
  healthy_elite = elite_index[passed[elite_index]]
  if healthy_elite.size:
    elite_mean = float(np.mean(np.asarray(raw, dtype=np.float64)[healthy_elite]))
  else:
    elite_mean = float('nan')
  if passed.any():
    fitness_mean = float(np.mean(np.asarray(raw, dtype=np.float64)[passed]))
  else:
    fitness_mean = float('nan')
  if not archive:
    return {
        'generation': int(generation),
        'fitness_np': fitness_np,
        'masked': masked,
        'settled': False,
        'kept': np.array([], dtype=np.int64),
        'n_duplicate': 0,
        'n_kept': 0,
        'n_scored': int(fitness_np.shape[0]),
        'discarded': 0,
        'replaced': 0,
        'added': 0,
        'kept_qpos': None,
        'kept_qvel': None,
        'elite_mean': elite_mean,
        'fitness_mean': fitness_mean,
        'raw': raw,
    }
  kept, n_duplicate = select_unique(
      assessment.keys, fitness_np, state.seen
  )
  state.n_duplicate_skipped += int(n_duplicate)
  n_kept = int(kept.size)
  n_scored = int(fitness_np.shape[0])
  discarded, replaced, added = suite_turnover(
      generation,
      kept,
      n_scored,
      elite_count,
      0 if int(generation) == 0 else n_immigrants,
  )
  state.turnover_window['discarded'] += int(discarded)
  state.turnover_window['replaced'] += int(replaced)
  state.turnover_window['added'] += int(added)
  state.turnover_window['elites_replaced'] += int(state.stop.replaced)
  kept_qpos = kept_qvel = None
  if n_kept > 0 and state.spec.save_path:
    kept_index = np.asarray(kept, dtype=np.int64)
    kept_qpos = np.array(np.asarray(suite.qpos)[kept_index], copy=True)
    kept_qvel = np.array(np.asarray(suite.qvel)[kept_index], copy=True)
  return {
      'generation': int(generation),
      'fitness_np': fitness_np,
      'masked': masked,
      'settled': settled,
      'kept': kept,
      'n_duplicate': int(n_duplicate),
      'n_kept': n_kept,
      'n_scored': n_scored,
      'discarded': int(discarded),
      'replaced': int(replaced),
      'added': int(added),
      'kept_qpos': kept_qpos,
      'kept_qvel': kept_qvel,
      'elite_mean': elite_mean,
      'fitness_mean': fitness_mean,
      'raw': raw,
  }


def _archive_lane(
    state: _LaneState,
    prepared: dict,
    assessment: _Assessment,
    population: Population,
    *,
    failure_rate_every: int,
    config: GAConfig,
    attribution_config,
    policy_ids,
    save_every: int,
    archive: bool = True,
) -> None:
  generation = int(prepared['generation'])
  if not archive:
    state.breed_fitness = jp.asarray(prepared['masked'])
    print(
        f'{state.spec.slug} gen {generation:4d}  '
        f'rescore {prepared["n_scored"]} (not stored)  '
        f'unique={len(state.seen)}',
        flush=True,
    )
    return
  if (
      assessment.failure_matrix is not None
      and state.spec.store_search_failures is not None
  ):
    state.spec.store_search_failures(
        assessment.survivor_qpos,
        assessment.survivor_qvel,
        assessment.survivor_policy_rng,
        assessment.failure_matrix,
    )
  state.n_zero_action_fails += int(assessment.n_step_fail)
  state.n_search_all_fails += int(assessment.n_all_fail)
  if prepared['kept_qpos'] is not None:
    shard = save_case_states(
        prepared['kept_qpos'],
        prepared['kept_qvel'],
        state.spec.save_path,
        generation,
        counts=np.asarray([len(state.seen)], dtype=np.int32),
    )
    print(
        f'      wrote {shard}  n_cases={int(prepared["kept_qpos"].shape[0])}',
        flush=True,
    )
    state.n_generations_stored += 1
  fitness_np = prepared['fitness_np']
  fitness = jp.asarray(fitness_np)
  nan_difficulty = jp.full(fitness.shape, jp.nan)
  if state.best is None:
    index = int(np.argmax(fitness_np))
    state.best = BestIndividual(
        genome=jax.tree.map(lambda x: x[index], population.genomes),
        policy_rng=population.policy_rng[index],
        difficulty=jp.nan,
        fitness=fitness[index],
    )
  else:
    state.best = _update_best(state.best, population, fitness, nan_difficulty)
  passed = assessment.passed
  scored = jp.asarray(fitness_np[passed]) if np.any(passed) else fitness
  state.history_records.append(
      _compute_generation_record(
          generation,
          scored,
          jp.full(scored.shape, jp.nan),
          state.best,
      )
  )
  elite_mean = prepared['elite_mean']
  elite_mean_text = f'{elite_mean:.4f}' if np.isfinite(elite_mean) else 'n/a'
  fitness_mean = prepared['fitness_mean']
  fitness_mean_text = (
      f'{fitness_mean:.4f}' if np.isfinite(fitness_mean) else 'n/a'
  )
  print(
      f'{state.spec.slug} gen {generation:4d}  '
      f'fitness mean={fitness_mean_text}  '
      f'elite_mean={elite_mean_text}  '
      f'max={float(np.max(fitness_np)):.4f}  '
      f'{prepared.get("means", "")}  '
      f'kept {prepared["n_kept"]}/{prepared["n_scored"]}  '
      f'unique={len(state.seen)}  duplicates={prepared["n_duplicate"]}  '
      f'unhealthy={assessment.n_unhealthy}  '
      f'zero_action_fails={assessment.n_step_fail}  '
      f'all_fail={assessment.n_all_fail}  '
      f'discarded={prepared["discarded"]}  replaced={prepared["replaced"]}  '
      f'added={prepared["added"]}  '
      f'{state.stop.format_progress()}',
      flush=True,
  )
  state.breed_fitness = jp.asarray(prepared['masked'])
  state.settled = bool(prepared['settled'])
  state.last_generation = generation
  _report_suite(
      state,
      generation,
      force=False,
      failure_rate_every=failure_rate_every,
  )
  if (
      state.spec.save_path
      and save_every > 0
      and (generation == 0 or generation % int(save_every) == 0)
  ):
    _checkpoint(
        state.best,
        population,
        state.history_records,
        config,
        attribution_config,
        policy_ids,
        state.spec.save_path,
        len(state.seen),
        state.n_generations_stored,
    )


def _report_suite(state: _LaneState, generation: int, *, force: bool, failure_rate_every: int) -> None:
  generation = int(generation)
  if state.last_reported_generation == generation:
    return
  if not force and generation % int(failure_rate_every) != 0:
    return
  if state.current_n_cases > ELITE_COUNT:
    raise RuntimeError(
        f'{state.spec.slug} has {state.current_n_cases} elite cases; '
        f'evaluation is limited to {ELITE_COUNT}.'
    )
  qpos = qvel = policy_rng = None
  if state.spec.on_suite_checkpoint is not None:
    if state.current_states is None:
      raise TypeError(
          f'{state.spec.slug} has no elite states to roll out.'
      )
    qpos, qvel, policy_rng = state.current_states
  report = {
      'generation': generation,
      'candidates_discarded': int(state.turnover_window['discarded']),
      'tests_replaced': int(state.turnover_window['replaced']),
      'tests_added': int(state.turnover_window['added']),
      'elites_replaced': int(state.turnover_window['elites_replaced']),
      'n_cases': int(state.current_n_cases),
  }
  if state.spec.on_suite_checkpoint is not None:
    state.spec.on_suite_checkpoint(report, qpos, qvel, policy_rng)
  state.suite_reports.append(report)
  for key in state.turnover_window:
    state.turnover_window[key] = 0
  state.last_reported_generation = generation
  print(
      f'{state.spec.slug} suite gen {generation:4d}  '
      f'n_cases={report["n_cases"]}  '
      f'search_fail_rate={_format_rate(report.get("search_fail_rate"))}  '
      f'eval_fail_rate={_format_rate(report.get("eval_fail_rate"))}  '
      f'discarded={report["candidates_discarded"]}  '
      f'replaced={report["tests_replaced"]}  '
      f'added={report["tests_added"]}  '
      f'elites_replaced={report["elites_replaced"]}',
      flush=True,
  )


def _copy_population(population: Population) -> Population:
  """A private population. Later breeding must not alias another search."""
  genomes = jax.tree.map(
      lambda leaf: jp.asarray(np.array(leaf, copy=True)), population.genomes
  )
  return Population(
      genomes=genomes,
      policy_rng=jp.asarray(np.array(population.policy_rng, copy=True)),
  )


def _means_text(state: _LaneState, assessment: _Assessment, objectives, raw, passed) -> str:
  if state.spec.kind == FEATURE and assessment.feature_columns:
    return '  '.join(
        _format_feature(
            item,
            np.asarray(assessment.feature_columns[item.name], dtype=np.float64)[passed],
        )
        for item in objectives
        if np.any(passed)
    )
  return ''


def run_quad_fitness_search(
    bundle,
    rng: jax.Array,
    objectives: Sequence[FeatureObjective],
    lanes: Sequence[QuadLane],
    *,
    seed: int,
    best_policy_index: int = 0,
    elite_count: int = ELITE_COUNT,
    candidates_per_generation: int = CANDIDATES_PER_GENERATION,
    elite_replace_limit: int = DEFAULT_ELITE_REPLACE_LIMIT,
    elite_replace_window: int = DEFAULT_ELITE_REPLACE_WINDOW,
    failure_rate_every: int = DEFAULT_FAILURE_RATE_EVERY,
    max_generations: int | None = None,
    immigrant_fraction: float = 0.25,
    save_every: int = 1,
    resumes: Mapping[str, FeatureResume | None] | None = None,
) -> list[dict]:
  """Run the four searches from one seeded generation-0 population.

  ``rng`` is the only source of randomness. Generation 0 draws
  ``elite_count`` individuals, and every search scores that population.
  Each later generation splits one tournament key and, when immigrants
  are on, one immigrant population. Every search applies that draw to
  its own population. Selection uses that search's fitness, so the four
  populations diverge after generation 0.
  """
  objectives = tuple(objectives)
  if len(lanes) != 4:
    raise ValueError(f'Quad search runs four fitnesses, got {len(lanes)}.')
  kinds = [lane.kind for lane in lanes]
  if kinds != list(LANE_KINDS):
    raise ValueError(f'Lanes must be {LANE_KINDS}, got {kinds}.')
  if not 0.0 <= float(immigrant_fraction) < 1.0:
    raise ValueError(
        f'immigrant_fraction must be in [0, 1), got {immigrant_fraction}'
    )
  if max_generations is None:
    max_generations = DEFAULT_MAX_GENERATIONS
  max_generations = int(max_generations)
  if max_generations < 1:
    raise ValueError(f'max_generations must be >= 1, got {max_generations}')
  failure_rate_every = int(failure_rate_every)
  if failure_rate_every < 1:
    raise ValueError(
        f'failure_rate_every must be >= 1, got {failure_rate_every}'
    )
  best_policy_index = int(best_policy_index)
  discrete = False
  needs = requirements(objectives, discrete_actions=discrete) | frozenset({'policy'})
  config: GAConfig = feature_search_config(
      bundle.ga_config,
      elite_count=elite_count,
      candidates_per_generation=candidates_per_generation,
  ).replace(n_generations=max_generations)
  operators = _operator_kwargs(bundle)
  suite_fn = operators.get('genomes_to_suite_fn', _default_genomes_to_test_suite)
  random_genome_fn = operators.get('random_genome_fn')
  repair_genome_fn = operators.get('repair_genome_fn')
  next_kwargs = {}
  crossover_fn = operators.get('crossover_genomes_fn')
  mutate_fn = operators.get('mutate_genome_fn')
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
  states = [
      _LaneState(
          lane,
          int(config.elite_count),
          int(elite_replace_limit),
          int(elite_replace_window),
      )
      for lane in lanes
  ]
  for state in states:
    if state.shard_dir:
      os.makedirs(state.shard_dir, exist_ok=True)
  policy_ids = list(bundle.policy_ids)
  if best_policy_index < 0 or best_policy_index >= len(policy_ids):
    raise IndexError(
        f'Best-policy index {best_policy_index} is outside the search '
        f'ensemble of {len(policy_ids)}.'
    )
  print(
      f'Quad fitness search {bundle.environment}/{bundle.algorithm}  '
      f'seed={int(seed)}  experiment draws from this seed  '
      f'max_generations={max_generations}  '
      f'elites={config.elite_count}  '
      f'best_policy={policy_ids[best_policy_index]} '
      f'(search row {best_policy_index})',
      flush=True,
  )
  print(
      'Generation 0 is one shared population. From generation 1 each '
      'search breeds only from its own population. The tournament key '
      f'and the {n_immigrants} immigrants are drawn once per generation '
      'from the experiment seed and applied to every search.',
      flush=True,
  )
  for state in states:
    print(f'  {state.spec.name}: {state.spec.slug}', flush=True)
  print(ELITE_STOP, flush=True)

  def consume(active: list[_LaneState], population_for_screen: Population, search_failures, generation: int, *, archive: bool = True) -> None:
    assessment = _assess(
        population_for_screen,
        suite_fn,
        bundle,
        needs,
        subspaces,
        discrete,
        search_failures,
    )
    prepared = []
    for state in active:
      item = _prepare_lane(
          state,
          assessment,
          generation,
          objectives=objectives,
          algorithm=bundle.algorithm,
          environment=bundle.environment,
          policy_index=best_policy_index,
          elite_count=int(config.elite_count),
          n_immigrants=n_immigrants,
          archive=archive,
      )
      item['means'] = _means_text(
          state, assessment, objectives, item['raw'], assessment.passed
      )
      prepared.append(item)
    assessment.signals = None
    assessment.feature_columns = None
    for state, item in zip(active, prepared):
      _archive_lane(
          state,
          item,
          assessment,
          population_for_screen,
          failure_rate_every=failure_rate_every,
          config=config,
          attribution_config=attribution_config,
          policy_ids=policy_ids,
          save_every=save_every,
          archive=archive,
      )
    assessment.suite = None

  def advance(step_rng, immigrants, active: list[_LaneState], step: int) -> None:
    for state in active:
      parent = state.population
      state.population = next_generation(
          step_rng,
          parent,
          state.breed_fitness,
          config,
          **next_kwargs,
      )
      if immigrants is not None:
        state.population = _replace_tail(state.population, immigrants)
      if state.population.policy_rng is parent.policy_rng:
        raise RuntimeError(
            f'{state.spec.slug} bred in place. Each search needs its own '
            'population after generation 0.'
        )
      consume(
          [state],
          state.population,
          state.spec.search_failures,
          step,
      )
    owned = [
        state.population.policy_rng
        for state in active
        if not state.settled
    ]
    if len(owned) > 1 and len({id(item) for item in owned}) != len(owned):
      raise RuntimeError(
          'Two searches still share a population after generation 0.'
      )

  if resumes is None:
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
    for state in states:
      state.population = population
    consume(states, population, lanes[0].search_failures, 0)
    # Generation 0 scored one population. Each search now owns a copy, and
    # every later generation breeds that copy from its own fitness.
    for state in states:
      state.population = _copy_population(population)
    for step in range(1, max_generations):
      if all(state.settled for state in states):
        break
      rng, step_rng = jax.random.split(rng)
      immigrants = None
      if n_immigrants > 0:
        rng, immigrants = sample_population(
            rng, bundle, immigrant_sampler, n_immigrants
        )
      advance(
          step_rng,
          immigrants,
          [state for state in states if not state.settled],
          step,
      )
  else:
    missing = [kind for kind in LANE_KINDS if kind not in resumes]
    if missing:
      raise ValueError(f'Resume is missing lanes: {missing}.')
    print(
        'Continuing saved searches. Each one resumes from its own last '
        'generation. The elite-replacement window starts again. A search '
        'already at the generation ceiling stays as it was.',
        flush=True,
    )
    for state in states:
      resume = resumes[state.spec.kind]
      if resume is None:
        state.carry_on = False
        state.settled = True
        print(
            f'  {state.spec.slug}: already at the generation ceiling',
            flush=True,
        )
        continue
      state.population = resume.population
      state.best = resume.best
      state.history_records = list(resume.history_records)
      state.seen = set(resume.seen)
      state.n_generations_stored = int(resume.n_generations_stored)
      state.last_generation = int(resume.generation)
      state.last_reported_generation = int(resume.generation)
      if int(resume.generation) + 1 >= max_generations:
        state.carry_on = False
        state.settled = True
        print(
            f'  {state.spec.slug}: generation {int(resume.generation)} '
            'is already at the ceiling',
            flush=True,
        )
        continue
      print(
          f'  {state.spec.slug}: continue from generation '
          f'{int(resume.generation)} ({len(state.seen)} unique states)',
          flush=True,
      )
    for state in states:
      if not state.carry_on:
        continue
      consume(
          [state],
          state.population,
          state.spec.search_failures,
          int(state.last_generation),
          archive=False,
      )
    for step in range(1, max_generations):
      active = [
          state for state in states
          if state.carry_on and not state.settled
          and state.last_generation == step - 1
      ]
      pending = [
          state for state in states
          if state.carry_on and not state.settled
          and state.last_generation is not None
          and int(state.last_generation) + 1 < max_generations
      ]
      if not pending:
        break
      if not active:
        continue
      step_rng = jax.random.fold_in(rng, step)
      immigrants = None
      if n_immigrants > 0:
        _ignored, immigrants = sample_population(
            jax.random.fold_in(step_rng, 1),
            bundle,
            immigrant_sampler,
            n_immigrants,
        )
      advance(jax.random.fold_in(step_rng, 2), immigrants, active, step)

  infos = []
  for state in states:
    if not state.carry_on:
      print(
          f'{state.spec.slug}: left at its saved generation',
          flush=True,
      )
      infos.append({
          'slug': state.spec.slug,
          'fitness': state.spec.kind,
          'continued': False,
          'stopped_because': 'already_complete',
      })
      continue
    if state.last_generation is not None:
      _report_suite(
          state,
          state.last_generation,
          force=True,
          failure_rate_every=failure_rate_every,
      )
    if state.settled:
      print(
          f'{state.spec.slug}: fewer than {state.stop.limit} elites replaced '
          f'across the last {state.stop.window} generations '
          f'({state.stop.window_total} replaced, {state.stop.n_elites} elites, '
          f'{len(state.seen)} unique states).',
          flush=True,
      )
    else:
      print(
          f'{state.spec.slug}: stopped at the generation ceiling '
          f'({int(state.last_generation) + 1 if state.last_generation is not None else 0} '
          f'generations, {state.stop.n_elites} elites, {len(state.seen)} unique states).',
          flush=True,
      )
    if state.spec.save_path and state.population is not None and state.best is not None:
      _checkpoint(
          state.best,
          state.population,
          state.history_records,
          config,
          attribution_config,
          policy_ids,
          state.spec.save_path,
          len(state.seen),
          state.n_generations_stored,
      )
    placeholder = jp.full((state.population.policy_rng.shape[0],), jp.nan)
    _make_collection(
        [],
        state.best,
        state.population,
        placeholder,
        state.history_records,
        config,
        attribution_config,
        policy_ids,
        shard_dir=state.shard_dir,
    )
    infos.append(
        {
            'slug': state.spec.slug,
            'fitness': state.spec.kind,
            'continued': True,
            'fitness_name': state.spec.name,
            'rng_seed': int(seed),
            'best_policy_index': int(best_policy_index),
            'best_policy_id': int(policy_ids[best_policy_index]),
            'search_kind': 'quad_fitness',
            'stopped_because': (
                'elite_replacements' if state.settled else 'max_generations'
            ),
            'elite_stop': ELITE_STOP,
            'elite_replace_limit': int(state.stop.limit),
            'elite_replace_window': int(state.stop.window),
            'elite_replace_window_total': int(state.stop.window_total),
            'elites_replaced': int(state.stop.replaced),
            'n_elites': int(state.stop.n_elites),
            'elite_count': int(config.elite_count),
            'n_unique_states': int(len(state.seen)),
            'n_duplicate_skipped': int(state.n_duplicate_skipped),
            'zero_action_fails': int(state.n_zero_action_fails),
            'search_all_fails': int(state.n_search_all_fails),
            'failure_rate_every': int(failure_rate_every),
            'suite_report': SUITE_REPORT,
            'suite_reports': state.suite_reports,
            'features': [
                {'name': item.name, 'direction': item.direction_name}
                for item in objectives
            ],
            'max_generations': int(max_generations),
            'n_immigrants': int(n_immigrants),
            'shard_dir': state.shard_dir,
        }
    )
  return infos
