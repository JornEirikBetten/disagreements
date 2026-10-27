"""GA collection that fills a fixed quota of *distinct* test cases per difficulty bin.

A case is identified by its initial state. Elites that the genetic search
re-evaluates are not written twice; they remain in the population so they can
still mutate into new states.
"""

from __future__ import annotations

from dataclasses import replace
import gc
import os

import jax
import numpy as np
from jax import numpy as jp

from attributions.types import AttributionConfig
from ga.collect import (
    GASearchCollection,
    collect_population_metrics,
    generation_shard_dir,
    save_collection,
    save_generation_shard,
)
from ga.fitness import evaluate_population
from ga.ga import BestIndividual, GAConfig, GenerationRecord, Population, SearchResult
from ga.search import (
    _compute_generation_record,
    _log_generation_record,
    _stack_generation_history,
    _update_best,
    init_population,
    next_generation,
)
from ga.stratified_init import (
    SamplerSettings,
    sample_population,
    sampler_from_bundle,
    take_indices,
)
from mutators.humanoid import (
    genomes_to_test_suite as _default_genomes_to_test_suite,
    mutate_genome as _default_mutate_genome,
)
from test_case_execution.types import TestSuite

ALL_FAIL_SENTINEL = -0.1
DEFAULT_BIN_CAPACITY = 10_000


def difficulty_levels(n_policies: int) -> np.ndarray:
  """All-fail sentinel plus ``k / P`` for ``k = 0, ..., P-1``."""
  n = int(n_policies)
  if n < 1:
    raise ValueError(f'n_policies must be >= 1, got {n_policies!r}')
  return np.array(
      [ALL_FAIL_SENTINEL] + [k / n for k in range(n)],
      dtype=np.float64,
  )


def snap_difficulty_bins(difficulty: np.ndarray, n_policies: int) -> np.ndarray:
  """Map continuous difficulties to integer bin ids in ``0, ..., P``."""
  levels = difficulty_levels(n_policies)
  values = np.asarray(difficulty, dtype=np.float64).reshape(-1)
  dists = np.abs(values[:, None] - levels[None, :])
  return np.argmin(dists, axis=1).astype(np.int32)


def fail_counts_from_bins(bin_ids: np.ndarray, n_policies: int) -> np.ndarray:
  """Map bin id to number of failing policies (all-fail → ``P``)."""
  n = int(n_policies)
  ids = np.asarray(bin_ids, dtype=np.int32).reshape(-1)
  return np.where(ids == 0, n, ids - 1).astype(np.int32)


def _state_rows(suite) -> np.ndarray:
  """One row per case: concatenated ``(qpos, qvel)``.

  Float32 is enough to treat a re-collected elite as the same case while
  leaving mutated neighbours distinct.
  """
  qpos = np.asarray(suite.qpos, dtype=np.float32)
  qvel = np.asarray(suite.qvel, dtype=np.float32)
  return np.concatenate(
      [qpos.reshape(qpos.shape[0], -1), qvel.reshape(qvel.shape[0], -1)],
      axis=1,
  )


def state_keys(suite) -> list[bytes]:
  """Hashable identity of each case in ``suite``."""
  rows = np.ascontiguousarray(_state_rows(suite))
  width = int(rows.dtype.itemsize * rows.shape[1])
  packed = rows.view(np.dtype((np.void, width))).reshape(rows.shape[0])
  return [bytes(item) for item in packed]


def select_unfilled(
    bin_ids: np.ndarray,
    counts: np.ndarray,
    capacity: int,
    *,
    keys: list[bytes] | None = None,
    seen: set[bytes] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Keep new cases whose bins still have room, in population order.

  A case whose initial state has already been written --- the search
  re-collects elites verbatim --- is skipped, so a bin fills with distinct
  test cases rather than with copies of the same one. ``seen`` is updated
  in place.

  Returns:
    kept indices, the updated per-bin counts, and how many cases were
    skipped as duplicates.
  """
  bin_ids = np.asarray(bin_ids, dtype=np.int32).reshape(-1)
  new_counts = np.asarray(counts, dtype=np.int32).copy()
  remaining = int(capacity) - new_counts
  if keys is not None and len(keys) != int(bin_ids.shape[0]):
    raise ValueError(
        f'keys has length {len(keys)}, expected {int(bin_ids.shape[0])}'
    )
  kept = []
  n_duplicate = 0
  for i, bin_id in enumerate(bin_ids):
    if remaining[bin_id] <= 0:
      continue
    if keys is not None and seen is not None:
      key = keys[i]
      if key in seen:
        n_duplicate += 1
        continue
      seen.add(key)
    kept.append(i)
    remaining[bin_id] -= 1
    new_counts[bin_id] += 1
  return np.asarray(kept, dtype=np.int32), new_counts, n_duplicate


def selection_fitness(
    bin_ids: np.ndarray,
    counts: np.ndarray,
    capacity: int,
    n_policies: int,
) -> np.ndarray:
  """Prefer unfilled bins, then higher fail counts as stepping stones.

  Remaining capacity is scaled by ``P + 1`` so any case from an unfilled bin
  outranks every filled-bin case. The fail-count tie-break keeps 15/16 (and
  other near-misses) above easy filled bins once only all-fail still has room.
  """
  n = int(n_policies)
  bin_ids = np.asarray(bin_ids, dtype=np.int32).reshape(-1)
  remaining = np.maximum(int(capacity) - np.asarray(counts, dtype=np.int32), 0)
  fail_count = fail_counts_from_bins(bin_ids, n)
  return (remaining[bin_ids].astype(np.float32) * (n + 1) + fail_count).astype(
      np.float32
  )


def stepping_stone_indices(
    bin_ids: np.ndarray,
    counts: np.ndarray,
    capacity: int,
    n_policies: int,
    n_keep: int,
) -> np.ndarray:
  """Filled-bin cases closest to an unfilled bin, highest fail count first.

  Used to keep 15/16 genomes in the population after that bin is full so they
  can still mutate into all-fail.
  """
  n_keep = int(n_keep)
  if n_keep < 1:
    return np.zeros(0, dtype=np.int32)
  n = int(n_policies)
  bin_ids = np.asarray(bin_ids, dtype=np.int32).reshape(-1)
  remaining = np.maximum(int(capacity) - np.asarray(counts, dtype=np.int32), 0)
  unfilled = np.flatnonzero(remaining > 0)
  if unfilled.size == 0:
    return np.zeros(0, dtype=np.int32)
  filled = remaining[bin_ids] <= 0
  if not np.any(filled):
    return np.zeros(0, dtype=np.int32)
  fail_count = fail_counts_from_bins(bin_ids, n)
  target = fail_counts_from_bins(unfilled, n)
  dist = np.min(
      np.abs(fail_count[:, None].astype(np.int32) - target[None, :]), axis=1
  )
  score = np.where(
      filled,
      -dist.astype(np.float64) * (n + 1) + fail_count,
      -np.inf,
  )
  order = np.argsort(-score)
  picked = [int(i) for i in order if np.isfinite(score[i])]
  return np.asarray(picked[:n_keep], dtype=np.int32)


def bins_full(counts: np.ndarray, capacity: int) -> bool:
  return bool(np.all(np.asarray(counts) >= int(capacity)))


def take_suite(suite, indices: np.ndarray):
  """Slice a test suite along the case axis."""
  idx = jp.asarray(indices, dtype=jp.int32)
  return TestSuite(
      qpos=suite.qpos[idx],
      qvel=suite.qvel[idx],
      policy_rng=suite.policy_rng[idx],
  )


def _overwrite_head(population: Population, stones: Population) -> Population:
  n_stone = int(stones.policy_rng.shape[0])
  if n_stone < 1:
    return population
  n_pop = int(population.policy_rng.shape[0])
  if n_stone > n_pop:
    raise ValueError('Cannot overwrite more individuals than the population size.')
  return Population(
      genomes=jax.tree.map(
          lambda stone, pop: jp.concatenate([stone, pop[n_stone:]], axis=0),
          stones.genomes,
          population.genomes,
      ),
      policy_rng=jp.concatenate(
          [stones.policy_rng, population.policy_rng[n_stone:]], axis=0
      ),
  )


def _immigrant_sampler(
    sampler: SamplerSettings, environment: str
) -> SamplerSettings:
  """Sample immigrants uniformly in the repaired genome box, not reset noise.

  Humanoid and Ant keep reset-scale noise: the free-joint quaternion and
  root pose are not a planar clip box, and sampling ``±max_dqpos`` there
  NaNs MJX.
  """
  if environment in ('humanoid', 'ant'):
    return sampler
  return replace(sampler, noise_scale=float(sampler.max_dqpos))


def _mutate_population(
    rng: jax.Array,
    population: Population,
    config: GAConfig,
    mutate_fn,
) -> tuple[jax.Array, Population]:
  n = int(population.policy_rng.shape[0])
  if n < 1 or mutate_fn is None:
    return rng, population
  rng, mut_rng = jax.random.split(rng)
  keys = jax.random.split(mut_rng, n)
  genomes = jax.vmap(
      lambda key, genome: mutate_fn(
          key,
          genome,
          config.sigma_dqpos,
          config.sigma_qvel,
          config.max_dqpos,
          config.max_qvel,
      )
  )(keys, population.genomes)
  return rng, Population(genomes=genomes, policy_rng=population.policy_rng)


def _replace_tail(population: Population, immigrants: Population) -> Population:
  n_imm = int(immigrants.policy_rng.shape[0])
  if n_imm < 1:
    return population
  n_keep = int(population.policy_rng.shape[0]) - n_imm
  if n_keep < 0:
    raise ValueError('Cannot replace more individuals than the population size.')
  return Population(
      genomes=jax.tree.map(
          lambda kept, new: jp.concatenate([kept[:n_keep], new], axis=0),
          population.genomes,
          immigrants.genomes,
      ),
      policy_rng=jp.concatenate(
          [population.policy_rng[:n_keep], immigrants.policy_rng], axis=0
      ),
  )


def _bin_label(level: float, n_policies: int) -> str:
  if level < 0 or np.isclose(level, ALL_FAIL_SENTINEL):
    return 'all-fail'
  k = int(np.round(level * n_policies))
  return f'{k}/{n_policies}'


def format_bin_counts(
    counts: np.ndarray, capacity: int, n_policies: int
) -> str:
  levels = difficulty_levels(n_policies)
  parts = []
  for count, level in zip(counts, levels):
    mark = '*' if int(count) >= int(capacity) else ''
    parts.append(f'{_bin_label(float(level), n_policies)}={int(count)}{mark}')
  n_full = int(np.sum(np.asarray(counts) >= int(capacity)))
  return (
      f'filled {n_full}/{len(counts)}  '
      f'{" ".join(parts)}'
  )


def _operator_kwargs(bundle) -> dict:
  return {
      key: bundle.operators[key]
      for key in (
          'genomes_to_suite_fn',
          'random_genome_fn',
          'repair_genome_fn',
          'crossover_genomes_fn',
          'mutate_genome_fn',
      )
      if key in bundle.operators
  }


def run_binned_ga_search(
    bundle,
    rng: jax.Array,
    *,
    n_policies: int,
    bin_capacity: int = DEFAULT_BIN_CAPACITY,
    max_generations: int | None = None,
    immigrant_fraction: float = 0.25,
    initial_population: Population | None = None,
    save_path: str | None = None,
    save_every: int = 1,
) -> tuple[GASearchCollection, dict]:
  """Evolve a population, keeping at most ``bin_capacity`` *distinct* cases
  per difficulty.

  Evaluated cases whose bins are already full, or whose initial state has
  already been written, are discarded from the collection. The search still
  breeds from elites --- they stay in the population --- but a re-evaluated
  elite is not a new test case and does not fill a bin slot. Attributions
  are extracted only for kept cases, in fixed microbatches so the IG kernel
  is not recompiled when the kept count changes. Each kept generation is
  written to disk immediately so RAM and the GPU only hold the current
  population. Selection prefers unfilled bins, then higher fail counts so
  near-misses (e.g. 15/16) keep breeding toward all-fail. A slice of elites is
  reserved for those stepping stones, mutated each generation. Immigrants are
  drawn from the full genome box, except Humanoid and Ant, which keep
  reset-scale noise on the free joint.
  """
  config: GAConfig = bundle.ga_config
  if max_generations is None:
    max_generations = int(config.n_generations)
  if bin_capacity < 1:
    raise ValueError(f'bin_capacity must be >= 1, got {bin_capacity}')
  if max_generations < 0:
    raise ValueError(f'max_generations must be >= 0, got {max_generations}')
  if not 0.0 <= float(immigrant_fraction) < 1.0:
    raise ValueError(
        f'immigrant_fraction must be in [0, 1), got {immigrant_fraction}'
    )

  n_bins = n_policies + 1
  counts = np.zeros(n_bins, dtype=np.int32)
  seen_states: set[bytes] = set()
  n_duplicate_skipped = 0
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

  config = config.replace(n_generations=int(max_generations))
  n_immigrants = int(round(float(immigrant_fraction) * config.population_size))
  n_immigrants = min(
      n_immigrants, max(0, int(config.population_size - config.elite_count))
  )
  sampler = sampler_from_bundle(bundle)
  immigrant_sampler = _immigrant_sampler(sampler, bundle.environment)
  n_stepping_stones = max(0, int(config.elite_count) // 4)
  attribution_config = bundle.attribution_config
  if attribution_config is None:
    attribution_config = AttributionConfig()

  rng, init_rng = jax.random.split(rng)
  if initial_population is None:
    init_kwargs = {}
    if random_genome_fn is not None:
      init_kwargs['random_genome_fn'] = random_genome_fn
    if repair_genome_fn is not None:
      init_kwargs['repair_genome_fn'] = repair_genome_fn
    population = init_population(
        init_rng, bundle.nq, bundle.nv, config, **init_kwargs
    )
  else:
    population = initial_population

  log_config = config
  generation_records: list = []
  history_records: list[GenerationRecord] = []
  best: BestIndividual | None = None
  difficulty = None
  n_generations_stored = 0
  shard_dir = generation_shard_dir(save_path) if save_path else None
  if shard_dir:
    os.makedirs(shard_dir, exist_ok=True)

  def _evaluate_and_collect(population: Population, generation: int):
    nonlocal counts, best, difficulty, n_generations_stored, n_duplicate_skipped
    eval_fitness, difficulty, failed = evaluate_population(
        bundle.execute_fn,
        population.genomes,
        population.policy_rng,
        bundle.qpos0,
        config,
        genomes_to_suite_fn=suite_fn,
    )
    suite = suite_fn(population.genomes, bundle.qpos0, population.policy_rng)
    keys = state_keys(suite)
    bin_ids = snap_difficulty_bins(difficulty, n_policies)
    kept, counts, n_duplicate = select_unfilled(
        bin_ids,
        counts,
        bin_capacity,
        keys=keys,
        seen=seen_states,
    )
    n_duplicate_skipped += int(n_duplicate)
    select_fit = jp.asarray(
        selection_fitness(bin_ids, counts, bin_capacity, n_policies),
        dtype=jp.float32,
    )
    n_pop = int(bin_ids.shape[0])
    n_kept = int(kept.size)
    if n_kept > 0:
      kept_j = jp.asarray(kept, dtype=jp.int32)
      record = collect_population_metrics(
          take_suite(suite, kept),
          jp.asarray(failed)[:, kept_j],
          jp.asarray(eval_fitness)[kept_j],
          generation=generation,
          extract_attributions_fn=bundle.extract_attributions_fn,
      )
      del suite
      if save_path:
        shard = save_generation_shard(record, save_path, counts=counts)
        print(f'      wrote {shard}  n_cases={record.n_cases}', flush=True)
        del record
        gc.collect()
      else:
        generation_records.append(record)
      n_generations_stored += 1
    else:
      del suite
    if best is None:
      best = BestIndividual(
          genome=jax.tree.map(
              lambda x: x[jp.argmax(eval_fitness)], population.genomes
          ),
          policy_rng=population.policy_rng[jp.argmax(eval_fitness)],
          difficulty=difficulty[jp.argmax(eval_fitness)],
          fitness=eval_fitness[jp.argmax(eval_fitness)],
      )
    else:
      best = _update_best(best, population, eval_fitness, difficulty)
    history = _compute_generation_record(
        generation, select_fit, difficulty, best
    )
    history_records.append(history)
    _log_generation_record(history, log_config)
    print(
        f'      kept {n_kept}/{n_pop}  duplicates={int(n_duplicate)}  '
        f'collected={int(counts.sum())}/'
        f'{n_bins * int(bin_capacity)}  {format_bin_counts(counts, bin_capacity, n_policies)}',
        flush=True,
    )
    return select_fit, bin_ids

  select_fitness_arr, bin_ids = _evaluate_and_collect(population, 0)
  if save_path:
    _checkpoint_sidecar(
        best,
        population,
        difficulty,
        history_records,
        config,
        attribution_config,
        bundle.policy_ids,
        counts,
        save_path,
        n_generations_stored,
    )

  for gen in range(int(max_generations)):
    if bins_full(counts, bin_capacity):
      print('All difficulty bins full; stopping.', flush=True)
      break
    rng, step_rng = jax.random.split(rng)
    parents = population
    population = next_generation(
        step_rng,
        parents,
        select_fitness_arr,
        config,
        **next_kwargs,
    )
    stone_idx = stepping_stone_indices(
        bin_ids, counts, bin_capacity, n_policies, n_stepping_stones
    )
    if stone_idx.size > 0:
      rng, stones = _mutate_population(
          rng,
          take_indices(parents, stone_idx),
          config,
          mutate_fn if mutate_fn is not None else _default_mutate_genome,
      )
      population = _overwrite_head(population, stones)
    if n_immigrants > 0:
      rng, immigrants = sample_population(
          rng, bundle, immigrant_sampler, n_immigrants
      )
      population = _replace_tail(population, immigrants)
    select_fitness_arr, bin_ids = _evaluate_and_collect(population, gen + 1)
    if save_path and save_every > 0 and (gen + 1) % int(save_every) == 0:
      _checkpoint_sidecar(
          best,
          population,
          difficulty,
          history_records,
          config,
          attribution_config,
          bundle.policy_ids,
          counts,
          save_path,
          n_generations_stored,
      )

  collection = _make_collection(
      generation_records,
      best,
      population,
      difficulty,
      history_records,
      config,
      attribution_config,
      bundle.policy_ids,
      shard_dir=shard_dir,
  )
  if save_path:
    _checkpoint_sidecar(
        best,
        population,
        difficulty,
        history_records,
        config,
        attribution_config,
        bundle.policy_ids,
        counts,
        save_path,
        n_generations_stored,
    )
  info = {
      'bin_capacity': int(bin_capacity),
      'n_policies': int(n_policies),
      'n_bins': int(n_bins),
      'binning_variable': 'search_fail_rate',
      'evaluation_used_for_selection': False,
      'counts': [int(x) for x in counts],
      'levels': [float(x) for x in difficulty_levels(n_policies)],
      'n_duplicate_skipped': int(n_duplicate_skipped),
      'n_unique_states': int(len(seen_states)),
      'unique_states': True,
      'n_generations_stored': int(n_generations_stored),
      'shard_dir': shard_dir,
      'all_full': bins_full(counts, bin_capacity),
      'immigrant_fraction': float(immigrant_fraction),
      'n_immigrants': int(n_immigrants),
      'n_stepping_stones': int(n_stepping_stones),
  }
  return collection, info


def _make_collection(
    generation_records: list,
    best: BestIndividual | None,
    population: Population,
    difficulty: jax.Array | None,
    history_records: list[GenerationRecord],
    config: GAConfig,
    attribution_config: AttributionConfig,
    policy_ids: list[int],
    shard_dir: str | None = None,
) -> GASearchCollection:
  if best is None or difficulty is None or not history_records:
    raise ValueError('Binned search produced no evaluated generations.')
  search_result = SearchResult(
      best=best,
      final_population=population,
      history=_stack_generation_history(history_records),
      final_difficulties=difficulty,
  )
  return GASearchCollection(
      generations=generation_records,
      policy_ids=list(policy_ids),
      ga_config=config,
      attribution_config=attribution_config,
      search_result=search_result,
      shard_dir=shard_dir,
  )


def _checkpoint_sidecar(
    best: BestIndividual | None,
    population: Population,
    difficulty: jax.Array | None,
    history_records: list[GenerationRecord],
    config: GAConfig,
    attribution_config: AttributionConfig,
    policy_ids: list[int],
    counts: np.ndarray,
    save_path: str,
    n_generations_stored: int,
) -> None:
  if best is None or difficulty is None or not history_records:
    return
  collection = _make_collection(
      [],
      best,
      population,
      difficulty,
      history_records,
      config,
      attribution_config,
      policy_ids,
      shard_dir=generation_shard_dir(save_path),
  )
  save_collection(collection, save_path)
  print(
      f'      checkpoint {save_path}  '
      f'generations={int(n_generations_stored)}  '
      f'encounters={collection.n_encounters}  '
      f'counts={counts.tolist()}',
      flush=True,
  )
