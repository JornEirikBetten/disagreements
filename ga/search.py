"""Genetic algorithm search for locomotion test cases."""

from typing import Callable

import jax
from jax import numpy as jp

from ga.fitness import evaluate_population
from ga.ga import (
    BestIndividual,
    GAConfig,
    GenerationRecord,
    GenerationStats,
    Genome,
    Population,
    SearchResult,
)
from ga.selection import select_elites, tournament_select
from mutators.humanoid import (
    crossover_genomes as _default_crossover_genomes,
    mutate_genome as _default_mutate_genome,
    random_genome as _default_random_genome,
    repair_genome as _default_repair_genome,
)
from mutators.humanoid import genomes_to_test_suite as _default_genomes_to_test_suite
from test_case_execution.types import TestSuite


def init_population(
    rng: jax.Array,
    nq: int,
    nv: int,
    config: GAConfig,
    random_genome_fn=_default_random_genome,
    repair_genome_fn=_default_repair_genome,
) -> Population:
  """Create an initial random population."""
  rng, pop_rng, policy_rng = jax.random.split(rng, 3)
  genome_keys = jax.random.split(pop_rng, config.population_size)
  genomes = jax.vmap(
      lambda key: random_genome_fn(key, nq, nv, config.noise_scale)
  )(genome_keys)
  genomes = jax.vmap(
      lambda g: repair_genome_fn(g, config.max_dqpos, config.max_qvel)
  )(genomes)
  return Population(
      genomes=genomes,
      policy_rng=jax.random.split(policy_rng, config.population_size),
  )


def _breed_children(
    rng: jax.Array,
    population: Population,
    parent_idx: jax.Array,
    config: GAConfig,
    crossover_genomes_fn=_default_crossover_genomes,
    mutate_genome_fn=_default_mutate_genome,
) -> Population:
  """Create children from selected parent indices."""
  n_children = parent_idx.shape[0] // 2

  def make_child(
      child_rng: jax.Array, idx_a: jax.Array, idx_b: jax.Array
  ) -> tuple[Genome, jax.Array]:
    genome_a = jax.tree.map(lambda x: x[idx_a], population.genomes)
    genome_b = jax.tree.map(lambda x: x[idx_b], population.genomes)
    child_rng, crossover_rng, mutate_rng, cross_coin, mutate_coin = (
        jax.random.split(child_rng, 5)
    )

    child = jax.lax.cond(
        jax.random.bernoulli(cross_coin, p=config.crossover_rate),
        lambda _: crossover_genomes_fn(
            crossover_rng,
            genome_a,
            genome_b,
            config.max_dqpos,
            config.max_qvel,
        ),
        lambda _: genome_a,
        operand=None,
    )
    child = jax.lax.cond(
        jax.random.bernoulli(mutate_coin, p=config.mutation_rate),
        lambda c: mutate_genome_fn(
            mutate_rng,
            c,
            config.sigma_dqpos,
            config.sigma_qvel,
            config.max_dqpos,
            config.max_qvel,
        ),
        lambda c: c,
        operand=child,
    )
    return child, population.policy_rng[idx_a]

  child_keys = jax.random.split(rng, n_children)
  child_genomes, child_policy_rng = jax.vmap(make_child)(
      child_keys,
      parent_idx[:n_children],
      parent_idx[n_children:],
  )
  return Population(genomes=child_genomes, policy_rng=child_policy_rng)


def next_generation(
    rng: jax.Array,
    population: Population,
    fitness: jax.Array,
    config: GAConfig,
    crossover_genomes_fn=_default_crossover_genomes,
    mutate_genome_fn=_default_mutate_genome,
) -> Population:
  """Produce the next generation via elitism, crossover, and mutation."""
  elites = select_elites(population, fitness, config.elite_count)
  n_children = config.population_size - config.elite_count
  rng, select_rng, breed_rng = jax.random.split(rng, 3)
  parent_idx = tournament_select(
      select_rng, fitness, n_children * 2, config.tournament_k
  )
  children = _breed_children(
      breed_rng,
      population,
      parent_idx,
      config,
      crossover_genomes_fn=crossover_genomes_fn,
      mutate_genome_fn=mutate_genome_fn,
  )
  return Population(
      genomes=jax.tree.map(
          lambda e, c: jp.concatenate([e, c], axis=0),
          elites.genomes,
          children.genomes,
      ),
      policy_rng=jp.concatenate([elites.policy_rng, children.policy_rng], axis=0),
  )


def _compute_generation_record(
    generation: int,
    fitness: jax.Array,
    difficulty: jax.Array,
    best: BestIndividual,
) -> GenerationRecord:
  """Summarize fitness and difficulty for one population evaluation."""
  return GenerationRecord(
      generation=generation,
      mean_difficulty=jp.mean(difficulty),
      max_difficulty=jp.max(difficulty),
      min_difficulty=jp.min(difficulty),
      std_difficulty=jp.std(difficulty),
      mean_fitness=jp.mean(fitness),
      max_fitness=jp.max(fitness),
      min_fitness=jp.min(fitness),
      std_fitness=jp.std(fitness),
      best_fitness=best.fitness,
      best_difficulty=best.difficulty,
  )


def _stack_generation_history(records: list[GenerationRecord]) -> GenerationStats:
  return GenerationStats(
      generation=jp.array([r.generation for r in records]),
      mean_difficulty=jp.array([float(r.mean_difficulty) for r in records]),
      max_difficulty=jp.array([float(r.max_difficulty) for r in records]),
      min_difficulty=jp.array([float(r.min_difficulty) for r in records]),
      std_difficulty=jp.array([float(r.std_difficulty) for r in records]),
      mean_fitness=jp.array([float(r.mean_fitness) for r in records]),
      max_fitness=jp.array([float(r.max_fitness) for r in records]),
      min_fitness=jp.array([float(r.min_fitness) for r in records]),
      std_fitness=jp.array([float(r.std_fitness) for r in records]),
      best_fitness=jp.array([float(r.best_fitness) for r in records]),
      best_difficulty=jp.array([float(r.best_difficulty) for r in records]),
  )


def _log_generation_record(record: GenerationRecord, config: GAConfig) -> None:
  if config.log_every <= 0:
    return
  if record.generation % config.log_every != 0:
    return
  print(
      f'gen {record.generation:4d}/{config.n_generations}  '
      f'fitness mean={float(record.mean_fitness):.4f}  '
      f'max={float(record.max_fitness):.4f}  '
      f'std={float(record.std_fitness):.4f}  |  '
      f'difficulty mean={float(record.mean_difficulty):.4f}  '
      f'max={float(record.max_difficulty):.4f}  '
      f'min={float(record.min_difficulty):.4f}  |  '
      f'best_fitness={float(record.best_fitness):.4f}  '
      f'best_difficulty={float(record.best_difficulty):.4f}'
  )


def _update_best(
    best: BestIndividual,
    population: Population,
    fitness: jax.Array,
    difficulty: jax.Array,
) -> BestIndividual:
  idx = jp.argmax(fitness)
  is_better = fitness[idx] > best.fitness
  return BestIndividual(
      genome=jax.tree.map(
          lambda new, old: jp.where(is_better, new, old),
          jax.tree.map(lambda x: x[idx], population.genomes),
          best.genome,
      ),
      policy_rng=jp.where(is_better, population.policy_rng[idx], best.policy_rng),
      difficulty=jp.where(is_better, difficulty[idx], best.difficulty),
      fitness=jp.where(is_better, fitness[idx], best.fitness),
  )


def build_ga_search(
    qpos0: jax.Array,
    nq: int,
    nv: int,
    execute_fn: Callable,
    config: GAConfig | None = None,
    genomes_to_suite_fn: Callable[
        [Genome, jax.Array, jax.Array], TestSuite
    ] = _default_genomes_to_test_suite,
    random_genome_fn=_default_random_genome,
    repair_genome_fn=_default_repair_genome,
    crossover_genomes_fn=_default_crossover_genomes,
    mutate_genome_fn=_default_mutate_genome,
):
  """Build a GA search runner.

  Args:
    qpos0: Default environment pose.
    nq: Number of generalized positions.
    nv: Number of generalized velocities.
    execute_fn: Cached executor mapping TestSuite -> failed (P, G).
    config: GA hyperparameters.
    genomes_to_suite_fn: Converts batched genomes to a TestSuite.
    random_genome_fn / repair_genome_fn / crossover_genomes_fn /
      mutate_genome_fn: Environment-specific genetic operators.

  Returns:
    Callable ``run(rng) -> SearchResult``.
  """
  if config is None:
    config = GAConfig()

  def run(rng: jax.Array) -> SearchResult:
    rng, init_rng = jax.random.split(rng)
    population = init_population(
        init_rng,
        nq,
        nv,
        config,
        random_genome_fn=random_genome_fn,
        repair_genome_fn=repair_genome_fn,
    )

    fitness, difficulty, failed = evaluate_population(
        execute_fn,
        population.genomes,
        population.policy_rng,
        qpos0,
        config,
        genomes_to_suite_fn=genomes_to_suite_fn,
    )
    best = BestIndividual(
        genome=jax.tree.map(lambda x: x[jp.argmax(fitness)], population.genomes),
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
      best = _update_best(best, population, fitness, difficulty)
      record = _compute_generation_record(gen + 1, fitness, difficulty, best)
      history_records.append(record)
      _log_generation_record(record, config)

    return SearchResult(
        best=best,
        final_population=population,
        history=_stack_generation_history(history_records),
        final_difficulties=difficulty,
    )

  return run
