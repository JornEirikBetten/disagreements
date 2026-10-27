"""Fitness evaluation for GA populations."""

from typing import Callable

import jax
from jax import numpy as jp

from ga.ga import GAConfig, Genome
from mutators.humanoid import genomes_to_test_suite as _default_genomes_to_test_suite
from test_case_execution.types import TestSuite


def difficulty_per_case(failed: jax.Array) -> jax.Array:
  """Compute per-test-case difficulty from a (P, N) failure matrix.

  Difficulty is the fraction of policies that fail. If every policy fails,
  difficulty is set to -0.1 so likely-unsolvable cases are not ranked highly.
  """
  frac = jp.mean(failed.astype(jp.float32), axis=0)
  return jp.where(frac >= 1.0, -0.1, frac)


def difficulty_to_fitness(difficulty: jax.Array, config: GAConfig) -> jax.Array:
  """Map difficulty values to fitness scores (higher is better)."""
  if config.fitness_mode == 'maximize':
    return difficulty
  if config.fitness_mode == 'target':
    return -jp.abs(difficulty - config.target_difficulty)
  raise ValueError(f'Unknown fitness_mode: {config.fitness_mode}')


def search_criterion_to_fitness(search_criterion: jax.Array, config: GAConfig) -> jax.Array:
  """Map search criterion values to fitness scores (higher is better)."""
  if config.fitness_mode == 'maximize':
    return search_criterion
  if config.fitness_mode == 'target':
    return -jp.abs(search_criterion - config.target_search_criterion)
  raise ValueError(f'Unknown fitness_mode: {config.fitness_mode}')


def evaluate_population(
    execute_fn,
    genomes: Genome,
    policy_rng: jax.Array,
    qpos0: jax.Array,
    config: GAConfig,
    genomes_to_suite_fn: Callable[
        [Genome, jax.Array, jax.Array], TestSuite
    ] = _default_genomes_to_test_suite,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Evaluate a population and return fitness, difficulty, and failed matrix.

  Args:
    execute_fn: Callable TestSuite -> failed array of shape (P, G).
    genomes: Batched genomes with leading axis G.
    policy_rng: Policy RNG keys, shape (G, 2).
    qpos0: Default environment pose.
    config: GA hyperparameters.
    genomes_to_suite_fn: Converts batched genomes to a TestSuite.

  Returns:
    fitness: (G,)
    difficulty: (G,)
    failed: (P, G)
  """
  suite = genomes_to_suite_fn(genomes, qpos0, policy_rng)
  failed = execute_fn(suite)
  difficulty = difficulty_per_case(failed)
  fitness = difficulty_to_fitness(difficulty, config)
  return fitness, difficulty, failed
