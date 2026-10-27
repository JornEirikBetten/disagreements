"""Selection operators for the genetic algorithm."""

import jax
from jax import numpy as jp

from ga.ga import Genome, Population


def tournament_select(
    rng: jax.Array,
    fitness: jax.Array,
    n_select: int,
    tournament_k: int,
) -> jax.Array:
  """Select parent indices via tournament selection."""
  pop_size = fitness.shape[0]

  def one_tournament(key: jax.Array) -> jax.Array:
    candidates = jax.random.randint(key, (tournament_k,), 0, pop_size)
    winner = candidates[jp.argmax(fitness[candidates])]
    return winner

  keys = jax.random.split(rng, n_select)
  return jax.vmap(one_tournament)(keys)


def _take_genomes(genomes: Genome, indices: jax.Array) -> Genome:
  return jax.tree.map(lambda x: x[indices], genomes)


def select_elites(
    population: Population,
    fitness: jax.Array,
    elite_count: int,
) -> Population:
  """Return the top elite_count individuals by fitness."""
  elite_idx = jp.argsort(fitness)[-elite_count:]
  return Population(
      genomes=_take_genomes(population.genomes, elite_idx),
      policy_rng=population.policy_rng[elite_idx],
  )
