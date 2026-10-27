"""Genetic operators for Ant test-case genomes.

Position noise is uniform and velocity noise is Gaussian, matching
``Ant.reset``. The free-root quaternion is renormalized when a genome
becomes a test suite.
"""

import jax
from jax import numpy as jp

from ga.ga import Genome
from test_case_execution.types import TestSuite


def random_genome(rng: jax.Array, nq: int, nv: int, noise_scale: float) -> Genome:
  """Sample a genome from the same distribution as Ant.reset."""
  rng, k1, k2 = jax.random.split(rng, 3)
  low, hi = -noise_scale, noise_scale
  dqpos = jax.random.uniform(k1, (nq,), minval=low, maxval=hi)
  qvel = hi * jax.random.normal(k2, (nv,))
  return Genome(dqpos=dqpos, qvel=qvel)


def repair_genome(
    genome: Genome,
    max_dqpos: float,
    max_qvel: float,
) -> Genome:
  """Clip genome values to valid ranges."""
  return Genome(
      dqpos=jp.clip(genome.dqpos, -max_dqpos, max_dqpos),
      qvel=jp.clip(genome.qvel, -max_qvel, max_qvel),
  )


def mutate_genome(
    rng: jax.Array,
    genome: Genome,
    sigma_dqpos: float,
    sigma_qvel: float,
    max_dqpos: float,
    max_qvel: float,
) -> Genome:
  """Apply Gaussian mutation to a genome."""
  rng, k1, k2 = jax.random.split(rng, 3)
  dqpos = genome.dqpos + jax.random.normal(k1, genome.dqpos.shape) * sigma_dqpos
  qvel = genome.qvel + jax.random.normal(k2, genome.qvel.shape) * sigma_qvel
  return repair_genome(Genome(dqpos=dqpos, qvel=qvel), max_dqpos, max_qvel)


def crossover_genomes(
    rng: jax.Array,
    genome_a: Genome,
    genome_b: Genome,
    max_dqpos: float,
    max_qvel: float,
) -> Genome:
  """Uniform crossover between two genomes."""
  rng, k1, k2 = jax.random.split(rng, 3)
  mask_dqpos = jax.random.bernoulli(k1, 0.5, shape=genome_a.dqpos.shape)
  mask_qvel = jax.random.bernoulli(k2, 0.5, shape=genome_a.qvel.shape)
  dqpos = jp.where(mask_dqpos, genome_a.dqpos, genome_b.dqpos)
  qvel = jp.where(mask_qvel, genome_a.qvel, genome_b.qvel)
  return repair_genome(Genome(dqpos=dqpos, qvel=qvel), max_dqpos, max_qvel)


def _normalize_quaternion(qpos: jax.Array) -> jax.Array:
  quat = qpos[3:7]
  quat = quat / (jp.linalg.norm(quat) + 1e-8)
  return qpos.at[3:7].set(quat)


def genomes_to_test_suite(
    genomes: Genome,
    qpos0: jax.Array,
    policy_rng: jax.Array,
) -> TestSuite:
  """Convert a batched population of genomes into an executable test suite."""
  qpos = qpos0 + genomes.dqpos
  qpos = jax.vmap(_normalize_quaternion)(qpos)
  return TestSuite(qpos=qpos, qvel=genomes.qvel, policy_rng=policy_rng)
