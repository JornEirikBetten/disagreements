import jax
from flax import struct

from test_case_execution.types import TestCase


@struct.dataclass
class Genome:
  dqpos: jax.Array  # (nq,) or batched (G, nq)
  qvel: jax.Array   # (nv,) or batched (G, nv)

  def to_test_case(
      self, policy_rng: jax.Array, qpos0: jax.Array
  ) -> TestCase:
    return TestCase(
        qpos=qpos0 + self.dqpos,
        qvel=self.qvel,
        policy_rng=policy_rng,
    )


@struct.dataclass
class Population:
  """A batched population of genomes with fixed per-slot policy RNGs."""

  genomes: Genome
  policy_rng: jax.Array  # (G, 2)


@struct.dataclass
class GAConfig:
  population_size: int = 32
  n_generations: int = 50
  rollout_length: int = 200
  elite_count: int = 4
  tournament_k: int = 3
  mutation_rate: float = 0.8
  crossover_rate: float = 0.7
  noise_scale: float = 1e-2
  sigma_dqpos: float = 5e-3
  sigma_qvel: float = 5e-3
  max_dqpos: float = 0.1
  max_qvel: float = 0.1
  fitness_mode: str = 'maximize'  # 'maximize' or 'target'
  target_difficulty: float = 0.5
  log_every: int = 1


@struct.dataclass
class GenerationRecord:
  """Summary statistics for a single generation evaluation."""

  generation: int = struct.field(pytree_node=False)
  mean_difficulty: jax.Array
  max_difficulty: jax.Array
  min_difficulty: jax.Array
  std_difficulty: jax.Array
  mean_fitness: jax.Array
  max_fitness: jax.Array
  min_fitness: jax.Array
  std_fitness: jax.Array
  best_fitness: jax.Array
  best_difficulty: jax.Array


@struct.dataclass
class GenerationStats:
  """Per-generation stats stacked across the full search."""

  generation: jax.Array          # (num_gens,)
  mean_difficulty: jax.Array     # (num_gens,)
  max_difficulty: jax.Array
  min_difficulty: jax.Array
  std_difficulty: jax.Array
  mean_fitness: jax.Array
  max_fitness: jax.Array
  min_fitness: jax.Array
  std_fitness: jax.Array
  best_fitness: jax.Array        # global best so far at each generation
  best_difficulty: jax.Array


@struct.dataclass
class BestIndividual:
  genome: Genome
  policy_rng: jax.Array
  difficulty: jax.Array
  fitness: jax.Array


@struct.dataclass
class SearchResult:
  best: BestIndividual
  final_population: Population
  history: GenerationStats
  final_difficulties: jax.Array  # (G,)
