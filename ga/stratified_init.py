"""Stratified initial populations with capped tries and sampler widening."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import numpy as np
from jax import numpy as jp

from ga.env_setup import SearchBundle
from ga.ga import GAConfig, Population
from ga.search import init_population
from mutators.humanoid import (
    random_genome as _default_random_genome,
    repair_genome as _default_repair_genome,
)

WIDEN_FACTOR = 1.5
MID_LOW = 0.3
MID_HIGH = 0.7


@dataclass
class SamplerSettings:
  """Bounds used when drawing random genomes."""

  noise_scale: float
  max_dqpos: float
  max_qvel: float
  sigma_dqpos: float
  sigma_qvel: float
  kind: str


def sampler_from_bundle(bundle: SearchBundle) -> SamplerSettings:
  cfg = bundle.ga_config
  return SamplerSettings(
      noise_scale=float(cfg.noise_scale),
      max_dqpos=float(cfg.max_dqpos),
      max_qvel=float(cfg.max_qvel),
      sigma_dqpos=float(cfg.sigma_dqpos),
      sigma_qvel=float(cfg.sigma_qvel),
      kind=bundle.kind,
  )


def apply_sampler_to_bundle(bundle: SearchBundle, sampler: SamplerSettings) -> None:
  """Write sampler bounds into the GA config."""
  bundle.ga_config = bundle.ga_config.replace(
      noise_scale=sampler.noise_scale,
      max_dqpos=sampler.max_dqpos,
      max_qvel=sampler.max_qvel,
      sigma_dqpos=sampler.sigma_dqpos,
      sigma_qvel=sampler.sigma_qvel,
  )


def widen_sampler(sampler: SamplerSettings) -> SamplerSettings:
  return replace(
      sampler,
      noise_scale=sampler.noise_scale * WIDEN_FACTOR,
      max_dqpos=sampler.max_dqpos * WIDEN_FACTOR,
      max_qvel=sampler.max_qvel * WIDEN_FACTOR,
      sigma_dqpos=sampler.sigma_dqpos * WIDEN_FACTOR,
      sigma_qvel=sampler.sigma_qvel * WIDEN_FACTOR,
  )


def _sampling_config(base: GAConfig, sampler: SamplerSettings, n: int) -> GAConfig:
  return base.replace(
      population_size=int(n),
      noise_scale=sampler.noise_scale,
      max_dqpos=sampler.max_dqpos,
      max_qvel=sampler.max_qvel,
      sigma_dqpos=sampler.sigma_dqpos,
      sigma_qvel=sampler.sigma_qvel,
  )


def sample_population(
    rng: jax.Array,
    bundle: SearchBundle,
    sampler: SamplerSettings,
    n: int,
) -> tuple[jax.Array, Population]:
  rng, sample_rng = jax.random.split(rng)
  cfg = _sampling_config(bundle.ga_config, sampler, n)
  operators = bundle.operators
  population = init_population(
      sample_rng,
      bundle.nq,
      bundle.nv,
      cfg,
      random_genome_fn=operators.get('random_genome_fn', _default_random_genome),
      repair_genome_fn=operators.get('repair_genome_fn', _default_repair_genome),
  )
  return rng, population


def evaluate_fail_rates(bundle: SearchBundle, population: Population) -> np.ndarray:
  from mutators.humanoid import genomes_to_test_suite as default_suite_fn

  suite_fn = bundle.operators.get('genomes_to_suite_fn', default_suite_fn)
  suite = suite_fn(population.genomes, bundle.qpos0, population.policy_rng)
  failed = bundle.execute_fn(suite)
  return np.asarray(jp.mean(failed.astype(jp.float32), axis=0))


def concat_populations(parts: list[Population]) -> Population:
  if not parts:
    raise ValueError('No populations to concatenate')
  if len(parts) == 1:
    return parts[0]
  return Population(
      genomes=jax.tree.map(
          lambda *xs: jp.concatenate(xs, axis=0),
          *[p.genomes for p in parts],
      ),
      policy_rng=jp.concatenate([p.policy_rng for p in parts], axis=0),
  )


def take_indices(population: Population, indices: np.ndarray) -> Population:
  idx = jp.asarray(indices, dtype=jp.int32)
  return Population(
      genomes=jax.tree.map(lambda x: x[idx], population.genomes),
      policy_rng=population.policy_rng[idx],
  )


def stratified_initial_population(
    rng: jax.Array,
    bundle: SearchBundle,
    *,
    n_policies: int,
    tries_per_widen: int = 8,
    max_widens: int = 6,
    batch_size: int | None = None,
) -> tuple[jax.Array, Population, SamplerSettings, dict]:
  """Fill easy / any-fail / mid-difficulty slots, widening the sampler on caps.

  Quotas are one quarter of the population each for:
    * easy: fail rate 0
    * any_fail: at least one policy fails (and not in the mid band)
    * mid: fail rate in [0.3, 0.7]
  Remaining slots are filled with fresh random samples.
  """
  pop_size = int(bundle.ga_config.population_size)
  if batch_size is None:
    batch_size = pop_size
  n_easy = max(1, pop_size // 4)
  n_any = max(1, pop_size // 4)
  n_mid = max(1, pop_size // 4)
  quotas = {'easy': n_easy, 'any_fail': n_any, 'mid': n_mid}
  kept: dict[str, list[Population]] = {
      'easy': [],
      'any_fail': [],
      'mid': [],
  }
  counts = {k: 0 for k in quotas}
  sampler = sampler_from_bundle(bundle)
  apply_sampler_to_bundle(bundle, sampler)
  n_batches = 0
  n_widens = 0
  found_any_failure = False

  def _absorb(population: Population, rates: np.ndarray) -> None:
    nonlocal found_any_failure
    if np.any(rates > 0):
      found_any_failure = True
    for name in ('mid', 'any_fail', 'easy'):
      if counts[name] >= quotas[name]:
        continue
      need = quotas[name] - counts[name]
      if name == 'mid':
        mask = (rates >= MID_LOW) & (rates <= MID_HIGH)
      elif name == 'any_fail':
        mask = (rates >= 1.0 / n_policies) & ~(
            (rates >= MID_LOW) & (rates <= MID_HIGH)
        )
      else:
        mask = rates <= 0.0
      idx = np.flatnonzero(mask)[:need]
      if idx.size == 0:
        continue
      kept[name].append(take_indices(population, idx))
      counts[name] += int(idx.size)

  def _filled() -> bool:
    return all(counts[k] >= quotas[k] for k in quotas)

  while not _filled() and n_widens <= max_widens:
    for _ in range(tries_per_widen):
      rng, population = sample_population(rng, bundle, sampler, batch_size)
      rates = evaluate_fail_rates(bundle, population)
      n_batches += 1
      _absorb(population, rates)
      print(
          f'  stratified batch {n_batches}: easy={counts["easy"]}/{n_easy}  '
          f'any_fail={counts["any_fail"]}/{n_any}  mid={counts["mid"]}/{n_mid}  '
          f'mean_fail={float(rates.mean()):.3f} max_fail={float(rates.max()):.3f}  '
          f'sampler noise={sampler.noise_scale:.3f} '
          f'max_dqpos={sampler.max_dqpos:.3f}'
      )
      if _filled():
        break
    if _filled():
      break
    if n_widens >= max_widens:
      break
    sampler = widen_sampler(sampler)
    apply_sampler_to_bundle(bundle, sampler)
    n_widens += 1
    print(
        f'  widen {n_widens}/{max_widens}: noise={sampler.noise_scale:.3f}  '
        f'max_dqpos={sampler.max_dqpos:.3f}  max_qvel={sampler.max_qvel:.3f}'
    )

  parts: list[Population] = []
  for name in ('easy', 'any_fail', 'mid'):
    parts.extend(kept[name])
  n_kept = sum(int(p.policy_rng.shape[0]) for p in parts) if parts else 0
  if n_kept < pop_size:
    rng, extra = sample_population(rng, bundle, sampler, pop_size - n_kept)
    parts.append(extra)
  if not parts:
    rng, population = sample_population(rng, bundle, sampler, pop_size)
  else:
    population = concat_populations(parts)
  n_now = int(population.policy_rng.shape[0])
  if n_now > pop_size:
    population = take_indices(population, np.arange(pop_size))
  elif n_now < pop_size:
    rng, extra = sample_population(rng, bundle, sampler, pop_size - n_now)
    population = concat_populations([population, extra])

  apply_sampler_to_bundle(bundle, sampler)
  info = {
      'counts': dict(counts),
      'quotas': dict(quotas),
      'n_batches': n_batches,
      'n_widens': n_widens,
      'found_any_failure': found_any_failure,
      'sampler': {
          'noise_scale': sampler.noise_scale,
          'max_dqpos': sampler.max_dqpos,
          'max_qvel': sampler.max_qvel,
          'kind': sampler.kind,
      },
  }
  print(
      f'Stratified init done: easy={counts["easy"]}/{n_easy}  '
      f'any_fail={counts["any_fail"]}/{n_any}  mid={counts["mid"]}/{n_mid}  '
      f'widens={n_widens} batches={n_batches}'
  )
  return rng, population, sampler, info
