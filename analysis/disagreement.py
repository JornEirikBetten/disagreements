"""Disagreement metrics between policy attribution profiles."""

import jax
import numpy as np
from jax import numpy as jp

from attributions.types import SuiteAttributionResult


def state_ig_vectors(attributions: SuiteAttributionResult) -> jax.Array:
  """Concatenate state IG attributions. Shape (P, N, nq + nv)."""
  return jp.concatenate(
      [
          attributions.state_integrated_gradient_qpos,
          attributions.state_integrated_gradient_qvel,
      ],
      axis=-1,
  )


def observation_ig_vectors(attributions: SuiteAttributionResult) -> jax.Array:
  """Observation IG attributions. Shape (P, N, obs_dim)."""
  return attributions.observation_integrated_gradient


def mean_pairwise_cosine_distance(vectors: jax.Array) -> jax.Array:
  """Mean pairwise cosine distance for P attribution vectors.

  Cosine distance is scale-invariant and suited to comparing IG attribution
  directions across policies that may differ in overall sensitivity magnitude.

  Args:
    vectors: (P, D) attribution vectors, one per policy.

  Returns:
    Scalar mean pairwise distance in [0, 2].
  """
  norms = jp.linalg.norm(vectors, axis=-1, keepdims=True)
  normed = vectors / (norms + 1e-8)
  similarity = normed @ normed.T
  distance = 1.0 - similarity
  p = vectors.shape[0]
  if p < 2:
    return jp.array(0.0)
  mask = jp.triu(jp.ones((p, p), dtype=bool), k=1)
  return jp.sum(distance[mask]) / jp.sum(mask)


def mean_pairwise_l2_distance(vectors: jax.Array) -> jax.Array:
  """Mean pairwise L2 distance for P attribution vectors."""
  p = vectors.shape[0]
  if p < 2:
    return jp.array(0.0)

  def pair_distance(i, j):
    return jp.linalg.norm(vectors[i] - vectors[j])

  idx_i, idx_j = jp.triu_indices(p, k=1)
  distances = jax.vmap(pair_distance)(idx_i, idx_j)
  return jp.mean(distances)


def attribution_disagreement(
    policy_attributions: jax.Array,
    metric: str = 'cosine',
) -> jax.Array:
  """Compute per-test-case disagreement across policies.

  Args:
    policy_attributions: (P, N, D) IG vectors.
    metric: ``'cosine'`` (recommended for IG) or ``'l2'``.

  Returns:
    (N,) disagreement scores.
  """
  if metric == 'cosine':
    distance_fn = mean_pairwise_cosine_distance
  elif metric == 'l2':
    distance_fn = mean_pairwise_l2_distance
  else:
    raise ValueError(f'Unknown metric: {metric}')

  return jax.vmap(distance_fn, in_axes=1)(policy_attributions)


def spearman_correlation(x, y) -> float:
  """Spearman rank correlation between two 1D arrays."""
  x = np.asarray(x)
  y = np.asarray(y)
  if x.size < 2:
    return float('nan')
  rx = np.argsort(np.argsort(x))
  ry = np.argsort(np.argsort(y))
  return float(np.corrcoef(rx, ry)[0, 1])


def pearson_correlation(x, y) -> float:
  """Pearson correlation between two 1D arrays."""
  x = np.asarray(x)
  y = np.asarray(y)
  if x.size < 2:
    return float('nan')
  return float(np.corrcoef(x, y)[0, 1])

def mean_pairwise_cosine_distance_np(vectors: np.ndarray) -> float:
  """NumPy mean pairwise cosine distance for P vectors of shape (P, D)."""
  p = vectors.shape[0]
  if p < 2:
    return 0.0
  norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
  normed = vectors / (norms + 1e-8)
  similarity = normed @ normed.T
  distance = 1.0 - similarity
  idx_i, idx_j = np.triu_indices(p, k=1)
  return float(np.mean(distance[idx_i, idx_j]))


def pairwise_disagreement_np(policy_vectors: np.ndarray) -> np.ndarray:
  """Per-case disagreement across policies. Input shape (P, N, D), output (N,)."""
  _, n_cases, _ = policy_vectors.shape
  return np.array([
      mean_pairwise_cosine_distance_np(policy_vectors[:, i, :])
      for i in range(n_cases)
  ])


def pairwise_topk_attribution_cosine_all_k(ig: np.ndarray) -> np.ndarray:
  """Mean pairwise cosine distance on per-policy top-k masked IG vectors.

  For each policy and test case, the ``k`` coordinates of largest ``|IG|``
  are kept and the rest are zeroed. Disagreement at ``k`` is then the usual
  mean pairwise cosine distance. At ``k = D`` this matches
  :func:`pairwise_disagreement_np`.

  Args:
    ig: ``(P, N, D)`` signed feature attributions (typically state IG).

  Returns:
    ``(N, D)`` distances. Column ``k - 1`` uses top-``k``.
  """
  ig = np.asarray(ig, dtype=np.float64)
  if ig.ndim != 3:
    raise ValueError(f'ig must have shape (P, N, D), got {ig.shape}')
  n_policies, n_cases, n_features = ig.shape
  if n_features < 1:
    raise ValueError('ig must have at least one feature')
  if n_policies < 2:
    return np.zeros((n_cases, n_features), dtype=np.float64)

  order = np.argsort(-np.abs(ig), axis=-1)
  ranks = np.empty_like(order)
  policy_ix = np.arange(n_policies)[:, None, None]
  case_ix = np.arange(n_cases)[None, :, None]
  ranks[policy_ix, case_ix, order] = np.arange(n_features)

  sq_ordered = np.take_along_axis(ig * ig, order, axis=-1)
  energy = np.cumsum(sq_ordered, axis=-1)

  idx_i, idx_j = np.triu_indices(n_policies, k=1)
  n_pairs = int(idx_i.size)
  both_from = np.maximum(ranks[idx_i], ranks[idx_j])
  products = ig[idx_i] * ig[idx_j]
  dots = np.zeros((n_pairs, n_cases, n_features), dtype=np.float64)
  pair_ix = np.arange(n_pairs)[:, None, None]
  np.add.at(dots, (pair_ix, case_ix, both_from), products)
  dots = np.cumsum(dots, axis=-1)

  norms_i = np.sqrt(energy[idx_i]) + 1e-8
  norms_j = np.sqrt(energy[idx_j]) + 1e-8
  distance = 1.0 - dots / (norms_i * norms_j)
  return np.mean(distance, axis=0)


def mean_pairwise_abs_difference_np(values: np.ndarray) -> float:
  """Mean pairwise absolute difference for P scalar values."""
  p = values.shape[0]
  if p < 2:
    return 0.0
  idx_i, idx_j = np.triu_indices(p, k=1)
  return float(np.mean(np.abs(values[idx_i] - values[idx_j])))


def _policy_scale(action_scale: np.ndarray) -> np.ndarray:
  """Gaussian scale σ as ``(P, N, n_actuators)``."""
  scale = np.asarray(action_scale, dtype=np.float64)
  if scale.ndim == 2:
    scale = scale[:, :, None]
  if scale.ndim != 3:
    raise ValueError(
        'action_scale must have shape (P, N) or (P, N, n_actuators), '
        f'got {scale.shape}'
    )
  return scale


def _mean_pairwise_abs_rows(values: np.ndarray) -> np.ndarray:
  """Mean pairwise absolute difference across policies.

  ``values`` is ``(P, N)``. The result is ``(N,)``.
  """
  n_policies = values.shape[0]
  if n_policies < 2:
    return np.zeros(values.shape[1], dtype=np.float64)
  idx_i, idx_j = np.triu_indices(n_policies, k=1)
  return np.mean(np.abs(values[idx_i] - values[idx_j]), axis=0)


def criticality_log_volume_np(action_scale: np.ndarray) -> np.ndarray:
  """``sum_i log σ_i`` for each policy and case. Shape ``(P, N)``.

  For a diagonal Gaussian this is the log-volume of the action covariance,
  and the differential entropy up to an additive constant.
  ``action_scale`` is σ with shape ``(P, N)`` or ``(P, N, n_actuators)``.
  """
  scale = _policy_scale(action_scale)
  return np.sum(np.log(np.maximum(scale, 1e-12)), axis=-1)


def criticality_disagreement_np(action_scale: np.ndarray) -> np.ndarray:
  """Mean pairwise absolute difference of ``sum_i log σ_i``.

  The additive entropy constant drops out of the disagreement.
  ``action_scale`` is σ with shape ``(P, N)`` or ``(P, N, n_actuators)``.

  Returns:
    ``(N,)`` disagreement scores.
  """
  return _mean_pairwise_abs_rows(criticality_log_volume_np(action_scale))


def standardized_criticality_disagreement_np(log_volume: np.ndarray) -> np.ndarray:
  """Mean pairwise absolute difference of per-policy z-scored criticality.

  ``log_volume`` has shape ``(P, N)`` and is ``sum_i log σ_i``. Each policy
  is standardized across the N cases, then disagreement is the mean pairwise
  absolute difference. A constant offset between policies, such as the
  PPO–SAC entropy gap, drops out. The sign of the log-volume does not
  matter: absolute differences are unchanged if every entry is negated.

  Returns:
    ``(N,)`` disagreement scores.
  """
  volume = np.asarray(log_volume, dtype=np.float64)
  if volume.ndim != 2:
    raise ValueError(
        f'log_volume must have shape (P, N), got {volume.shape}'
    )
  return _mean_pairwise_abs_rows(standardize_per_policy_np(volume))


def std_magnitude_disagreement_np(action_scale: np.ndarray) -> np.ndarray:
  """Mean pairwise absolute difference of ``||σ||_2``.

  Returns:
    ``(N,)`` disagreement scores.
  """
  scale = _policy_scale(action_scale)
  magnitude = np.linalg.norm(scale, axis=-1)
  return _mean_pairwise_abs_rows(magnitude)


def std_direction_disagreement_np(action_scale: np.ndarray) -> np.ndarray:
  """Mean pairwise cosine distance of the per-policy σ vectors.

  Returns:
    ``(N,)`` disagreement scores.
  """
  return pairwise_disagreement_np(_policy_scale(action_scale))


def action_variance_disagreement_np(action_scale: np.ndarray) -> np.ndarray:
  """Alias of :func:`criticality_disagreement_np`."""
  return criticality_disagreement_np(action_scale)


def standardize_per_policy_np(values: np.ndarray) -> np.ndarray:
  """Z-score each policy's values across cases. Input (P, N), output (P, N)."""
  mean = np.mean(values, axis=1, keepdims=True)
  std = np.std(values, axis=1, keepdims=True)
  return (values - mean) / (std + 1e-8)


def minimum_value_zscore_np(values: np.ndarray) -> np.ndarray:
  """Lowest per-policy z-score at each case. Input (P, N), output (N,).

  Each policy is standardized across cases, then the minimum is taken over
  policies. A low score means some policy's value sits far below that
  policy's own mean.
  """
  standardized = standardize_per_policy_np(np.asarray(values, dtype=np.float64))
  if standardized.ndim != 2:
    raise ValueError(
        f'values must have shape (P, N), got {standardized.shape}'
    )
  return np.min(standardized, axis=0)


def value_estimates_np(
    values: np.ndarray,
    q_values: np.ndarray | None = None,
    *,
    algorithm: str | None = None,
) -> np.ndarray:
  """Scalar state value for disagreement. Shape ``(P, N)``.

  A SAC policy uses the minimum of its two critics. A PPO policy keeps the
  stored state value. In a mixed ensemble the two sit on different rows:
  critic rows are finite in ``q_values``, and PPO rows are NaN.
  """
  estimates = np.asarray(values, dtype=np.float64)
  if estimates.ndim != 2:
    raise ValueError(
        f'values must have shape (P, N), got {estimates.shape}'
    )
  needs_critics = algorithm in ('sac', 'ppo_sac')
  if q_values is None:
    if needs_critics:
      raise ValueError(
          'SAC value disagreement uses the minimum of the two critics. '
          'q_values is missing; run add_critic_values.py.'
      )
    return estimates
  critics = np.asarray(q_values, dtype=np.float64)
  expected = estimates.shape + (2,)
  if critics.shape != expected:
    raise ValueError(
        f'q_values must have shape {expected}, got {critics.shape}'
    )
  critic_rows = np.any(np.isfinite(critics), axis=(1, 2))
  if needs_critics and algorithm == 'sac' and not np.all(critic_rows):
    raise ValueError('A SAC policy is missing its twin critics.')
  if needs_critics and algorithm == 'ppo_sac' and not np.any(critic_rows):
    raise ValueError('The mixed ensemble has no SAC critics on q_values.')
  if np.any(critic_rows) and not np.all(np.isfinite(critics[critic_rows])):
    raise ValueError('A SAC critic row is only partly filled.')
  if not np.any(critic_rows):
    return estimates
  out = np.array(estimates, copy=True)
  out[critic_rows] = np.min(critics[critic_rows], axis=-1)
  return out


def standardized_value_disagreement_np(values: np.ndarray) -> np.ndarray:
  """Disagreement of per-policy standardized value estimates.

  Each policy's value estimates are z-scored across the N test cases, then
  disagreement is the mean pairwise absolute difference across policies.
  For SAC, pass the minimum of the two critics as ``values``.

  Args:
    values: (P, N) value estimates.

  Returns:
    (N,) disagreement scores.
  """
  standardized = standardize_per_policy_np(values)
  return _mean_pairwise_abs_rows(standardized)
