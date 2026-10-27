"""Extract disagreement metrics from a GA search collection."""

import numpy as np

from analysis.disagreement import (
    criticality_log_volume_np,
    mean_pairwise_abs_difference_np,
    pairwise_disagreement_np,
    standardized_criticality_disagreement_np,
    standardized_value_disagreement_np,
    value_estimates_np,
)
from analysis.energy import energy_metrics_from_collection
from ga.collect import GASearchCollection


def _is_categorical_actions(actions: np.ndarray) -> bool:
  """True when ``actions`` is a batch of categorical probability vectors."""
  if actions.ndim != 3:
    return False
  totals = np.sum(actions, axis=-1)
  return bool(np.allclose(totals, 1.0, atol=1e-3))


def _flatten_trailing(vectors: np.ndarray) -> np.ndarray:
  """Ensure shape (P, N, D) for pairwise metrics."""
  if vectors.ndim == 3:
    return vectors
  p, n = vectors.shape[:2]
  return np.asarray(vectors).reshape(p, n, -1)


def empirical_action_choice_entropy_np(action_probs: np.ndarray) -> np.ndarray:
  """Entropy of the argmax histogram across policies.

  Args:
    action_probs: (P, N, A) categorical probabilities.

  Returns:
    (N,) entropies in nats. Zero when every policy picks the same action.
  """
  argmax = np.argmax(action_probs, axis=-1)
  n_policies, n_cases = argmax.shape
  n_actions = action_probs.shape[-1]
  one_hot = np.eye(n_actions, dtype=np.float64)[argmax]
  counts = np.sum(one_hot, axis=0)
  probs = counts / n_policies
  safe = np.clip(probs, 1e-12, 1.0)
  return -np.sum(np.where(counts > 0, probs * np.log(safe), 0.0), axis=-1)


def _pairwise_abs_np(values: np.ndarray) -> np.ndarray:
  """Mean pairwise |diff| for (P, N) or (P, N, 1) arrays -> (N,)."""
  values = np.asarray(values)
  if values.ndim == 3:
    values = np.mean(values, axis=-1)
  n_cases = values.shape[1]
  return np.array([
      mean_pairwise_abs_difference_np(values[:, i])
      for i in range(n_cases)
  ])


def disagreement_metrics_from_collection(
    collection: GASearchCollection,
    *,
    algorithm: str | None = None,
) -> dict[str, np.ndarray]:
  """Flatten per-test-case difficulty and disagreement scores from a collection.

  Returns arrays of equal length (total encounters across all generations):
    - difficulty
    - action_disagreement: mean pairwise cosine distance on policy actions
    - action_std_mean: mean over action dims of per-case policy action std
    - action_choice_spread: entropy of the empirical argmax histogram across
      policies (discrete analogue of action-location spread)
    - action_scale_mean: mean over policies and action dims of Gaussian scale,
      or mean categorical entropy for discrete policies
    - action_scale_var: mean over action dims of variance of scale across policies
    - action_scale_log2_sq: mean over policies/dims of log2(scale^2) (~entropy)
    - state_ig_disagreement: mean pairwise cosine distance on state IG vectors
    - state_importance_disagreement: cosine disagreement on |state IG|, or
      mean pairwise |Δentropy| when importance is action-distribution entropy
    - criticality_disagreement: mean pairwise |Δ| of per-policy z-scored
      sum_i log σ_i. Each policy is standardized across every case in the
      collection, which is one experiment.
      for Gaussian policies; mean pairwise |Δentropy| for categorical policies
    - obs_ig_disagreement: mean pairwise cosine distance on observation IG vectors
    - value_disagreement: mean pairwise |diff| of per-policy standardized values.
      Each policy is z-scored across every case in the collection. SAC uses
      the minimum of the two critics; PPO uses V(s)
    - values_mean: mean value estimate across policies
    - values_std: std of raw value estimates across policies
  """
  difficulties = []
  action_disagreements = []
  action_std_means = []
  action_choice_spreads = []
  action_scale_means = []
  action_scale_vars = []
  action_scale_log2_sqs = []
  state_ig_disagreements = []
  state_importance_disagreements = []
  criticality_disagreements = []
  criticality_volumes = []
  obs_ig_disagreements = []
  value_pieces = []
  values_means = []
  values_stds = []
  missing_values = False
  discrete = False

  for record in collection.generations:
    difficulties.append(record.difficulty)
    actions = np.asarray(record.actions)
    discrete = _is_categorical_actions(actions)
    action_disagreements.append(
        pairwise_disagreement_np(_flatten_trailing(actions))
    )
    # action_std: (N, nu) — std across policies per (case, dim), then mean over dims
    action_std_means.append(np.mean(record.action_std, axis=-1))
    if discrete:
      action_choice_spreads.append(empirical_action_choice_entropy_np(actions))
    # action_scale: (P, N, nu) Gaussian σ, or (P, N, 1) categorical entropy
    action_scale_means.append(np.mean(record.action_scale, axis=(0, -1)))
    action_scale_vars.append(
        np.mean(np.var(record.action_scale, axis=0), axis=-1)
    )
    # log2(σ²) is proportional to Gaussian entropy per action dim
    action_scale_log2_sqs.append(
        np.mean(np.log2(np.square(record.action_scale) + 1e-12), axis=(0, -1))
    )
    state_ig = np.concatenate(
        [
            _flatten_trailing(record.state_ig_qpos),
            _flatten_trailing(record.state_ig_qvel),
        ],
        axis=-1,
    )
    state_ig_disagreements.append(pairwise_disagreement_np(state_ig))
    if hasattr(record, 'state_importance_qpos'):
      state_importance = np.concatenate(
          [
              _flatten_trailing(record.state_importance_qpos),
              _flatten_trailing(record.state_importance_qvel),
          ],
          axis=-1,
      )
    else:
      state_importance = np.abs(state_ig)
    if discrete:
      state_importance_disagreements.append(_pairwise_abs_np(state_importance))
      criticality_disagreements.append(_pairwise_abs_np(record.action_scale))
    else:
      state_importance_disagreements.append(
          pairwise_disagreement_np(state_importance)
      )
      criticality_volumes.append(
          criticality_log_volume_np(record.action_scale)
      )
    obs_ig_disagreements.append(
        pairwise_disagreement_np(_flatten_trailing(record.obs_ig))
    )

    if hasattr(record, 'values'):
      values = value_estimates_np(
          record.values,
          getattr(record, 'q_values', None),
          algorithm=algorithm,
      )
      value_pieces.append(values)
      values_means.append(np.mean(values, axis=0))
      values_stds.append(np.std(values, axis=0))
    else:
      missing_values = True

  if criticality_volumes and criticality_disagreements:
    raise ValueError(
        'A collection mixes categorical and Gaussian action scales.'
    )
  if criticality_volumes:
    criticality = standardized_criticality_disagreement_np(
        np.concatenate(criticality_volumes, axis=1)
    )
  else:
    criticality = np.concatenate(criticality_disagreements)
  metrics = {
      'difficulty': np.concatenate(difficulties),
      'action_disagreement': np.concatenate(action_disagreements),
      'action_std_mean': np.concatenate(action_std_means),
      'action_scale_mean': np.concatenate(action_scale_means),
      'action_scale_var': np.concatenate(action_scale_vars),
      'action_scale_log2_sq': np.concatenate(action_scale_log2_sqs),
      'state_ig_disagreement': np.concatenate(state_ig_disagreements),
      'state_importance_disagreement': np.concatenate(
          state_importance_disagreements
      ),
      'criticality_disagreement': criticality,
      'obs_ig_disagreement': np.concatenate(obs_ig_disagreements),
  }
  if action_choice_spreads:
    metrics['action_choice_spread'] = np.concatenate(action_choice_spreads)
  if value_pieces and missing_values:
    raise ValueError(
        'Value estimates are present on some generations and missing on others.'
    )
  if value_pieces:
    # One scale per policy across the whole collection, which is one experiment.
    stacked = np.concatenate(value_pieces, axis=1)
    metrics['value_disagreement'] = standardized_value_disagreement_np(stacked)
    metrics['values_mean'] = np.concatenate(values_means)
    metrics['values_std'] = np.concatenate(values_stds)
  energy = energy_metrics_from_collection(collection)
  for key, values in energy.items():
    if values.shape[0] != metrics['difficulty'].shape[0]:
      raise ValueError(
          f'Energy observable {key} has length {values.shape[0]}, '
          f'expected {metrics["difficulty"].shape[0]}'
      )
    metrics[key] = values
  return metrics


def gait_action_disagreement_metrics_from_collection(
    collection: GASearchCollection,
    environment: str,
    algorithm: str,
):
  """Pairwise cosine disagreement of gait and non-gait action components.

  Each policy's first-step action is projected with that policy's saved
  PCA gait subspace. Disagreement is then the usual mean pairwise cosine
  distance across the ensemble, computed separately in the gait subspace
  and in its orthogonal complement.
  """
  from analysis.gait_subspace import (
      decompose_ensemble_actions,
      load_gait_subspaces_for_policies,
  )

  policy_ids = list(collection.policy_ids)
  if not policy_ids:
    raise ValueError(
        'Collection has no policy_ids; cannot load gait subspaces.'
    )
  subspaces = load_gait_subspaces_for_policies(
      environment, algorithm, policy_ids
  )
  difficulties = []
  gait_disagreements = []
  nongait_disagreements = []
  for record in collection.generations:
    difficulties.append(record.difficulty)
    gait, residual = decompose_ensemble_actions(record.actions, subspaces)
    gait_disagreements.append(pairwise_disagreement_np(gait))
    nongait_disagreements.append(pairwise_disagreement_np(residual))
  return {
      'difficulty': np.concatenate(difficulties),
      'gait_action_disagreement': np.concatenate(gait_disagreements),
      'nongait_action_disagreement': np.concatenate(nongait_disagreements),
  }

