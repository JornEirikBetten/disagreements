r"""Gait subspace from PCA on nominal locomotion actions.

Locomotion actions mix a quasi-periodic gait with aperiodic stabilization.
Of the usual ways to separate those, PCA on unperturbed successful actions
is the one that produces a *saveable subspace* \(W_{\mathrm{gait}}\) that
can be applied later to any action vector \(a_t \in \mathbb{R}^d\)
(including single-step disagreement modes):

1. **PCA gait synergies (this module).** Fit PCA on stacked actions from
   unperturbed, successful rollouts. The top \(k\) components
   (\(k \le 3\), \(\ge 80\%\) variance when possible) are the gait
   subspace. Residual energy \((I - W W^T)(a - \bar a)\) is non-gait
   torque. Same pipeline for all locomotion envs; no phase clock.
2. **FFT / band-pass.** Isolates a stride frequency on a *trajectory*,
   not a subspace you can apply to a single \(a_t\). Needs a time series
   at every later evaluation.
3. **Phase-locked limit cycle \(\bar a(\theta)\).** Natural for a planar
   walker; ambiguous in 3D Humanoid (multi-limb, yaw drift). Stores
   a lookup table, not \(W_{\mathrm{gait}}\).
4. **Attitude vs propulsive joint masks.** A static partition of actuators,
   not cyclic vs aperiodic energy *within* a joint.

Decomposition is mean-centered PCA (the fit that produced \(W\)):

    a_gait = \\bar a + W W^T (a - \\bar a)
    a_stabilize = (I - W W^T)(a - \\bar a)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from collections.abc import Sequence

import numpy as np

VARIANCE_THRESHOLD = 0.80
MAX_COMPONENTS = 3
MIN_COMPONENTS = 1
DEFAULT_ROOT = os.path.join('datasets', 'gait_subspaces')


@dataclass
class GaitSubspace:
  """Orthonormal gait basis for one policy."""

  mean_action: np.ndarray  # (d,)
  W_gait: np.ndarray  # (d, k), orthonormal columns
  explained_variance_ratio: np.ndarray  # (k,)
  full_explained_variance_ratio: np.ndarray  # (r,) all SVD modes
  n_samples: int
  n_successful_rollouts: int
  n_rollouts: int
  environment: str
  algorithm: str
  policy_id: int
  dt: float
  burnin_steps: int
  rollout_length: int
  used_partial_rollouts: bool = False
  stride_hz: float | None = None

  @property
  def k(self) -> int:
    return int(self.W_gait.shape[1])

  @property
  def action_dim(self) -> int:
    return int(self.W_gait.shape[0])

  @property
  def cumulative_explained(self) -> float:
    return float(np.sum(self.explained_variance_ratio))

  def project_gait(self, actions: np.ndarray) -> np.ndarray:
    """Reconstruct the gait component. ``actions`` is ``(..., d)``."""
    centered = np.asarray(actions, dtype=np.float64) - self.mean_action
    scores = centered @ self.W_gait
    return scores @ self.W_gait.T + self.mean_action

  def residual(self, actions: np.ndarray) -> np.ndarray:
    """Non-gait (stabilization) residual. ``actions`` is ``(..., d)``."""
    return np.asarray(actions, dtype=np.float64) - self.project_gait(actions)

  def decompose(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a_gait, a_stabilize)`` with the same leading shape as ``actions``."""
    gait = self.project_gait(actions)
    return gait, np.asarray(actions, dtype=np.float64) - gait

  def centered_gait_and_residual(
      self, actions: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean-centered gait and orthogonal residual in action space.

    These live in the subspaces \(W W^T\) and \(I - W W^T\), so they are
    the right representation for pairwise cosine disagreement (the mean
    gait offset is not mixed into the gait vectors).
    """
    centered = np.asarray(actions, dtype=np.float64) - self.mean_action
    gait = (centered @ self.W_gait) @ self.W_gait.T
    return gait, centered - gait


def choose_n_components(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float = VARIANCE_THRESHOLD,
    max_components: int = MAX_COMPONENTS,
    min_components: int = MIN_COMPONENTS,
) -> int:
  r"""Smallest \(k\) reaching ``variance_threshold``, clipped to ``[min, max]``."""
  ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
  if ratios.size == 0:
    return min_components
  cumsum = np.cumsum(ratios)
  k = int(np.searchsorted(cumsum, variance_threshold) + 1)
  k = max(min_components, min(max_components, k, ratios.size))
  return k


def fit_gait_pca(
    actions: np.ndarray,
    variance_threshold: float = VARIANCE_THRESHOLD,
    max_components: int = MAX_COMPONENTS,
    min_components: int = MIN_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
  r"""Fit PCA on stacked action rows.

  Args:
    actions: Nominal action matrix \(A \in \mathbb{R}^{N \times d}\).

  Returns:
    ``(mean_action, W_gait, full_explained_variance_ratio, k)``.
  """
  x = np.asarray(actions, dtype=np.float64)
  if x.ndim != 2:
    raise ValueError(f'actions must be 2-D (N, d), got shape {x.shape}')
  n, d = x.shape
  if n < 2:
    raise ValueError(f'Need at least 2 action samples, got {n}')
  mean = x.mean(axis=0)
  centered = x - mean
  _, singular, vt = np.linalg.svd(centered, full_matrices=False)
  eigvals = (singular ** 2) / max(n - 1, 1)
  total = float(eigvals.sum())
  if total <= 0:
    ratios = np.zeros_like(eigvals)
  else:
    ratios = eigvals / total
  k = choose_n_components(
      ratios,
      variance_threshold=variance_threshold,
      max_components=min(max_components, d),
      min_components=min(min_components, d),
  )
  w_gait = vt[:k].T
  return mean, w_gait, ratios, k


def gait_subspace_path(
    environment: str,
    algorithm: str,
    policy_id: int,
    root: str = DEFAULT_ROOT,
) -> str:
  return os.path.join(
      root, environment, algorithm, f'policy_{int(policy_id)}.npz'
  )


def save_gait_subspace(path: str, subspace: GaitSubspace) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  np.savez_compressed(
      path,
      mean_action=np.asarray(subspace.mean_action, dtype=np.float32),
      W_gait=np.asarray(subspace.W_gait, dtype=np.float32),
      explained_variance_ratio=np.asarray(
          subspace.explained_variance_ratio, dtype=np.float32
      ),
      full_explained_variance_ratio=np.asarray(
          subspace.full_explained_variance_ratio, dtype=np.float32
      ),
      n_samples=np.int32(subspace.n_samples),
      n_successful_rollouts=np.int32(subspace.n_successful_rollouts),
      n_rollouts=np.int32(subspace.n_rollouts),
      environment=np.asarray(subspace.environment),
      algorithm=np.asarray(subspace.algorithm),
      policy_id=np.int32(subspace.policy_id),
      dt=np.float32(subspace.dt),
      burnin_steps=np.int32(subspace.burnin_steps),
      rollout_length=np.int32(subspace.rollout_length),
      used_partial_rollouts=np.bool_(subspace.used_partial_rollouts),
      stride_hz=(
          np.float32(subspace.stride_hz)
          if subspace.stride_hz is not None
          else np.array(np.nan, dtype=np.float32)
      ),
  )


def load_gait_subspace(path: str) -> GaitSubspace:
  with np.load(path, allow_pickle=False) as data:
    stride = float(data['stride_hz'])
    return GaitSubspace(
        mean_action=np.asarray(data['mean_action'], dtype=np.float64),
        W_gait=np.asarray(data['W_gait'], dtype=np.float64),
        explained_variance_ratio=np.asarray(
            data['explained_variance_ratio'], dtype=np.float64
        ),
        full_explained_variance_ratio=np.asarray(
            data['full_explained_variance_ratio'], dtype=np.float64
        ),
        n_samples=int(data['n_samples']),
        n_successful_rollouts=int(data['n_successful_rollouts']),
        n_rollouts=int(data['n_rollouts']),
        environment=str(data['environment']),
        algorithm=str(data['algorithm']),
        policy_id=int(data['policy_id']),
        dt=float(data['dt']),
        burnin_steps=int(data['burnin_steps']),
        rollout_length=int(data['rollout_length']),
        used_partial_rollouts=bool(data['used_partial_rollouts']),
        stride_hz=None if np.isnan(stride) else stride,
    )


def load_gait_subspace_for_policy(
    environment: str,
    algorithm: str,
    policy_id: int,
    root: str = DEFAULT_ROOT,
) -> GaitSubspace:
  return load_gait_subspace(
      gait_subspace_path(environment, algorithm, policy_id, root=root)
  )


def load_gait_subspaces_for_policies(
    environment: str,
    algorithm: str,
    policy_ids: Sequence[int],
    root: str = DEFAULT_ROOT,
    algorithms: Sequence[str] | None = None,
) -> list[GaitSubspace]:
  """Load one subspace per policy, in ``policy_ids`` order.

  ``algorithms`` overrides ``algorithm`` per policy. A mixed PPO/SAC
  ensemble passes one algorithm name per id so each subspace is read from
  the directory of the checkpoint that was actually in the search.
  """
  policy_ids = [int(policy_id) for policy_id in policy_ids]
  if algorithms is None:
    algorithm_names = [algorithm] * len(policy_ids)
  else:
    algorithm_names = [str(name) for name in algorithms]
    if len(algorithm_names) != len(policy_ids):
      raise ValueError(
          f'Need one algorithm per policy ({len(policy_ids)}), '
          f'got {len(algorithm_names)}'
      )
  subspaces = []
  for name, policy_id in zip(algorithm_names, policy_ids):
    subspace = load_gait_subspace_for_policy(
        environment, name, policy_id, root=root
    )
    if (
        subspace.environment != environment
        or subspace.algorithm != name
        or subspace.policy_id != policy_id
    ):
      raise ValueError(
          f'Gait subspace {environment}/{name}/policy_{policy_id} '
          f'is labeled {subspace.environment}/{subspace.algorithm}/'
          f'policy_{subspace.policy_id}'
      )
    subspaces.append(subspace)
  return subspaces


def decompose_ensemble_actions(
    actions: np.ndarray,
    subspaces: Sequence[GaitSubspace],
) -> tuple[np.ndarray, np.ndarray]:
  """Project each policy's actions with that policy's gait subspace.

  Args:
    actions: ``(P, N, d)`` first-step (or per-case) actions.
    subspaces: length ``P``, one ``GaitSubspace`` per policy.

  Returns:
    ``(gait, residual)`` each ``(P, N, d)``, mean-centered components.
  """
  stacked = np.asarray(actions, dtype=np.float64)
  if stacked.ndim != 3:
    raise ValueError(f'actions must be (P, N, d), got shape {stacked.shape}')
  n_policies, _n_cases, action_dim = stacked.shape
  if len(subspaces) != n_policies:
    raise ValueError(
        f'Need {n_policies} gait subspaces, got {len(subspaces)}'
    )
  gait = np.empty_like(stacked)
  residual = np.empty_like(stacked)
  for i, subspace in enumerate(subspaces):
    if subspace.action_dim != action_dim:
      raise ValueError(
          f'Policy {subspace.policy_id} gait subspace has d='
          f'{subspace.action_dim}, actions have d={action_dim}'
      )
    gait[i], residual[i] = subspace.centered_gait_and_residual(stacked[i])
  return gait, residual


def dominant_frequency_hz(signal: np.ndarray, dt: float) -> float | None:
  """Peak frequency of a 1-D signal, skipping the DC bin."""
  x = np.asarray(signal, dtype=np.float64)
  if x.size < 8 or dt <= 0:
    return None
  centered = x - x.mean()
  spectrum = np.abs(np.fft.rfft(centered))
  freqs = np.fft.rfftfreq(centered.size, d=dt)
  if spectrum.size < 2:
    return None
  peak = 1 + int(np.argmax(spectrum[1:]))
  return float(freqs[peak])
