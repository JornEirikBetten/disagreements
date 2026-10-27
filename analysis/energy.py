r"""Energy-based adequacy observables from ensemble actions and joint velocities.

These are *not* the Gaussian ``action_scale`` (σ) stored with policy
attributions. Ensemble action scale is the mean L2 control effort, and
disagreement is computed over per-policy effort rather than over action
direction.

For a start state \(s_0\), ensemble size \(E\), horizon \(K\), and action
dimension \(d\):

.. math::

    \mathcal{E}_{scale}(s_0)
    = \frac{1}{E K} \sum_{e=1}^{E} \sum_{t=0}^{K-1}
      \lVert a_t^{(e)} \rVert_2

    \mathcal{P}(s_0)
    = \frac{1}{E K} \sum_{e=1}^{E} \sum_{t=0}^{K-1}
      \lvert a_t^{(e)} \cdot v_t^{(e)} \rvert

Actuated velocities \(v_t\) are the last \(d\) components of generalized
velocity (the hinge joints commanded by the policy). First-step collections
use \(K=1\) and the test-case \(qvel\) at \(t=0\).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.disagreement import mean_pairwise_abs_difference_np, pairwise_disagreement_np


def actuated_velocities(qvel: np.ndarray, n_actuators: int) -> np.ndarray:
  """Joint velocities commanded by the policy. Last ``n_actuators`` of ``qvel``."""
  qvel = np.asarray(qvel)
  if n_actuators < 1:
    raise ValueError(f'n_actuators must be >= 1, got {n_actuators}')
  if qvel.shape[-1] < n_actuators:
    raise ValueError(
        f'qvel last axis {qvel.shape[-1]} is smaller than n_actuators='
        f'{n_actuators}'
    )
  return qvel[..., -int(n_actuators) :]


def _as_time_actions(actions: np.ndarray, horizon: int | None) -> np.ndarray:
  """Return actions as ``(P, N, K, d)``."""
  stacked = np.asarray(actions, dtype=np.float64)
  if stacked.ndim == 3:
    stacked = stacked[:, :, None, :]
  elif stacked.ndim != 4:
    raise ValueError(
        f'actions must be (P, N, d) or (P, N, T, d), got {stacked.shape}'
    )
  k_max = stacked.shape[2]
  k = k_max if horizon is None else int(horizon)
  if k < 1:
    raise ValueError(f'horizon must be >= 1, got {k}')
  if k > k_max:
    raise ValueError(f'horizon {k} exceeds stored length {k_max}')
  return stacked[:, :, :k, :]


def _as_time_qvel(
    qvel: np.ndarray,
    n_policies: int,
    n_cases: int,
    horizon: int,
) -> np.ndarray:
  """Return qvel as ``(P, N, K, nv)``, broadcasting a shared start-state qvel."""
  stacked = np.asarray(qvel, dtype=np.float64)
  if stacked.ndim == 2:
    if stacked.shape[0] != n_cases:
      raise ValueError(
          f'qvel has {stacked.shape[0]} cases, expected {n_cases}'
      )
    stacked = np.broadcast_to(
        stacked[None, :, None, :],
        (n_policies, n_cases, horizon, stacked.shape[-1]),
    )
    return np.asarray(stacked)
  if stacked.ndim == 3:
    if stacked.shape[0] == n_policies and stacked.shape[1] == n_cases:
      stacked = stacked[:, :, None, :]
    elif stacked.shape[0] == n_cases:
      stacked = np.broadcast_to(
          stacked[None, :, :, :],
          (n_policies, n_cases, stacked.shape[1], stacked.shape[-1]),
      )
      stacked = np.asarray(stacked)
    else:
      raise ValueError(
          f'qvel shape {tuple(np.asarray(qvel).shape)} is not (N, nv), '
          f'(P, N, nv), or (N, T, nv) for P={n_policies}, N={n_cases}'
      )
  elif stacked.ndim != 4:
    raise ValueError(
        f'qvel must be (N, nv), (P, N, nv), (N, T, nv), or (P, N, T, nv), '
        f'got {stacked.shape}'
    )
  if stacked.shape[0] != n_policies or stacked.shape[1] != n_cases:
    raise ValueError(
        f'qvel leading axes {stacked.shape[:2]} != (P, N)=({n_policies}, {n_cases})'
    )
  if stacked.shape[2] < horizon:
    raise ValueError(
        f'qvel horizon {stacked.shape[2]} is shorter than K={horizon}'
    )
  return stacked[:, :, :horizon, :]


def per_policy_action_scale(
    actions: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  r"""Per-policy time-averaged L2 action norm. ``(P, N)``.

  \(\frac{1}{K} \sum_t \lVert a_t^{(e)} \rVert_2\).
  """
  timed = _as_time_actions(actions, horizon)
  return np.linalg.norm(timed, axis=-1).mean(axis=2)


def action_scale_series(
    actions: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  """Per-step L2 action norms. ``(P, N, K)``."""
  timed = _as_time_actions(actions, horizon)
  return np.linalg.norm(timed, axis=-1)


def ensemble_action_scale(
    actions: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  r"""Ensemble action scale \(\mathcal{E}_{scale}(s_0)\). Shape ``(N,)``."""
  return per_policy_action_scale(actions, horizon=horizon).mean(axis=0)


def action_scale_disagreement(
    actions: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  r"""Mean pairwise |Δ| of per-policy action scale. Shape ``(N,)``.

  Magnitude disagreement, not directional cosine distance. For \(K>1\) the
  per-policy scale is the time average of \(\lVert a_t \rVert_2\).
  """
  scales = per_policy_action_scale(actions, horizon=horizon)
  n_cases = scales.shape[1]
  return np.array(
      [mean_pairwise_abs_difference_np(scales[:, i]) for i in range(n_cases)]
  )


def action_scale_trajectory_disagreement(
    actions: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  r"""Cosine disagreement of per-policy \(\lVert a_t \rVert_2\) series. ``(N,)``.

  Falls back to pairwise |Δ| when \(K=1\) (cosine on a scalar is degenerate).
  """
  series = action_scale_series(actions, horizon=horizon)
  if series.shape[-1] == 1:
    return action_scale_disagreement(actions, horizon=horizon)
  return pairwise_disagreement_np(series)


def mechanical_power_proxy(
    actions: np.ndarray,
    qvel: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
  r"""Mechanical power proxy \(\mathcal{P}(s_0)\). Shape ``(N,)``.

  Instantaneous work rate \(|a_t \cdot v_t|\) averaged over policies and
  the first ``horizon`` steps. ``v_t`` is the actuated-joint slice of ``qvel``.
  """
  timed = _as_time_actions(actions, horizon)
  n_policies, n_cases, k, n_actuators = timed.shape
  velocities = actuated_velocities(
      _as_time_qvel(qvel, n_policies, n_cases, k), n_actuators
  )
  power = np.abs(np.sum(timed * velocities, axis=-1))
  return power.mean(axis=(0, 2))


def energy_observables(
    actions: np.ndarray,
    qvel: np.ndarray,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
  """Ensemble action scale, scale disagreement, and mechanical power."""
  return {
      'ensemble_action_scale': ensemble_action_scale(actions, horizon=horizon),
      'action_scale_disagreement': action_scale_disagreement(
          actions, horizon=horizon
      ),
      'mechanical_power_proxy': mechanical_power_proxy(
          actions, qvel, horizon=horizon
      ),
  }


def energy_observables_from_traces(
    traces: Any,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
  """Energy observables from stacked multi-policy traces ``(P, N, T, ...)``."""
  actions = np.asarray(traces.actions)
  if not hasattr(traces, 'qvel') or traces.qvel is None:
    raise ValueError('traces have no qvel; cannot compute mechanical power.')
  return energy_observables(actions, np.asarray(traces.qvel), horizon=horizon)


def energy_observables_from_generation(
    record: Any,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
  """Energy observables from a GenerationCollection (typically K=1)."""
  actions = np.asarray(record.actions)
  qvel = np.asarray(record.suite.qvel)
  return energy_observables(actions, qvel, horizon=horizon)


def energy_metrics_from_collection(
    collection,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
  """Flatten energy observables across every encounter in a GA collection."""
  parts = [
      energy_observables_from_generation(record, horizon=horizon)
      for record in collection.generations
  ]
  if not parts:
    empty = np.zeros(0, dtype=np.float64)
    return {
        'ensemble_action_scale': empty,
        'action_scale_disagreement': empty,
        'mechanical_power_proxy': empty,
    }
  return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
