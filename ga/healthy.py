"""Initial-state health for locomotion feature search.

A candidate is healthy when it lies in the same set the environment uses to
clear ``done``. Hopper and Walker2d use strict bounds, so a torso angle or
height sitting on the clip is already a fall. Humanoid and Ant include the
boundary. Every candidate also takes a zero-action rollout of 3 steps. A
state that leaves the healthy set on any of those steps is discarded, so
the agent is only tested from a state that is still alive after three
uncontrolled transitions. Hopper and Walker2d also discard a candidate
whose limited joints start more than 0.18 rad past their stops.
"""

from __future__ import annotations

import mujoco
import numpy as np
import jax
from jax import numpy as jp

# Three physics steps over the whole candidate pool. A generation holds
# 20000 states, so each step screens them in two calls.
ZERO_ACTION_HORIZON = 3
ZERO_ACTION_BATCH = 10_000
# A joint may sit this far past its stop. Hopper's thigh and leg stop at
# 0 rad while the sampler reaches +0.2, and 0.001 rad of slack removed
# that whole side. 0.18 rad still rejects a joint pinned at the +0.2
# clip, which is as far as the Hopper sampler can go.
JOINT_LIMIT_TOLERANCE = 0.18
_JOINT_LIMIT_ENVS = frozenset({'hopper', 'walker2d'})

_zero_action_done = {}


def healthy_initial_mask(env, qpos, qvel) -> np.ndarray:
  """True for each initial state the environment would not already have failed.

  ``qpos`` and ``qvel`` are the suite arrays, one row per candidate. An
  environment without a healthy range keeps every finite state.
  """
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)
  if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
    raise ValueError(
        f'qpos {qpos.shape} and qvel {qvel.shape} must be aligned 2D arrays.'
    )
  finite = np.all(np.isfinite(qpos), axis=1) & np.all(np.isfinite(qvel), axis=1)
  z_range = getattr(env, '_healthy_z_range', None)
  model = getattr(getattr(env, 'sys', None), 'mj_model', None)
  if z_range is None or model is None:
    return finite
  height, pitch = _height_and_pitch(model, qpos)
  min_z, max_z = float(z_range[0]), float(z_range[1])
  angle_range = getattr(env, '_healthy_angle_range', None)
  if angle_range is None:
    healthy = (height >= min_z) & (height <= max_z)
  else:
    if pitch is None:
      raise ValueError(
          f'{getattr(env, "unwrapped", env)} publishes an angle range but '
          'has no root pitch joint.'
      )
    min_angle, max_angle = float(angle_range[0]), float(angle_range[1])
    healthy = (
        (height > min_z)
        & (height < max_z)
        & (pitch > min_angle)
        & (pitch < max_angle)
    )
  state_range = getattr(env, '_healthy_state_range', None)
  if state_range is not None:
    lo, hi = float(state_range[0]), float(state_range[1])
    state = np.concatenate([qpos[:, 2:], qvel], axis=1)
    healthy &= np.all((state > lo) & (state < hi), axis=1)
  return finite & healthy & np.isfinite(height)


def zero_action_fall_mask(
    env,
    qpos,
    qvel,
    *,
    batch_size: int = ZERO_ACTION_BATCH,
    horizon: int = ZERO_ACTION_HORIZON,
) -> np.ndarray:
  """True where a zero-action rollout leaves the healthy set.

  ``done`` is the environment's own check, evaluated after every step. A
  candidate fails if any of the ``horizon`` steps is a fall or is
  non-finite. Hopper and Walker2d also fail when a limited joint starts
  more than 0.18 rad outside its range. Integrated states are not checked
  again, so a joint that later rides its stop is kept. ``batch_size``
  states are stepped together; a short tail is padded so the compiled
  rollout keeps that size.
  """
  qpos = np.asarray(qpos, dtype=np.float32)
  qvel = np.asarray(qvel, dtype=np.float32)
  if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
    raise ValueError(
        f'qpos {qpos.shape} and qvel {qvel.shape} must be aligned 2D arrays.'
    )
  batch_size = int(batch_size)
  if batch_size < 1:
    raise ValueError(f'batch_size must be >= 1, got {batch_size}')
  horizon = int(horizon)
  if horizon < 1:
    raise ValueError(f'horizon must be >= 1, got {horizon}')
  n = int(qpos.shape[0])
  fell = np.zeros(n, dtype=bool)
  if n == 0:
    return fell
  if not hasattr(env, 'reset_from_initial_state') or not hasattr(env, 'action_size'):
    raise TypeError(
        f'{type(env).__name__} cannot take a zero-action step from qpos.'
    )
  step = _compiled_zero_action_done(env, horizon)
  for start in range(0, n, batch_size):
    stop = min(start + batch_size, n)
    count = stop - start
    block_q = qpos[start:stop]
    block_v = qvel[start:stop]
    if count < batch_size:
      block_q = np.repeat(block_q[:1], batch_size, axis=0)
      block_v = np.repeat(block_v[:1], batch_size, axis=0)
      block_q[:count] = qpos[start:stop]
      block_v[:count] = qvel[start:stop]
    done = step(jp.asarray(block_q), jp.asarray(block_v))
    done = np.asarray(jax.block_until_ready(done), dtype=bool)[:count]
    fell[start:stop] = done
  return fell


def _compiled_zero_action_done(env, horizon: int):
  """Jitted map from ``(qpos, qvel)`` to a fall anywhere in the rollout."""
  key = (id(env), int(horizon))
  compiled = _zero_action_done.get(key)
  if compiled is not None:
    return compiled
  action = jp.zeros((int(env.action_size),))
  limits = _joint_limit_spec(env)

  def outside(q):
    if limits is None:
      return jp.array(False)
    address, low, high, limited = limits
    joint = q[address]
    slack = JOINT_LIMIT_TOLERANCE
    bad = (~jp.isfinite(joint)) | (joint < low - slack) | (joint > high + slack)
    return jp.any(bad & limited)

  def one(qpos, qvel):
    state = env.reset_from_initial_state(qpos, qvel)
    fell = outside(state.pipeline_state.q)

    def body(carry, _):
      state, fell = carry
      state = env.step(state, action)
      done = state.done
      fell = fell | (~jp.isfinite(done)) | (done > 0)
      return (state, fell), None

    (_, fell), _ = jax.lax.scan(
        body, (state, fell), None, length=int(horizon)
    )
    return fell

  compiled = jax.jit(jax.vmap(one))
  _zero_action_done[key] = compiled
  return compiled


def _joint_limit_spec(env):
  """Limited-joint addresses and ranges for Hopper and Walker2d."""
  if not _enforces_joint_limits(env):
    return None
  model = getattr(getattr(env, 'sys', None), 'mj_model', None)
  if model is None:
    return None
  limited = np.asarray(model.jnt_limited, dtype=bool)
  if not np.any(limited):
    return None
  return (
      jp.asarray(np.asarray(model.jnt_qposadr, dtype=np.int32)),
      jp.asarray(np.asarray(model.jnt_range[:, 0], dtype=np.float32)),
      jp.asarray(np.asarray(model.jnt_range[:, 1], dtype=np.float32)),
      jp.asarray(limited),
  )


def _enforces_joint_limits(env) -> bool:
  current = env
  for _ in range(4):
    if current is None:
      return False
    if type(current).__name__.lower() in _JOINT_LIMIT_ENVS:
      return True
    current = getattr(current, 'env', None)
  return False


def _height_and_pitch(model, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
  """Torso height and, for planar bodies, root pitch.

  A free root stores height in ``qpos``. A planar root stores the vertical
  slide relative to the torso's compiled position.
  """
  names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
      for index in range(int(model.njnt))
  ]
  if 'rootz' in names and 'rooty' in names:
    z_index = int(model.jnt_qposadr[names.index('rootz')])
    pitch_index = int(model.jnt_qposadr[names.index('rooty')])
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if torso < 0:
      raise ValueError('Planar body has rootz but no torso body.')
    height = float(model.body_pos[torso, 2]) + qpos[:, z_index]
    return height, qpos[:, pitch_index]
  for index, name in enumerate(names):
    free = int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE)
    if free or name == 'root':
      address = int(model.jnt_qposadr[index])
      return qpos[:, address + 2], None
  raise ValueError('Cannot find a root joint for the healthy-height check.')
