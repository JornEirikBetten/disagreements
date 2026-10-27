"""Walker2d locomotion environment (MJX backend)."""

import jax

from brax import envs
from brax.envs.base import State
from brax.envs.walker2d import Walker2d as BraxWalker2d

from envs.mjx_metrics import remap_brax_locomotion_metrics, zero_rollout_metrics


class Walker2d(BraxWalker2d):
  """Brax Walker2d with ``reset_from_initial_state`` for GA test-case search."""

  def __init__(self, **kwargs):
    kwargs.setdefault('backend', 'mjx')
    super().__init__(**kwargs)

  def reset(self, rng: jax.Array) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    )
    return self.reset_from_initial_state(qpos, qvel)

  def reset_from_initial_state(
      self, qpos: jax.Array, qvel: jax.Array
  ) -> State:
    data = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(data)
    metrics, zero = zero_rollout_metrics()
    return State(data, obs, zero, zero, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    state = super().step(state, action)
    return state.replace(metrics=remap_brax_locomotion_metrics(state.metrics))


def make_walker2d_env(**kwargs) -> Walker2d:
  """Create and register a Walker2d environment."""
  env_name = 'walker2d'
  envs.register_environment(env_name, Walker2d, **kwargs)
  return envs.get_environment(env_name)
