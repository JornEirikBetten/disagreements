"""Ant locomotion environment (MJX backend)."""

import jax

from brax import envs
from brax.envs.ant import Ant as BraxAnt
from brax.envs.base import State

from envs.mjx_metrics import zero_rollout_metrics


def _ant_metrics(
    *,
    forward,
    ctrl,
    survive,
    contact,
    x,
    y,
    dist,
    x_vel,
    y_vel,
) -> dict:
  """Shared rollout fields plus Brax Ant's survive and contact terms.

  Reset and step must expose the same metric keys so a JAX scan can carry
  the state.
  """
  return {
      'reward_forward': forward,
      'reward_ctrl': ctrl,
      'reward_healthy': survive,
      'forward_reward': forward,
      'reward_linvel': forward,
      'reward_quadctrl': ctrl,
      'reward_alive': survive,
      'x_position': x,
      'y_position': y,
      'distance_from_origin': dist,
      'x_velocity': x_vel,
      'y_velocity': y_vel,
      'reward_survive': survive,
      'reward_contact': contact,
  }


class Ant(BraxAnt):
  """Brax Ant with ``reset_from_initial_state`` for GA test-case search."""

  def __init__(self, **kwargs):
    kwargs.setdefault('backend', 'mjx')
    super().__init__(**kwargs)

  def reset(self, rng: jax.Array) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
    return self.reset_from_initial_state(qpos, qvel)

  def reset_from_initial_state(
      self, qpos: jax.Array, qvel: jax.Array
  ) -> State:
    data = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(data)
    _, zero = zero_rollout_metrics()
    metrics = _ant_metrics(
        forward=zero,
        ctrl=zero,
        survive=zero,
        contact=zero,
        x=zero,
        y=zero,
        dist=zero,
        x_vel=zero,
        y_vel=zero,
    )
    return State(data, obs, zero, zero, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    state = super().step(state, action)
    metrics = state.metrics
    x = metrics['x_position']
    y = metrics['y_position']
    return state.replace(
        metrics=_ant_metrics(
            forward=metrics['forward_reward'],
            ctrl=metrics['reward_ctrl'],
            survive=metrics['reward_survive'],
            contact=metrics['reward_contact'],
            x=x,
            y=y,
            dist=metrics['distance_from_origin'],
            x_vel=metrics['x_velocity'],
            y_vel=metrics['y_velocity'],
        )
    )


def make_ant_env(**kwargs) -> Ant:
  """Create and register an Ant environment."""
  env_name = 'ant'
  envs.register_environment(env_name, Ant, **kwargs)
  return envs.get_environment(env_name)
