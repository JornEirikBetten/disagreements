"""Shared rollout-metric names expected by the GA execution stack."""

from jax import numpy as jp


def zero_rollout_metrics():
  """Brax locomotion rollout metric names."""
  zero = jp.zeros(())
  return {
      'reward_forward': zero,
      'reward_ctrl': zero,
      'reward_healthy': zero,
      'forward_reward': zero,
      'reward_linvel': zero,
      'reward_quadctrl': zero,
      'reward_alive': zero,
      'x_position': zero,
      'y_position': zero,
      'distance_from_origin': zero,
      'x_velocity': zero,
      'y_velocity': zero,
  }, zero


def remap_brax_locomotion_metrics(metrics: dict) -> dict:
  """Copy Brax Hopper/Walker2d step metrics into the shared rollout fields."""
  zero = jp.zeros(())
  x = metrics['x_position']
  forward = metrics['reward_forward']
  updates = dict(metrics)
  updates.update(
      forward_reward=forward,
      reward_linvel=forward,
      reward_quadctrl=metrics['reward_ctrl'],
      reward_alive=metrics['reward_healthy'],
      y_position=zero,
      distance_from_origin=jp.abs(x),
      y_velocity=zero,
  )
  return updates
