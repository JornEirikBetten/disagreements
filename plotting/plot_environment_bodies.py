"""The four locomotion bodies, one standing pose each, at a shared scale.

Hopper, Humanoid, and Ant are the center of the reset distribution: ``init_q``
with zero velocity. Ant's center is ``init_q`` (torso height 0.55), which is
also the origin of the search genomes. Walker2d is in a stride, because the
asset stacks its two legs in one plane and a straight pose hides one of them.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import matplotlib.pyplot as plt
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image

import plotting.style as figs
from envs.ant import make_ant_env
from envs.hopper import make_hopper_env
from envs.humanoid import make_humanoid_env
from envs.walker2d import make_walker2d_env

OUTPUT = 'figures/environments/representative_states.pdf'
_MAKERS = {
    'hopper': make_hopper_env,
    'walker2d': make_walker2d_env,
    'humanoid': make_humanoid_env,
    'ant': make_ant_env,
}
# A low three-quarter view, facing the direction of locomotion.
_AZIMUTH = 112.0
_ELEVATION = -12.0
# rootx, rootz, pitch, thigh, leg, foot, thigh_left, leg_left, foot_left.
# One leg reaches forward and the other trails, with both feet near the ground.
_WALKER_STRIDE = np.array([0.0, 0.0, -0.15, 0.0, 0.0, -0.2, -0.5, 0.0, -0.3])
_RENDER = 960


def _pose(name: str, env) -> jnp.ndarray:
  if name == 'walker2d':
    return jnp.asarray(_WALKER_STRIDE)
  return jnp.asarray(env.sys.init_q)


def _body_bounds(model, data) -> tuple[np.ndarray, np.ndarray]:
  """Axis-aligned bounds of every geom except the floor."""
  points = []
  for index in range(model.ngeom):
    if int(model.geom_bodyid[index]) == 0:
      continue
    center = np.asarray(data.geom_xpos[index], dtype=float)
    radius = float(np.max(model.geom_size[index, :2]))
    points.append(center)
    points.append(center + radius)
    points.append(center - radius)
  points = np.asarray(points)
  return points.min(axis=0), points.max(axis=0)


def render_center(env, qpos: jnp.ndarray, size: int = _RENDER) -> tuple[np.ndarray, float]:
  """RGB render on white, and the standing height in meters."""
  model = env.sys.mj_model
  for index in range(model.ngeom):
    if int(model.geom_bodyid[index]) == 0:
      model.geom_rgba[index, 3] = 0.0
  model.vis.headlight.ambient[:] = 0.62
  model.vis.headlight.diffuse[:] = 0.5
  model.vis.headlight.specular[:] = 0.12
  model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), size)
  model.vis.global_.offheight = max(int(model.vis.global_.offheight), size)

  qvel = jnp.zeros(env.sys.qd_size())
  pipeline = env.reset_from_initial_state(qpos, qvel).pipeline_state
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(pipeline.q)
  data.qvel[:] = np.asarray(pipeline.qd)
  mujoco.mj_forward(model, data)

  low, high = _body_bounds(model, data)
  fovy = float(model.vis.global_.fovy) * np.pi / 180.0
  distance = float(np.linalg.norm(high - low)) / (2 * np.tan(fovy / 2) * 0.72)
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)
  camera.lookat[:] = 0.5 * (low + high)
  camera.distance = distance
  camera.azimuth = _AZIMUTH
  camera.elevation = _ELEVATION

  renderer = mujoco.Renderer(model, height=size, width=size)
  renderer.update_scene(data, camera=camera)
  renderer.enable_segmentation_rendering()
  segment = np.asarray(renderer.render())
  renderer.disable_segmentation_rendering()
  renderer.update_scene(data, camera=camera)
  color = np.asarray(renderer.render())
  renderer.close()

  mask = segment[..., 0] >= 0
  if mask.mean() > 0.85 or not np.any(mask):
    raise RuntimeError('The robot did not separate from the background.')
  white = np.full_like(color, 255)
  covered = mask[..., None]
  standing = float(high[2] - max(float(low[2]), 0.0))
  return np.where(covered, color, white), standing


def _crop(image: np.ndarray) -> np.ndarray:
  rgb = np.asarray(image)
  content = np.any(rgb < 250, axis=2)
  rows = np.flatnonzero(content.any(axis=1))
  cols = np.flatnonzero(content.any(axis=0))
  return rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _place(
    aspects: list[float],
    relative: list[float],
) -> tuple[float, list[float]]:
  """Tallest height and the gap before each robot, in inches.

  Shorter bodies keep their share of that height. The row fills the text width.
  """
  count = len(aspects)
  minimum = 0.12

  def layout(height: float) -> tuple[float, list[float]]:
    widths = [aspect * scale * height for aspect, scale in zip(aspects, relative)]
    gaps = [0.0] + [minimum] * (count - 1)
    return sum(widths) + sum(gaps), gaps

  low, high = 0.3, 3.0
  for _ in range(24):
    mid = (low + high) / 2
    if layout(mid)[0] > figs.TEXT_WIDTH:
      high = mid
    else:
      low = mid
  return low, layout(low)[1]


def _row(
    crops: list[np.ndarray],
    relative: list[float],
    gaps: list[float],
    height: float,
) -> tuple[np.ndarray, list[float]]:
  """Feet on one line. Each body is ``relative`` times the tallest height."""
  pixels = 420
  heights = [max(1, int(round(scale * height * pixels))) for scale in relative]
  tallest = max(heights)
  gap_px = [int(round(gap * pixels)) for gap in gaps]
  pad = int(round(0.04 * tallest))
  width = 0
  placed = []
  for crop, robot_h, gap in zip(crops, heights, gap_px):
    robot_w = max(1, int(round(crop.shape[1] * robot_h / crop.shape[0])))
    image = np.asarray(Image.fromarray(crop).resize(
        (robot_w, robot_h), Image.Resampling.LANCZOS,
    ))
    width += gap + robot_w
    placed.append((image, gap))
  canvas = np.full((tallest + pad, width, 3), 255, dtype=np.uint8)
  rule = (
      int(figs.RULE[1:3], 16),
      int(figs.RULE[3:5], 16),
      int(figs.RULE[5:7], 16),
  )
  x = 0
  lefts = []
  for image, gap in placed:
    x += gap
    top = tallest - image.shape[0]
    canvas[top:tallest, x:x + image.shape[1]] = image
    lefts.append(x / width)
    x += image.shape[1]
  ground = tallest + max(1, pad // 6)
  if ground < canvas.shape[0]:
    canvas[ground:ground + max(1, pixels // 280), :] = rule
  return canvas, lefts


@figs.styled
def environment_bodies_figure(
    panels: dict[str, np.ndarray] | None = None,
) -> plt.Figure:
  """Four poses on one row, scaled by standing height."""
  names = list(figs.ENVIRONMENT_ORDER)
  if panels is None:
    rendered = {}
    for name in names:
      env = _MAKERS[name]()
      rendered[name] = render_center(env, _pose(name, env))
      print(f'  {name} standing height {rendered[name][1]:.3f} m', flush=True)
  else:
    rendered = panels
  crops = [_crop(rendered[name][0]) for name in names]
  standing = [rendered[name][1] for name in names]
  relative = [value / max(standing) for value in standing]
  aspects = [crop.shape[1] / crop.shape[0] for crop in crops]
  height, gaps = _place(aspects, relative)
  image, lefts = _row(crops, relative, gaps, height)
  letter_band = 0.20
  ground = 0.04 * height
  figure_height = height + ground + letter_band
  image_span = (height + ground) / figure_height
  fig = plt.figure(figsize=(figs.TEXT_WIDTH, figure_height))
  axis = fig.add_axes((0.0, 0.0, 1.0, image_span))
  axis.imshow(image, interpolation='lanczos')
  axis.set_axis_off()
  for index, left in enumerate(lefts):
    axis.text(
        left, 1.02, 'abcdefgh'[index],
        transform=axis.transAxes, ha='left', va='bottom', clip_on=False,
        fontweight='bold', fontsize=8.5,
    )
  return fig


def save_representative_states() -> None:
  os.chdir(_ROOT)
  fig = environment_bodies_figure()
  figs.save(fig, OUTPUT, png=True)
  print(OUTPUT)


if __name__ == '__main__':
  save_representative_states()
