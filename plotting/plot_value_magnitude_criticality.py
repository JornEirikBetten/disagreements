"""Distributions of value z-scores, action magnitudes, and criticality.

Criticality is the negative of the per-policy Gaussian action entropy,
``-(0.5 * d * log(2 π e) + sum_i log σ_i)``. A larger value means the
policy concentrates its action more tightly.

The rows are the quota-filled difficulty-bin searches: every search-fail
quota, including all-fail, holds the same number of distinct states. Each
policy's value estimates are z-scored across the states of one search, using
the minimum of the two critics for SAC. Action magnitude is the L2 norm of
that policy's mode action at the initial state.

Usage (from the repository root):

  python plotting/plot_value_magnitude_criticality.py
  python plotting/plot_value_magnitude_criticality.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import plotting.style as figs
from analysis.disagreement import standardize_per_policy_np, value_estimates_np

ENVIRONMENTS = figs.ENVIRONMENT_ORDER
ALGORITHMS = figs.POLICY_SET_ORDER
N_POLICIES = 16
ROLLOUT_LENGTH = 15
# Bin 0 .. 15 are fail counts 0/16 .. 15/16. Bin 16 is the all-fail quota.
N_BINS = N_POLICIES + 1
CACHE_DIR = os.path.join('figures', 'quota_distributions', 'cache')
FIGURE_DIR = os.path.join('figures', 'quota_distributions')

_SKIP_FIELDS = frozenset({
    'qpos',
    'qvel',
    'policy_rng',
    'failed',
    'fitness',
    'action_loc',
    'action_mean',
    'action_std',
    'state_ig_qpos',
    'state_ig_qvel',
    'obs_ig',
    'state_importance_qpos',
    'state_importance_qvel',
    'eval_fail_rate',
    'eval_failed',
})
_VIOLIN_BINS = 36
_DODGE = {'ppo': -0.018, 'sac': 0.0, 'ppo_sac': 0.018}
_HALF_WIDTH = 0.0085


class _SkippedArray:
  def __setstate__(self, state):
    return


class _ShardUnpickler(pickle._Unpickler):
  """Load a generation shard without materializing attribution arrays."""

  dispatch = pickle._Unpickler.dispatch.copy()

  def __init__(self, file):
    super().__init__(file)
    self._skip_nd = 0
    self._discard_big = 0

  def find_class(self, module, name):
    if name == '_reconstruct' and 'numpy' in module:
      real = super().find_class(module, name)

      def reconstruct(subtype, shape, dtype, _real=real, unpickler=self):
        if unpickler._skip_nd > 0:
          unpickler._skip_nd -= 1
          unpickler._discard_big += 1
          return _SkippedArray()
        return _real(subtype, shape, dtype)

      return reconstruct
    return super().find_class(module, name)

  def _remember(self, text: str) -> None:
    self.append(text)
    if text in _SKIP_FIELDS:
      self._skip_nd += 1

  def load_short_binunicode(self):
    length = self.read(1)[0]
    self._remember(str(self.read(length), 'utf-8', 'surrogatepass'))

  def load_binunicode(self):
    length, = struct.unpack('<I', self.read(4))
    self._remember(str(self.read(length), 'utf-8', 'surrogatepass'))

  def _discard(self, length: int) -> None:
    remaining = length
    while remaining:
      chunk = self.read(min(remaining, 1 << 20))
      if not chunk:
        raise EOFError('truncated generation shard')
      remaining -= len(chunk)

  def _take_bytes(self, length: int) -> None:
    # A skipped array may store its buffer as a short or a long bytes object.
    # Drop the next one, then stop at BUILD so a later kept array is intact.
    if self._discard_big:
      self._discard(length)
      self.append(b'')
      self._discard_big -= 1
      return
    self.append(self.read(length))

  def load_short_binbytes(self):
    self._take_bytes(self.read(1)[0])

  def load_binbytes(self):
    length, = struct.unpack('<I', self.read(4))
    self._take_bytes(length)

  def load_binbytes8(self):
    length, = struct.unpack('<Q', self.read(8))
    self._take_bytes(length)

  def load_build(self):
    pickle._Unpickler.load_build(self)
    self._discard_big = 0


_ShardUnpickler.dispatch[pickle.SHORT_BINUNICODE[0]] = (
    _ShardUnpickler.load_short_binunicode
)
_ShardUnpickler.dispatch[pickle.BINUNICODE[0]] = _ShardUnpickler.load_binunicode
_ShardUnpickler.dispatch[pickle.SHORT_BINBYTES[0]] = (
    _ShardUnpickler.load_short_binbytes
)
_ShardUnpickler.dispatch[pickle.BINBYTES[0]] = _ShardUnpickler.load_binbytes
_ShardUnpickler.dispatch[pickle.BINBYTES8[0]] = _ShardUnpickler.load_binbytes8
_ShardUnpickler.dispatch[pickle.BUILD[0]] = _ShardUnpickler.load_build


def _collection_dir(environment: str, algorithm: str) -> str:
  return os.path.join(
      'datasets',
      'ga_data',
      environment,
      algorithm,
      'difficulty_bins',
      f'rollout_length={ROLLOUT_LENGTH}',
  )


def _finished_experiments(environment: str, algorithm: str) -> list[tuple[int, str]]:
  directory = _collection_dir(environment, algorithm)
  found = []
  for name in sorted(os.listdir(directory)):
    if not name.endswith('_init.json'):
      continue
    path = os.path.join(directory, name)
    with open(path) as handle:
      meta = json.load(handle)
    if not meta.get('bins', {}).get('all_full'):
      continue
    experiment = int(name.split('_')[1])
    collection = path.replace('_init.json', '.pkl')
    found.append((experiment, collection))
  if len(found) != 5:
    raise FileNotFoundError(
        f'{directory} has {len(found)} quota-filled experiments, expected 5.'
    )
  return found


def _load_shard(path: str):
  with open(path, 'rb') as handle:
    return _ShardUnpickler(handle).load()


def _bin_index(difficulty: np.ndarray) -> np.ndarray:
  values = np.asarray(difficulty, dtype=np.float64)
  all_fail = values < 0.0
  counts = np.rint(values * N_POLICIES).astype(np.int64)
  if np.any(~all_fail & ((counts < 0) | (counts >= N_POLICIES))):
    bad = values[~all_fail & ((counts < 0) | (counts >= N_POLICIES))]
    raise ValueError(f'difficulty outside 0 .. {N_POLICIES - 1}/16: {bad[:5]}')
  counts[all_fail] = N_POLICIES
  return counts


def action_entropy(action_scale: np.ndarray) -> np.ndarray:
  """Per-policy Gaussian action entropy in nats. Shape ``(P, N)``.

  Matches :func:`ga.quad_fitness_search.gaussian_action_entropy`. This is
  the criticality score ``sum_i log σ_i`` plus ``0.5 * d * log(2 π e)``.
  """
  scale = np.asarray(action_scale, dtype=np.float64)
  if scale.ndim == 2:
    scale = scale[:, :, None]
  if scale.ndim != 3:
    raise ValueError(
        f'action_scale must have shape (P, N) or (P, N, n_actuators), got {scale.shape}'
    )
  sigma = np.maximum(scale, 1e-12)
  n_actuators = int(sigma.shape[-1])
  return (
      0.5 * n_actuators * np.log(2.0 * np.pi * np.e)
      + np.sum(np.log(sigma), axis=-1)
  )


def _cache_path(environment: str, algorithm: str) -> str:
  return os.path.join(CACHE_DIR, f'{environment}_{algorithm}.npz')


def extract_combo(environment: str, algorithm: str) -> str:
  """Z-scores, action magnitudes, and action entropies for one ensemble."""
  difficulties = []
  experiments = []
  zscores = []
  magnitudes = []
  entropies = []
  for experiment, collection in _finished_experiments(environment, algorithm):
    shard_dir = collection.replace('.pkl', '_generations')
    with open(os.path.join(shard_dir, 'manifest.json')) as handle:
      manifest = json.load(handle)
    diff_parts = []
    value_parts = []
    critic_parts = []
    magnitude_parts = []
    entropy_parts = []
    for entry in manifest['shards']:
      record = _load_shard(os.path.join(shard_dir, entry['file']))
      actions = np.asarray(record.actions, dtype=np.float32)
      diff_parts.append(np.asarray(record.difficulty, dtype=np.float32))
      value_parts.append(np.asarray(record.values, dtype=np.float32))
      magnitude_parts.append(np.linalg.norm(actions, axis=-1))
      entropy_parts.append(action_entropy(record.action_scale))
      critics = getattr(record, 'q_values', None)
      if critics is not None:
        critic_parts.append(np.asarray(critics, dtype=np.float32))
      del record
    difficulty = np.concatenate(diff_parts)
    values = np.concatenate(value_parts, axis=1)
    q_values = np.concatenate(critic_parts, axis=1) if critic_parts else None
    magnitude = np.concatenate(magnitude_parts, axis=1)
    entropy = np.concatenate(entropy_parts, axis=1)
    estimates = value_estimates_np(values, q_values, algorithm=algorithm)
    zscore = standardize_per_policy_np(estimates)
    if zscore.shape != magnitude.shape or zscore.shape != entropy.shape:
      raise ValueError(
          f'{environment}/{algorithm} experiment {experiment}: '
          f'z-score {zscore.shape} != magnitude {magnitude.shape}'
      )
    if zscore.shape[0] != N_POLICIES:
      raise ValueError(
          f'{environment}/{algorithm} experiment {experiment} has '
          f'{zscore.shape[0]} policies, expected {N_POLICIES}.'
      )
    bins = _bin_index(difficulty)
    counts = np.bincount(bins, minlength=N_BINS)
    if np.any(counts != counts[0]):
      raise ValueError(
          f'{environment}/{algorithm} experiment {experiment} quotas are '
          f'uneven: {counts.tolist()}'
      )
    difficulties.append(difficulty)
    experiments.append(np.full(difficulty.shape[0], experiment, dtype=np.int32))
    zscores.append(np.asarray(zscore, dtype=np.float32).T)
    magnitudes.append(np.asarray(magnitude, dtype=np.float32).T)
    entropies.append(np.asarray(entropy, dtype=np.float32).T)
    print(
        f'  {environment}/{algorithm} experiment {experiment}: '
        f'{difficulty.shape[0]} states, {int(counts[0])} per quota',
        flush=True,
    )
  path = _cache_path(environment, algorithm)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez_compressed(
      path,
      difficulty=np.concatenate(difficulties),
      experiment=np.concatenate(experiments),
      zscore=np.concatenate(zscores, axis=0),
      magnitude=np.concatenate(magnitudes, axis=0),
      entropy=np.concatenate(entropies, axis=0),
  )
  print(f'saved {path}', flush=True)
  return path


def load_cache() -> dict[tuple[str, str], dict[str, np.ndarray]]:
  loaded = {}
  for environment in ENVIRONMENTS:
    for algorithm in ALGORITHMS:
      path = _cache_path(environment, algorithm)
      if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} is missing. Run without --plot-only first.'
        )
      with np.load(path) as payload:
        loaded[(environment, algorithm)] = {
            'difficulty': payload['difficulty'],
            'zscore': payload['zscore'],
            'magnitude': payload['magnitude'],
            'entropy': payload['entropy'],
        }
  return loaded


def _groups(difficulty: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
  """One flat sample per quota. ``values`` is ``(N, P)``."""
  bins = _bin_index(difficulty)
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  bins = np.repeat(bins, values.shape[1])
  return [flat[bins == index] for index in range(N_BINS)]


def _violin(ax, x: float, sample: np.ndarray, color: str, y_range: tuple[float, float]):
  lo, hi = y_range
  hist, edges = np.histogram(sample, bins=_VIOLIN_BINS, range=(lo, hi))
  if hist.max() <= 0:
    return
  width = hist / hist.max() * _HALF_WIDTH
  centers = 0.5 * (edges[:-1] + edges[1:])
  xs = np.concatenate([x - width, (x + width)[::-1]])
  ys = np.concatenate([centers, centers[::-1]])
  ax.add_patch(Polygon(
      np.column_stack([xs, ys]),
      closed=True,
      facecolor=color,
      edgecolor='none',
      alpha=0.55,
      zorder=3,
  ))
  q25, median, q75 = np.quantile(sample, (0.25, 0.5, 0.75))
  ax.plot([x, x], [q25, q75], color=color, linewidth=1.15, solid_capstyle='butt', zorder=4)
  ax.plot(x, median, marker='_', color=color, markersize=3.2, markeredgewidth=0.9, zorder=5)


def _ylim(samples: list[np.ndarray], *, floor: float | None = None) -> tuple[float, float]:
  pooled = np.concatenate([sample for sample in samples if sample.size])
  lo, hi = np.quantile(pooled, (0.005, 0.995))
  pad = 0.06 * (hi - lo if hi > lo else 1.0)
  lo -= pad
  hi += pad
  if floor is not None:
    lo = min(lo, floor)
  return float(lo), float(hi)


def plot_quantity(
    cache: dict,
    field: str,
    ylabel: str,
    path: str,
    *,
    share_y: bool,
    zero_line: bool,
) -> str:
  """One row per environment, violins dodged by ensemble."""
  levels = np.arange(N_BINS, dtype=np.float64) / N_POLICIES
  with figs.style():
    return _draw_quantity(cache, field, ylabel, path, share_y=share_y, zero_line=zero_line, levels=levels)


def _draw_quantity(cache, field, ylabel, path, *, share_y, zero_line, levels) -> str:
  grouped = {
      (environment, algorithm): _groups(
          cache[(environment, algorithm)]['difficulty'],
          cache[(environment, algorithm)][field],
      )
      for environment in ENVIRONMENTS
      for algorithm in ALGORITHMS
  }
  if share_y:
    shared = _ylim([
        sample
        for groups in grouped.values()
        for sample in groups
    ])
  fig = figs.new_figure(height=6.15)
  axes = fig.subplots(len(ENVIRONMENTS), 1, sharex=True, sharey=share_y)
  for axis, environment in zip(axes, ENVIRONMENTS, strict=True):
    if share_y:
      y_range = shared
    else:
      y_range = _ylim([
          sample
          for algorithm in ALGORITHMS
          for sample in grouped[(environment, algorithm)]
      ])
      axis.set_ylim(*y_range)
    if zero_line:
      axis.axhline(0.0, color=figs.REFERENCE, linewidth=0.6, zorder=1)
    for algorithm in ALGORITHMS:
      style = figs.POLICY_SETS[algorithm]
      samples = grouped[(environment, algorithm)]
      medians = np.array([float(np.median(sample)) for sample in samples])
      xs = levels + _DODGE[algorithm]
      axis.plot(
          xs,
          medians,
          color=style.color,
          linestyle=style.linestyle,
          linewidth=0.9,
          zorder=2,
      )
      for x, sample in zip(xs, samples, strict=True):
        _violin(axis, float(x), sample, style.color, y_range)
    axis.set_ylabel(figs.ENVIRONMENTS[environment].label)
    axis.set_axisbelow(True)
    axis.grid(True, axis='y')
  if share_y:
    axes[0].set_ylim(*shared)
  axes[-1].set_xlabel(figs.SEARCH_RATE)
  axes[-1].set_xlim(-0.06, 1.06)
  axes[-1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
  fig.supylabel(ylabel)
  figs.add_legend(fig, figs.policy_set_handles(ALGORITHMS))
  return figs.save(fig, path, png=True)


_MEASURES = (
    ('zscore', 'Value z-score', True),
    ('magnitude', r'Action magnitude $\Vert a \Vert_2$', False),
    ('criticality', 'Criticality (nats)', False),
)


def _grouped(cache: dict, field: str) -> dict:
  return {
      (environment, algorithm): _groups(
          cache[(environment, algorithm)]['difficulty'],
          cache[(environment, algorithm)][field],
      )
      for environment in ENVIRONMENTS
      for algorithm in ALGORITHMS
  }


def _draw_panel(axis, groups: list[np.ndarray], levels: np.ndarray, y_range, *, zero_line: bool):
  if zero_line:
    axis.axhline(0.0, color=figs.REFERENCE, linewidth=0.6, zorder=1)
  axis.set_ylim(*y_range)
  for algorithm, samples in zip(ALGORITHMS, groups, strict=True):
    style = figs.POLICY_SETS[algorithm]
    medians = np.array([float(np.median(sample)) for sample in samples])
    xs = levels + _DODGE[algorithm]
    axis.plot(
        xs,
        medians,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=0.9,
        zorder=2,
    )
    for x, sample in zip(xs, samples, strict=True):
      _violin(axis, float(x), sample, style.color, y_range)
  axis.set_axisbelow(True)
  axis.grid(True, axis='y')
  axis.set_xlim(-0.06, 1.06)


# The combined figure is three text-widths wide, so the 8 pt paper style
# is too small when the page is read as a whole.
_LARGE_TYPE = {
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'figure.labelsize': 15,
    'figure.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'axes.titlepad': 8,
    'axes.labelpad': 6,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.constrained_layout.h_pad': 8 / 72,
    'figure.constrained_layout.w_pad': 10 / 72,
}


def with_criticality(cache: dict) -> dict:
  """Store criticality as the negative of the cached action entropy."""
  for payload in cache.values():
    payload['criticality'] = -np.asarray(payload['entropy'], dtype=np.float64)
  return cache


def save_combined(path: str) -> str:
  """Value, action magnitude, and criticality by search quota."""
  return plot_measures(with_criticality(load_cache()), path)


def plot_measures(cache: dict, path: str) -> str:
  """Environments in rows, the three measures side by side."""
  levels = np.arange(N_BINS, dtype=np.float64) / N_POLICIES
  with figs.style():
    plt.rcParams.update(_LARGE_TYPE)
    grouped = {field: _grouped(cache, field) for field, _label, _zero in _MEASURES}
    z_range = _ylim([
        sample
        for groups in grouped['zscore'].values()
        for sample in groups
    ])
    fig = figs.new_figure(height=7.2, width=figs.TEXT_WIDTH * 3)
    axes = fig.subplots(len(ENVIRONMENTS), len(_MEASURES), sharex=True)
    for column, (field, label, zero_line) in enumerate(_MEASURES):
      axes[0, column].set_title(label)
      for row, environment in enumerate(ENVIRONMENTS):
        axis = axes[row, column]
        if field == 'zscore':
          y_range = z_range
        else:
          y_range = _ylim([
              sample
              for algorithm in ALGORITHMS
              for sample in grouped[field][(environment, algorithm)]
          ])
        _draw_panel(
            axis,
            [grouped[field][(environment, algorithm)] for algorithm in ALGORITHMS],
            levels,
            y_range,
            zero_line=zero_line,
        )
        if column == 0:
          axis.set_ylabel(figs.ENVIRONMENTS[environment].label)
    for axis in axes[-1]:
      axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.supxlabel(figs.SEARCH_RATE)
    figs.add_legend(fig, figs.policy_set_handles(ALGORITHMS))
    # figs.save re-enters the 8 pt style and would shrink text at draw time.
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    fig.savefig(os.path.splitext(path)[0] + '.png', dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Plot value z-score, action-magnitude, and criticality '
          'distributions against difficulty on the quota-filled searches.'
      ),
  )
  parser.add_argument(
      '--plot-only',
      action='store_true',
      help='Draw the figures from the extracted cache.',
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if not args.plot_only:
    for environment in ENVIRONMENTS:
      for algorithm in ALGORITHMS:
        extract_combo(environment, algorithm)
  cache = load_cache()
  z_path = plot_quantity(
      cache,
      'zscore',
      'Value z-score',
      os.path.join(FIGURE_DIR, 'value_zscore_vs_difficulty.pdf'),
      share_y=True,
      zero_line=True,
  )
  mag_path = plot_quantity(
      cache,
      'magnitude',
      r'Action magnitude $\Vert a \Vert_2$',
      os.path.join(FIGURE_DIR, 'action_magnitude_vs_difficulty.pdf'),
      share_y=False,
      zero_line=False,
  )
  with_criticality(cache)
  criticality_path = plot_quantity(
      cache,
      'criticality',
      r'Criticality $-H$ (nats)',
      os.path.join(FIGURE_DIR, 'criticality_vs_difficulty.pdf'),
      share_y=False,
      zero_line=False,
  )
  measures_path = plot_measures(
      cache,
      os.path.join(FIGURE_DIR, 'measures_vs_difficulty.pdf'),
  )
  print(f'wrote {z_path}', flush=True)
  print(f'wrote {mag_path}', flush=True)
  print(f'wrote {criticality_path}', flush=True)
  print(f'wrote {measures_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
