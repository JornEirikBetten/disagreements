"""Figure tools for the results of the paper.

Every figure is drawn at the size it is printed: the full width is the
Springer Nature text block (31 pc) and the lettering is 7 to 8 pt. Include
the PDFs with ``\\includegraphics{...}`` and no scaling (``width=\\textwidth``
gives the same size).

The encodings are the same in every figure:

  color and line style  policy set: PPO blue solid, SAC orange dashed,
                        PPO+SAC green dash-dot
  marker shape          environment: Hopper diamond, Walker2d triangle,
                        Humanoid square, Ant circle
  marker fill           filled for the evaluation ensemble E, open for the
                        search ensemble S and for ceilings
  gray                  references: the search quota and zero

Results go in as plain dictionaries of ``Curve`` and ``Estimate``, keyed by
the identifiers of the analysis scripts ('hopper', 'ppo_sac',
'criticality_disagreement', 'legal_contact', ...). Each figure function
returns a matplotlib Figure, and ``save`` writes it as a PDF:

  import plotting.style as figs
  fig = figs.feature_curves_figure(curves)
  figs.save(fig, 'figures/predictors/feature_curves.pdf')  # png=True adds a PNG

``python plotting/style.py --demo DIR`` draws every figure type from synthetic
numbers, stamped as such, into DIR.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

TEXT_WIDTH = 372.0 / 72.27  # inches

INK = '#25282D'
AXIS = '#3D424A'
MUTED = '#68707C'
GRID = '#E7EAEE'
RULE = '#C9D0D9'
REFERENCE = '#A2A9B2'
SHADE = '#F4F6F9'

LINE_WIDTH = 1.1
MARKER_SIZE = 3.6
BAND_ALPHA = 0.16
SET_PITCH = 0.25
ENV_PITCH = 0.058

STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 8.0,
    'axes.titlesize': 8.0,
    'axes.labelsize': 8.0,
    'figure.labelsize': 8.0,
    'figure.titlesize': 8.0,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.5,
    'text.color': INK,
    'axes.labelcolor': INK,
    'axes.titlecolor': INK,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlepad': 3.0,
    'axes.labelpad': 2.0,
    'axes.axisbelow': True,
    'axes.xmargin': 0.02,
    'axes.ymargin': 0.07,
    'axes.unicode_minus': True,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK,
    'ytick.labelcolor': INK,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.pad': 2.0,
    'ytick.major.pad': 2.0,
    'lines.linewidth': LINE_WIDTH,
    'lines.markersize': MARKER_SIZE,
    'lines.solid_capstyle': 'round',
    'errorbar.capsize': 1.1,
    'legend.frameon': False,
    'legend.handlelength': 2.2,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.2,
    'legend.borderpad': 0.2,
    'legend.borderaxespad': 0.2,
    'figure.dpi': 150,
    'figure.constrained_layout.h_pad': 2.0 / 72,
    'figure.constrained_layout.w_pad': 2.0 / 72,
    'figure.constrained_layout.hspace': 0.03,
    'figure.constrained_layout.wspace': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


@dataclass(frozen=True)
class SeriesStyle:
  label: str
  color: str
  linestyle: str = '-'


@dataclass(frozen=True)
class EnvironmentStyle:
  label: str
  marker: str
  scale: float = 1.0  # evens out the apparent size of the marker shapes


@dataclass(frozen=True)
class RoleStyle:
  label: str
  filled: bool


POLICY_SETS = {
    'ppo': SeriesStyle('PPO', '#0072B2', '-'),
    'sac': SeriesStyle('SAC', '#D55E00', '--'),
    'ppo_sac': SeriesStyle('PPO+SAC', '#009E73', '-.'),
}
POLICY_SET_ORDER = tuple(POLICY_SETS)

ENVIRONMENTS = {
    'hopper': EnvironmentStyle('Hopper', 'D', 0.84),
    'walker2d': EnvironmentStyle('Walker2d', '^', 1.15),
    'humanoid': EnvironmentStyle('Humanoid', 's', 0.9),
    'ant': EnvironmentStyle('Ant', 'o', 1.0),
}
ENVIRONMENT_ORDER = tuple(ENVIRONMENTS)

ROLES = {
    'eval': RoleStyle('evaluation $E$', True),
    'search': RoleStyle('search $S$', False),
}

DISAGREEMENT = {
    'action_scale_disagreement': 'Action-magnitude',
    'action_disagreement': 'Action-direction',
    'value_disagreement': 'Value estimate',
    'feature_attribution_disagreement': 'Attributions',
    'criticality_disagreement': 'Criticality',
}

# Two-line names for the five-across grids, where the one-line names collide.
DISAGREEMENT_LINES = {
    'action_scale_disagreement': 'Action\nmagnitude',
    'action_disagreement': 'Action\ndirection',
    'value_disagreement': 'Value\nestimate',
    'feature_attribution_disagreement': 'Attributions',
    'criticality_disagreement': 'Criticality',
    'minimum_value_zscore': 'Min.\nvalue z',
}

ASSOCIATIONS = {
    'spearman': r'Spearman $\rho$',
    'mi': 'Mutual information (nats)',
    'semipartial_spearman': r'Semi-partial $\rho$',
    'pearson': r'Pearson $r$',
}

STRATA = {
    'all': 'Full quota',
    'legal_contact': 'Healthy and in contact',
}

MODELS = {
    **DISAGREEMENT,
    'disagreement': 'Disagreement',
    'ceiling': 'Ceiling',
}

RELIABILITY = {
    'rho_search_eval': r'$\rho(r_S, r_E)$',
    'split_half': r'Split-half $\rho_E$',
    'ceiling': r'Ceiling $\sqrt{\rho_E}$',
}

COSTS = {
    **DISAGREEMENT,
    'rollout': 'Rollouts',
}

SEARCH_RATE = r'Search fail rate $r_S$'
EVAL_RATE = r'Held-out fail rate $r_E$'
STANDARDIZED_MEAN = 'Mean (SD units)'

CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    'opus_correlation', ('#6B4C9A', '#FFFFFF', '#2B3A4A'),
)


@dataclass(frozen=True)
class Curve:
  """A series y(x) with an optional band from ``low`` to ``high``."""
  x: np.ndarray
  y: np.ndarray
  low: np.ndarray | None = None
  high: np.ndarray | None = None

  def __post_init__(self):
    for name in ('x', 'y', 'low', 'high'):
      value = getattr(self, name)
      if value is not None:
        object.__setattr__(self, name, np.asarray(value, dtype=float))

  @classmethod
  def from_runs(
      cls, x: Sequence[float], runs: Sequence[Sequence[float]], band: str = 'se',
  ) -> Curve:
    """Mean over independent runs (rows), such as the five searches.

    The band is the standard error ('se'), the standard deviation ('sd'), or
    a 95% t-interval ('ci95') across runs.
    """
    runs = np.asarray(runs, dtype=float)
    mean = np.nanmean(runs, axis=0)
    count = np.sum(np.isfinite(runs), axis=0)
    spread = np.nanstd(runs, axis=0, ddof=1)
    if band == 'sd':
      half = spread
    elif band == 'se':
      half = spread / np.sqrt(count)
    elif band == 'ci95':
      from scipy import stats
      half = stats.t.ppf(0.975, count - 1) * spread / np.sqrt(count)
    else:
      raise ValueError(f'Unknown band {band!r}')
    return cls(x, mean, mean - half, mean + half)


@dataclass(frozen=True)
class Estimate:
  """A point estimate with an optional interval."""
  value: float
  low: float | None = None
  high: float | None = None

  @classmethod
  def from_replicates(
      cls, value: float, replicates: Sequence[float], level: float = 0.95,
  ) -> Estimate:
    """Percentile interval from bootstrap replicates."""
    tail = (1.0 - level) / 2.0
    low, high = np.nanquantile(np.asarray(replicates, float), (tail, 1 - tail))
    return cls(float(value), float(low), float(high))

  @classmethod
  def from_se(cls, value: float, se: float, z: float = 1.96) -> Estimate:
    return cls(float(value), float(value - z * se), float(value + z * se))

  def yerr(self) -> list[list[float]] | None:
    if self.low is None or self.high is None:
      return None
    return [[max(self.value - self.low, 0.0)], [max(self.high - self.value, 0.0)]]


def style():
  """Context manager for drawing custom figures in the same style."""
  return matplotlib.rc_context(STYLE)


def styled(function: Callable[..., plt.Figure]) -> Callable[..., plt.Figure]:
  """Build the figure under STYLE and lay it out before returning it."""
  @functools.wraps(function)
  def wrapper(*args, **kwargs):
    with style():
      fig = function(*args, **kwargs)
      fig.draw_without_rendering()
    return fig
  return wrapper


def new_figure(height: float, width: float = TEXT_WIDTH) -> plt.Figure:
  return plt.figure(figsize=(width, height), layout='constrained')


def save(fig: plt.Figure, path: str, *, png: bool = False, dpi: int = 300) -> str:
  """Write the figure as a PDF at its drawn size and close it.

  ``png=True`` also writes a 300 dpi PNG next to the PDF, for drafts.
  """
  if not path.lower().endswith('.pdf'):
    raise ValueError(f'Figures are saved as PDF: {path}')
  directory = os.path.dirname(path)
  if directory:
    os.makedirs(directory, exist_ok=True)
  with style():
    fig.savefig(path, metadata={'CreationDate': None})
    if png:
      fig.savefig(os.path.splitext(path)[0] + '.png', dpi=dpi)
  plt.close(fig)
  return path


def label(key: Hashable) -> str:
  """Display name of an identifier: 'ppo_sac' is 'PPO+SAC'."""
  if isinstance(key, tuple):
    return ', '.join(label(part) for part in key)
  for table in (ENVIRONMENTS, POLICY_SETS, ROLES):
    if key in table:
      return table[key].label
  for table in (DISAGREEMENT, MODELS, ASSOCIATIONS, STRATA,
                RELIABILITY, COSTS):
    if key in table:
      return table[key]
  return str(key)


def _number(value: float, _position=None) -> str:
  text = f'{value:.6g}'
  if text in ('-0', '-0.0'):
    text = '0'
  return text.replace('-', '\u2212')


def _per_panel(value, panel):
  return value.get(panel) if isinstance(value, Mapping) else value


def style_axis(ax: plt.Axes, grid: str | None = 'both') -> None:
  """Light grid behind the data and no top or right spine."""
  ax.spines[['top', 'right']].set_visible(False)
  if grid:
    ax.grid(True, axis=grid, color=GRID, linewidth=0.5)
  ax.set_axisbelow(True)


def draw_reference(ax: plt.Axes, reference) -> None:
  """'diagonal' draws y = x on [0, 1], a number draws a horizontal line."""
  if reference is None:
    return
  if isinstance(reference, str) and reference == 'diagonal':
    ax.plot((0, 1), (0, 1), color=REFERENCE, linewidth=0.8, zorder=1.5,
            solid_capstyle='butt')
  else:
    ax.axhline(float(reference), color=REFERENCE, linewidth=0.6, zorder=1.5)


def draw_curve(
    ax: plt.Axes,
    curve: Curve,
    series: SeriesStyle,
    *,
    filled: bool = True,
    markers: int | Sequence[int] | None = None,
    markersize: float = 3.1,
    linewidth: float = LINE_WIDTH,
    band: bool = True,
) -> None:
  """A band and a line. ``markers`` is Matplotlib's ``markevery``."""
  if band and curve.low is not None and curve.high is not None:
    ax.fill_between(curve.x, curve.low, curve.high, color=series.color,
                    alpha=BAND_ALPHA, linewidth=0, zorder=2)
  ax.plot(
      curve.x, curve.y,
      color=series.color,
      linestyle=series.linestyle,
      linewidth=linewidth,
      marker='o' if markers is not None else None,
      markevery=markers,
      markersize=markersize,
      markerfacecolor=series.color if filled else 'white',
      markeredgecolor=series.color,
      markeredgewidth=0.45 if markersize < 2.4 else 0.8,
      zorder=3,
  )


def policy_set_handles(
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    kind: str = 'line',
    linewidth: float = LINE_WIDTH,
) -> list[Line2D]:
  """Legend entries for policy sets, as lines or as error-bar strokes."""
  handles = []
  for key in policy_sets:
    series = POLICY_SETS[key]
    if kind == 'line':
      handles.append(Line2D([], [], color=series.color, linestyle=series.linestyle,
                            linewidth=linewidth * 1.2, label=series.label))
    else:
      handles.append(Line2D([], [], color=series.color, marker='|', linestyle='none',
                            markersize=8.0, markeredgewidth=1.5, label=series.label))
  return handles


def environment_handles(
    environments: Sequence[str] = ENVIRONMENT_ORDER,
) -> list[Line2D]:
  """Legend entries for environments: filled black marker shapes."""
  return [
      Line2D([], [], color=INK, marker=ENVIRONMENTS[key].marker, linestyle='none',
             markersize=4.2 * ENVIRONMENTS[key].scale, markeredgewidth=0,
             label=ENVIRONMENTS[key].label)
      for key in environments
  ]


def role_handles(
    roles: Sequence[str] = ('eval', 'search'),
    markersize: float = 3.6,
) -> list[Line2D]:
  """Legend entries for ensemble roles: filled and open markers."""
  edge = 0.45 if markersize < 2.4 else 0.8
  return [
      Line2D([], [], color=INK, marker='o', linestyle='none', markersize=markersize,
             markerfacecolor=INK if ROLES[key].filled else 'white',
             markeredgecolor=INK, markeredgewidth=edge, label=ROLES[key].label)
      for key in roles
  ]


def reference_handle(text: str) -> Line2D:
  return Line2D([], [], color=REFERENCE, linewidth=0.9, label=text)


def legend_gap() -> Line2D:
  """A blank legend entry that separates two groups."""
  return Line2D([], [], linestyle='none', label=' ')


def _legend_ncols(fig: plt.Figure, n: int, per_inch: float = 0.92) -> int:
  """How many legend entries fit in one row of this figure."""
  return max(1, min(n, int(fig.get_figwidth() / per_inch)))


def add_legend(
    fig: plt.Figure,
    handles: Sequence[Line2D],
    ncols: int | None = None,
    loc: str = 'outside upper center',
):
  """A frameless legend. Entries wrap onto a second row when the figure is narrow."""
  handles = [handle for handle in handles if handle.get_label() not in ('', ' ')]
  if not handles:
    return None
  return fig.legend(
      handles=handles,
      loc=loc,
      ncols=ncols or _legend_ncols(fig, len(handles)),
      frameon=False,
  )


def add_legend_rows(fig: plt.Figure, rows: Sequence[Sequence[Line2D]]):
  """Each sequence of handles becomes one legend row.

  Matplotlib fills a legend down each column, so the handles are stored
  in column-major order.
  """
  width = max((len(row) for row in rows), default=0)
  if width == 0:
    return None
  handles = []
  for column in range(width):
    for row in rows:
      handles.append(row[column] if column < len(row) else legend_gap())
  return fig.legend(handles=handles, loc='outside upper center', ncols=width,
                    frameon=False)


def label_panels(axes: Sequence[plt.Axes], letters: str = 'abcdefghij') -> None:
  """Bold panel letters at the top left, as Springer figures cite them."""
  for ax, letter in zip(axes, letters):
    ax.set_title(letter, loc='left', fontweight='bold', fontsize=8.5)


@styled
def curve_grid(
    curves: Mapping[tuple, Curve],
    *,
    rows: Sequence[Hashable],
    columns: Sequence[Hashable],
    series: Sequence[Hashable] = POLICY_SET_ORDER,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    series_style: Callable[[Hashable], tuple[SeriesStyle, bool]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    reference=None,
    markers: int | None = None,
    markersize: float = 3.1,
    linewidth: float = LINE_WIDTH,
    xlim: tuple[float, float] | None = (-0.02, 1.02),
    xticks: Sequence[float] | None = (0, 0.5, 1),
    xscale: str = 'linear',
    ylim: tuple[float, float] | None = None,
    yticks: Sequence[float] | None = None,
    sharey: bool | str = 'row',
    legend: Sequence[Line2D] | bool = True,
    legend_rows: Sequence[Sequence[Line2D]] | None = None,
    width: float = TEXT_WIDTH,
    panel_height: float = 0.78,
) -> plt.Figure:
  """Small multiples: one panel per (row, column) and one line per series.

  ``curves[(row, column, series)]`` is a Curve. ``series_style(series)``
  returns the SeriesStyle and whether its markers are filled; by default the
  series are policy sets. ``xscale='log2'`` puts x on a base-2 log scale.
  """
  style_of = series_style or (lambda key: (POLICY_SETS[key], True))
  fig = new_figure(len(rows) * panel_height + 0.78, width)
  axes = fig.subplots(len(rows), len(columns), sharex=True, sharey=sharey,
                      squeeze=False)
  for i, row in enumerate(rows):
    for j, column in enumerate(columns):
      ax = axes[i, j]
      style_axis(ax)
      draw_reference(ax, reference)
      for key in series:
        curve = curves.get((row, column, key))
        if curve is not None:
          line_style, filled = style_of(key)
          draw_curve(
              ax, curve, line_style, filled=filled, markers=markers,
              markersize=markersize, linewidth=linewidth,
          )
      if yticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 2.5, 5, 10)))
      else:
        ax.set_yticks(yticks)
      ax.yaxis.set_major_formatter(FuncFormatter(_number))
    axes[i, 0].set_ylabel(row_labels[i] if row_labels else label(row))
  for j, column in enumerate(columns):
    axes[0, j].set_title(column_labels[j] if column_labels else label(column))
  ax = axes[0, 0]
  if xscale == 'log2':
    ax.set_xscale('log', base=2)
    ax.xaxis.set_minor_locator(NullLocator())
  if xticks is not None:
    ax.set_xticks(xticks)
  ax.xaxis.set_major_formatter(FuncFormatter(_number))
  if xlim is not None:
    ax.set_xlim(xlim)
  if ylim is not None:
    for axis in axes[:, 0]:
      axis.set_ylim(ylim)
  if xlabel:
    fig.supxlabel(xlabel)
  if ylabel:
    fig.supylabel(ylabel)
  if legend_rows:
    add_legend_rows(fig, legend_rows)
  else:
    if legend is True:
      legend = policy_set_handles(
          [key for key in series if key in POLICY_SETS], linewidth=linewidth,
      )
    if legend:
      add_legend(fig, legend)
  return fig


def feature_curves_figure(
    curves: Mapping[tuple, Curve],
    *,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    measures: Sequence[str] = tuple(DISAGREEMENT),
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    xlabel: str = r'Evaluation failure rate, $r_E$',
    ylabel: str = STANDARDIZED_MEAN,
    **kwargs,
) -> plt.Figure:
  """Mean disagreement at each held-out fail rate (RQ2).

  ``curves[(environment, measure, policy_set)]`` runs over the fail-rate
  levels, for example ``Curve.from_runs(levels, per_search_means)``.
  """
  return curve_grid(
      curves,
      rows=environments,
      columns=measures,
      series=policy_sets,
      column_labels=[DISAGREEMENT_LINES.get(key, label(key)) for key in measures],
      xlabel=xlabel,
      ylabel=ylabel,
      reference=0.0,
      **{'panel_height': 0.88, 'linewidth': 1.6, **kwargs},
  )




def environment_curves_figure(
    curves: Mapping[tuple, Curve],
    *,
    measures: Sequence[str] = tuple(DISAGREEMENT),
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    xlabel: str = EVAL_RATE,
    ylabel: str = STANDARDIZED_MEAN,
    reference=0.0,
    **kwargs,
) -> plt.Figure:
  """One row of curves for a single environment, one panel per measure.

  ``curves[(measure, policy_set)]`` runs over the r_E levels. Use this for
  the close-up of one task; ``feature_curves_figure`` is the four-task grid.
  """
  keyed = {(None, measure, key): curve
           for (measure, key), curve in curves.items()}
  return curve_grid(
      keyed,
      rows=(None,),
      columns=measures,
      series=policy_sets,
      row_labels=[''],
      column_labels=[DISAGREEMENT_LINES.get(key, label(key)) for key in measures],
      xlabel=xlabel,
      ylabel=ylabel,
      reference=reference,
      **{'panel_height': 1.15, **kwargs},
  )


def transfer_figure(
    curves: Mapping[tuple, Curve],
    *,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    search_sets: Sequence[str] = ('ppo', 'sac', 'ppo_sac'),
    series: Sequence[tuple[str, str]] | None = None,
    panel_height: float = 1.0,
    **kwargs,
) -> plt.Figure:
  """Fail rate of each ensemble on the cases a search found (RQ1).

  ``curves[(environment, search_set, (policy_set, role))]`` is the fail rate
  of the ``role`` ensemble ('eval' or 'search') of ``policy_set`` against the
  quota r_S of ``search_set``. The column's own search ensemble lies on the
  diagonal by construction, so the gray diagonal stands in for it. Columns are
  the three searches. PPO and SAC share no policies, so each search fail rate
  is drawn on the other's cases. The mixed search reuses eight policies from
  each, so its fail rate is not drawn on the PPO or SAC searches, and theirs
  is not drawn on the mixed search. A filled circle is the evaluation
  ensemble and an open circle is the search ensemble, drawn at every quota.
  """
  if series is None:
    series = tuple((key, 'eval') for key in search_sets)
    series = (
        *series,
        *(
            (key, 'search')
            for key in search_sets
            if key in ('ppo', 'sac')
        ),
    )
  shown = list(dict.fromkeys(key for key, _role in series))
  fig = curve_grid(
      curves,
      rows=environments,
      columns=search_sets,
      series=series,
      column_labels=[f'{POLICY_SETS[key].label} search' for key in search_sets],
      series_style=lambda key: (POLICY_SETS[key[0]], ROLES[key[1]].filled),
      xlabel=SEARCH_RATE,
      ylabel='Fail rate',
      reference='diagonal',
      markers=1,
      markersize=1.8,
      ylim=(-0.03, 1.03),
      yticks=(0, 0.5, 1),
      legend=False,
      legend_rows=(
          policy_set_handles(shown),
          (*role_handles(markersize=2.2), reference_handle('search quota')),
      ),
      panel_height=panel_height,
      **kwargs,
  )
  return fig


def ensemble_size_figure(
    curves: Mapping[tuple, Curve],
    *,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    measures: Sequence[str] = tuple(DISAGREEMENT),
    sizes: Sequence[int] = (2, 4, 8, 16),
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    ylabel: str = r'Spearman $\rho$ with $r_E$',
    **kwargs,
) -> plt.Figure:
  """Association with r_E when the measures use P of the 16 policies (RQ2).

  ``curves[(environment, measure, policy_set)]`` runs over ``sizes``.
  """
  return curve_grid(
      curves,
      rows=environments,
      columns=measures,
      series=policy_sets,
      column_labels=[DISAGREEMENT_LINES.get(key, label(key)) for key in measures],
      xlabel='Policies in the ensemble $P$',
      ylabel=ylabel,
      markers=1,
      xscale='log2',
      xticks=sizes,
      xlim=(sizes[0] / 1.3, sizes[-1] * 1.3),
      **kwargs,
  )


def _dot_marker_size(axes_width: float, slots: int, pitch: float) -> float:
  points = axes_width * 72.0 / slots * pitch
  return float(np.clip(0.92 * points, 2.3, MARKER_SIZE))


def _cluster_pitches(n_slots: int, n_sets: int, n_envs: int) -> tuple[float, float]:
  """Group one slot by environment, with the policy-set colors spread inside.

  Returns the color spacing and the distance between environment centers.
  A wider gap separates the shapes than separates the colors.
  """
  room = 0.90 if n_slots <= 6 else 0.86
  group_gap = 1.7
  n_sets = max(n_sets, 1)
  n_envs = max(n_envs, 1)
  span = (n_envs - 1) * (n_sets - 1 + group_gap) + (n_sets - 1)
  set_pitch = room / span if span else 0.0
  env_pitch = (n_sets - 1 + group_gap) * set_pitch
  return set_pitch, env_pitch


def _slot_label(key: Hashable) -> str:
  if key in DISAGREEMENT_LINES:
    return DISAGREEMENT_LINES[key]
  return label(key)


def draw_estimates(
    ax: plt.Axes,
    estimates: Mapping[tuple, Estimate],
    *,
    slots: Sequence[Hashable],
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    open_slots: Sequence[Hashable] = (),
    set_pitch: float = SET_PITCH,
    env_pitch: float = ENV_PITCH,
    marker_size: float = MARKER_SIZE,
) -> None:
  """Dot-and-whisker estimates at categorical slots 0, 1, 2, ...

  ``estimates[(slot, policy_set, environment)]`` is an Estimate. Inside a
  slot the environments sit side by side, and inside an environment the
  policy-set colors are spread.
  """
  set_center = (len(policy_sets) - 1) / 2
  env_center = (len(environments) - 1) / 2
  for s, slot in enumerate(slots):
    hollow = slot in open_slots
    for e, environment in enumerate(environments):
      shape = ENVIRONMENTS[environment]
      for p, policy_set in enumerate(policy_sets):
        estimate = estimates.get((slot, policy_set, environment))
        if estimate is None or not np.isfinite(estimate.value):
          continue
        color = POLICY_SETS[policy_set].color
        ax.errorbar(
            s + (e - env_center) * env_pitch + (p - set_center) * set_pitch,
            estimate.value,
            yerr=estimate.yerr(),
            fmt=shape.marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=1.1,
            capthick=0.6,
            markersize=marker_size * shape.scale,
            markerfacecolor='white' if hollow else color,
            markeredgecolor=color if hollow else 'white',
            markeredgewidth=0.75 if hollow else 0.4,
            zorder=3,
        )


@styled
def dot_figure(
    estimates: Mapping[tuple, Estimate],
    *,
    panels: Sequence[Hashable],
    slots: Sequence[Hashable],
    slot_labels: Sequence[str] | None = None,
    ylabels: str | Mapping | None = None,
    panel_titles: str | Mapping | None = None,
    references: float | Mapping | None = 0.0,
    ylims: tuple | Mapping | None = None,
    yscale: str = 'linear',
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    open_slots: Sequence[Hashable] = (),
    dividers: Sequence[int] = (),
    letters: bool | None = None,
    legend: bool = True,
    width: float = TEXT_WIDTH,
    panel_height: float = 1.4,
    marker_scale: float = 1.0,
) -> plt.Figure:
  """Stacked dot-and-whisker panels over one shared row of slots.

  ``estimates[(panel, slot, policy_set, environment)]`` is an Estimate.
  ``ylabels``, ``panel_titles``, ``references`` and ``ylims`` take a single
  value for every panel or a mapping from panel to value. Slots alternate
  with a light shade; ``dividers`` draws a rule before each listed slot.
  """
  fig = new_figure(len(panels) * panel_height + 0.72, width)
  axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
  set_pitch, env_pitch = _cluster_pitches(len(slots), len(policy_sets),
                                         len(environments))
  size = marker_scale * _dot_marker_size(width * 0.86, len(slots), set_pitch)
  for ax, panel in zip(axes, panels):
    style_axis(ax, grid='y')
    for s in range(1, len(slots), 2):
      ax.axvspan(s - 0.5, s + 0.5, color=SHADE, linewidth=0, zorder=0)
    for index in dividers:
      ax.axvline(index - 0.5, color=RULE, linewidth=0.6, zorder=0.6)
    draw_reference(ax, _per_panel(references, panel))
    subset = {key[1:]: value for key, value in estimates.items() if key[0] == panel}
    draw_estimates(ax, subset, slots=slots, policy_sets=policy_sets,
                   environments=environments, open_slots=open_slots,
                   set_pitch=set_pitch, env_pitch=env_pitch, marker_size=size)
    ax.set_yscale(yscale)
    if yscale == 'linear':
      ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10)))
      ax.yaxis.set_major_formatter(FuncFormatter(_number))
    limits = _per_panel(ylims, panel)
    if limits is not None:
      ax.set_ylim(*limits)
    ylabel = _per_panel(ylabels, panel)
    if ylabel:
      ax.set_ylabel(ylabel)
    title = _per_panel(panel_titles, panel)
    if title:
      ax.set_title(title)
    ax.tick_params(axis='x', length=0)
  axes[-1].set_xticks(range(len(slots)),
                      labels=slot_labels or [_slot_label(slot) for slot in slots])
  axes[-1].set_xlim(-0.5, len(slots) - 0.5)
  if letters or (letters is None and len(panels) > 1):
    label_panels(axes)
  if legend:
    add_legend_rows(fig, [
        environment_handles(environments),
        policy_set_handles(policy_sets, kind='bar'),
    ])
  return fig


def _draw_baseline_band(
    ax: plt.Axes,
    baseline: Mapping[tuple, Estimate],
    text: str,
) -> None:
  """Shade the range of a baseline and name it inside the band."""
  values = [estimate.value for estimate in baseline.values()
            if np.isfinite(estimate.value)]
  if not values:
    return
  low, high = min(values), max(values)
  if high - low < 1e-6:
    low, high = low - 0.02, high + 0.02
  ax.axhspan(low, high, facecolor='#DCE4EE', edgecolor='none',
             linewidth=0, zorder=0.5)
  bottom, top = ax.get_ylim()
  margin = 0.045 * (max(top, high) - min(bottom, 0.0))
  ax.set_ylim(min(bottom, 0.0), max(top, high) + margin)
  ax.text(
      0.5, (low + high) / 2, text,
      transform=ax.get_yaxis_transform(),
      ha='center', va='center', fontsize=7, color=MUTED, zorder=1,
  )


def association_figure(
    estimates: Mapping[tuple, Estimate],
    *,
    measures: Sequence[str] = ('spearman', 'mi', 'semipartial_spearman'),
    features: Sequence[str] = tuple(DISAGREEMENT),
    baseline: Mapping[tuple, Estimate] | None = None,
    baseline_panel: str = 'mi',
    baseline_text: str = (
        r'region for mutual information between $r_S$ and $r_E$'
    ),
    spearman_baseline: Mapping[tuple, Estimate] | None = None,
    spearman_text: str = (
        r'region for correlation between $r_S$ and $r_E$'
    ),
    **kwargs,
) -> plt.Figure:
  """Association of each disagreement measure with r_E (RQ2).

  ``estimates[(measure, feature, policy_set, environment)]``, one panel per
  association measure in ``measures``. Mutual information starts at zero;
  the other panels carry a zero line.

  ``baseline[(policy_set, environment)]`` is I(r_S; r_E), drawn on
  ``baseline_panel``. ``spearman_baseline`` is ρ(r_S, r_E), drawn on the
  Spearman panel. Each band is labeled inside its range.
  """
  fig = dot_figure(
      estimates,
      panels=measures,
      slots=features,
      ylabels={key: ASSOCIATIONS.get(key, key) for key in measures},
      references={key: None if key == 'mi' else 0.0 for key in measures},
      ylims={'mi': (0, None)},
      **kwargs,
  )
  if baseline and baseline_panel in measures:
    _draw_baseline_band(
        fig.axes[list(measures).index(baseline_panel)],
        baseline,
        baseline_text,
    )
  if spearman_baseline and 'spearman' in measures:
    _draw_baseline_band(
        fig.axes[list(measures).index('spearman')],
        spearman_baseline,
        spearman_text,
    )
  return fig




def _model_group(model: Hashable) -> str:
  if model in DISAGREEMENT or model == 'disagreement':
    return 'disagreement'
  return 'ceiling' if model == 'ceiling' else 'other'


def _marker_x(
    slot: int,
    environment: str,
    policy_set: str,
    *,
    n_slots: int,
    policy_sets: Sequence[str],
    environments: Sequence[str],
) -> float:
  """Horizontal position of one dot inside a categorical slot."""
  set_pitch, env_pitch = _cluster_pitches(
      n_slots, len(policy_sets), len(environments)
  )
  set_center = (len(policy_sets) - 1) / 2
  env_center = (len(environments) - 1) / 2
  return (
      slot
      + (list(environments).index(environment) - env_center) * env_pitch
      + (list(policy_sets).index(policy_set) - set_center) * set_pitch
  )


def _set_environment_ticks(
    figure: plt.Figure,
    n_panels: int,
    models: Sequence[str],
    policy_sets: Sequence[str],
    environments: Sequence[str],
) -> None:
  """Name each environment under its cluster."""
  _set_pitch, env_pitch = _cluster_pitches(
      len(models), len(policy_sets), len(environments),
  )
  env_center = (len(environments) - 1) / 2
  positions = []
  labels = []
  for slot in range(len(models)):
    for index, environment in enumerate(environments):
      positions.append(slot + (index - env_center) * env_pitch)
      labels.append(ENVIRONMENTS[environment].label)
  axis = figure.axes[n_panels - 1]
  axis.set_xticks(positions, labels=labels)
  axis.tick_params(axis='x', length=0)


def model_score_figure(
    estimates: Mapping[tuple, Estimate],
    *,
    models: Sequence[str] = ('disagreement',),
    panels: Sequence[Hashable] = ('all',),
    ylabel: str = r'Cross-validated $\rho$ with $r_E$',
    ylabels: str | Mapping | None = None,
    search_correlations: Mapping[tuple, float] | None = None,
    search_by_panel: Mapping[Hashable, Mapping[tuple, float]] | None = None,
    **kwargs,
) -> plt.Figure:
  """Agreement of out-of-fold predictions with r_E (RQ2 and RQ3).

  ``estimates[(panel, model, policy_set, environment)]``; the panels are
  usually the strata. The 'ceiling' slot, the square root of the split-half
  reliability of r_E, is drawn open. A rule separates a disagreement model
  from the ceiling when both are drawn.

  ``search_correlations[(policy_set, environment)]`` is Spearman
  $\\rho(r_S, r_E)$ for that search, drawn on every panel. ``search_by_panel``
  supplies a separate map per panel, so a concordance panel can mark
  $\\rho_c(r_S, r_E)$ instead. Each value is a black star at the same
  horizontal position as that search's predictive score.

  ``environment_labels`` replaces the slot names with Hopper, Walker2d,
  Humanoid, and Ant under each environment's cluster.
  """
  dividers = [
      index for index in range(1, len(models))
      if _model_group(models[index]) != _model_group(models[index - 1])
  ]
  policy_sets = kwargs.pop('policy_sets', POLICY_SET_ORDER)
  environments = kwargs.pop('environments', ENVIRONMENT_ORDER)
  environment_labels = bool(kwargs.pop('environment_labels', False))
  star_markersize = float(kwargs.pop('star_markersize', 6.5))
  kwargs.setdefault('panel_titles', {key: STRATA[key] for key in panels
                                     if key in STRATA})
  figure = dot_figure(
      estimates,
      panels=panels,
      slots=models,
      ylabels=ylabel if ylabels is None else ylabels,
      open_slots=('ceiling',),
      dividers=dividers,
      policy_sets=policy_sets,
      environments=environments,
      **kwargs,
  )
  if environment_labels:
    _set_environment_ticks(
        figure, len(panels), models, policy_sets, environments,
    )
  if search_by_panel is None:
    if not search_correlations:
      return figure
    search_by_panel = {panel: search_correlations for panel in panels}
    star_label = r'$\rho(r_S, r_E)$'
  else:
    star_label = (
        r'$r_S$ with $r_E$' if len(search_by_panel) > 1 else r'$\rho(r_S, r_E)$'
    )
  slot = list(models).index('disagreement') if 'disagreement' in models else 0
  drew_star = False
  for axis, panel in zip(figure.axes[:len(panels)], panels):
    mapping = search_by_panel.get(panel)
    if not mapping:
      continue
    for (policy_set, environment), value in mapping.items():
      if not np.isfinite(value):
        continue
      drew_star = True
      axis.plot(
          _marker_x(
              slot, environment, policy_set,
              n_slots=len(models),
              policy_sets=policy_sets,
              environments=environments,
          ),
          value,
          marker='*',
          color='black',
          linestyle='none',
          markersize=star_markersize,
          markeredgewidth=0.4,
          zorder=4,
      )
  if not drew_star:
    return figure
  for legend in list(figure.legends):
    legend.remove()
  add_legend_rows(figure, [
      environment_handles(environments),
      [
          *policy_set_handles(policy_sets, kind='bar'),
          Line2D(
              [], [], color='black', marker='*', linestyle='none',
              markersize=star_markersize + 0.4, markeredgewidth=0.4,
              label=star_label,
          ),
      ],
  ])
  return figure


@styled
def pooled_environment_score_figure(
    estimates: Mapping[str, Estimate],
    search_correlations: Mapping[tuple, float] | None = None,
    *,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    ylabel: str = r'cross-validated $\rho$ with $r_E$',
    slot_label: str = 'all disagreement features',
    panel_title: str = 'All policy sets',
    marker_scale: float = 2.0,
    width: float = TEXT_WIDTH,
    panel_height: float = 1.4,
) -> plt.Figure:
  """One pooled extra-trees score per environment.

  PPO, SAC, and PPO+SAC rows of one environment share a model. The marker
  is that model's cross-validated correlation with $r_E$. Each black star
  is Spearman $\\rho(r_S, r_E)$ for one of the searches in the pool, drawn
  at the same horizontal position as the environment's score.
  """
  figure = new_figure(panel_height + 0.72, width)
  axis = figure.subplots(1, 1)
  style_axis(axis, grid='y')
  draw_reference(axis, 0.0)
  set_pitch, _env_pitch = _cluster_pitches(1, 1, len(environments))
  size = marker_scale * _dot_marker_size(width * 0.86, 1, set_pitch)
  policy_sets = ('pooled',)
  for environment in environments:
    estimate = estimates.get(environment)
    if estimate is None or not np.isfinite(estimate.value):
      continue
    shape = ENVIRONMENTS[environment]
    x = _marker_x(
        0, environment, 'pooled',
        n_slots=1, policy_sets=policy_sets, environments=environments,
    )
    axis.errorbar(
        x,
        estimate.value,
        yerr=estimate.yerr(),
        fmt=shape.marker,
        color=INK,
        ecolor=INK,
        elinewidth=0.8,
        capsize=1.1,
        capthick=0.6,
        markersize=size * shape.scale,
        markerfacecolor=INK,
        markeredgecolor='white',
        markeredgewidth=0.4,
        zorder=3,
    )
    if not search_correlations:
      continue
    for key, value in search_correlations.items():
      if key[0] != environment or not np.isfinite(value):
        continue
      axis.plot(
          x, value,
          marker='*', color='black', linestyle='none',
          markersize=6.5, markeredgewidth=0.4, zorder=4,
      )
  axis.set_xlim(-0.5, 0.5)
  axis.set_xticks([0], labels=[slot_label])
  axis.tick_params(axis='x', length=0)
  axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10)))
  axis.yaxis.set_major_formatter(FuncFormatter(_number))
  axis.set_ylabel(ylabel)
  if panel_title:
    axis.set_title(panel_title)
  add_legend(figure, [
      *environment_handles(environments),
      Line2D(
          [], [], color='black', marker='*', linestyle='none',
          markersize=7.0, markeredgewidth=0.4,
          label=r'$\rho(r_S, r_E)$',
      ),
  ])
  return figure


def ceiling_figure(
    estimates: Mapping[tuple, Estimate],
    *,
    quantities: Sequence[str] = tuple(RELIABILITY),
    ylabel: str = 'Rank correlation',
    width: float = TEXT_WIDTH,
    **kwargs,
) -> plt.Figure:
  """How well r_S orders cases by r_E, against the reliability of r_E (RQ1).

  ``estimates[(quantity, policy_set, environment)]`` for the quantities in
  RELIABILITY: rho(r_S, r_E), the Spearman-Brown split-half reliability of
  r_E, and its square root, the ceiling for any predictor of r_E.
  """
  return dot_figure(
      {('ceiling', *key): value for key, value in estimates.items()},
      panels=('ceiling',),
      slots=quantities,
      ylabels=ylabel,
      references=None,
      open_slots=('ceiling',),
      width=width,
      **kwargs,
  )


def cost_figure(
    costs: Mapping[tuple, Estimate],
    *,
    items: Sequence[str] = tuple(COSTS),
    ylabel: str = 'Seconds per test case',
    **kwargs,
) -> plt.Figure:
  """Compute time of each measure next to the rollouts it stands in for.

  ``costs[(item, policy_set, environment)]`` is an Estimate, for example the
  median and quartiles of repeated timings. The y-axis is logarithmic.
  """
  return dot_figure(
      {('cost', *key): value for key, value in costs.items()},
      panels=('cost',),
      slots=items,
      ylabels=ylabel,
      references=None,
      yscale='log',
      dividers=[index for index, key in enumerate(items) if key == 'rollout'],
      **kwargs,
  )


def _correlation_text(value: float) -> str:
  text = f'{value:.2f}'
  text = text.replace('0.', '.', 1) if abs(value) < 1 else text
  return text.replace('-', '\u2212')


@styled
def redundancy_figure(
    matrices: Mapping[Hashable, np.ndarray],
    *,
    panels: Sequence[Hashable],
    panel_titles: Mapping | None = None,
    measures: Sequence[str] = tuple(DISAGREEMENT),
    colorbar_label: str = r'Spearman $\rho$',
    width: float = TEXT_WIDTH,
    height: float | None = None,
) -> plt.Figure:
  """Rank correlation between the disagreement measures (RQ2).

  ``matrices[panel]`` is a square matrix ordered like ``measures``; each
  panel shows its lower triangle.
  """
  names = [_slot_label(key).replace('\n', ' ') for key in measures]
  count = len(measures)
  n_panels = len(panels)
  nrows = 1 if n_panels <= 2 else 2
  ncols = math.ceil(n_panels / nrows)
  fig = new_figure(height or (2.35 * nrows + 0.15), width)
  axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
  image = None
  for ax, panel in zip(axes, panels):
    lower = np.asarray(matrices[panel], dtype=float)[1:, :-1]
    hidden = np.triu(np.ones(lower.shape, dtype=bool), k=1)
    image = ax.imshow(
        np.ma.masked_array(lower, hidden),
        cmap=CORRELATION_CMAP,
        vmin=-1,
        vmax=1,
        aspect='equal',
    )
    for i in range(count - 1):
      for j in range(i + 1):
        value = lower[i, j]
        if np.isfinite(value):
          ax.text(j, i, _correlation_text(value), ha='center', va='center',
                  fontsize=6.5, color='white' if abs(value) > 0.55 else INK)
    ax.set_xticks(range(count - 1), labels=names[:-1], rotation=35,
                  ha='right', rotation_mode='anchor')
    ax.set_yticks(range(count - 1), labels=names[1:])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
      spine.set_visible(False)
    title = _per_panel(panel_titles, panel) if panel_titles else label(panel)
    ax.set_title(title)
  for ax in axes[n_panels:]:
    ax.set_visible(False)
  colorbar = fig.colorbar(image, ax=list(axes[:n_panels]), shrink=0.68,
                          aspect=18, pad=0.03)
  colorbar.set_label(colorbar_label)
  colorbar.set_ticks((-1, -0.5, 0, 0.5, 1))
  colorbar.ax.yaxis.set_major_formatter(FuncFormatter(_number))
  colorbar.ax.tick_params(length=2, width=0.5)
  colorbar.outline.set_visible(False)
  return fig


def _lanes(values: np.ndarray, lanes: int = 3, pitch: float = 0.05) -> np.ndarray:
  ranks = np.argsort(np.argsort(values))
  return (ranks % lanes - (lanes - 1) / 2) * pitch


@styled
def ensemble_returns_figure(
    returns: Mapping[tuple, Sequence[float]],
    *,
    environments: Sequence[str] = ENVIRONMENT_ORDER,
    policy_sets: Sequence[str] = POLICY_SET_ORDER,
    ylabel: str = 'Best evaluation return',
    width: float = TEXT_WIDTH,
    height: float = 2.15,
) -> plt.Figure:
  """Return of every policy in the search and evaluation ensembles.

  ``returns[(environment, policy_set, role)]`` holds one return per policy.
  Each strip is one ensemble, and the black bar is its mean.
  """
  fig = new_figure(height, width)
  axes = fig.subplots(1, len(environments), squeeze=False)[0]
  for ax, environment in zip(axes, environments):
    shape = ENVIRONMENTS[environment]
    style_axis(ax, grid='y')
    for i, key in enumerate(policy_sets):
      color = POLICY_SETS[key].color
      for role, offset in (('search', -0.18), ('eval', 0.18)):
        values = np.asarray(returns.get((environment, key, role), ()), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
          continue
        x = i + offset
        ax.plot(x + _lanes(values), values, linestyle='none', marker=shape.marker,
                markersize=2.9 * shape.scale,
                markerfacecolor=color if ROLES[role].filled else 'white',
                markeredgecolor=color, markeredgewidth=0.6, zorder=3)
        ax.plot((x - 0.11, x + 0.11), (values.mean(),) * 2, color=INK,
                linewidth=1.1, solid_capstyle='butt', zorder=4)
    tick_labels = [
        POLICY_SETS[key].label.replace('+', '+\n') for key in policy_sets
    ]
    ax.set_xticks(range(len(policy_sets)), labels=tick_labels)
    ax.tick_params(axis='x', length=0, labelsize=6.5, pad=1.5)
    ax.set_xlim(-0.55, len(policy_sets) - 0.45)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 2.5, 5, 10)))
    ax.yaxis.set_major_formatter(FuncFormatter(_number))
    ax.set_title(shape.label)
  fig.supylabel(ylabel)
  mean = Line2D([], [], color=INK, linewidth=1.1, solid_capstyle='butt',
                label='mean')
  add_legend(fig, [*role_handles(), mean])
  return fig


LEVELS = np.arange(17) / 16


def _demo_transfer(rng):
  curves = {}
  sets = ('ppo', 'sac', 'ppo_sac')
  for rank, environment in enumerate(ENVIRONMENT_ORDER):
    for search_set in sets:
      roles = [(name, 'eval') for name in sets]
      if search_set in ('ppo', 'sac'):
        other = 'sac' if search_set == 'ppo' else 'ppo'
        roles.append((other, 'search'))
      for key, role in roles:
        same = key == search_set
        gain = (0.93 if same else 0.6 + 0.07 * rank) * rng.uniform(0.95, 1.05)
        floor = 0.02 if same else rng.uniform(0.03, 0.12)
        y = floor + gain * LEVELS ** rng.uniform(0.9, 1.3)
        y = np.clip(y + rng.normal(0, 0.008, y.shape), 0, 1)
        half = 0.01 + 0.03 * np.sqrt(LEVELS * (1 - LEVELS))
        curves[(environment, search_set, (key, role))] = Curve(
            LEVELS, y, y - half, y + half,
        )
  return transfer_figure(curves)


def _demo_ceiling(rng):
  estimates = {}
  for key in POLICY_SET_ORDER:
    for environment in ENVIRONMENT_ORDER:
      split = rng.uniform(0.88, 0.97)
      rho = split * rng.uniform(0.86, 0.96)
      for quantity, value in (('rho_search_eval', rho), ('split_half', split),
                              ('ceiling', math.sqrt(split))):
        half = rng.uniform(0.004, 0.012)
        estimates[(quantity, key, environment)] = Estimate(
            value, value - half, value + half,
        )
  return ceiling_figure(estimates)


def _demo_feature_curves(rng):
  curves = {}
  for environment in ENVIRONMENT_ORDER:
    for measure in DISAGREEMENT:
      slope, bend = rng.normal(0.45, 0.3), rng.normal(0, 0.25)
      for key in POLICY_SET_ORDER:
        mean = (slope * rng.uniform(0.7, 1.3) * (2 * LEVELS - 1)
                + bend * (4 * LEVELS * (1 - LEVELS) - 2 / 3))
        runs = mean + rng.normal(0, 0.12, (5, LEVELS.size))
        curves[(environment, measure, key)] = Curve.from_runs(LEVELS, runs)
  return feature_curves_figure(curves)


def _demo_environment_curves(rng):
  curves = {}
  for measure in DISAGREEMENT:
    slope, bend = rng.normal(0.45, 0.3), rng.normal(0, 0.25)
    for key in POLICY_SET_ORDER:
      mean = (slope * rng.uniform(0.7, 1.3) * (2 * LEVELS - 1)
              + bend * (4 * LEVELS * (1 - LEVELS) - 2 / 3))
      runs = mean + rng.normal(0, 0.12, (5, LEVELS.size))
      curves[(measure, key)] = Curve.from_runs(LEVELS, runs)
  return environment_curves_figure(curves)


def _demo_associations(rng):
  estimates = {}
  centers = dict(zip(DISAGREEMENT, (0.32, 0.27, 0.16, 0.41, 0.52)))
  for feature, center in centers.items():
    for key in POLICY_SET_ORDER:
      for environment in ENVIRONMENT_ORDER:
        rho = float(np.clip(center + rng.normal(0, 0.1), -0.2, 0.9))
        half = rng.uniform(0.008, 0.025)
        mi = max(0.0, 0.5 * rho ** 2 + rng.normal(0.02, 0.01))
        partial = 0.35 * rho + rng.normal(0, 0.04)
        for measure, value, spread in (('spearman', rho, half),
                                       ('mi', mi, half / 2),
                                       ('semipartial_spearman', partial, half)):
          estimates[(measure, feature, key, environment)] = Estimate(
              value, max(value - spread, 0.0) if measure == 'mi' else value - spread,
              value + spread,
          )
  baseline = {
      (key, environment): Estimate(float(rng.uniform(0.55, 0.9)))
      for key in POLICY_SET_ORDER
      for environment in ENVIRONMENT_ORDER
  }
  spearman_baseline = {
      (key, environment): Estimate(float(rng.uniform(0.80, 0.92)))
      for key in POLICY_SET_ORDER
      for environment in ENVIRONMENT_ORDER
  }
  return association_figure(
      estimates, baseline=baseline, spearman_baseline=spearman_baseline,
  )


def _demo_model_scores(rng):
  estimates = {}
  models = ('disagreement', 'ceiling')
  centers = {'disagreement': 0.55, 'ceiling': 0.93}
  for model in models:
    for key in POLICY_SET_ORDER:
      for environment in ENVIRONMENT_ORDER:
        value = float(np.clip(centers[model] + rng.normal(0, 0.04), 0, 0.99))
        half = rng.uniform(0.006, 0.02 if model == 'ceiling' else 0.035)
        estimates[('spearman', model, key, environment)] = Estimate(
            value, value - half, value + half,
        )
  return model_score_figure(estimates, models=models, panels=('spearman',))


def _demo_ensemble_size(rng):
  sizes = np.array((2, 4, 8, 16), dtype=float)
  curves = {}
  for environment in ENVIRONMENT_ORDER:
    for measure in DISAGREEMENT:
      for key in POLICY_SET_ORDER:
        final = rng.uniform(0.15, 0.6)
        rho = final * (1 - 0.9 / sizes) / (1 - 0.9 / 16)
        half = 0.015 + 0.06 / sizes
        curves[(environment, measure, key)] = Curve(sizes, rho, rho - half,
                                                    rho + half)
  return ensemble_size_figure(curves)


def _demo_cost(rng):
  base = dict(zip(COSTS, (2e-5, 2e-5, 4e-5, 8e-4, 2.5e-3, 6e-2)))
  scale = {'hopper': 1.0, 'walker2d': 1.3, 'humanoid': 3.0, 'ant': 1.8}
  width = {'ppo': 1.0, 'sac': 2.2, 'ppo_sac': 1.6}
  costs = {}
  for item, seconds in base.items():
    for key in POLICY_SET_ORDER:
      for environment in ENVIRONMENT_ORDER:
        value = seconds * scale[environment] * width[key] * math.exp(
            rng.normal(0, 0.1))
        costs[(item, key, environment)] = Estimate(value, 0.9 * value,
                                                   1.12 * value)
  return cost_figure(costs)






def _demo_redundancy(rng):
  matrices = {}
  for environment in ENVIRONMENT_ORDER:
    loadings = rng.normal(0.55, 0.35, (len(DISAGREEMENT), 2))
    covariance = loadings @ loadings.T + np.diag(
        rng.uniform(0.2, 0.6, len(DISAGREEMENT)))
    scale = np.sqrt(np.diag(covariance))
    matrices[environment] = covariance / np.outer(scale, scale)
  return redundancy_figure(matrices, panels=ENVIRONMENT_ORDER)


def _demo_returns(rng):
  centers = {'hopper': 3300, 'walker2d': 5450, 'humanoid': 11800, 'ant': 4300}
  returns = {}
  for environment, center in centers.items():
    for key in POLICY_SET_ORDER:
      mean = center * rng.uniform(0.85, 1.1)
      for role in ROLES:
        returns[(environment, key, role)] = rng.normal(mean, 0.07 * center, 16)
  return ensemble_returns_figure(returns)


__all__ = (
    'Curve',
    'Estimate',
    'TEXT_WIDTH',
    'POLICY_SETS',
    'ENVIRONMENTS',
    'DISAGREEMENT',
    'style',
    'save',
    'label',
    'style_axis',
    'draw_curve',
    'draw_estimates',
    'curve_grid',
    'dot_figure',
    'feature_curves_figure',
    'environment_curves_figure',
    'transfer_figure',
    'ensemble_size_figure',
    'association_figure',
    'model_score_figure',
    'ceiling_figure',
    'cost_figure',
    'redundancy_figure',
    'ensemble_returns_figure',
)


DEMOS = {
    'transfer': _demo_transfer,
    'ceiling': _demo_ceiling,
    'feature_curves': _demo_feature_curves,
    'environment_curves': _demo_environment_curves,
    'associations': _demo_associations,
    'model_scores': _demo_model_scores,
    'ensemble_size': _demo_ensemble_size,
    'cost': _demo_cost,
    'redundancy': _demo_redundancy,
    'ensemble_returns': _demo_returns,
}


def write_demo(directory: str, names: Sequence[str] = tuple(DEMOS)) -> list[str]:
  """Draw the chosen figure types from synthetic numbers into ``directory``."""
  paths = []
  for name in names:
    fig = DEMOS[name](np.random.default_rng(7))
    fig.text(0.004, 0.996, 'synthetic data', ha='left', va='top', fontsize=6,
             color=MUTED, style='italic',
             bbox=dict(facecolor='white', edgecolor='none', pad=1.2, alpha=0.85))
    paths.append(save(fig, os.path.join(directory, f'{name}.pdf'), png=True))
    print(f'  wrote {paths[-1]}', flush=True)
  return paths


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--demo', metavar='DIR',
                      help='draw every figure type from synthetic numbers')
  parser.add_argument('--only', nargs='+', choices=tuple(DEMOS),
                      default=tuple(DEMOS), help='figure types for --demo')
  args = parser.parse_args(argv)
  if args.demo:
    write_demo(args.demo, args.only)
  else:
    parser.print_help()


if __name__ == '__main__':
  main()
