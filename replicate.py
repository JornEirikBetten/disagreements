"""Draw the result figures from this package's experiment outputs.

Writes PDFs to ``results/``. Reads the search archives in ``test_suites/``,
the analysis tables in ``figures/``, and the MuJoCo models.

  python replicate.py
  python replicate.py --only fig03 fig07
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
os.chdir(ROOT)
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import plotting.style as figs


def _pdf(name: str) -> Path:
  OUT.mkdir(parents=True, exist_ok=True)
  return OUT / name


def _require(path: Path, rebuild: str) -> Path:
  if not path.is_file():
    raise FileNotFoundError(f'{path} is missing. {rebuild}')
  return path


def _read_csv(path: Path) -> list[dict]:
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _save(figure, name: str) -> Path:
  path = _pdf(name)
  figs.save(figure, str(path))
  _check(path)
  print(f'wrote {path.relative_to(ROOT)}', flush=True)
  return path


def _check(path: Path) -> None:
  data = path.read_bytes()[:5]
  if data != b'%PDF-' or path.stat().st_size < 1000:
    raise RuntimeError(f'{path} is not a usable PDF ({path.stat().st_size} bytes).')
  draft = path.with_suffix('.png')
  if draft.is_file():
    draft.unlink()


def figure_bodies() -> Path:
  """Figure 2. Hopper, Walker2d, Humanoid, and Ant at a shared scale."""
  from plotting.plot_environment_bodies import environment_bodies_figure

  return _save(environment_bodies_figure(), 'fig02_bodies.pdf')


def figure_transfer() -> Path:
  """Figure 3. Fail rate of every other ensemble against the search quota."""
  from plotting.plot_failure_rate_transfer import failure_rate_transfer_figure

  return _save(failure_rate_transfer_figure(), 'fig03_transfer.pdf')


def figure_associations() -> Path:
  """Figure 4. Rank correlation and mutual information with r_E."""
  from plotting.plot_disagreement_associations import disagreement_associations_figure

  return _save(disagreement_associations_figure(), 'fig04_associations.pdf')


def figure_curves() -> Path:
  """Figure 5. Mean disagreement at each evaluation fail rate."""
  from plotting.plot_disagreement_by_difficulty import disagreement_by_difficulty_figure

  return _save(disagreement_by_difficulty_figure(), 'fig05_disagreement_curves.pdf')


def figure_predictors() -> Path:
  """Figure 6. Cross-validated extra-trees agreement with r_E."""
  from plotting.plot_predictor_agreement import write_score_figure

  archived = ROOT / 'figures/predictors/extra_trees_10_foldcv_experiment'
  retrained = ROOT / 'figures/predictors/extra_trees_5_foldcv_experiment'
  stem = archived if (archived.parent / (archived.name + '_scores.csv')).is_file() else retrained
  scores = _read_csv(_require(
      stem.parent / (stem.name + '_scores.csv'),
      'Rebuild it with `python cv_difficulty_predictors.py --folds 5`.',
  ))
  ceiling = _read_csv(_require(
      stem.parent / (stem.name + '_search_eval_spearman.csv'),
      'The r_S ceiling is written beside the extra-trees scores.',
  ))
  path = _pdf('fig06_predictors.pdf')
  write_score_figure(scores, str(path), search_eval=ceiling)
  _check(path)
  print(f'wrote {path.relative_to(ROOT)}', flush=True)
  return path


def figure_search() -> Path:
  """Figure 7. Search and evaluation fail rates over 100 generations."""
  from plotting.plot_search_fail_rates import search_fail_rates_figure

  return _save(search_fail_rates_figure(), 'fig07_search.pdf')


def figure_distributions() -> Path:
  """Figure B1. Value, action magnitude, and criticality by search quota."""
  from plotting.plot_value_magnitude_criticality import save_combined

  path = _pdf('figB1_distributions.pdf')
  save_combined(str(path))
  _check(path)
  print(f'wrote {path.relative_to(ROOT)}', flush=True)
  return path


def figure_episodes() -> Path:
  """Figure B2. Humanoid evaluation length and return."""
  from plotting.plot_humanoid_episode_lengths import (
      episode_length_figure,
      load,
  )

  _require(
      ROOT / 'figures/policy_performances/humanoid_episode_lengths.csv',
      'The policy table is figures/policy_performances/humanoid_episode_lengths.csv.',
  )
  return _save(episode_length_figure(load()), 'figB2_humanoid_episodes.pdf')


FIGURES = (
    ('fig02', 'Fig. 2', 'Environment bodies', figure_bodies),
    ('fig03', 'Fig. 3', 'Difficulty transfer', figure_transfer),
    ('fig04', 'Fig. 4', 'Association with difficulty', figure_associations),
    ('fig05', 'Fig. 5', 'Disagreement curves', figure_curves),
    ('fig06', 'Fig. 6', 'Extra-trees predictors', figure_predictors),
    ('fig07', 'Fig. 7', 'Disagreement search', figure_search),
    ('figB1', 'Fig. B1', 'Quota distributions', figure_distributions),
    ('figB2', 'Fig. B2', 'Humanoid episodes', figure_episodes),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument(
      '--only',
      nargs='+',
      choices=[key for key, *_rest in FIGURES],
      help='Figure ids to draw. The default is every figure.',
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  selected = set(args.only) if args.only else {key for key, *_rest in FIGURES}
  written = []
  for key, number, title, draw in FIGURES:
    if key not in selected:
      continue
    print(f'=== {number}  {title} ===', flush=True)
    written.append(draw())
  print(f'{len(written)} figures in {OUT.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
  main()
