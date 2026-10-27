# Replication repository

This directory reproduces *Initial-State Policy Disagreements to Estimate Test Case Difficulty in Continuous Control*. It contains the experiment code, the trained policies, the search archives, and the commands that rebuild the result figures.

Run the commands below with the project environment. From the parent checkout that is `../venv/bin/python`. From a fresh environment, install `requirements.txt` first. MuJoCo needs an EGL device for rendering and for the searches.

```bash
cd disagreements
python reproduce.py check
python reproduce.py figures
```

`check` confirms the released archives. `figures` writes the eight result PDFs to `results/`.

## Layout

| Path | Contents |
| --- | --- |
| `search_for_tests/`, `ga/`, `envs/`, `policies/`, `analysis/` | Search, environments, policy loading, disagreement measures |
| `trained_policies/{environment}/{algorithm}/best_control_{id}/` | The 50 PPO and SAC checkpoints kept for each environment |
| `datasets/policy_performances.csv` | Evaluation return used to rank those checkpoints |
| `datasets/ga_data/{environment}/{algorithm}/difficulty_bins/` | The five quota searches, 34,000 states each |
| `datasets/{environment}/{algorithm}/predictor_data_balanced.pkl` | Disagreement features and fail rates for the analyses |
| `test_suites/` | Elite suites and per-generation fail rates for the disagreement searches |
| `figures/` | Transfer tables, association tables, predictor scores, distribution caches |
| `results/` | The paper figures, written by `python reproduce.py figures` |

`datasets/`, `trained_policies/`, `test_suites/`, and `figures/` are links to the study archive next to this directory, so the archives are not stored twice. A portable copy follows the links:

```bash
cp -aL disagreements disagreements-portable
```

## Policies

Fifty PPO policies and fifty SAC policies were trained in Hopper, Walker2d, Humanoid, and Ant. Seeds are `123 × j` for `j = 0, …, 49`. The checkpoint is the evaluation with the highest return. The network shapes, reward scales, and optimiser settings are the ones in Appendix C of the paper and are summarised below. The training loop that wrote the checkpoints ran against the Brax PPO and SAC implementations and logged to the W&B projects named in `extract_wandb_policy_performances.py`. Those checkpoints are the starting point of every experiment in this package.

The best 32 checkpoints of each algorithm are split by alternating down the return ranking: odd ranks form the search set `S`, even ranks form the held-out set. The mixed set takes the eight best of each algorithm. Print the split that the searches use:

```bash
python reproduce.py policy-sets
```

| | Hopper | Walker2d | Ant | Humanoid |
| --- | --- | --- | --- | --- |
| SAC steps | 6,553,600 | 7,864,320 | 10,000,000 | 10,000,000 |
| SAC discount | 0.997 | 0.997 | 0.99 | 0.99 |
| SAC reward scale | 30 | 5 | 0.1 | 0.1 |
| PPO steps | 50,000,000 | 50,000,000 | 50,000,000 | 200,000,000 |
| PPO discount | 0.99 | 0.99 | 0.97 | 0.95 |
| PPO reward scale | 5 | 1 | 10 | 10 |

Both algorithms use a tanh-Gaussian policy. SAC uses two-layer ReLU networks of width 256. PPO uses a four-layer Swish actor of width 32 and a five-layer Swish value network of width 256. The remaining shared settings, including PPO unroll length and SAC warm-up, are in Appendix C.

## Quota searches

These searches build the pools behind Figures 3–6 and B1. One search keeps 2,000 distinct initial states in each of 17 bins, from all-pass through all-fail, 34,000 states in total. The horizon is 15 control steps. The population is 1,000. Five independent searches are run for PPO, SAC, and PPO+SAC in each of the four environments. The held-out policies score the stored states after the search stops. They do not guide it.

One finished search is skipped. This reruns Hopper, PPO, experiment 1:

```bash
python reproduce.py quota --environment hopper --algorithm ppo --experiment 1
```

The paper grid is 60 searches:

```bash
python reproduce.py quota --all
```

`--overwrite` deletes a finished collection and searches again. Outputs:

```text
datasets/ga_data/{environment}/{algorithm}/difficulty_bins/rollout_length=15/experiment_{1-5}_ga_collection.pkl
```

Pool the five searches of each policy set into the table the analyses read:

```bash
python reproduce.py datasets
```

## Disagreement searches

These searches produce Figure 7. Each one keeps 10,000 elites for 100 generations and is repeated as experiments 100–104. Humanoid uses the PPO search set. Hopper, Walker2d, and Ant use SAC. The six fitnesses are the search fail rate, action-direction disagreement, action-magnitude disagreement, value disagreement, criticality disagreement, and attribution disagreement. Attribution is maximised everywhere except Humanoid, where it is minimised. Disagreement is measured at the initial state. Candidates that fail every search policy are discarded.

The fail-rate and criticality searches share their initial population:

```bash
python reproduce.py quad-search --environment hopper --experiment 100
python reproduce.py feature-search --environment hopper --experiment 100 --feature value_disagreement
python reproduce.py attribution-search --environment humanoid --experiment 100
```

`--all` runs every environment and experiment. A suite that already has `failure_rates.csv` is skipped. The per-generation fail rates are under `test_suites/`.

## Analyses

Each command reads the quota archives and writes the table the corresponding figure uses. The released tables are already in `figures/`.

```bash
python reproduce.py transfer --source ppo          # also sac, ppo_sac; Figure 3
python reproduce.py associations                   # Figures 4 and 5, B = 1000
python reproduce.py predictors                      # Figure 6, 5-fold, 500 trees
python reproduce.py distributions                   # Figure B1 from the cache
python reproduce.py distributions --extract         # rebuild that cache from the shards
```

`transfer` rolls the other policy sets out on the saved states. `associations` bootstraps over the five searches. Both are long runs. `associations --environment hopper --algorithm ppo` scores one dataset.

## Figures

```bash
python reproduce.py figures
python reproduce.py figures --only fig03 fig07
```

| Paper | File | Plot |
| --- | --- | --- |
| Fig. 2 | `results/fig02_bodies.pdf` | `plotting/plot_environment_bodies.py` |
| Fig. 3 | `results/fig03_transfer.pdf` | `plotting/plot_failure_rate_transfer.py` |
| Fig. 4 | `results/fig04_associations.pdf` | `plotting/plot_disagreement_associations.py` |
| Fig. 5 | `results/fig05_disagreement_curves.pdf` | `plotting/plot_disagreement_by_difficulty.py` |
| Fig. 6 | `results/fig06_predictors.pdf` | `plotting/plot_predictor_agreement.py` |
| Fig. 7 | `results/fig07_search.pdf` | `plotting/plot_search_fail_rates.py` |
| Fig. B1 | `results/figB1_distributions.pdf` | `plotting/plot_value_magnitude_criticality.py` |
| Fig. B2 | `results/figB2_humanoid_episodes.pdf` | `plotting/plot_humanoid_episode_lengths.py` |

Figure 6’s archived score file is the 5-fold leave-one-search-out result. Retraining with `python reproduce.py predictors` writes `figures/predictors/extra_trees_5_foldcv_experiment_scores.csv`, and the figure stage uses that file when the archived one is absent. Humanoid experiment 100 of the search-fail-rate fitness has an empty suite directory, so Figure 7 averages the other four repetitions of that fitness.
