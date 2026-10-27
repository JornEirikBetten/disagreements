#!/usr/bin/env python3
"""Train the locomotion policies used in the paper.

One script covers Hopper, Walker2d, Ant, and Humanoid, with both Soft
Actor-Critic (SAC) and Proximal Policy Optimization (PPO). The numbers below
are the hyperparameters in Appendix C, Table C3. The learning algorithms
themselves are the Brax 0.14.2 agents in ``src/sac_modified`` and
``src/ppo_modified``: SAC with a learned temperature, and PPO with a clipped
surrogate, a value loss on the lambda-return, and an entropy bonus.

Examples
--------
Train the 50 Hopper SAC policies (seeds 0, 123, ..., 123*49)::

    python train_policies.py hopper sac

Train Walker2d PPO policies 11 through 20, without Weights & Biases::

    python train_policies.py walker2d ppo --num-seeds 10 --seed-start 10 --no-wandb

What one run does
-----------------
1. Build the MJX environment and the network for the chosen algorithm.
2. Evaluate the randomly initialized policy (this is evaluation 1 of
   ``num_evals``).
3. Train. SAC fills the replay buffer, then alternates environment steps with
   gradient updates. PPO collects a fresh rollout and optimizes it for ``K``
   epochs. Each remaining evaluation follows one of these blocks.
4. Keep the parameters with the highest mean evaluation return, and also save
   the parameters from the end of training.

An evaluation is the mean undiscounted return of 128 episodes of 1,000 control
steps. Actions are sampled from the stochastic policy. The checkpoint used
later in the paper is the best one, not necessarily the final one.

Seeds are ``123 * j`` for ``j = seed_start, ..., seed_start + num_seeds - 1``.
The paper uses ``j = 0, ..., 49``.
"""

import argparse
import functools
import os

# Set before JAX is imported. SAC and PPO both train on one GPU; the flag
# enables the GEMM kernels used for these runs.
os.environ.setdefault('MUJOCO_GL', 'glfw')
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

import numpy as np
from flax import linen

from brax import envs
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks

import src.ppo_modified as ppo
import src.sac_modified as sac
from src.envs.humanoid import register_humanoid

try:
  import wandb
except ImportError:
  wandb = None


# ---------------------------------------------------------------------------
# Hyperparameters from Appendix C, Table C3.
#
# Shared by every environment and both algorithms:
#   episode length 1,000 control steps, action repeat 1,
#   128 stochastic evaluation episodes,
#   observation standardization (running mean and standard deviation),
#   Adam with beta1=0.9, beta2=0.999, eps=1e-8, constant step size,
#   no gradient clipping.
#
# The policy is a diagonal Gaussian. The network emits 2 * dim(A) values.
# The standard deviation is softplus(s_sigma) + 1e-3, and the action is
# tanh of a sample from that Gaussian. Both of those are the Brax defaults
# for the tanh-normal distribution; they are not extra arguments here.
# ---------------------------------------------------------------------------

ENVS = ('hopper', 'walker2d', 'ant', 'humanoid')
ALGOS = ('sac', 'ppo')

# Fixed by the implementation, and the same for every run.
EPISODE_LENGTH = 1000
ACTION_REPEAT = 1
NUM_EVAL_ENVS = 128
DETERMINISTIC_EVAL = False  # evaluation samples the stochastic policy
NORMALIZE_OBSERVATIONS = True

# PPO loss constants. The value coefficient c_V = 0.5 is the default inside
# brax.training.agents.ppo.losses.compute_ppo_loss (the loss multiplies the
# mean squared error by c_V / 2). It is not an argument of ppo.train.
PPO_CLIP_EPSILON = 0.3
PPO_GAE_LAMBDA = 0.95
PPO_NORMALIZE_ADVANTAGE = True
PPO_POLICY_HIDDEN = (32, 32, 32, 32)          # Swish
PPO_VALUE_HIDDEN = (256, 256, 256, 256, 256)  # Swish

# SAC constants. The temperature Adam step size is fixed at 3e-4 inside
# src/sac_modified/modified_training_sac.py. The target entropy is
# -dim(A) / 2 and the temperature starts at 1, also inside that trainer.
SAC_HIDDEN = (256, 256)  # ReLU, actor and both critics
SAC_TAU = 0.005          # Polyak factor for the target critics
SAC_REPLAY_CAPACITY = 1_048_576  # 2**20
SAC_MAX_DEVICES = 1      # SAC updates stay on one device
SAC_ALPHA_LEARNING_RATE = 3e-4  # documented here; applied in the trainer

# Per-environment SAC settings. ``grad_updates_per_step`` is G in Table C3:
# after each step of the 128 parallel environments, the learner takes G
# minibatch updates (temperature, then critics, then actor).
SAC = {
    'hopper': dict(
        num_timesteps=6_553_600,
        num_evals=20,
        reward_scaling=30.0,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        min_replay_size=8_192,
        grad_updates_per_step=64,
        wandb_project='sac-hopper',
    ),
    'walker2d': dict(
        num_timesteps=7_864_320,
        num_evals=20,
        reward_scaling=5.0,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=128,
        min_replay_size=8_192,
        grad_updates_per_step=32,
        wandb_project='sac-walker2d',
    ),
    # Ant and Humanoid share one SAC configuration.
    'ant': dict(
        num_timesteps=10_000_000,
        num_evals=50,
        reward_scaling=0.1,
        discounting=0.99,
        learning_rate=3e-4,
        num_envs=128,
        batch_size=256,
        min_replay_size=10_000,
        grad_updates_per_step=64,
        wandb_project='sac-ant',
    ),
    'humanoid': dict(
        num_timesteps=10_000_000,
        num_evals=50,
        reward_scaling=0.1,
        discounting=0.99,
        learning_rate=3e-4,
        num_envs=128,
        batch_size=256,
        min_replay_size=10_000,
        grad_updates_per_step=64,
        wandb_project='sac-humanoid',
    ),
}

# Per-environment PPO settings.
# One iteration collects B * M * L transitions (minibatch size, number of
# minibatches, unroll length), then reuses that rollout for K epochs.
# num_evals counts the initial evaluation plus one evaluation after each
# training block, so there are num_evals - 1 blocks.
PPO = {
    'hopper': dict(
        num_timesteps=50_000_000,
        num_evals=50,
        reward_scaling=5.0,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        num_minibatches=16,
        unroll_length=10,
        num_updates_per_batch=8,
        wandb_project='hopper_ppo',
    ),
    'walker2d': dict(
        num_timesteps=50_000_000,
        num_evals=50,
        reward_scaling=1.0,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=512,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=8,
        wandb_project='walker2d_ppo',
    ),
    'ant': dict(
        num_timesteps=50_000_000,
        num_evals=50,
        reward_scaling=10.0,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=2048,
        num_minibatches=32,
        unroll_length=5,
        num_updates_per_batch=4,
        wandb_project='ant_ppo',
    ),
    'humanoid': dict(
        num_timesteps=200_000_000,
        num_evals=100,
        reward_scaling=10.0,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        num_minibatches=16,
        unroll_length=5,
        num_updates_per_batch=4,
        wandb_project='humanoid_ppo',
    ),
}


def sac_network_factory():
  """Actor and twin critics: two hidden layers of width 256, ReLU."""
  return functools.partial(
      sac_networks.make_sac_networks,
      hidden_layer_sizes=SAC_HIDDEN,
      activation=linen.relu,
  )


def ppo_network_factory():
  """Actor (32, 32, 32, 32) and value (256,) * 5, both with Swish."""
  return functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=PPO_POLICY_HIDDEN,
      value_hidden_layer_sizes=PPO_VALUE_HIDDEN,
      activation=linen.swish,
  )


def make_environment(env_name):
  """Return ``(env, wrap_for_training)``.

  Hopper, Walker2d, and Ant are the Brax MJX environments. Their reward,
  reset noise, and solver settings are the Brax 0.14.2 defaults described in
  Appendix C.

  Humanoid is the custom MJX environment in ``src/envs/humanoid.py``: the
  MuJoCo humanoid model, conjugate-gradient solver with 6 iterations and 6
  line-search iterations, and 5 physics steps of 5 ms per control step.
  Reward weights are forward 1.25, control cost 0.1, healthy reward 5.

  ``wrap_for_training`` tells the trainer to apply the Brax training wrapper:
  vectorized episodes of 1,000 steps and an automatic reset when an episode ends.
  """
  if env_name == 'humanoid':
    register_humanoid()
    return envs.get_environment('humanoid'), True
  if env_name in ('hopper', 'walker2d', 'ant'):
    return envs.get_environment(env_name, backend='mjx'), True
  raise ValueError(f'Unknown environment {env_name!r}. Expected one of {ENVS}.')


def paper_seeds(num_seeds, seed_start):
  """Seeds 123*j for j in [seed_start, seed_start + num_seeds)."""
  return [int(123 * (seed_start + i)) for i in range(num_seeds)]


def make_progress_fn(checkpoint_path, use_wandb):
  """Save the policy whenever the mean evaluation return improves.

  The trainer calls this as
  ``progress_fn(env_steps, metrics, parameters, best_so_far)`` and expects
  the updated best return back.
  """
  def progress_fn(num_steps, metrics, parameters, best_eval_reward):
    metrics['num_env_steps'] = num_steps
    if use_wandb:
      wandb.log(metrics)
    if 'eval/episode_reward' in metrics:
      current = metrics['eval/episode_reward']
      if current > best_eval_reward:
        best_eval_reward = current
        model.save_params(checkpoint_path + 'params.json', parameters)
        print(
            f'  new best return {float(current):.3f} at {int(num_steps)} steps'
            f' -> {checkpoint_path}params.json'
        )
    return best_eval_reward

  return progress_fn


def train_sac(env, wrap_env, cfg, seed, progress_fn):
  """Run one SAC training.

  After a warm-up filled by the initial stochastic policy, every step of the
  parallel environments is followed by ``grad_updates_per_step`` minibatch
  updates. The warm-up is rounded up to a multiple of ``num_envs`` inside the
  trainer, so Ant and Humanoid collect 10,112 transitions rather than 10,000.
  """
  train_fn = functools.partial(
      sac.train,
      num_timesteps=cfg['num_timesteps'],
      num_evals=cfg['num_evals'],
      reward_scaling=cfg['reward_scaling'],
      episode_length=EPISODE_LENGTH,
      normalize_observations=NORMALIZE_OBSERVATIONS,
      action_repeat=ACTION_REPEAT,
      discounting=cfg['discounting'],
      learning_rate=cfg['learning_rate'],
      num_envs=cfg['num_envs'],
      batch_size=cfg['batch_size'],
      seed=seed,
      min_replay_size=cfg['min_replay_size'],
      max_replay_size=SAC_REPLAY_CAPACITY,
      grad_updates_per_step=cfg['grad_updates_per_step'],
      tau=SAC_TAU,
      max_devices_per_host=SAC_MAX_DEVICES,
      num_eval_envs=NUM_EVAL_ENVS,
      deterministic_eval=DETERMINISTIC_EVAL,
      network_factory=sac_network_factory(),
      wrap_env=wrap_env,
  )
  return train_fn(environment=env, progress_fn=progress_fn)


def train_ppo(env, wrap_env, cfg, seed, progress_fn):
  """Run one PPO training.

  Each block rolls out ``batch_size * num_minibatches * unroll_length``
  transitions with the current policy, standardizes advantages inside each
  minibatch, and takes ``num_updates_per_batch`` epochs over that rollout.
  ``num_envs`` must be divisible by the number of devices.
  """
  train_fn = functools.partial(
      ppo.train,
      num_timesteps=cfg['num_timesteps'],
      num_evals=cfg['num_evals'],
      reward_scaling=cfg['reward_scaling'],
      episode_length=EPISODE_LENGTH,
      normalize_observations=NORMALIZE_OBSERVATIONS,
      action_repeat=ACTION_REPEAT,
      unroll_length=cfg['unroll_length'],
      num_minibatches=cfg['num_minibatches'],
      num_updates_per_batch=cfg['num_updates_per_batch'],
      discounting=cfg['discounting'],
      learning_rate=cfg['learning_rate'],
      entropy_cost=cfg['entropy_cost'],
      num_envs=cfg['num_envs'],
      batch_size=cfg['batch_size'],
      seed=seed,
      clipping_epsilon=PPO_CLIP_EPSILON,
      gae_lambda=PPO_GAE_LAMBDA,
      normalize_advantage=PPO_NORMALIZE_ADVANTAGE,
      num_eval_envs=NUM_EVAL_ENVS,
      deterministic_eval=DETERMINISTIC_EVAL,
      network_factory=ppo_network_factory(),
      wrap_env=wrap_env,
  )
  return train_fn(environment=env, progress_fn=progress_fn)


def train_one(env_name, algo, seed, run_idx, use_wandb):
  """Train a single policy and write its best and final parameters."""
  cfg = dict(SAC[env_name] if algo == 'sac' else PPO[env_name])
  env, wrap_env = make_environment(env_name)

  best_dir = os.path.join(
      os.getcwd(), 'trained_policies', env_name, algo, f'best_control_{run_idx}'
  )
  final_dir = os.path.join(
      os.getcwd(), 'trained_policies', env_name, algo, f'final_control_{run_idx}'
  )
  os.makedirs(best_dir, exist_ok=True)
  os.makedirs(final_dir, exist_ok=True)
  best_path = best_dir + os.sep
  final_path = final_dir + os.sep

  print(
      f'{env_name} {algo} run {run_idx}: seed {seed}, '
      f'{cfg["num_timesteps"]} steps, {cfg["num_evals"]} evaluations'
  )

  if use_wandb:
    wandb.init(
        project=cfg['wandb_project'],
        config={
            'env': env_name,
            'algo': algo,
            'seed': seed,
            **{k: v for k, v in cfg.items() if k != 'wandb_project'},
            'episode_length': EPISODE_LENGTH,
            'num_eval_envs': NUM_EVAL_ENVS,
            'sac_tau': SAC_TAU if algo == 'sac' else None,
            'sac_alpha_learning_rate': (
                SAC_ALPHA_LEARNING_RATE if algo == 'sac' else None
            ),
            'ppo_clip_epsilon': PPO_CLIP_EPSILON if algo == 'ppo' else None,
            'ppo_gae_lambda': PPO_GAE_LAMBDA if algo == 'ppo' else None,
        },
        name=(
            f'{env_name}-{algo}-{run_idx}'
            f'-batch_size={cfg["batch_size"]}'
            f'-lr={cfg["learning_rate"]:.2e}'
        ),
    )

  progress_fn = make_progress_fn(best_path, use_wandb)
  train = train_sac if algo == 'sac' else train_ppo
  _, params, _ = train(env, wrap_env, cfg, seed, progress_fn)

  model.save_params(final_path + 'params.json', params)
  print(f'  final parameters -> {final_path}params.json')
  if use_wandb:
    wandb.finish()


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          'Train SAC or PPO on Hopper, Walker2d, Ant, or Humanoid '
          'with the paper hyperparameters.'
      ),
  )
  parser.add_argument('env', choices=ENVS, help='Locomotion environment.')
  parser.add_argument('algo', choices=ALGOS, help='sac or ppo.')
  parser.add_argument(
      '--num-seeds',
      type=int,
      default=50,
      help='Number of independent policies. The paper trains 50.',
  )
  parser.add_argument(
      '--seed-start',
      type=int,
      default=0,
      help=(
          'Index of the first seed. Run j uses seed 123 * (seed_start + j) '
          'and is saved as best_control_{seed_start + j + 1}.'
      ),
  )
  parser.add_argument(
      '--no-wandb',
      action='store_true',
      help='Do not log to Weights & Biases. Checkpoints are still saved.',
  )
  args = parser.parse_args()
  if args.num_seeds < 1:
    parser.error('--num-seeds must be at least 1')
  if args.seed_start < 0:
    parser.error('--seed-start must be non-negative')
  if not args.no_wandb and wandb is None:
    parser.error('wandb is not installed; pass --no-wandb')
  return args


def main():
  np.set_printoptions(precision=3, suppress=True, linewidth=100)
  args = parse_args()
  seeds = paper_seeds(args.num_seeds, args.seed_start)
  for offset, seed in enumerate(seeds):
    run_idx = args.seed_start + offset + 1
    train_one(
        args.env,
        args.algo,
        seed,
        run_idx,
        use_wandb=not args.no_wandb,
    )


if __name__ == '__main__':
  main()
