# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import cast

from debug.sac_debug import *
from debug.split_replay_buffer import SplitReplayBuffer

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    debug: bool = False
    """if toggled, log additional diagnostic metrics (grad norms, Q-value stats, log-pi stats)"""
    holdout_fraction: float = 0.1
    """fraction of the replay buffer held out (debug mode only) for separate metric logging"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))
        self.relu_conv = nn.ReLU()
        self.relu_fc1 = nn.ReLU()

    def forward(self, x):
        x = self.relu_conv(self.conv(x / 255.0))
        x = self.relu_fc1(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))
        self.relu_conv = nn.ReLU()
        self.relu_fc1 = nn.ReLU()

    def forward(self, x):
        x = self.relu_conv(self.conv(x))
        x = self.relu_fc1(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    if args.debug:
        rb = SplitReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
            holdout_fraction=args.holdout_fraction,
            min_train_size=args.batch_size,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
    start_time = time.time()
    if args.debug:
        qf_grad_norm = actor_grad_norm = alpha_grad_norm = 0.0
        # Fixed reference batch for Hessian sharpness only: sampled once from
        # holdout, with the actor's frozen next-state action-probability
        # distribution (and its log-probs) at cache time, so later sharpness
        # comparisons aren't confounded by actor drift.
        ref_data = None
        ref_next_state_action_probs = None
        ref_next_state_log_pi = None

        # Debug-mode logging cadences (see CrossQ investigation metrics table)
        DEBUG_WEIGHT_NORM_FREQ = 1_000      # critic weight norms / effective scale
        DEBUG_HOLDOUT_BATCH_FREQ = 5_000    # effective rank, dead-ReLU, actor entropy, generalization gap (fresh holdout sample)
        DEBUG_LIPSCHITZ_FREQ = 5_000        # critic Lipschitz (fresh holdout sample)
        DEBUG_GRAD_VARIANCE_FREQ = 5_000    # gradient variance/SNR (train buffer)
        DEBUG_SHARPNESS_FREQ = 25_000       # critic Hessian sharpness (fixed reference batch, frozen action distribution)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                if args.debug:
                    qf_grad_norm = compute_grad_norm(qf1, qf2)
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                if args.debug:
                    actor_grad_norm = compute_grad_norm(actor)
                actor_optimizer.step()

                if args.autotune:
                    # reuse action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    if args.debug:
                        alpha_grad_norm = compute_grad_norm(log_alpha)
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                if args.debug:
                    log_grad_norms(writer, global_step, qf_grad_norm, actor_grad_norm,
                                   alpha_grad_norm if args.autotune else None)
                    log_q_stats(writer, global_step, qf1_a_values, qf2_a_values, next_q_value)

                    split_rb = cast(SplitReplayBuffer, rb)

                    # Critic weight norms / effective scale: free, param-only.
                    if global_step % DEBUG_WEIGHT_NORM_FREQ == 0:
                        log_weight_norms(writer, global_step, qf1, qf2)
                        log_normalization_scales(writer, global_step, qf1, qf2)

                    # Effective rank, dead-ReLU rate, actor entropy/action
                    # distribution, and generalization gap all use a fresh
                    # sample from the holdout buffer at the same cadence.
                    if split_rb.holdout_size() >= args.batch_size and global_step % DEBUG_HOLDOUT_BATCH_FREQ == 0:
                        holdout_data = split_rb.sample_holdout(args.batch_size)
                        with torch.no_grad():
                            _, holdout_next_state_log_pi, holdout_next_state_action_probs = actor.get_action(
                                holdout_data.next_observations
                            )
                            holdout_qf1_next_target = qf1_target(holdout_data.next_observations)
                            holdout_qf2_next_target = qf2_target(holdout_data.next_observations)
                            holdout_min_qf_next_target = holdout_next_state_action_probs * (
                                torch.min(holdout_qf1_next_target, holdout_qf2_next_target)
                                - alpha * holdout_next_state_log_pi
                            )
                            holdout_min_qf_next_target = holdout_min_qf_next_target.sum(dim=1)
                            holdout_next_q_value = holdout_data.rewards.flatten() + (
                                1 - holdout_data.dones.flatten()
                            ) * args.gamma * holdout_min_qf_next_target

                            holdout_qf1_values = qf1(holdout_data.observations)
                            holdout_qf2_values = qf2(holdout_data.observations)
                            holdout_qf1_a_values = holdout_qf1_values.gather(1, holdout_data.actions.long()).view(-1)
                            holdout_qf2_a_values = holdout_qf2_values.gather(1, holdout_data.actions.long()).view(-1)
                            holdout_qf1_loss = F.mse_loss(holdout_qf1_a_values, holdout_next_q_value)
                            holdout_qf2_loss = F.mse_loss(holdout_qf2_a_values, holdout_next_q_value)

                            _, holdout_log_pi, holdout_action_probs = actor.get_action(holdout_data.observations)
                            holdout_qf1_pi = qf1(holdout_data.observations)
                            holdout_qf2_pi = qf2(holdout_data.observations)
                            holdout_min_qf_pi = torch.min(holdout_qf1_pi, holdout_qf2_pi)
                            holdout_actor_loss = (holdout_action_probs * ((alpha * holdout_log_pi) - holdout_min_qf_pi)).mean()

                        log_q_stats(
                            writer, global_step, holdout_qf1_a_values, holdout_qf2_a_values, holdout_next_q_value,
                            prefix="holdout",
                        )
                        log_losses(
                            writer, global_step,
                            holdout_qf1_loss.item(), holdout_qf2_loss.item(), holdout_actor_loss.item(),
                            prefix="holdout",
                        )
                        log_generalization_gap(
                            writer, global_step,
                            qf1_loss.item(), qf2_loss.item(), actor_loss.item(),
                            holdout_qf1_loss.item(), holdout_qf2_loss.item(), holdout_actor_loss.item(),
                            prefix="holdout",
                        )
                        log_log_pi_stats_discrete(
                            writer, global_step, holdout_log_pi, holdout_action_probs,
                            holdout_next_state_log_pi, holdout_next_state_action_probs, prefix="holdout",
                        )
                        log_action_distribution_stats(writer, global_step, holdout_action_probs, prefix="holdout")
                        log_layer_activation_stats(
                            writer, global_step, qf1, qf2, holdout_data.observations, prefix="holdout"
                        )

                    # Critic Lipschitz: fresh holdout sample, coarser cadence.
                    if split_rb.holdout_size() >= args.batch_size and global_step % DEBUG_LIPSCHITZ_FREQ == 0:
                        lipschitz_data = split_rb.sample_holdout(args.batch_size)
                        log_gradient_lipschitz(
                            writer, global_step, qf1, qf2, lipschitz_data.observations, prefix="holdout"
                        )

                    # Gradient variance/SNR: fresh resamples from the train
                    # buffer, since this is specifically about training-loss
                    # gradient noise.
                    if global_step % DEBUG_GRAD_VARIANCE_FREQ == 0:
                        def make_sample_qf_loss_fn(critic):
                            def sample_qf_loss_fn():
                                sample = rb.sample(args.batch_size)
                                with torch.no_grad():
                                    _, sample_next_state_log_pi, sample_next_state_action_probs = actor.get_action(
                                        sample.next_observations
                                    )
                                    sample_qf1_next_target = qf1_target(sample.next_observations)
                                    sample_qf2_next_target = qf2_target(sample.next_observations)
                                    sample_min_qf_next_target = sample_next_state_action_probs * (
                                        torch.min(sample_qf1_next_target, sample_qf2_next_target)
                                        - alpha * sample_next_state_log_pi
                                    )
                                    sample_min_qf_next_target = sample_min_qf_next_target.sum(dim=1)
                                    sample_next_q_value = sample.rewards.flatten() + (
                                        1 - sample.dones.flatten()
                                    ) * args.gamma * sample_min_qf_next_target
                                return F.mse_loss(
                                    critic(sample.observations).gather(1, sample.actions.long()).view(-1),
                                    sample_next_q_value,
                                )
                            return sample_qf_loss_fn

                        log_gradient_variance_stats(writer, global_step, qf1, make_sample_qf_loss_fn(qf1), name="qf1")
                        log_gradient_variance_stats(writer, global_step, qf2, make_sample_qf_loss_fn(qf2), name="qf2")

                    # Critic Hessian sharpness: fixed reference batch with the
                    # actor's frozen next-state action-probability
                    # distribution (cached once), but current target networks
                    # and alpha — isolates critic-only drift from actor drift.
                    if ref_data is None and split_rb.holdout_size() >= args.batch_size:
                        ref_data = split_rb.sample_holdout(args.batch_size)
                        with torch.no_grad():
                            _, ref_next_state_log_pi, ref_next_state_action_probs = actor.get_action(
                                ref_data.next_observations
                            )

                    if ref_data is not None and global_step % DEBUG_SHARPNESS_FREQ == 0:
                        with torch.no_grad():
                            ref_qf1_next_target = qf1_target(ref_data.next_observations)
                            ref_qf2_next_target = qf2_target(ref_data.next_observations)
                            ref_min_qf_next_target = ref_next_state_action_probs * (
                                torch.min(ref_qf1_next_target, ref_qf2_next_target) - alpha * ref_next_state_log_pi
                            )
                            ref_min_qf_next_target = ref_min_qf_next_target.sum(dim=1)
                            ref_next_q_value = ref_data.rewards.flatten() + (
                                1 - ref_data.dones.flatten()
                            ) * args.gamma * ref_min_qf_next_target

                        def ref_qf1_loss_fn():
                            return F.mse_loss(
                                qf1(ref_data.observations).gather(1, ref_data.actions.long()).view(-1), ref_next_q_value
                            )

                        def ref_qf2_loss_fn():
                            return F.mse_loss(
                                qf2(ref_data.observations).gather(1, ref_data.actions.long()).view(-1), ref_next_q_value
                            )

                        log_sharpness_stats(writer, global_step, qf1, ref_qf1_loss_fn, name="qf1", prefix="holdout")
                        log_sharpness_stats(writer, global_step, qf2, ref_qf2_loss_fn, name="qf2", prefix="holdout")

    envs.close()
    writer.close()
