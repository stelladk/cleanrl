# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
from warnings import warn
from torch.utils.tensorboard import SummaryWriter

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
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    debug: bool = False
    """if toggled, log additional diagnostic metrics (grad norms, Q-value stats, log-pi stats)"""
    holdout_fraction: float = 0.1
    """fraction of the replay buffer held out (debug mode only) for separate metric logging"""

    # Architecture
    selu: bool = False
    """use selu activation function instead of relu"""
    batchnorm: bool = False
    """use BatchNorm in the critic"""
    layernorm: bool = False
    """use LayerNorm in the critic"""
    critic_simba: bool = False
    """use the SimBa residual critic architecture (https://github.com/SonyResearch/simba) instead of the plain MLP critic"""
    critic_hidden_dim: int = 512
    """hidden dimension of the critic network (only used when --critic-simba is set; SimBa paper uses 512)"""
    critic_num_blocks: int = 2
    """number of residual blocks in the critic (only used when --critic-simba is set; SimBa paper uses 2)"""
    critic_weight_decay: float = 0.0
    """weight decay for the critic optimizer (AdamW); SimBa paper uses 1e-2 together with --critic-simba"""
    actor_simba: bool = False
    """use the SimBa residual actor architecture (https://github.com/SonyResearch/simba) instead of the plain MLP actor"""
    actor_hidden_dim: int = 128
    """hidden dimension of the actor network (only used when --actor-simba is set; SimBa paper uses 128)"""
    actor_num_blocks: int = 1
    """number of residual blocks in the actor (only used when --actor-simba is set; SimBa paper uses 1)"""
    actor_weight_decay: float = 0.0
    """weight decay for the actor optimizer (AdamW); SimBa paper uses 1e-2 together with --actor-simba"""
    normalize_observation: bool = False
    """normalize observations with a running mean/std wrapper; SimBa paper enables this together with --critic-simba/--actor-simba"""
    temp_target_entropy_coef: float = -1.0
    """target entropy = temp_target_entropy_coef * action_dim (autotune only); SimBa paper uses -0.5 instead of -1.0"""
    temp_initial_value: float = 1.0
    """initial entropy coefficient alpha (autotune only); SimBa paper uses 0.01 instead of 1.0"""


def make_env(env_id, seed, idx, capture_video, run_name, normalize_observation):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, selu: bool, batchnorm: bool, layernorm: bool):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu1 = nn.SELU() if selu else nn.ReLU()
        self.relu2 = nn.SELU() if selu else nn.ReLU()

        if layernorm:
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)
            if selu:
                warn("SeLU is self-normalizing and normalization is redundant", UserWarning)
        elif batchnorm:
            self.norm1 = nn.BatchNorm1d(256)
            self.norm2 = nn.BatchNorm1d(256)
            if selu:
                warn("SeLU is self-normalizing and normalization is redundant", UserWarning)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.relu1(self.norm1(self.fc1(x)))
        x = self.relu2(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with a 4x-expansion ReLU MLP, as used in SimBa
    (https://github.com/SonyResearch/simba) to let depth scale without losing the
    identity path."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class SimbaQNetwork(nn.Module):
    """SAC critic using the SimBa architecture (https://github.com/SonyResearch/simba):
    an input projection, a stack of residual blocks, a closing LayerNorm, and a linear
    head, orthogonal-initialized at the projection/head so the residual stream starts
    near-identity."""

    def __init__(self, env, hidden_dim: int, num_blocks: int):
        super().__init__()
        input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)


def make_critic(envs: gym.vector.VectorEnv, args: Args) -> nn.Module:
    if args.critic_simba:
        return SimbaQNetwork(envs, hidden_dim=args.critic_hidden_dim, num_blocks=args.critic_num_blocks)
    return SoftQNetwork(envs, selu=args.selu, batchnorm=args.batchnorm, layernorm=args.layernorm)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, selu: bool):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.relu1 = nn.SELU() if selu else nn.ReLU()
        self.relu2 = nn.SELU() if selu else nn.ReLU()
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SimbaActor(Actor):
    """SAC actor using the SimBa architecture (https://github.com/SonyResearch/simba):
    input projection, residual blocks, closing LayerNorm, then orthogonal-initialized
    mean/log-std heads. Subclasses `Actor` purely to reuse `get_action`; the
    architecture (`forward`) is unrelated to `Actor.__init__`, so it is not called."""

    def __init__(self, env, hidden_dim: int, num_blocks: int):
        nn.Module.__init__(self)
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.fc_mean.weight, gain=1.0)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.orthogonal_(self.fc_logstd.weight, gain=1.0)
        nn.init.zeros_(self.fc_logstd.bias)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


def make_actor(envs: gym.vector.VectorEnv, args: Args) -> nn.Module:
    if args.actor_simba:
        return SimbaActor(envs, hidden_dim=args.actor_hidden_dim, num_blocks=args.actor_num_blocks)
    return Actor(envs, selu=args.selu)


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
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.normalize_observation)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = make_actor(envs, args).to(device)
    qf1 = make_critic(envs, args).to(device)
    qf2 = make_critic(envs, args).to(device)
    qf1_target = make_critic(envs, args).to(device)
    qf2_target = make_critic(envs, args).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.critic_weight_decay
    )
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.policy_lr, weight_decay=args.actor_weight_decay)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = args.temp_target_entropy_coef * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.full((1,), float(np.log(args.temp_initial_value)), requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    if args.debug:
        rb = SplitReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
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
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
    start_time = time.time()
    if args.debug:
        qf_grad_norm = actor_grad_norm = alpha_grad_norm = 0.0
        # Fixed reference batch for Hessian sharpness only: sampled once from
        # holdout, with the actor's frozen next-state action a' (and its
        # log-prob) at cache time, so later sharpness comparisons aren't
        # confounded by actor drift.
        ref_data = None
        ref_next_actions = None
        ref_next_log_pi = None

        # Debug-mode logging cadences (see CrossQ investigation metrics table)
        DEBUG_WEIGHT_NORM_FREQ = 1_000      # critic weight norms / effective scale
        DEBUG_HOLDOUT_BATCH_FREQ = 5_000   # effective rank, dead-ReLU, actor entropy, generalization gap (fresh holdout sample)
        DEBUG_LIPSCHITZ_FREQ = 5_000       # critic Lipschitz (fresh holdout sample)
        DEBUG_GRAD_VARIANCE_FREQ = 5_000   # gradient variance/SNR (train buffer)
        DEBUG_SHARPNESS_FREQ = 25_000      # critic Hessian sharpness (fixed reference batch, frozen a')

        # Simplicity bias score (SimBa, Appendix A/B) characterizes the
        # critic *architecture's* init distribution, not the trained network,
        # so it's logged once at the start rather than on a training cadence.
        log_simplicity_bias(
            writer,
            global_step=0,
            model_fn=lambda: make_critic(envs, args),
            obs_dim=int(np.array(envs.single_observation_space.shape).prod()),
            action_dim=int(np.prod(envs.single_action_space.shape)),
            device=device,
            name="critic",
        )

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
                if info is not None:
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
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            if args.debug:
                qf_grad_norm = compute_grad_norm(qf1, qf2)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.debug:
                        actor_grad_norm = compute_grad_norm(actor)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

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
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
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
                            holdout_next_state_actions, holdout_next_state_log_pi, _ = actor.get_action(
                                holdout_data.next_observations
                            )
                            holdout_qf1_next_target = qf1_target(holdout_data.next_observations, holdout_next_state_actions)
                            holdout_qf2_next_target = qf2_target(holdout_data.next_observations, holdout_next_state_actions)
                            holdout_min_qf_next_target = (
                                torch.min(holdout_qf1_next_target, holdout_qf2_next_target)
                                - alpha * holdout_next_state_log_pi
                            )
                            holdout_next_q_value = holdout_data.rewards.flatten() + (
                                1 - holdout_data.dones.flatten()
                            ) * args.gamma * holdout_min_qf_next_target.view(-1)

                            holdout_qf1_a_values = qf1(holdout_data.observations, holdout_data.actions).view(-1)
                            holdout_qf2_a_values = qf2(holdout_data.observations, holdout_data.actions).view(-1)
                            holdout_qf1_loss = F.mse_loss(holdout_qf1_a_values, holdout_next_q_value)
                            holdout_qf2_loss = F.mse_loss(holdout_qf2_a_values, holdout_next_q_value)

                            holdout_pi, holdout_log_pi, _ = actor.get_action(holdout_data.observations)
                            holdout_qf1_pi = qf1(holdout_data.observations, holdout_pi)
                            holdout_qf2_pi = qf2(holdout_data.observations, holdout_pi)
                            holdout_min_qf_pi = torch.min(holdout_qf1_pi, holdout_qf2_pi)
                            holdout_actor_loss = ((alpha * holdout_log_pi) - holdout_min_qf_pi).mean()

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
                        log_log_pi_stats(writer, global_step, holdout_log_pi, holdout_next_state_log_pi, prefix="holdout")
                        log_action_stats(writer, global_step, holdout_pi, prefix="holdout")
                        log_layer_activation_stats(
                            writer, global_step, qf1, qf2, holdout_data.observations, holdout_data.actions, prefix="holdout"
                        )

                    # Critic Lipschitz: fresh holdout sample, coarser cadence.
                    if split_rb.holdout_size() >= args.batch_size and global_step % DEBUG_LIPSCHITZ_FREQ == 0:
                        lipschitz_data = split_rb.sample_holdout(args.batch_size)
                        log_gradient_lipschitz(
                            writer, global_step, qf1, qf2, lipschitz_data.observations, lipschitz_data.actions, prefix="holdout"
                        )

                    # Gradient variance/SNR: fresh resamples from the train
                    # buffer, since this is specifically about training-loss
                    # gradient noise.
                    if global_step % DEBUG_GRAD_VARIANCE_FREQ == 0:
                        def make_sample_qf_loss_fn(critic):
                            def sample_qf_loss_fn():
                                sample = rb.sample(args.batch_size)
                                with torch.no_grad():
                                    sample_next_actions, sample_next_log_pi, _ = actor.get_action(sample.next_observations)
                                    sample_qf1_next_target = qf1_target(sample.next_observations, sample_next_actions)
                                    sample_qf2_next_target = qf2_target(sample.next_observations, sample_next_actions)
                                    sample_min_qf_next_target = (
                                        torch.min(sample_qf1_next_target, sample_qf2_next_target)
                                        - alpha * sample_next_log_pi
                                    )
                                    sample_next_q_value = sample.rewards.flatten() + (
                                        1 - sample.dones.flatten()
                                    ) * args.gamma * sample_min_qf_next_target.view(-1)
                                return F.mse_loss(critic(sample.observations, sample.actions).view(-1), sample_next_q_value)
                            return sample_qf_loss_fn

                        log_gradient_variance_stats(writer, global_step, qf1, make_sample_qf_loss_fn(qf1), name="qf1")
                        log_gradient_variance_stats(writer, global_step, qf2, make_sample_qf_loss_fn(qf2), name="qf2")

                    # Critic Hessian sharpness: fixed reference batch with the
                    # actor's frozen next-state action a' (cached once), but
                    # current target networks and alpha — isolates critic-only
                    # drift from actor drift.
                    if ref_data is None and split_rb.holdout_size() >= args.batch_size:
                        ref_data = split_rb.sample_holdout(args.batch_size)
                        with torch.no_grad():
                            ref_next_actions, ref_next_log_pi, _ = actor.get_action(ref_data.next_observations)

                    if ref_data is not None and global_step % DEBUG_SHARPNESS_FREQ == 0:
                        with torch.no_grad():
                            ref_qf1_next_target = qf1_target(ref_data.next_observations, ref_next_actions)
                            ref_qf2_next_target = qf2_target(ref_data.next_observations, ref_next_actions)
                            ref_min_qf_next_target = (
                                torch.min(ref_qf1_next_target, ref_qf2_next_target) - alpha * ref_next_log_pi
                            )
                            ref_next_q_value = ref_data.rewards.flatten() + (
                                1 - ref_data.dones.flatten()
                            ) * args.gamma * ref_min_qf_next_target.view(-1)

                        def ref_qf1_loss_fn():
                            return F.mse_loss(qf1(ref_data.observations, ref_data.actions).view(-1), ref_next_q_value)

                        def ref_qf2_loss_fn():
                            return F.mse_loss(qf2(ref_data.observations, ref_data.actions).view(-1), ref_next_q_value)

                        log_sharpness_stats(writer, global_step, qf1, ref_qf1_loss_fn, name="qf1", prefix="holdout")
                        log_sharpness_stats(writer, global_step, qf2, ref_qf2_loss_fn, name="qf2", prefix="holdout")

    envs.close()
    writer.close()
