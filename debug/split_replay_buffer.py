import numpy as np
from cleanrl_utils.buffers import ReplayBuffer


class SplitReplayBuffer:
    """
    Wraps two SB3 ReplayBuffers — one for training, one for holdout.
    Episode-level splitting: at each episode boundary, decide which buffer
    the next episode's transitions go into.
    
    Drop-in replacement for cleanrl's `rb` with one extra method: sample_holdout().
    """
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        handle_timeout_termination=False,
        holdout_fraction=0.1,
        holdout_seed=42,
        min_train_size=0,
    ):
        # Training buffer gets the bulk of the capacity
        train_size = int(buffer_size * (1 - holdout_fraction))
        holdout_size = buffer_size - train_size
        
        # Sub-buffers are always fed one env's transition at a time (see `add`),
        # so they're each sized for n_envs=1 regardless of the outer n_envs.
        self.train_buffer = ReplayBuffer(
            train_size, observation_space, action_space,
            device, n_envs=1,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.holdout_buffer = ReplayBuffer(
            holdout_size, observation_space, action_space,
            device, n_envs=1,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        self.holdout_fraction = holdout_fraction
        self.rng = np.random.RandomState(holdout_seed)
        self.n_envs = n_envs
        # Until the train buffer holds this many transitions, every episode is
        # routed to it (never holdout), so `sample()` always has data once
        # `learning_starts >= min_train_size` transitions have been collected.
        self.min_train_size = min_train_size
        
        # Per-environment holdout assignment for the current episode
        # Initialized to all training; updated when an episode ends
        self._is_holdout = np.zeros(n_envs, dtype=bool)
        # Track whether we need to assign on the next add (i.e., new episode starting)
        self._needs_assignment = np.ones(n_envs, dtype=bool)
    
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        """
        Adds a transition. Episode boundary is detected from `dones`:
        when done=True, the NEXT add() for that env will start a new episode
        and get a fresh holdout assignment.
        """
        # Assign holdout status for any envs that need it (start of episode)
        for env_idx in range(self.n_envs):
            if self._needs_assignment[env_idx]:
                if self.train_buffer.size() < self.min_train_size:
                    self._is_holdout[env_idx] = False
                else:
                    self._is_holdout[env_idx] = (self.rng.random() < self.holdout_fraction)
                self._needs_assignment[env_idx] = False
        
        # Route each env's transition to the appropriate buffer
        # SB3's add expects all envs at once, so we have to split by env
        for env_idx in range(self.n_envs):
            target_buffer = self.holdout_buffer if self._is_holdout[env_idx] else self.train_buffer
            
            # Extract this env's slice and add as if n_envs=1
            # (SB3's add is happy to take properly-shaped arrays)
            target_buffer.add(
                obs[env_idx:env_idx+1],
                next_obs[env_idx:env_idx+1],
                actions[env_idx:env_idx+1],
                rewards[env_idx:env_idx+1],
                dones[env_idx:env_idx+1],
                [infos[env_idx]] if isinstance(infos, list) else infos,
            )
            
            # If this transition ended an episode, flag for reassignment next time.
            # `dones` alone misses time-limit truncations (cleanrl passes `terminations`
            # only, so SAC's bootstrap isn't zeroed on truncation), so also check
            # gymnasium's `final_info`, which is set on termination OR truncation.
            episode_ended = bool(dones[env_idx])
            if isinstance(infos, dict) and infos.get("final_info") is not None:
                episode_ended = episode_ended or (infos["final_info"][env_idx] is not None)
            if episode_ended:
                self._needs_assignment[env_idx] = True
    
    def sample(self, batch_size):
        """Sample from the training buffer (drop-in replacement for rb.sample)."""
        return self.train_buffer.sample(batch_size)
    
    def sample_holdout(self, batch_size):
        """Sample from the holdout buffer."""
        return self.holdout_buffer.sample(batch_size)
    
    @property
    def size(self):
        return self.train_buffer.size() + self.holdout_buffer.size()
    
    def train_size(self):
        return self.train_buffer.size()
    
    def holdout_size(self):
        return self.holdout_buffer.size()