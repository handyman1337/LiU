"""
Utility functions and data structures for Lab 3: Actor-Critic Methods.

Provided to students fully implemented — nothing to modify here.
Students implement the agent classes and training loops in the notebook.

TDDE78 — Deep Reinforcement Learning
Linköping University, Spring 2026
"""

import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from collections import deque


# =============================================================================
#  GAE and Rollout Buffer — used by A2C
# =============================================================================

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE) — Schulman et al., 2016.

    GAE(gamma, lambda):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}

    Special cases:
        lambda = 0  →  A_t = delta_t          (one-step TD, low variance, high bias)
        lambda = 1  →  A_t = G_t - V(s_t)     (Monte Carlo, high variance, no bias)

    Args:
        rewards    (list/array): Rewards [r_0, ..., r_{T-1}]
        values     (list/array): Value estimates [V(s_0), ..., V(s_{T-1})]
        dones      (list/array): Episode-end flags [d_0, ..., d_{T-1}]
        last_value (float):      V(s_T) — bootstrapped value after the last step
        gamma      (float):      Discount factor
        gae_lambda (float):      GAE lambda parameter

    Returns:
        advantages (np.ndarray): Shape (T,), dtype float32
        returns    (np.ndarray): advantages + values — value-function targets, shape (T,)
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value        = last_value if t == T - 1 else values[t + 1]
        delta             = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae               = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t]     = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class RolloutBuffer:
    """
    Buffer for collecting on-policy rollout data for A2C.

    Stores a fixed number of environment steps collected under the current
    policy, then computes GAE advantages and yields mini-batches for the
    A2C update.

    Args:
        n_steps    (int): Number of environment steps per rollout.
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Dimension of the continuous action space.
    """

    def __init__(self, n_steps: int, state_dim: int, action_dim: int):
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.clear()

    def clear(self):
        """Reset the buffer to empty."""
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.dones      = []
        self.log_probs  = []
        self.values     = []
        self.advantages = None
        self.returns    = None

    def store(self, state, action, reward, done, log_prob, value):
        """
        Store a single transition.

        Args:
            state:    numpy array of shape (state_dim,)
            action:   numpy array of shape (action_dim,)
            reward:   float
            done:     bool — True if the episode ended on this step
            log_prob: float — log π_θ(a | s) at collection time
            value:    float — V(s) estimate at collection time
        """
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32).reshape(-1))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Compute GAE advantages and value-function targets.

        Must be called after filling the buffer and before get_batches().

        Args:
            last_value (float): Bootstrapped V(s_{T+1}) after the last step.
            gamma      (float): Discount factor.
            gae_lambda (float): GAE lambda.
        """
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_value, gamma, gae_lambda
        )

    def get_batches(self, batch_size):
        """
        Yield randomly shuffled mini-batches as PyTorch tensors.

        compute_returns_and_advantages() must be called first.

        Args:
            batch_size (int): Number of transitions per mini-batch.

        Yields:
            Tuple of tensors: (states, actions, old_log_probs, advantages, returns)
                states:        FloatTensor (batch_size, state_dim)
                actions:       FloatTensor (batch_size, action_dim)
                old_log_probs: FloatTensor (batch_size,)
                advantages:    FloatTensor (batch_size,)
                returns:       FloatTensor (batch_size,)
        """
        n       = len(self)
        indices = np.random.permutation(n)

        states_t     = torch.FloatTensor(np.array(self.states))
        actions_t    = torch.FloatTensor(np.array(self.actions, dtype=np.float32))
        log_probs_t  = torch.FloatTensor(np.array(self.log_probs, dtype=np.float32))
        advantages_t = torch.FloatTensor(self.advantages)
        returns_t    = torch.FloatTensor(self.returns)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                states_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                advantages_t[idx],
                returns_t[idx],
            )

    def __len__(self):
        return len(self.states)


# =============================================================================
#  Replay Buffer — used by SAC
# =============================================================================

class ReplayBuffer:
    """
    Fixed-size replay buffer for off-policy SAC.

    Stores transitions (s, a, r, s', done) and samples random mini-batches
    for the critic and actor updates.

    Args:
        capacity   (int): Maximum number of transitions to store.
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Dimension of the continuous action space.
        seed       (int): Random seed for reproducibility.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 42):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.buffer     = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state:      numpy array of shape (state_dim,)
            action:     numpy array of shape (action_dim,)
            reward:     float
            next_state: numpy array of shape (state_dim,)
            done:       bool or float
        """
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of FloatTensors: (states, actions, rewards, next_states, dones)
                rewards and dones are shaped (batch_size, 1) for SAC target computation.
        """
        batch                                         = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones  = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
#  Plotting utilities
# =============================================================================

def _save_plot(fig, title, experiments_dir):
    """Save *fig* to experiments_dir/plots/ as a PNG file."""
    if experiments_dir is None:
        return
    try:
        plots_dir = os.path.join(experiments_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        filename = (
            title.lower()
            .replace(" ", "_").replace("/", "_")
            .replace("—", "").replace(":", "").replace("≥", "")
            .replace("λ", "lambda").strip("_") + ".png"
        )
        filepath = os.path.join(plots_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {filepath}")
    except Exception as e:
        print(f"Could not save plot: {e}")


def smooth(data, window=10):
    """Moving-average smoothing."""
    data = np.array(data)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_a2c_results(results, title="A2C Training", window=20, experiments_dir=None):
    """
    Plot A2C training curves: episode rewards (vs environment steps),
    policy loss, and value loss.

    Expected keys in *results*:
        'episode_rewards', 'episode_timesteps', 'policy_losses', 'value_losses'

    Args:
        results         : dict returned by train_a2c().
        title           : Plot title (also used as filename when saving).
        window          : Moving-average window for episode rewards.
        experiments_dir : Root experiments directory; None = don't save to disk.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rewards = results['episode_rewards']
    steps   = results['episode_timesteps']

    axes[0].plot(steps, rewards, alpha=0.3, color='steelblue', label='Raw')
    if len(rewards) >= window:
        sm_r = smooth(rewards, window)
        sm_s = steps[window - 1:]
        axes[0].plot(sm_s, sm_r, color='steelblue', linewidth=2,
                     label=f'{window}-ep MA')
    axes[0].set_xlabel('Environment Steps'); axes[0].set_ylabel('Reward')
    axes[0].set_title(f'{title} — Episode Rewards')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['policy_losses'], alpha=0.5, color='red')
    axes[1].set_xlabel('Update'); axes[1].set_ylabel('Policy Loss')
    axes[1].set_title(f'{title} — Policy Loss'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(results['value_losses'], alpha=0.5, color='green')
    axes[2].set_xlabel('Update'); axes[2].set_ylabel('Value Loss')
    axes[2].set_title(f'{title} — Value Loss'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


def plot_sac_results(results, title="SAC Training", experiments_dir=None):
    """
    Plot SAC training curves: episode rewards, critic loss, actor loss,
    and entropy temperature α.

    Expected keys in *results*:
        'episode_rewards', 'episode_timesteps',
        'critic_losses', 'actor_losses', 'alpha_history'

    Args:
        results         : dict returned by train_sac().
        title           : Plot title (also used as filename when saving).
        experiments_dir : Root experiments directory; None = don't save to disk.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    rewards = results['episode_rewards']
    steps   = results['episode_timesteps']

    axes[0].plot(steps, rewards, alpha=0.3, color='steelblue', label='Raw')
    sm_r = smooth(rewards, 10)
    if len(sm_r) > 0 and len(steps) >= 10:
        axes[0].plot(steps[9:], sm_r, color='steelblue', linewidth=2, label='10-ep MA')
    axes[0].set_xlabel('Environment Steps'); axes[0].set_ylabel('Episode Reward')
    axes[0].set_title(f'{title} — Episode Rewards')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if results.get('critic_losses'):
        axes[1].plot(results['critic_losses'], alpha=0.5, color='red')
    axes[1].set_xlabel('Gradient Step'); axes[1].set_ylabel('Critic Loss')
    axes[1].set_title(f'{title} — Critic Loss'); axes[1].grid(True, alpha=0.3)

    if results.get('actor_losses'):
        axes[2].plot(results['actor_losses'], alpha=0.5, color='purple')
    axes[2].set_xlabel('Gradient Step'); axes[2].set_ylabel('Actor Loss')
    axes[2].set_title(f'{title} — Actor Loss'); axes[2].grid(True, alpha=0.3)

    if results.get('alpha_history'):
        axes[3].plot(results['alpha_history'], alpha=0.7, color='orange')
    axes[3].set_xlabel('Gradient Step'); axes[3].set_ylabel('α (temperature)')
    axes[3].set_title(f'{title} — Entropy Temperature α'); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


def plot_comparison(all_results, title="Method Comparison", window=20,
                    x_key='episode_timesteps', experiments_dir=None):
    """
    Plot mean ± std reward across multiple methods/seeds on the same axes.

    Args:
        all_results     (dict): {label: list_of_result_dicts}
        title           (str):  Plot title and saved filename.
        window          (int):  Moving-average window.
        x_key           (str):  Key used for x-axis ('episode_timesteps').
        experiments_dir :       Root experiments directory; None = don't save.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (name, results_list) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]

        min_eps        = min(len(r['episode_rewards']) for r in results_list)
        rewards_matrix = np.array([r['episode_rewards'][:min_eps] for r in results_list])
        steps          = results_list[0][x_key][:min_eps]

        mean_r = rewards_matrix.mean(axis=0)
        std_r  = rewards_matrix.std(axis=0)

        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        sm_x    = steps[window - 1:]

        ax.plot(sm_x, sm_mean, color=color, label=name, linewidth=2)
        ax.fill_between(sm_x, sm_mean - sm_std, sm_mean + sm_std,
                        color=color, alpha=0.15)

    ax.set_xlabel('Environment Steps'); ax.set_ylabel('Episode Reward')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


# =============================================================================
#  Video recording
# =============================================================================

def record_agent_video(
    agent,
    env_name: str = "LunarLanderContinuous-v3",
    agent_type: str = "a2c",
    num_episodes: int = 3,
    seed: int = 0,
    name_prefix: str = "actor_critic",
    experiments_dir: str = None,
    video_dir: str = None,
):
    """
    Record video(s) of a trained agent using its deterministic policy.

    For A2C the deterministic action is tanh(mean) of the Gaussian.
    For SAC it calls agent.select_action(..., deterministic=True).

    Videos are saved to *video_dir* (default: experiments_dir/videos/).
    Returns the path of the last recorded MP4, or None if no file was found.

    Args:
        agent           : A2CAgent or SACAgent instance.
        env_name        : Gymnasium environment id.
        agent_type      : "a2c" or "sac".
        num_episodes    : Number of episodes to record.
        seed            : Base seed (seed+ep used per episode).
        name_prefix     : Filename prefix for the saved MP4 files.
        experiments_dir : Root experiments directory (used when video_dir is None).
        video_dir       : Explicit path for videos; overrides experiments_dir/videos/.
    """
    import gymnasium as gym

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if video_dir is None:
        base = experiments_dir if experiments_dir else "experiments"
        video_dir = os.path.join(base, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder    = video_dir,
        episode_trigger = lambda ep: True,
        name_prefix     = name_prefix,
    )

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0

        for _ in range(10_000):
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(_device)
                if agent_type == "a2c":
                    mean, _ = agent.network(state_t)
                    action  = torch.tanh(mean).cpu().numpy().flatten()
                else:
                    action = agent.select_action(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        print(f"Episode {ep + 1}: reward = {episode_reward:.1f}")

    env.close()

    video_files = sorted(
        [f for f in os.listdir(video_dir)
         if f.startswith(name_prefix) and f.endswith('.mp4')]
    )
    if video_files:
        video_path = os.path.join(video_dir, video_files[-1])
        print(f"Video saved → {video_path}")
        return video_path
    return None
