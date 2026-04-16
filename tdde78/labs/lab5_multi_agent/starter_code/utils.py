"""
Utility functions for Multi-Agent RL.

Provided to students fully implemented — nothing to modify here.

TDDE78 — Lab 5: Multi-Agent Deep RL
Linköping University, Spring 2026
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# =============================================================================
#  GAE — Reused from Lab 2
# =============================================================================

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE) — Schulman et al., 2016.

    Reused from Lab 2 — see lab2 utils.py for full documentation.

    Args:
        rewards    (array): Rewards          [r_0, ..., r_{T-1}]
        values     (array): Value estimates  [V(s_0), ..., V(s_{T-1})]
        dones      (array): Episode-end flags [d_0, ..., d_{T-1}]
        last_value (float): Bootstrapped V(s_T) after the last step
        gamma      (float): Discount factor
        gae_lambda (float): GAE lambda parameter

    Returns:
        advantages (np.ndarray): Shape (T,)
        returns    (np.ndarray): Shape (T,) — advantages + values
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae   = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# =============================================================================
#  Plotting utilities
# =============================================================================

def smooth(data, window=10):
    """Moving-average smoothing."""
    data = np.array(data)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def _save_plot(fig, title, experiments_dir):
    """Save fig to experiments_dir/plots/ as PNG."""
    if experiments_dir is None:
        return
    try:
        plots_dir = os.path.join(experiments_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        filename = (
            title.lower()
            .replace(" ", "_").replace("/", "_")
            .replace("—", "").replace(":", "").strip("_") + ".png"
        )
        fig.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches="tight")
        print(f"Plot saved → {os.path.join(plots_dir, filename)}")
    except Exception as e:
        print(f"Could not save plot: {e}")


def plot_mappo_results(results, title="MAPPO Training", window=20, experiments_dir=None):
    """
    Plot MAPPO training curves: total team reward and training losses.

    Expected keys in results:
        'episode_rewards', 'policy_losses', 'value_losses'
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rewards = np.array(results['episode_rewards'])
    axes[0].plot(rewards, alpha=0.3, color='steelblue', label='Raw')
    if len(rewards) >= window:
        sm_r = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), sm_r,
                     color='steelblue', linewidth=2, label=f'{window}-ep MA')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Team Reward')
    axes[0].set_title(f'{title} — Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if results.get('policy_losses'):
        sm = smooth(results['policy_losses'], 50)
        axes[1].plot(sm, color='darkorange', linewidth=1.5, label='Policy Loss')
    if results.get('value_losses'):
        sm = smooth(results['value_losses'], 50)
        axes[1].plot(sm, color='red', linewidth=1.5, label='Value Loss')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{title} — Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


def plot_maddpg_results(results, title="MADDPG Training", window=20, experiments_dir=None):
    """
    Plot MADDPG training curves: total team reward and agent losses.

    Expected keys in results:
        'episode_rewards', 'actor_losses', 'critic_losses'
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rewards = np.array(results['episode_rewards'])
    axes[0].plot(rewards, alpha=0.3, color='seagreen', label='Raw')
    if len(rewards) >= window:
        sm_r = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), sm_r,
                     color='seagreen', linewidth=2, label=f'{window}-ep MA')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Team Reward')
    axes[0].set_title(f'{title} — Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if results.get('actor_losses'):
        sm = smooth(results['actor_losses'], 50)
        axes[1].plot(sm, color='darkorange', linewidth=1.5, label='Actor Loss (mean agents)')
    if results.get('critic_losses'):
        sm = smooth(results['critic_losses'], 50)
        axes[1].plot(sm, color='red', linewidth=1.5, label='Critic Loss (mean agents)')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{title} — Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


# =============================================================================
#  Replay Buffer — MADDPG off-policy training
# =============================================================================

class MultiAgentReplayBuffer:
    """
    Replay buffer for MADDPG: stores joint transitions.

    Each transition records observations, one-hot actions, rewards,
    next observations, and the done flag for ALL agents simultaneously,
    enabling off-policy centralized training.

    Args:
        capacity   (int): Maximum number of transitions stored.
        obs_dim    (int): Single-agent observation dimension.
        action_dim (int): Number of discrete actions.
        n_agents   (int): Number of agents.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, n_agents: int):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.pos        = 0
        self.size       = 0

        self.obs      = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, n_agents),             dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity,),                      dtype=np.float32)

    def push(self, obs_arr, actions_arr, rewards_arr, next_obs_arr, done):
        """
        Store one transition.

        Args:
            obs_arr      (array): (n_agents, obs_dim)
            actions_arr  (array): (n_agents, action_dim) — one-hot encoded
            rewards_arr  (array): (n_agents,)
            next_obs_arr (array): (n_agents, obs_dim)
            done         (bool/float): episode-end flag
        """
        self.obs[self.pos]      = obs_arr
        self.actions[self.pos]  = actions_arr
        self.rewards[self.pos]  = rewards_arr
        self.next_obs[self.pos] = next_obs_arr
        self.dones[self.pos]    = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.

        Returns CPU FloatTensors — move to device in the training loop:
            obs        (FloatTensor): (batch, n_agents, obs_dim)
            actions    (FloatTensor): (batch, n_agents, action_dim)
            rewards    (FloatTensor): (batch, n_agents)
            next_obs   (FloatTensor): (batch, n_agents, obs_dim)
            dones      (FloatTensor): (batch,)
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def __len__(self):
        return self.size


def plot_comparison(all_results, title="Comparison", window=20, experiments_dir=None):
    """
    Plot mean ± std reward across multiple methods / seeds on the same axes.

    Args:
        all_results (dict): {label: list_of_result_dicts}
        title       (str):  Plot title and saved filename.
        window      (int):  Moving-average window.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (name, results_list) in enumerate(all_results.items()):
        color   = colors[idx % len(colors)]
        min_eps = min(len(r['episode_rewards']) for r in results_list)
        mat     = np.array([r['episode_rewards'][:min_eps] for r in results_list])
        mean_r  = mat.mean(0)
        std_r   = mat.std(0)
        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        xs      = range(window - 1, min_eps)

        ax.plot(xs, sm_mean, color=color, label=name, linewidth=2)
        ax.fill_between(xs, sm_mean - sm_std, sm_mean + sm_std,
                        color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Team Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()
