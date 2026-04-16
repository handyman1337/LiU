"""
Utility functions and data structures for Model-Based RL.

Provided to students fully implemented — nothing to modify here.

TDDE78 — Lab 4: Model-Based Deep RL
Linköping University, Spring 2026
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# =============================================================================
#  Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Fixed-size circular replay buffer for DQN / Dyna.

    Stores transitions (s, a, r, s', done) as pre-allocated numpy arrays
    for fast batch sampling.

    Args:
        capacity  (int): Maximum number of transitions to store.
        state_dim (int): Dimension of the observation space.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity    = capacity
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int64)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)
        self.ptr         = 0
        self.size        = 0

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        idx                    = self.ptr
        self.states[idx]       = state
        self.actions[idx]      = int(action)
        self.rewards[idx]      = float(reward)
        self.next_states[idx]  = next_state
        self.dones[idx]        = float(done)
        self.ptr               = (self.ptr + 1) % self.capacity
        self.size              = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.

        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# =============================================================================
#  Plotting utilities
# =============================================================================

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


def smooth(data, window=10):
    """Moving-average smoothing."""
    data = np.array(data)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_dyna_results(results, title="Dyna Training", window=20, experiments_dir=None):
    """
    Plot Dyna training curves: episode rewards, world model loss, Q-loss.

    Expected keys in results:
        'episode_rewards', 'episode_timesteps', 'model_losses', 'q_losses'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rewards = results['episode_rewards']
    steps   = results['episode_timesteps']

    axes[0].plot(steps, rewards, alpha=0.3, color='steelblue', label='Raw')
    if len(rewards) >= window:
        sm_r = smooth(rewards, window)
        sm_s = steps[window - 1:]
        axes[0].plot(sm_s, sm_r, color='steelblue', linewidth=2, label=f'{window}-ep MA')
    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title(f'{title} — Episode Rewards')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if results.get('model_losses'):
        sm = smooth(results['model_losses'], 100)
        axes[1].plot(sm, color='darkorange', linewidth=1.5)
    axes[1].set_xlabel('Gradient Step')
    axes[1].set_ylabel('World Model Loss')
    axes[1].set_title(f'{title} — World Model Loss')
    axes[1].grid(True, alpha=0.3)

    if results.get('q_losses'):
        sm = smooth(results['q_losses'], 100)
        axes[2].plot(sm, color='red', linewidth=1.5)
    axes[2].set_xlabel('Gradient Step')
    axes[2].set_ylabel('Q-Network Loss')
    axes[2].set_title(f'{title} — Q-Network Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()


def plot_comparison(all_results, title="Comparison", window=20,
                    x_key='episode_timesteps', ylim=None, experiments_dir=None):
    """
    Plot mean ± std reward across multiple methods / seeds on the same axes.

    Args:
        all_results (dict): {label: list_of_result_dicts}
        title       (str):  Plot title and saved filename.
        window      (int):  Moving-average window.
        x_key       (str):  Key for x-axis values ('episode_timesteps').
        ylim        (tuple): Optional (ymin, ymax) to fix the y-axis range.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (name, results_list) in enumerate(all_results.items()):
        color    = colors[idx % len(colors)]
        min_eps  = min(len(r['episode_rewards']) for r in results_list)
        mat      = np.array([r['episode_rewards'][:min_eps] for r in results_list])
        steps    = results_list[0][x_key][:min_eps]
        mean_r   = mat.mean(0)
        std_r    = mat.std(0)
        sm_mean  = smooth(mean_r, window)
        sm_std   = smooth(std_r,  window)
        sm_x     = steps[len(steps) - len(sm_mean):]

        ax.plot(sm_x, sm_mean, color=color, label=name, linewidth=2)
        ax.fill_between(sm_x, sm_mean - sm_std, sm_mean + sm_std,
                        color=color, alpha=0.15)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, title, experiments_dir)
    plt.show()
