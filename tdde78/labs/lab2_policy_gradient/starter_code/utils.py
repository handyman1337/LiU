"""
Utility functions and data structures for Policy Gradient methods — SOLUTION.

TDDE78 — Lab 2: Policy Gradient
Linköping University, Spring 2026
"""

import numpy as np
import torch


def compute_returns(rewards, gamma):
    """
    Compute discounted returns for a single episode (Monte Carlo).

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards (list or np.ndarray): Sequence of rewards [r_0, r_1, ..., r_{T-1}]
        gamma   (float):              Discount factor in [0, 1]

    Returns:
        np.ndarray: Discounted returns of shape (T,), dtype float32
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


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
        rewards    (list/array): Rewards          [r_0, ..., r_{T-1}]
        values     (list/array): Value estimates  [V(s_0), ..., V(s_{T-1})]
        dones      (list/array): Episode-end flags [d_0, ..., d_{T-1}]
        last_value (float):      V(s_T) — bootstrapped value after the last step
        gamma      (float):      Discount factor
        gae_lambda (float):      GAE lambda parameter

    Returns:
        advantages (np.ndarray): Shape (T,), dtype float32
        returns    (np.ndarray): advantages + values, used as value targets, shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = last_value if t == T - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class RolloutBuffer:
    """
    Buffer for collecting on-policy rollout data for PPO.

    Stores a fixed number of environment steps collected under the current
    policy, then computes GAE advantages and yields mini-batches for
    multiple epochs of PPO updates.

    Args:
        n_steps       (int):  Number of environment steps per rollout.
        state_dim     (int):  Dimension of the state/observation space.
        action_dim    (int):  Dimension of the action space.
        is_continuous (bool): True for continuous actions (float arrays),
                              False for discrete actions (integers).
    """

    def __init__(self, n_steps: int, state_dim: int, action_dim: int, is_continuous: bool = False):
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.clear()

    def clear(self):
        """Reset the buffer to empty."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def store(self, state, action, reward, done, log_prob, value):
        """
        Store a single transition.

        Args:
            state:    numpy array of shape (state_dim,)
            action:   int (discrete) or numpy array of shape (action_dim,) (continuous)
            reward:   float
            done:     bool — True if the episode ended on this step
            log_prob: float — log π_θ(a | s) at collection time
            value:    float — V(s) estimate at collection time
        """
        self.states.append(np.array(state, dtype=np.float32))
        if self.is_continuous:
            self.actions.append(np.array(action, dtype=np.float32).reshape(-1))
        else:
            self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Compute GAE advantages and value-function targets.

        Must be called after collect_rollout() and before get_batches().

        Args:
            last_value (float): Bootstrapped V(s_{T+1}) after the last collected step.
            gamma      (float): Discount factor.
            gae_lambda (float): GAE lambda.
        """
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_value, gamma, gae_lambda
        )

    def get_batches(self, batch_size):
        """
        Yield randomly shuffled mini-batches of transitions as PyTorch tensors.

        compute_returns_and_advantages() must be called first.

        Args:
            batch_size (int): Number of transitions per mini-batch.

        Yields:
            Tuple of tensors: (states, actions, old_log_probs, advantages, returns, old_values)
                states:        FloatTensor (batch_size, state_dim)
                actions:       LongTensor or FloatTensor (batch_size,) or (batch_size, action_dim)
                old_log_probs: FloatTensor (batch_size,)
                advantages:    FloatTensor (batch_size,)
                returns:       FloatTensor (batch_size,)
                old_values:    FloatTensor (batch_size,)  — V(s) at collection time, for value clipping
        """
        n = len(self)
        indices = np.random.permutation(n)

        states_t = torch.FloatTensor(np.array(self.states))
        if self.is_continuous:
            actions_t = torch.FloatTensor(np.array(self.actions, dtype=np.float32))
        else:
            actions_t = torch.LongTensor(np.array(self.actions, dtype=np.int64))
        log_probs_t = torch.FloatTensor(np.array(self.log_probs, dtype=np.float32))
        advantages_t = torch.FloatTensor(self.advantages)
        returns_t = torch.FloatTensor(self.returns)
        old_values_t = torch.FloatTensor(np.array(self.values, dtype=np.float32))

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                states_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                advantages_t[idx],
                returns_t[idx],
                old_values_t[idx],
            )

    def __len__(self):
        return len(self.states)
