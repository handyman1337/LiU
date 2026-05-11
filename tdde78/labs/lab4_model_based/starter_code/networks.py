"""
Neural Network Architectures for Model-Based RL.

TDDE78 — Lab 4: Model-Based Deep RL
Linköping University, Spring 2026
"""

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    DQN Q-network: maps state → Q-values for all discrete actions.

    Reused from Lab 1 — students do NOT re-implement this.

    Architecture: state_dim → 128 → 128 → action_dim  (ReLU activations)

    Args:
        state_dim  (int): Observation space dimension.
        action_dim (int): Number of discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class WorldModel(nn.Module):
    """
    Neural network dynamics model: (s, a) → (s', r, done).

    *** This is the main new component you implement in Lab 4. ***

    Predicts the next-state residual Δs = s' - s, scalar reward, and
    a terminal logit. Trained with supervised learning on real transitions.

    Input:  concat(state, one_hot(action))  —  state_dim + action_dim dims
    Output: (next_state_pred, reward_pred, done_logit)

    Args:
        state_dim  (int): Observation space dimension.
        action_dim (int): Number of discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        # TODO: Define a shared trunk (two hidden layers with ReLU) and three
        # output heads: one for next-state prediction, one for reward, and one
        # for the done signal (binary logit).
        raise NotImplementedError("Implement WorldModel.__init__()")

    def forward(self, state, action_onehot):
        """
        Args:
            state         (FloatTensor): (batch, state_dim)
            action_onehot (FloatTensor): (batch, action_dim)

        Returns:
            next_state (FloatTensor): (batch, state_dim)
            reward     (FloatTensor): (batch,)
            done_logit (FloatTensor): (batch,)  — sigmoid gives done probability
        """
        # TODO: Concatenate state and action_onehot, pass through the trunk,
        # then apply each head. Next-state is a residual prediction (Δs + s).
        # Return (next_state, reward, done_logit).
        raise NotImplementedError("Implement WorldModel.forward()")
