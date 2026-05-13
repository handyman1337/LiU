"""
Neural Network Architectures for Model-Based RL.

TDDE78 — Lab 4: Model-Based Deep RL
Linköping University, Spring 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        #I chose 128 features for all layers here because Q-network defined above has the same amount
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.next_state_head = nn.Linear(128, state_dim) #logits over next states
        self.reward_head = nn.Linear(128, 1) #scalar reward
        self.done_head = nn.Linear(128, 1) #binary logit
        

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
        x = torch.cat([state, action_onehot], dim=1) #concat cols
        features = self.shared_trunk(x)
        
        #I tried doing Δs + s here but that broke training. I imagine it might be because
        #adding one-hot encoded states skews the logits outputed from the next_state_head.
        next_state = self.next_state_head(features)
        
        #Pass features through heads to get reward and done_logit
        reward = self.reward_head(features).squeeze(-1)
        done_logit = self.done_head(features).squeeze(-1)
        
        return next_state, reward, done_logit