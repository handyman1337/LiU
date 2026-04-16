"""
Neural Network Architectures for Policy Gradient methods.

TDDE78 — Lab 2: Policy Gradient
Linköping University, Spring 2026
"""

import numpy as np
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization (standard in PPO / CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActorCritic(nn.Module):
    """
    Actor-Critic network for discrete action spaces.

    Shared backbone with separate policy (actor) and value (critic) heads.
    Used for environments like CartPole-v1.

    Architecture:
        state_dim → 64 → 64  [shared, Tanh activations]
                            ├── → action_dim  [policy head: logits for Categorical]
                            └── → 1           [value head: V(s)]

    Args:
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Number of possible discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DiscreteActorCritic, self).__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Create:
        #   1. self.shared — a nn.Sequential with:
        #        layer_init(Linear(state_dim, 64)) → Tanh
        #        layer_init(Linear(64, 64))         → Tanh
        #   2. self.policy_head — Linear(64, action_dim), initialized with std=0.01
        #      (small std → nearly uniform initial policy)
        #   3. self.value_head  — Linear(64, 1),          initialized with std=1.0
        # =====================================================================
        raise NotImplementedError("Define DiscreteActorCritic layers")

    def forward(self, state):
        """
        Forward pass through shared layers and both heads.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            logits: Tensor of shape (batch_size, action_dim)
            value:  Tensor of shape (batch_size, 1)
        """
        # =====================================================================
        # TODO: Pass state through self.shared, then through each head.
        # =====================================================================
        raise NotImplementedError("Implement DiscreteActorCritic.forward()")

    def get_action(self, state, action=None):
        """
        Sample an action from the Categorical policy.

        Args:
            state:  Tensor of shape (batch_size, state_dim)
            action: Optional LongTensor of shape (batch_size,).
                    If provided, computes log_prob for these actions
                    (used during PPO update epochs).

        Returns:
            action:   LongTensor of shape (batch_size,)
            log_prob: Tensor of shape (batch_size,)
            entropy:  Tensor of shape (batch_size,) — policy entropy
            value:    Tensor of shape (batch_size, 1)
        """
        # =====================================================================
        # TODO: Implement action sampling with a Categorical distribution.
        #
        # Steps:
        # 1. Call self.forward(state) to get logits and value
        # 2. Create a Categorical distribution from logits:
        #      dist = Categorical(logits=logits)
        # 3. If action is None, sample:  action = dist.sample()
        # 4. Compute: log_prob = dist.log_prob(action)
        #             entropy  = dist.entropy()
        # 5. Return action, log_prob, entropy, value
        # =====================================================================
        raise NotImplementedError("Implement DiscreteActorCritic.get_action()")


