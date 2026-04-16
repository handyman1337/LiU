"""
Neural Network Architectures for Actor-Critic methods.

TDDE78 — Lab 3: Actor-Critic
Linköping University, Spring 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization (standard in A2C / CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
#  A2C Network — Continuous Control
# =============================================================================

class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces.

    Used by A2C on environments like LunarLanderContinuous-v3 and BipedalWalker-v3.

    The actor outputs the mean of a Gaussian policy. The log standard deviation
    is a separate learnable parameter (not state-dependent) — this is the
    standard approach in A2C / PPO for continuous control.

    Architecture:
        state_dim → 256 → Tanh → 256 → Tanh   [shared backbone]
                                                ├── → action_dim   [actor mean μ(s)]
                                                └── → 1            [critic V(s)]
        log_std: nn.Parameter of shape (action_dim,)  [learned log std, shared across states]

    Args:
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Dimension of the continuous action space.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(ContinuousActorCritic, self).__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Create:
        #   1. self.shared — a nn.Sequential with:
        #        layer_init(Linear(state_dim, 256)) → Tanh
        #        layer_init(Linear(256, 256))        → Tanh
        #
        #   2. self.actor_mean — Linear(256, action_dim), initialized with std=0.01
        #      (small std → near-zero initial actions)
        #
        #   3. self.critic — Linear(256, 1), initialized with std=1.0
        #
        #   4. self.log_std — nn.Parameter(torch.zeros(action_dim))
        #      (a learnable log std parameter, independent of state)
        # =====================================================================
        raise NotImplementedError("Define ContinuousActorCritic layers")

    def forward(self, state):
        """
        Forward pass: compute actor mean and critic value.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            mean:  Tensor of shape (batch_size, action_dim)
            value: Tensor of shape (batch_size, 1)
        """
        # =====================================================================
        # TODO: Pass state through self.shared, then through each head.
        #       Return (mean, value).
        # =====================================================================
        raise NotImplementedError("Implement ContinuousActorCritic.forward()")

    def get_action(self, state, action=None):
        """
        Sample an action from the Gaussian policy.

        Uses the reparameterization trick: a = mean + std * eps, eps ~ N(0, I)

        Args:
            state:  Tensor of shape (batch_size, state_dim)
            action: Optional FloatTensor of shape (batch_size, action_dim).
                    If provided, evaluates the log_prob of these actions
                    (used during the A2C update).

        Returns:
            action:   FloatTensor of shape (batch_size, action_dim)
            log_prob: Tensor of shape (batch_size,) — sum of log probs over action dims
            entropy:  Tensor of shape (batch_size,) — sum of entropies over action dims
            value:    Tensor of shape (batch_size, 1)
        """
        # =====================================================================
        # TODO: Implement Gaussian action sampling.
        #
        # Steps:
        # 1. Call self.forward(state) to get mean and value
        # 2. Get std by exponentiating self.log_std (clamped to valid range):
        #      std = self.log_std.exp().expand_as(mean)
        # 3. Create a Normal distribution: dist = Normal(mean, std)
        # 4. If action is None, sample: action = dist.sample()
        # 5. Compute log_prob:
        #      log_prob = dist.log_prob(action).sum(dim=-1)   # sum over action dims
        # 6. Compute entropy:
        #      entropy = dist.entropy().sum(dim=-1)           # sum over action dims
        # 7. Return action, log_prob, entropy, value
        # =====================================================================
        raise NotImplementedError("Implement ContinuousActorCritic.get_action()")


# =============================================================================
#  SAC Networks — Continuous Control
# =============================================================================

class SACActor(nn.Module):
    """
    SAC Actor: Gaussian policy with reparameterization trick and tanh squashing.

    Outputs a squashed Gaussian: a = tanh(μ + σ·ε), ε ~ N(0,I)

    The tanh squashing bounds actions to (-1, 1) and requires a change-of-variables
    correction to the log probability:
        log π(a|s) = log N(u; μ, σ) - Σ_i log(1 - tanh²(u_i) + ε)
    where u = arctanh(a) is the pre-squashing action.

    Architecture:
        state_dim → 256 → ReLU → 256 → ReLU
                                         ├── → action_dim  [mean μ(s)]
                                         └── → action_dim  [log std log σ(s), clamped to [-5, 2]]

    Args:
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Dimension of the continuous action space.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(SACActor, self).__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Create:
        #   1. self.net — a nn.Sequential:
        #        Linear(state_dim, 256) → ReLU
        #        Linear(256, 256)       → ReLU
        #
        #   2. self.mean_head    — Linear(256, action_dim)
        #   3. self.log_std_head — Linear(256, action_dim)
        # =====================================================================
        raise NotImplementedError("Define SACActor layers")

    def forward(self, state):
        """
        Compute mean and log_std of the Gaussian.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            mean:    Tensor of shape (batch_size, action_dim)
            log_std: Tensor of shape (batch_size, action_dim), clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        # =====================================================================
        # TODO: Pass state through self.net, then through each head.
        #       Clamp log_std to [LOG_STD_MIN, LOG_STD_MAX].
        # =====================================================================
        raise NotImplementedError("Implement SACActor.forward()")

    def get_action(self, state):
        """
        Sample a squashed Gaussian action with log probability.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            action:   FloatTensor of shape (batch_size, action_dim) — squashed to (-1, 1)
            log_prob: Tensor of shape (batch_size,) — log π(action|state) with tanh correction
        """
        # =====================================================================
        # TODO: Implement reparameterized sampling with tanh squashing.
        #
        # Steps:
        # 1. Call self.forward(state) to get mean and log_std
        # 2. Compute std = log_std.exp()
        # 3. Create a Normal distribution: dist = Normal(mean, std)
        # 4. Sample pre-squashing action using rsample() (reparameterized):
        #      x_t = dist.rsample()
        # 5. Squash: action = torch.tanh(x_t)
        # 6. Compute log_prob WITH tanh correction:
        #      log_prob = dist.log_prob(x_t).sum(dim=-1)
        #      log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)
        #    (This is the numerically stable form of: -log(1 - tanh²(x_t) + ε))
        # 7. Return action, log_prob
        # =====================================================================
        raise NotImplementedError("Implement SACActor.get_action()")


class SACCritic(nn.Module):
    """
    SAC Critic: Twin Q-networks Q(s, a).

    Uses two separate Q-networks to mitigate overestimation bias (clipped double-Q).

    Architecture (for each Q-network):
        [state_dim + action_dim] → 256 → ReLU → 256 → ReLU → 1

    Args:
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Dimension of the continuous action space.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(SACCritic, self).__init__()
        # =====================================================================
        # TODO: Define twin Q-networks.
        #
        # Create:
        #   1. self.q1 — first Q-network as nn.Sequential:
        #        Linear(state_dim + action_dim, 256) → ReLU
        #        Linear(256, 256)                    → ReLU
        #        Linear(256, 1)
        #
        #   2. self.q2 — second Q-network with the SAME architecture (independent)
        # =====================================================================
        raise NotImplementedError("Define SACCritic twin Q-networks")

    def forward(self, state, action):
        """
        Compute Q-values from both networks.

        Args:
            state:  Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)

        Returns:
            q1: Tensor of shape (batch_size, 1)
            q2: Tensor of shape (batch_size, 1)
        """
        # =====================================================================
        # TODO: Concatenate state and action, pass through each Q-network.
        #
        #   sa = torch.cat([state, action], dim=-1)
        #   q1 = self.q1(sa)
        #   q2 = self.q2(sa)
        #   return q1, q2
        # =====================================================================
        raise NotImplementedError("Implement SACCritic.forward()")
