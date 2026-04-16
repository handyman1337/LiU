"""
Neural Network Architectures for Multi-Agent RL.

TDDE78 — Lab 5: Multi-Agent Deep RL
Linköping University, Spring 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization (standard in PPO / CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
#  Shared Actor — Reused from Lab 2  (MAPPO)
# =============================================================================

class DiscreteActor(nn.Module):
    """
    Discrete-action policy network.

    Reused from Lab 2 — students do NOT re-implement this.

    Used as the shared actor across ALL agents (parameter sharing) in MAPPO.
    Each agent uses the same weights to select its local action from its
    own observation, enabling efficient multi-agent learning.

    Architecture: obs_dim → 64 → 64 → action_dim  (Tanh activations)

    Args:
        obs_dim    (int): Single-agent observation dimension.
        action_dim (int): Number of discrete actions.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.policy_head = layer_init(nn.Linear(64, action_dim), std=0.01)

    def forward(self, obs):
        return self.policy_head(self.net(obs))

    def get_action(self, obs, action=None):
        """
        Sample from (or evaluate) the Categorical policy.

        Args:
            obs:    FloatTensor (batch, obs_dim)
            action: Optional LongTensor (batch,) — if given, evaluates these actions

        Returns:
            action:   LongTensor (batch,)
            log_prob: FloatTensor (batch,)
            entropy:  FloatTensor (batch,)
        """
        logits = self.forward(obs)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


# =============================================================================
#  Centralized Critic — YOUR IMPLEMENTATION  (MAPPO)
# =============================================================================

class CentralizedCritic(nn.Module):
    """
    Centralized value function: V(o_1, o_2, ..., o_N).

    *** This is the main new component you implement in Lab 5. ***

    Takes the joint observation (all agents' observations concatenated)
    and outputs a scalar value estimate. Used during centralized training
    under the CTDE (Centralized Training, Decentralized Execution) paradigm.

    Architecture: (n_agents * obs_dim) → 128 → 128 → 1  (Tanh activations)

    Args:
        obs_dim  (int): Single-agent observation dimension.
        n_agents (int): Number of agents.
    """

    def __init__(self, obs_dim: int, n_agents: int):
        super().__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Suggested architecture:
        #   joint_dim = obs_dim * n_agents
        #   self.net = nn.Sequential(
        #       layer_init(nn.Linear(joint_dim, 128)), nn.Tanh(),
        #       layer_init(nn.Linear(128, 128)),       nn.Tanh(),
        #       layer_init(nn.Linear(128, 1), std=1.0),
        #   )
        # =====================================================================
        raise NotImplementedError("Implement CentralizedCritic.__init__()")

    def forward(self, joint_obs):
        """
        Args:
            joint_obs (FloatTensor): (batch, n_agents * obs_dim)

        Returns:
            value (FloatTensor): (batch, 1)
        """
        # =====================================================================
        # TODO: Implement the forward pass.
        #   return self.net(joint_obs)
        # =====================================================================
        raise NotImplementedError("Implement CentralizedCritic.forward()")


# =============================================================================
#  MADDPG Networks — Provided (do NOT modify)
# =============================================================================

class MADDPGActor(nn.Module):
    """
    Per-agent actor network for MADDPG (discrete actions).

    Each agent has its OWN independent actor weights (no parameter sharing,
    unlike MAPPO). Outputs action probabilities via softmax and supports
    Gumbel-Softmax for differentiable action sampling during training.

    Architecture: obs_dim → 64 → 64 → action_dim  (ReLU activations)

    Provided — students do NOT re-implement this.

    Args:
        obs_dim    (int): Single-agent observation dimension.
        action_dim (int): Number of discrete actions.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def forward(self, obs):
        """Returns raw logits."""
        return self.net(obs)

    def get_probs(self, obs):
        """Returns action probabilities via softmax."""
        return F.softmax(self.forward(obs), dim=-1)

    def get_action(self, obs, explore: bool = True):
        """
        Select action for execution (decentralized).

        Args:
            obs     (FloatTensor): (batch, obs_dim)
            explore (bool): sample if True, argmax if False

        Returns:
            action    (LongTensor):  (batch,) integer action
            action_oh (FloatTensor): (batch, action_dim) one-hot encoding
        """
        probs = self.get_probs(obs)
        if explore:
            dist   = Categorical(probs=probs)
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)
        action_oh = F.one_hot(action, num_classes=probs.shape[-1]).float()
        return action, action_oh

    def gumbel_softmax_action(self, obs, tau: float = 1.0, hard: bool = True):
        """
        Differentiable action sampling via Gumbel-Softmax trick.
        Used during policy gradient updates so gradients flow through
        the discrete action selection.

        Args:
            obs  (FloatTensor): (batch, obs_dim)
            tau  (float): temperature (lower → more peaked)
            hard (bool): straight-through estimator if True

        Returns:
            action_gs (FloatTensor): (batch, action_dim) — approx. one-hot
        """
        logits = self.forward(obs)
        return F.gumbel_softmax(logits, tau=tau, hard=hard)


class MADDPGCritic(nn.Module):
    """
    Centralized Q-function for MADDPG.

    Agent i's critic sees the joint observations AND joint actions of ALL
    agents, outputting a scalar Q-value specific to agent i.

    This is the CTDE component of MADDPG: during training, each agent's
    critic has full global access. During execution, only the actor is used.

    Input: [o_1, ..., o_N, a_1, ..., a_N]
    Architecture: n_agents*(obs_dim+action_dim) → 128 → 128 → 1  (ReLU)

    Provided — students do NOT re-implement this.

    Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed
    Cooperative-Competitive Environments", NeurIPS 2017.

    Args:
        obs_dim    (int): Single-agent observation dimension.
        action_dim (int): Number of discrete actions.
        n_agents   (int): Number of agents.
    """

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int):
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, joint_obs, joint_actions):
        """
        Args:
            joint_obs     (FloatTensor): (batch, n_agents * obs_dim)
            joint_actions (FloatTensor): (batch, n_agents * action_dim) — one-hot or probs

        Returns:
            Q-value (FloatTensor): (batch, 1)
        """
        x = torch.cat([joint_obs, joint_actions], dim=-1)
        return self.net(x)
