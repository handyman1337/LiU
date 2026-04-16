"""
Experience Replay Buffer for DQN.

TDDE78 — Lab 1: Value-Based Deep RL
Linköping University, Spring 2026
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Fixed-size replay buffer to store experience tuples.

    Each transition is stored as (state, action, reward, next_state, done).
    During training, random mini-batches are sampled to break temporal
    correlations between consecutive experiences.

    Args:
        capacity (int): Maximum number of transitions to store.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Whether the episode terminated.
        """
        # =====================================================================
        # TODO: Store the transition in the buffer.
        # Hint: Append a tuple (state, action, reward, next_state, done)
        # =====================================================================
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            Each array has batch_size as its first dimension.
        """
        # =====================================================================
        # TODO: Sample a random batch from the buffer and return as numpy arrays.
        #
        # Steps:
        # 1. Use random.sample() to get batch_size transitions
        # 2. Unzip the transitions into separate arrays
        # 3. Convert each to a numpy array with appropriate dtype:
        #    - states, next_states: np.float32
        #    - actions: np.int64
        #    - rewards: np.float32
        #    - dones: np.float32 (0.0 or 1.0)
        # =====================================================================
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """Return the current number of transitions stored."""
        return len(self.buffer)
