# Lab 1 — Value-Based Deep Reinforcement Learning

**TDDE78 — Deep Reinforcement Learning, Linköping University — Spring 2026**

---

## Overview

In this lab you implement **Deep Q-Networks (DQN)** and two of its key extensions: **Double DQN** and **Dueling DQN**. These algorithms approximate the optimal action-value function Q*(s, a) with a neural network and learn from raw observations using gradient descent.

---

## Theory

### Deep Q-Network (DQN)

The goal is to learn Q*(s, a) — the expected cumulative reward when taking action a in state s and following the optimal policy. DQN parameterizes this as a neural network Q(s, a; θ) and minimizes the **TD loss**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[\left(r + \gamma \max_{a'} Q(s', a';\, \theta^-) - Q(s, a;\, \theta)\right)^2\right]
$$

where **θ⁻** are the parameters of a periodically-copied **target network** and **D** is a **replay buffer** of past transitions.

Two key stabilization tricks make DQN work in practice:
- **Replay buffer:** random mini-batch sampling breaks temporal correlations between consecutive transitions
- **Target network:** a slowly-updated copy of Q prevents the training target from shifting at every step

![DQN algorithm](https://lilianweng.github.io/posts/2018-02-19-rl-overview/DQN_algorithm.png)

*Source: Lilian Weng, "A (Long) Peek into Reinforcement Learning"*

---

### Double DQN

The max operator in the DQN target uses the same network to both *select* and *evaluate* the best action — causing systematic **overestimation bias**. Double DQN fixes this by decoupling selection (online network θ) from evaluation (target network θ⁻):

$$
y^{\text{DDQN}} = r + \gamma\, Q\!\left(s',\, \underbrace{\arg\max_{a'} Q(s', a';\, \theta)}_{\text{select with } \theta},\; \theta^-\right)
$$

This is a two-line change from standard DQN — same architecture, different target computation.

---

### Dueling DQN

Dueling DQN decomposes Q into a **value stream** V(s) and an **advantage stream** A(s, a):

$$
Q(s, a;\, \theta) = V(s;\, \theta) + A(s, a;\, \theta) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\, \theta)
$$

The mean-centering makes the decomposition unique. The value stream learns *how good is this state*, while the advantage stream learns *which action is relatively better*.

![Dueling DQN architecture](https://lilianweng.github.io/posts/2018-05-05-drl-implementation/dueling-q-network.png)
*Source: Lilian Weng, "Implementing Deep Reinforcement Learning Models"*

---

## Environment

| Environment | Observation | Actions | Solved At |
|-------------|------------|---------|-----------|
| `CartPole-v1` | 4-dim vector (position, velocity, angle, angular velocity) | 2 discrete | Avg reward ≥ 475 over 100 episodes |
| `LunarLander-v3` | 8-dim vector (position, velocity, angle, leg contact) | 4 discrete | Avg reward ≥ 200 over 100 episodes |

---

## Tasks

### Part A — Implementation
- Implement the **replay buffer**: store and sample `(s, a, r, s', done)` transitions
- Implement **DQN**: Q-network, ε-greedy action selection, target network, TD loss
- Extend to **Double DQN**: modify target to decouple selection and evaluation
- Extend to **Dueling DQN**: add separate V and A heads with mean-centering

### Part B — Experiments
- Train **DQN on CartPole-v1** — verify the implementation produces a learning curve
- Compare **DQN vs Double DQN vs Dueling DQN** on CartPole-v1 (3 seeds, mean ± std)
- Scale to **LunarLander-v3** — compare DQN variants on a harder environment
- Ablation: **replay buffer size** ∈ {1 000, 10 000, 100 000}
- Ablation: **target network update frequency** ∈ {1, 10, 100, 1 000} steps

---

## Files

```
lab1_value_based/
├── README.md
├── starter_code/
│   ├── lab1_dqn.ipynb       # Starter notebook — fill in the TODOs
│   ├── networks.py          # DQN and DuelingDQN architectures
│   └── replay_buffer.py     # Experience replay buffer
└── experiments/
```

---

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. **Nature**, 518, 529–533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. **AAAI**.
3. Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. **ICML**.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
