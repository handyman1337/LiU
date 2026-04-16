# Lab 5 — Multi-Agent Deep Reinforcement Learning

**TDDE78 — Deep Reinforcement Learning, Linköping University — Spring 2026**

---

## Overview

In this lab you implement two **multi-agent** algorithms: **MAPPO** and **MADDPG**. Both follow the **CTDE principle** — Centralized Training, Decentralized Execution:

- **Decentralized execution:** each agent acts from its own local observation $o_i$ only
- **Centralized training:** critics access all agents' observations (and actions) during training

This allows critics to provide better credit assignment while keeping the policy fully decentralized at deployment time.

---

## Theory

### MAPPO — Multi-Agent PPO

MAPPO extends PPO to multi-agent settings using **parameter sharing** (one actor shared by all agents) and a **centralized value critic** that sees the joint observation.

![MAPPO architecture](https://marllib.readthedocs.io/en/latest/_images/mappo.png)

*Source: MARLlib documentation*

**Shared actor:** one set of weights, each agent runs it on its own observation $o_i$. **Centralized critic:** takes the concatenation of all agents' observations $[o_1, \dots, o_N]$ and outputs a scalar value $V(o_1,\dots,o_N)$.

**Objective:** same clipped PPO loss as Lab 2:
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\cdot\hat{A}_t,\;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\cdot\hat{A}_t\right)\right]
$$
$$
\mathcal{L}(\theta) = -\mathcal{L}^{\text{CLIP}} + c_V\cdot\mathcal{L}_V - c_H\cdot\mathcal{H}[\pi_\theta]
$$

Advantages are computed per-agent with GAE using the centralized value as bootstrap. Rewards are normalized **globally** across all agents and timesteps before GAE — needed because all agents share the same reward in simple_spread, making per-agent std ≈ 0.

---

### MADDPG — Multi-Agent DDPG

MADDPG extends DDPG to multi-agent settings. Each agent has its **own independent actor** and its own **centralized Q-critic** that sees joint observations and joint actions.

![MADDPG architecture](https://marllib.readthedocs.io/en/latest/_images/maddpg.png)

*Source: MARLlib documentation — Lowe et al., NeurIPS 2017*

**Critic loss** for agent $i$ (Bellman TD error with target networks):
$$
y_i = r_i + \gamma\, Q_i^{\text{target}}(o', a'_1,\dots,a'_N)\cdot(1-d), \quad a'_j \sim \pi_j^{\text{target}}(o'_j)
$$
$$
\mathcal{L}_{Q_i} = \mathbb{E}\!\left[(Q_i(o, a_1,\dots,a_N) - y_i)^2\right]
$$

**Actor loss** for agent $i$ — maximize $Q_i$ via **Gumbel-Softmax** (differentiable discrete actions):
$$
\mathcal{L}_{\pi_i} = -\mathbb{E}\!\left[Q_i\!\left(o,\; \tilde{a}_i^{\text{GS}},\; a_{j \neq i}\right)\right]
$$

Only agent $i$'s action $\tilde{a}_i^{\text{GS}}$ is Gumbel-Softmax (gradient flows through it); all other agents' actions $a_{j\neq i}$ come from the replay buffer (no gradient).

**Soft target update** applied every gradient step:
$$
\theta_{\text{target}} \leftarrow \tau\,\theta + (1-\tau)\,\theta_{\text{target}}, \quad \tau = 0.005
$$

Rewards are normalized with a **running Welford mean/std** before storing in the replay buffer — critical for preventing Q-value divergence.

---

## Environment

| Property | Value |
|----------|-------|
| Environment | `simple_spread_v3` (PettingZoo MPE) |
| Agents | 3 cooperative (`agent_0`, `agent_1`, `agent_2`) |
| Observation | 18-dim per agent |
| Actions | 5 discrete (no-op, left, right, up, down) |
| Reward | Shared: $-\sum_l \min_i \|a_i - l\|$ per step |
| Episode length | 25 steps |
| Random policy | ≈ −100 to −200 per episode |
| Trained policy | ≈ −60 (MAPPO), ≈ −72 (MADDPG) |

---

## Tasks

### Part A — Implementation
- Implement **`CentralizedCritic`** in `networks.py` — MLP taking joint observations → scalar value
- Implement **`MAPPOAgent.select_actions`**: run shared actor on each agent's obs; compute centralized value
- Implement **`MAPPOAgent.update`**: global reward normalization, per-agent GAE, advantage normalization, PPO update
- Implement **`MADDPGAgents.select_actions`**: run each agent's independent actor on its own obs
- Implement **`MADDPGAgents.update`**: critic TD update, Gumbel-Softmax actor update, soft target update

### Part B — Experiments
- Compare **MAPPO vs MADDPG** on simple_spread_v3 over 3 seeds — report mean ± std team reward
- Ablation: **MAPPO entropy coefficient** — ent_coef ∈ {0.0, 0.001, 0.01, 0.05}

---

## Files

```
lab5_multi_agent/
├── README.md
├── starter_code/
│   ├── lab5_marl.ipynb      # Starter notebook — fill in the TODOs
│   ├── networks.py          # CentralizedCritic (TODO), MADDPGActor, MADDPGCritic (provided)
│   └── utils.py             # compute_gae, MultiAgentReplayBuffer, plotting (all provided)
└── experiments/
```

---

## References

1. Lowe, R., et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. **NeurIPS**.
2. Yu, C., et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. **NeurIPS**.
3. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. **arXiv:1707.06347**.
4. Terry, J., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning*. **NeurIPS**.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
