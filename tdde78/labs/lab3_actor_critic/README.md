# Lab 3 — Actor-Critic Methods

**TDDE78 — Deep Reinforcement Learning, Linköping University — Spring 2026**

---

## Overview

In this lab you implement two **actor-critic** algorithms: **A2C** (on-policy) and **SAC** (off-policy). Both maintain a separate actor (policy) and critic (value function), but differ fundamentally in how they collect and reuse data.

---

## Theory

### A2C — Advantage Actor-Critic

A2C is on-policy. It collects a fixed-length rollout, computes GAE advantages, and performs a **single gradient update** before discarding the data. The actor is a Gaussian policy (continuous actions) and the critic is a shared value head.

![A2C architecture](https://huggingface.co/blog/assets/89_deep_rl_a2c/ac.jpg)

*Source: Hugging Face Deep RL Course — "Advantage Actor-Critic (A2C)"*

**GAE advantage** (same as Lab 2):
$$
A_t = \delta_t + \gamma\lambda(1-d_t)\cdot A_{t+1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**A2C loss:**
$$
\mathcal{L}(\theta) = \underbrace{-\mathbb{E}_t\!\left[\log\pi_\theta(a_t|s_t)\cdot \hat{A}_t\right]}_{\text{policy loss}} \;+\; c_V \cdot \underbrace{\mathbb{E}_t\!\left[(V_\theta(s_t) - R_t)^2\right]}_{\text{value loss}} \;-\; c_H \cdot \underbrace{\mathbb{E}_t\!\left[\mathcal{H}[\pi_\theta(\cdot|s_t)]\right]}_{\text{entropy bonus}}
$$

where $R_t = A_t + V(s_t)$ is the GAE return target. The actor defines a **Gaussian policy** $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)$ — actions are sampled as $a = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

---

### SAC — Soft Actor-Critic

![SAC algorithm](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/SAC_algo.png)

*Source: Lilian Weng — "Policy Gradient Algorithms"*

SAC is off-policy and maximizes a **maximum entropy objective** — the policy is rewarded for both collecting high return and remaining stochastic:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_t \left(r_t + \alpha\,\mathcal{H}[\pi(\cdot|s_t)]\right)\right]
$$

SAC uses **twin Q-networks** to reduce overestimation bias (taking the min of both critics when computing targets).

**Critic loss** (for each of the two Q-networks):
$$
y = r + \gamma\!\left(\min_{i=1,2} Q_{\phi_i^-}(s', \tilde{a}') - \alpha\log\pi_\theta(\tilde{a}'|s')\right), \quad \tilde{a}' \sim \pi_\theta(\cdot|s')
$$
$$
\mathcal{L}_Q(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[(Q_{\phi_i}(s,a) - y)^2\right]
$$

**Actor loss** (maximize Q, penalize low entropy):
$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\!\left[\alpha\log\pi_\theta(\tilde{a}|s) - \min_i Q_{\phi_i}(s,\tilde{a})\right]
$$

Actions use the **reparameterization trick** with tanh squashing: $\tilde{a} = \tanh(\mu + \sigma\cdot\epsilon)$. The log-probability must be corrected for this change of variables:
$$
\log\pi_\theta(a|s) = \log\mathcal{N}(\text{arctanh}(a);\mu,\sigma) - \sum_i\log(1 - a_i^2 + \varepsilon)
$$

The entropy temperature $\alpha$ can be **automatically tuned** by minimizing:
$$
\mathcal{L}(\alpha) = \mathbb{E}\!\left[-\alpha\left(\log\pi_\theta(a|s) + \bar{\mathcal{H}}\right)\right], \quad \bar{\mathcal{H}} = -|\mathcal{A}|
$$

Target networks are updated with **soft (Polyak) averaging**: $\phi^- \leftarrow \tau\phi + (1-\tau)\phi^-$.

---

## Environment

| Environment | Observation | Actions | Solved At |
|-------------|------------|---------|-----------|
| `LunarLanderContinuous-v3` | 8-dim vector | 2 continuous ∈ [−1, 1] | Avg reward ≥ 200 over 100 episodes |

---

## Tasks

### Part A — Implementation
- Implement **A2C**: Gaussian actor, shared value critic, GAE, combined loss with entropy bonus
- Implement **SAC**: replay buffer, twin Q-networks, reparameterization + tanh squashing, actor loss, soft target updates
- Implement **automatic entropy temperature tuning** for SAC

### Part B — Experiments
- Train **A2C** on LunarLanderContinuous-v3 — does it converge? Analyze stability and variance
- Train **SAC** on LunarLanderContinuous-v3 — compare sample efficiency and final performance vs A2C
- Ablation: **SAC entropy temperature** — fixed α ∈ {0.05, 0.2} vs auto-tuned α
- Ablation: **GAE lambda in A2C** — λ ∈ {0.0, 0.95, 1.0}

---

## Files

```
lab3_actor_critic/
├── README.md
├── starter_code/
│   ├── lab3_ac.ipynb        # Starter notebook — fill in the TODOs
│   ├── networks.py          # ContinuousActorCritic (A2C), SACActorNetwork, SACCriticNetwork
│   └── utils.py             # compute_gae, SAC ReplayBuffer, plotting
└── experiments/
```

---

## References

1. Mnih, V., et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. **ICML**.
2. Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. **ICLR**.
3. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. **ICML**.
4. Haarnoja, T., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. **arXiv:1812.05905**.
5. Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. **ICML**.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
