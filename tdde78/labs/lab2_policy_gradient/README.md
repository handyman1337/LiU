# Lab 2 — Policy Gradient Methods

**TDDE78 — Deep Reinforcement Learning, Linköping University — Spring 2026**

---

## Overview

In this lab you implement two **policy gradient** algorithms: **REINFORCE** (with and without a learned baseline) and **Proximal Policy Optimization (PPO)**. Unlike value-based methods, these algorithms directly optimize the policy by gradient ascent on expected return.

---

## Theory

### REINFORCE

REINFORCE collects a full episode, computes discounted returns, and updates the policy via the **Policy Gradient Theorem**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

where $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$ is the discounted return from step $t$.

**With baseline:** Subtracting the value estimate $V_\phi(s_t)$ reduces variance without introducing bias:

$$
\mathcal{L}(\theta,\phi) = -\sum_t \log \pi_\theta(a_t \mid s_t) \cdot \underbrace{(G_t - V_\phi(s_t))}_{\text{advantage}} \;+\; c_V \cdot \sum_t (G_t - V_\phi(s_t))^2
$$

Both actor and critic share the same backbone network (`DiscreteActorCritic`) with two output heads — logits for the policy and a scalar value estimate.

---

### Generalized Advantage Estimation (GAE)

GAE interpolates between one-step TD ($\lambda=0$, low variance, high bias) and Monte Carlo ($\lambda=1$, zero bias, high variance). Given $\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$:

$$
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\qquad \text{computed backwards: } A_t = \delta_t + \gamma\lambda(1-d_t)\cdot A_{t+1}
$$

The return (critic target) is $R_t = A_t + V(s_t)$.

---

### PPO (Proximal Policy Optimization)

PPO constrains how much the policy can change per update by clipping the probability ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

**Clipped surrogate objective:**

![PPO clipped objective](https://huggingface.co/blog/assets/93_deep_rl_ppo/clipped.jpg)
*Source: Hugging Face Deep RL Course — "Proximal Policy Optimization"*

**Full PPO loss** (actor + critic + entropy bonus):

$$
\mathcal{L}(\theta) = -\mathcal{L}^{\text{CLIP}}(\theta) \;+\; c_V \cdot \mathbb{E}_t\!\left[(V_\theta(s_t) - R_t)^2\right] \;-\; c_H \cdot \mathbb{E}_t\!\left[\mathcal{H}[\pi_\theta(\cdot\mid s_t)]\right]
$$

PPO runs $n\_epochs$ mini-batch updates over the collected rollout, then discards the data and collects a new one (on-policy).

---

## Environment

| Environment | Observation | Actions | Solved At |
|-------------|------------|---------|-----------|
| `CartPole-v1` | 4-dim vector | 2 discrete | Avg reward ≥ 475 over 100 episodes |

---

## Tasks

### Part A — Implementation
- Implement `compute_returns` (discounted Monte Carlo returns) and `compute_gae` (GAE)
- Implement **REINFORCE** with and without value baseline
- Implement **PPO**: rollout collection, clipped surrogate loss, value loss, entropy bonus, mini-batch updates

### Part B — Experiments
- Compare **REINFORCE vs PPO** on CartPole-v1 (3 seeds) — convergence speed and stability
- Ablation: **baseline** — REINFORCE with vs without V(s)
- Ablation: **GAE lambda** — λ ∈ {0.0, 0.95, 1.0}

---

## Files

```
lab2_policy_gradient/
├── README.md
├── starter_code/
│   ├── lab2_pg.ipynb        # Starter notebook — fill in the TODOs
│   ├── networks.py          # DiscreteActorCritic (shared backbone, two heads)
│   └── utils.py             # compute_returns, compute_gae, plotting
└── experiments/
```

---

## References

1. Williams, R.J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. **Machine Learning**, 8(3–4), 229–256.
2. Sutton, R.S., et al. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. **NeurIPS**.
3. Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. **ICLR**.
4. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. **arXiv:1707.06347**.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
