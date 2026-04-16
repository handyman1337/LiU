# Lab 4 — Model-Based Deep Reinforcement Learning

**TDDE78 — Deep Reinforcement Learning, Linköping University — Spring 2026**

---

## Overview

In this lab you implement two **model-based** approaches: **Dyna-Q** and **MCTS**. Unlike model-free methods, model-based RL explicitly learns or uses a model of the environment's dynamics to plan ahead and improve sample efficiency.

---

## Theory

### Dyna-Q: DQN + Neural WorldModel

Dyna-Q augments a standard DQN agent with a learned **WorldModel** that simulates transitions. After each real environment step, the WorldModel generates additional simulated transitions that are added to the replay buffer alongside real data.

![Dyna-Q architecture](https://lcalem.github.io/imgs/sutton/dynaq_agent.png)

*Source: Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 8*

#### WorldModel

The WorldModel is a neural network that learns the environment's dynamics from collected experience. It takes a state-action pair as input and predicts three outputs:

- **Next state** — a distribution over all possible next states (logits → cross-entropy loss)
- **Reward** — scalar prediction (MSE loss)
- **Done** — binary terminal flag (BCE loss)

For discrete state spaces (CliffWalking), states and actions are one-hot encoded before being concatenated as input:

$$
\text{input} = [\text{onehot}(s),\; \text{onehot}(a)] \in \mathbb{R}^{n_s + n_a}
$$

The WorldModel is trained by minimizing the combined prediction loss:
$$
\mathcal{L}_{\text{model}}(\psi) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}}\!\left[\underbrace{\text{CE}(\hat{s}', s')}_{\text{next state}} + \underbrace{(\hat{r} - r)^2}_{\text{reward}} + \underbrace{\text{BCE}(\hat{d}, d)}_{\text{done}}\right]
$$

Once trained, the WorldModel acts as a **differentiable simulator**: given any $(s, a)$ sampled from the replay buffer, it predicts $(r, s', d)$ without interacting with the real environment.

**Dyna planning loop:**
```
For each real step (s, a, r, s'):
    Store (s, a, r, s') in replay buffer D
    Update WorldModel on D
    Update DQN on D  (real experience)

    For k planning steps:
        Sample random (s̃, ã) from D
        Predict (r̃, s̃') from WorldModel
        Add (s̃, ã, r̃, s̃') to D
        Update DQN on simulated transition
```

More planning steps k → faster improvement per real interaction, as long as the WorldModel is accurate.

---

### MCTS — Monte Carlo Tree Search

MCTS builds a search tree from the current state by running many simulations. Each simulation has four phases:

![MCTS four phases](https://lcalem.github.io/imgs/sutton/mcts.png)

*Source: Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 8*

1. **Selection** — walk down the tree using the UCT formula until a leaf:

$$
a^* = \arg\max_a \left[\frac{Q(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right]
$$

- $Q/N$ — exploitation: average return of this action
- $c\sqrt{\ln N_\text{parent}/N}$ — exploration: prefer less-visited nodes ($c = \sqrt{2}$)

2. **Expansion** — add a new child node for an untried action
3. **Rollout** — simulate to episode end using a random policy
4. **Backpropagation** — update visit counts $N$ and total value $Q$ up the tree to the root

After $n\_simulations$, select $\arg\max_a N(\text{root}, a)$ (most-visited action). MCTS uses the environment as a **black-box simulator** via `deepcopy` — no model learning required.

---

## Environment

| Environment | Observation | Actions | Optimal Reward |
|-------------|------------|---------|----------------|
| `CliffWalking-v1` | integer (4×12 grid position) | 4 discrete (up/down/left/right) | ≈ −13 per episode |

The agent navigates from start (bottom-left) to goal (bottom-right) while avoiding a cliff along the bottom row. Step reward = −1; falling off the cliff = −100 and reset to start.

---

## Tasks

### Part A — Implementation
- Implement **WorldModel**: neural network predicting next state and reward from (s, a)
- Implement **Dyna-Q**: train WorldModel on real data, augment DQN replay with simulated rollouts
- Implement **MCTS**: UCT selection, node expansion, random rollout, backpropagation, action selection

### Part B — Experiments
- Train **DQN baseline** vs **Dyna-Q** — show Dyna converges faster with fewer real environment steps
- Ablation: **Dyna planning steps k** — vary number of simulated rollouts per real step
- Ablation: **MCTS simulation count** — sweep n_simulations and measure reward vs computation trade-off

---

## Files

```
lab4_model_based/
├── README.md
├── starter_code/
│   ├── lab4_mb.ipynb        # Starter notebook — fill in the TODOs
│   ├── networks.py          # WorldModel, DQN architectures
│   └── utils.py             # plotting, smooth, experiment save utilities
└── experiments/
```

---

## References

1. Sutton, R.S. (1991). *Dyna, an integrated architecture for learning, planning, and reacting*. **ACM SIGART Bulletin**, 2(4), 160–163.
2. Kocsis, L., & Szepesvári, C. (2006). *Bandit-based Monte-Carlo Planning*. **ECML**.
3. Silver, D., et al. (2016). *Mastering the game of Go with deep neural networks and tree search*. **Nature**, 529, 484–489.
4. Ha, D., & Schmidhuber, J. (2018). *World Models*. **NeurIPS**.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
