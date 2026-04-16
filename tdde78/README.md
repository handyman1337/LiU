# TDDE78: Deep Reinforcement Learning

## Linköping University — Spring 2026



## Course Overview

This course provides a foundational introduction to **deep reinforcement learning** from a computer science perspective. All methods use neural networks as function approximators — policies, value functions, and environment models are learned end-to-end in PyTorch. Students implement state-of-the-art algorithms and systematically evaluate them on standard benchmark environments using Gymnasium and PettingZoo.

### Prerequisites

- Practical programming skills in Python
- Working knowledge of deep learning, especially deep neural networks (e.g., from TDDC17 Artificial Intelligence)
- Understanding of common concepts in computer science and optimization

---

## Learning Outcomes

After completing all labs, students will be able to:

1. **Identify** basic concepts, terminology, theories, models, and methods in deep reinforcement learning
2. **Implement** core deep RL algorithms — including value-based, policy gradient, actor-critic, model-based, and multi-agent methods
3. **Design and execute** systematic experiments to evaluate and compare RL algorithms
4. **Analyze** the effect of architectural choices, hyperparameters, and algorithmic variants through ablation studies
5. **Interpret and document** experimental results with proper statistical methodology (multiple seeds, learning curves, performance metrics)
6. **Read and understand** current research literature in the field of deep reinforcement learning

---

## Lab Structure

The course consists of **5 labs**, each covering a major deep RL paradigm. Every lab is divided into two parts:

- **Part A — Implementation:** Build the algorithm(s) from scratch in PyTorch. Focus on understanding the core mechanics, loss functions, and training loops.
- **Part B — Experiments:** Run systematic experiments including ablation studies, hyperparameter sensitivity analysis, and cross-method comparisons. Produce a written report with visualizations and analysis.

### Lab Overview

| Lab | Topic | Core Methods | Environments |
|-----|-------|-------------|--------------|
| **Lab 1** | Value-Based Deep RL | DQN, Double DQN, Dueling DQN | CartPole-v1, LunarLander-v3 |
| **Lab 2** | Policy Gradient | REINFORCE (with/without baseline), PPO | CartPole-v1 |
| **Lab 3** | Actor-Critic | A2C (with GAE), SAC | LunarLanderContinuous-v3 |
| **Lab 4** | Model-Based Deep RL | Dyna-Q (neural WorldModel), MCTS | CliffWalking-v1 |
| **Lab 5** | Multi-Agent Deep RL | MAPPO, MADDPG | simple_spread_v3 (PettingZoo) |

### Lab Details

#### Lab 1 — Value-Based Deep RL
> Learn to approximate Q-values with neural networks

- **Part A:** Implement vanilla DQN with experience replay and target networks. Extend to Double DQN (reducing overestimation bias) and Dueling DQN (separate value and advantage streams).
- **Part B:** Ablation studies on replay buffer size, target network update frequency, and DQN variant comparison on CartPole-v1. Scale to LunarLander-v3 to measure performance on a harder continuous-observation task.

#### Lab 2 — Policy Gradient
> Learn policies directly through gradient ascent on expected return

- **Part A:** Implement REINFORCE with and without a learned value baseline for variance reduction. Implement Proximal Policy Optimization (PPO) with GAE and clipped surrogate objective.
- **Part B:** Ablation studies on CartPole-v1 — compare REINFORCE with and without baseline, sweep GAE lambda (λ ∈ {0.0, 0.95, 1.0}), and analyze variance reduction in practice.

#### Lab 3 — Actor-Critic
> Combine value estimation and policy optimization

- **Part A:** Implement Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE). Implement Soft Actor-Critic (SAC) for continuous control with entropy regularization and twin Q-networks.
- **Part B:** Compare A2C vs SAC on LunarLanderContinuous-v3. Study SAC entropy temperature (fixed α vs auto-tuned). Analyze GAE lambda sensitivity in A2C. Quantify the sample-efficiency gap between on-policy and off-policy methods.

#### Lab 4 — Model-Based Deep RL
> Learn a model of the environment and use it for planning

- **Part A:** Train a neural dynamics model (WorldModel) to predict next-state and reward. Integrate it with DQN via Dyna-style planning (real + simulated experience). Implement MCTS that uses the environment as a simulator for lookahead planning.
- **Part B:** Dyna planning steps ablation — measure how many simulated rollouts are needed to accelerate DQN on CliffWalking-v1. MCTS simulation count sweep — show the trade-off between planning depth and computational cost. Both evaluated on **CliffWalking-v1**.

#### Lab 5 — Multi-Agent Deep RL
> Extend deep RL to settings with multiple interacting agents

- **Part A:** Implement **MAPPO** (shared actor + centralized critic, on-policy PPO with parameter sharing) and **MADDPG** (per-agent independent actors + centralized Q-critics, off-policy with Gumbel-Softmax). Both follow the CTDE principle: centralized training, decentralized execution.
- **Part B:** Compare MAPPO vs MADDPG over 3 seeds on the cooperative task simple_spread_v3. MAPPO entropy coefficient ablation (ent_coef ∈ {0.0, 0.001, 0.01, 0.05}) to study the exploration-exploitation trade-off in cooperative settings.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| RL Environments (single-agent) | Gymnasium |
| RL Environments (multi-agent) | PettingZoo |
| Package Manager | uv |

---

## Setup

All dependencies are installed in an isolated virtual environment using `uv`.

```bash
# One-time setup
bash setup.sh

# Activate environment (every session)
source .venv/bin/activate

# Verify installation
python -c "import gymnasium; import torch; print('Ready!')"
```

See `setup.sh` for full details.

---

## Directory Structure

```
tdde78lab/
├── README.md                          # This file
├── setup.sh                           # Environment setup script
├── requirements.txt                   # Python dependencies
│
└── labs/
    ├── lab1_value_based/              # Lab 1: DQN and variants
    │   ├── starter_code/              # Notebook + networks.py, replay_buffer.py
    │   └── experiments/               # Saved plots and results
    │
    ├── lab2_policy_gradient/          # Lab 2: REINFORCE, PPO
    │   ├── starter_code/              # Notebook + networks.py, utils.py
    │   └── experiments/
    │
    ├── lab3_actor_critic/             # Lab 3: A2C, SAC
    │   ├── starter_code/              # Notebook + networks.py, utils.py
    │   └── experiments/
    │
    ├── lab4_model_based/              # Lab 4: Dyna-Q, MCTS
    │   ├── starter_code/              # Notebook + networks.py, utils.py
    │   └── experiments/
    │
    └── lab5_multi_agent/              # Lab 5: MAPPO, MADDPG
        ├── starter_code/              # Notebook + networks.py, utils.py
        └── experiments/
```

Each lab's `starter_code/` directory contains:
- A Jupyter notebook (`.ipynb`) — main implementation and experiments
- `networks.py` — neural network architectures
- `utils.py` — training helpers, GAE, plotting, replay buffers

---

## Resources

- **Course Syllabus:** [studieinfo.liu.se/en/kurs/TDDE78](https://studieinfo.liu.se/en/kurs/TDDE78)
- **Gymnasium:** [gymnasium.farama.org](https://gymnasium.farama.org/)
- **PettingZoo:** [pettingzoo.farama.org](https://pettingzoo.farama.org/)
- **Textbook (open access):** [arxiv.org/abs/2201.02135](https://arxiv.org/abs/2201.02135)
- **PyTorch:** [pytorch.org](https://pytorch.org/)

---

*Department of Computer Science (IDA), Linköping University — Spring 2026*

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
