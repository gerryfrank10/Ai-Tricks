# 🤖 Reinforcement Learning — Complete Guide

Reinforcement Learning (RL) is the science of decision-making: an **agent** learns by interacting with an **environment**, receiving **rewards** as feedback signals. From game-playing AIs to robotic control and aligning large language models, RL powers some of the most impressive achievements in modern AI.

---

## 🗺️ The MDP Framework

Every RL problem is formally a **Markov Decision Process (MDP)**, defined by the tuple *(S, A, P, R, γ)*.

| Symbol | Name | Description |
|--------|------|-------------|
| **S** | State space | All possible situations the agent can be in |
| **A** | Action space | All actions the agent can take |
| **P(s'│s,a)** | Transition model | Probability of reaching state s' from s via action a |
| **R(s,a,s')** | Reward function | Scalar feedback signal |
| **γ** | Discount factor | How much to value future rewards (0 < γ ≤ 1) |

**The Markov Property:** the future depends only on the current state, not history.

```python
import numpy as np

# Simple GridWorld MDP
class GridWorldMDP:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4          # up, down, left, right
        self.gamma = 0.99
        self.goal_state = self.n_states - 1
        self.reset()

    def reset(self):
        self.state = 0              # top-left corner
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)
        if action == 0: row = max(0, row - 1)           # up
        elif action == 1: row = min(self.size-1, row+1) # down
        elif action == 2: col = max(0, col - 1)         # left
        elif action == 3: col = min(self.size-1, col+1) # right

        next_state = row * self.size + col
        done = (next_state == self.goal_state)
        reward = 1.0 if done else -0.01   # small negative reward per step
        self.state = next_state
        return next_state, reward, done

env = GridWorldMDP(size=4)
state = env.reset()
print(f"Initial state: {state}")
next_state, reward, done = env.step(1)  # move down
print(f"After action: state={next_state}, reward={reward}, done={done}")
```

---

## 📐 Value Functions

Value functions measure how good it is to be in a state (or take an action in a state).

### State-Value Function V(s)

The expected cumulative discounted reward starting from state *s* following policy *π*:

```
V^π(s) = E_π [ Σ γ^t · R_{t+k+1} | S_t = s ]
```

### Action-Value Function Q(s, a)

Expected return starting from state *s*, taking action *a*, then following policy *π*:

```
Q^π(s,a) = E_π [ Σ γ^t · R_{t+k+1} | S_t = s, A_t = a ]
```

The relationship: `V^π(s) = Σ_a π(a|s) · Q^π(s,a)`

---

## ⚡ Bellman Equations

The Bellman equations express value functions **recursively** — the cornerstone of RL algorithms.

### Bellman Expectation Equation

```
V^π(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V^π(s')]

Q^π(s,a) = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · Σ_{a'} π(a'|s') · Q^π(s',a')]
```

### Bellman Optimality Equation

```
V*(s)   = max_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V*(s')]
Q*(s,a) = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · max_{a'} Q*(s',a')]
```

```python
# Value Iteration — solving Bellman Optimality iteratively
def value_iteration(env, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n_states)

    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            # Compute Q(s,a) for all actions, take max
            q_values = []
            for a in range(env.n_actions):
                env.state = s
                s_next, r, _ = env.step(a)
                q_values.append(r + gamma * V[s_next])
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Extract greedy policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q_values = []
        for a in range(env.n_actions):
            env.state = s
            s_next, r, _ = env.step(a)
            q_values.append(r + gamma * V[s_next])
        policy[s] = np.argmax(q_values)
    return V, policy

env = GridWorldMDP(size=4)
V_star, pi_star = value_iteration(env)
print("Optimal values:\n", V_star.reshape(4, 4).round(2))
```

---

## 🎯 Q-Learning

Q-Learning is a model-free, off-policy algorithm that directly learns Q*(s,a).

**Update rule:**
```
Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') - Q(s,a)]
```

```python
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training loop
env = GridWorldMDP(size=4)
agent = QLearningAgent(n_actions=4)
episode_rewards = []

for episode in range(2000):
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    episode_rewards.append(total_reward)

print(f"Mean reward (last 200 eps): {np.mean(episode_rewards[-200:]):.3f}")
print(f"Final epsilon: {agent.epsilon:.4f}")
```

---

## 🧠 Deep Q-Network (DQN) with PyTorch

DQN replaces the Q-table with a neural network, enabling RL on high-dimensional inputs (pixels, continuous states).

**Key innovations:** Experience Replay Buffer + Target Network.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Neural network approximating Q(s, a)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer — breaks temporal correlations
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 target_update_freq=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net: no gradients needed

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — robust to outliers

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
        return int(q.argmax())

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values — using frozen target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Periodic hard update of target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

# Usage with Gymnasium
# import gymnasium as gym
# env = gym.make("CartPole-v1")
# agent = DQNAgent(state_dim=4, action_dim=2)
```

---

## 🚀 Proximal Policy Optimization (PPO) — Overview

PPO is a policy-gradient method that directly optimizes the **policy** (instead of Q-values). It is the workhorse algorithm behind ChatGPT's RLHF fine-tuning.

**Core idea:** limit how much the policy changes each update to avoid catastrophic policy collapse.

```
L^CLIP(θ) = E[ min(r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t) ]
```

Where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio and `Â_t` is the advantage estimate.

| Component | Purpose |
|-----------|---------|
| Clipped ratio | Prevents large policy updates |
| Advantage function | Reduces variance in gradient estimates |
| Value function | Baseline for advantage computation |
| Entropy bonus | Encourages exploration |

---

## 🏋️ Gymnasium — Modern RL Environments

Gymnasium (successor to OpenAI Gym) is the standard API for RL environments.

```python
import gymnasium as gym
import numpy as np

# Classic control
env = gym.make("CartPole-v1", render_mode=None)

# Inspect environment
print(f"Observation space: {env.observation_space}")   # Box(4,)
print(f"Action space:      {env.action_space}")        # Discrete(2)
print(f"Reward range:      {env.reward_range}")

obs, info = env.reset(seed=42)
total_reward = 0

for step in range(500):
    action = env.action_space.sample()     # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward}")
env.close()

# Wrappers — composable environment modifications
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

env = gym.make("LunarLander-v3")
env = TimeLimit(env, max_episode_steps=500)
env = RecordEpisodeStatistics(env)

# Vector environments — run multiple envs in parallel
vec_env = gym.make_vec("CartPole-v1", num_envs=4)
obs, infos = vec_env.reset()
print(f"Batched obs shape: {obs.shape}")  # (4, 4)
```

---

## ⚙️ Stable-Baselines3 — Production-Ready RL

```python
from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Create vectorized environment (SB3 requires VecEnv)
vec_env = make_vec_env("CartPole-v1", n_envs=4, seed=42)

# Instantiate PPO — sensible defaults, easy to customize
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # entropy coefficient for exploration
    verbose=1,
)

# Callbacks for automated evaluation + early stopping
eval_env = Monitor(gym.make("CartPole-v1"))
stop_cb = StopTrainingOnRewardThreshold(reward_threshold=490, verbose=1)
eval_cb = EvalCallback(eval_env, callback_on_new_best=stop_cb,
                       eval_freq=5000, best_model_save_path="./logs/best_model")

model.learn(total_timesteps=200_000, callback=eval_cb)
model.save("ppo_cartpole")

# Reload and evaluate
loaded = PPO.load("ppo_cartpole")
obs, _ = gym.make("CartPole-v1").reset()

# SAC for continuous action spaces (MuJoCo, robotics)
sac_model = SAC("MlpPolicy", "Pendulum-v1", verbose=1)
sac_model.learn(total_timesteps=50_000)
```

---

## 🔗 LLM + RL — RLHF Overview

**Reinforcement Learning from Human Feedback (RLHF)** is the technique used to align GPT-4, Claude, and Gemini.

```
Stage 1: Supervised Fine-Tuning (SFT)
  LLM ──────────────────────────────→ SFT Model
         human demonstrations

Stage 2: Reward Model Training
  (Prompt, Response A, Response B) → Human preference label
  Reward Model learns: r(prompt, response)

Stage 3: RL Fine-Tuning with PPO
  SFT Model ──[PPO update]──→ Aligned Model
               ↑ reward signal from Reward Model
               + KL penalty to prevent distribution shift
```

```python
# Conceptual RLHF reward model structure
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Use last token's hidden state as reward signal
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

# Bradley-Terry preference loss
def preference_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

---

## 💡 Tips & Tricks

| Tip | Detail |
|-----|--------|
| **Reward shaping** | Add intermediate rewards to guide learning; avoid reward hacking |
| **Normalize observations** | Use `VecNormalize` in SB3; stabilizes training dramatically |
| **Tune γ carefully** | γ=0.99 for long horizons; γ=0.9 for short episodic tasks |
| **Start with SB3** | Don't implement DQN from scratch for production tasks |
| **Log with TensorBoard** | Pass `tensorboard_log="./tb_logs"` to SB3 models |
| **Parallel envs** | Use `n_envs=8` with PPO for 4-8x faster wall-clock training |
| **Seed everything** | `env.reset(seed=42)` + `torch.manual_seed(42)` for reproducibility |
| **Huber loss over MSE** | SmoothL1Loss is more robust to outlier TD errors in DQN |
| **Soft target updates** | `τ·θ_policy + (1-τ)·θ_target` instead of hard copy every N steps |
| **Clip gradients** | `clip_grad_norm_(params, 10.0)` prevents exploding gradients |

---

## 🗂️ Algorithm Comparison

| Algorithm | Type | Action Space | Sample Efficiency | Notes |
|-----------|------|-------------|-------------------|-------|
| Q-Learning | Model-free, off-policy | Discrete | Low | Tabular only |
| DQN | Model-free, off-policy | Discrete | Medium | Pixels, CNNs |
| PPO | Model-free, on-policy | Both | Medium | Stable, widely used |
| SAC | Model-free, off-policy | Continuous | High | Entropy regularization |
| TD3 | Model-free, off-policy | Continuous | High | Deterministic policy |
| DDPG | Model-free, off-policy | Continuous | Medium | Precursor to TD3 |
| Dreamer | Model-based | Both | Very High | Learns world model |

---

## 📚 Further Reading

- [Sutton & Barto — Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (free online)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Spinning Up in Deep RL — OpenAI](https://spinningup.openai.com/)
- [CleanRL — Single-file RL implementations](https://github.com/vwxyzjn/cleanrl)
