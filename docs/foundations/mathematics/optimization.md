# Optimization Theory for ML

Optimization is the process of finding parameters that minimize a loss function. Understanding the theory prevents you from blindly trusting default settings and helps diagnose training failures.

---

## Convex vs Non-Convex Optimization

```
Convex:            Non-Convex (typical neural network):
      ↓                      ↓      ↓
 ____/ \____        __/\__/  \_/\___/\__
     global min     local   saddle   local
                    min     point    min
```

- **Convex**: Any local minimum is the global minimum (linear/logistic regression, SVMs)
- **Non-convex**: Neural networks — local minima, saddle points, and flat regions abound
- **Good news**: In high-dimensional spaces, most critical points are saddle points, not poor local minima

```python
import numpy as np
import matplotlib.pyplot as plt

# Convex function: f(x) = x^2
x = np.linspace(-3, 3, 300)
plt.plot(x, x**2, label="Convex: x²")

# Non-convex: Rastrigin function (classic optimization benchmark)
def rastrigin(x, A=10):
    n = len(x)
    return A*n + sum(xi**2 - A*np.cos(2*np.pi*xi) for xi in x)

# Test with different starting points
for start in [-2.5, -1.0, 0.5, 1.8, 2.5]:
    val = rastrigin([start])
    print(f"start={start:5.1f}  →  f={val:.4f}")
```

---

## Gradient Descent Variants

### Batch Gradient Descent
Uses ALL data to compute one gradient update. Stable but slow.

```python
import torch
import torch.nn as nn

X = torch.randn(10000, 10)
y = torch.randn(10000, 1)
model = nn.Linear(10, 1)
criterion = nn.MSELoss()

# Batch GD — one update per full dataset pass
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### Stochastic Gradient Descent (SGD)
One sample at a time. Noisy but fast, good regularization effect.

```python
# Mini-batch SGD — best of both worlds
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

for epoch in range(50):
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
```

---

## Adam Optimizer — Derivation & Intuition

Adam = Adaptive Moment Estimation. Combines momentum (1st moment) with RMSProp (2nd moment).

```
m_t = β₁ m_{t-1} + (1-β₁) g_t          # 1st moment (mean of gradients)
v_t = β₂ v_{t-1} + (1-β₂) g_t²         # 2nd moment (variance of gradients)
m̂_t = m_t / (1 - β₁ᵗ)                  # bias correction
v̂_t = v_t / (1 - β₂ᵗ)                  # bias correction
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)  # parameter update
```

```python
# Adam from scratch
class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data   # weight decay

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Use PyTorch's built-in (numerically identical):
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# For transformers, prefer AdamW (decoupled weight decay):
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## Learning Rate Schedules

The learning rate is the single most important hyperparameter. These schedules improve convergence dramatically.

```python
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, LinearLR, SequentialLR
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 1. Cosine Annealing — standard for vision/NLP
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 2. OneCycleLR — fast.ai's "super convergence"
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=30,
    pct_start=0.3,      # 30% of training = warmup
    anneal_strategy='cos'
)

# 3. Reduce on Plateau — useful when training plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 4. Linear Warmup + Cosine Decay (standard for LLMs)
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=9000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1000])

# Training loop with scheduler
for epoch in range(100):
    train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)   # for ReduceLROnPlateau
    # scheduler.step()         # for others
```

---

## Loss Landscapes

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_loss_landscape(model, loader, criterion, n_points=30):
    """
    Plot loss in 2D parameter subspace (filter normalization method).
    Helps understand flatness/sharpness of minima.
    """
    # Get current params
    center = [p.data.clone() for p in model.parameters()]

    # Random directions in parameter space
    d1 = [torch.randn_like(p) for p in model.parameters()]
    d2 = [torch.randn_like(p) for p in model.parameters()]

    # Normalize directions (filter normalization)
    for p, c, dir1, dir2 in zip(model.parameters(), center, d1, d2):
        dir1 *= p.norm() / (dir1.norm() + 1e-10)
        dir2 *= p.norm() / (dir2.norm() + 1e-10)

    alphas = np.linspace(-1, 1, n_points)
    betas  = np.linspace(-1, 1, n_points)
    Z = np.zeros((n_points, n_points))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Perturb parameters
            with torch.no_grad():
                for p, c, d1_, d2_ in zip(model.parameters(), center, d1, d2):
                    p.data = c + a * d1_ + b * d2_

            # Compute loss
            total = 0
            for X_b, y_b in loader:
                total += criterion(model(X_b), y_b).item()
            Z[i, j] = total / len(loader)

    # Restore original params
    with torch.no_grad():
        for p, c in zip(model.parameters(), center):
            p.data = c

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    A, B = np.meshgrid(alphas, betas)
    ax.plot_surface(A, B, Z.T, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=150)
```

---

## Saddle Points & Escaping Them

```python
# Saddle point example: f(x,y) = x^2 - y^2
# Gradient at origin is zero, but it's NOT a minimum!

import torch

x = torch.tensor([0.01, 0.01], requires_grad=True)  # near saddle point

optimizer = torch.optim.SGD([x], lr=0.1)
trajectory = [x.data.clone()]

for step in range(50):
    optimizer.zero_grad()
    # Saddle: minimum in x-direction, maximum in y-direction
    loss = x[0]**2 - x[1]**2
    loss.backward()
    optimizer.step()
    trajectory.append(x.data.clone())

# SGD with noise will escape saddle points
# Adam is also good at escaping saddle points due to adaptive lr
print(f"Final position: {x.data}")  # Will move away from origin
```

---

## Second-Order Methods

```python
# L-BFGS: quasi-Newton method, excellent for small models / fine-tuning
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    return loss

# L-BFGS requires closure (re-evaluates loss)
optimizer.step(closure)
# Note: not suited for mini-batch training — use for full-batch fine-tuning
```

---

## Tips & Tricks

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| Loss NaN | Exploding gradients | `clip_grad_norm_(..., 1.0)` |
| Loss plateau | LR too low | Try OneCycleLR or increase LR 10x |
| Loss oscillating | LR too high | Reduce LR by 10x |
| Slow convergence | No momentum | Switch SGD → AdamW |
| Overfitting fast | Sharp minima | SAM optimizer, weight decay |
| LLM fine-tuning | AdamW default unstable | Use `lr=2e-5`, `warmup_steps=500` |

---

## Related Topics

- [Gradient Descent (in-depth)](../../optimization/gradient-descent.md)
- [Neural Network Training](../../deep-learning/neural-networks.md)
- [Deep Learning Optimization Tricks](../../deep-learning/optimization.md)
