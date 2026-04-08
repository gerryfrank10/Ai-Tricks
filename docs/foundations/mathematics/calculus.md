# Calculus & Automatic Differentiation

Calculus powers every gradient-based learning algorithm. Backpropagation is just the chain rule applied systematically across a computational graph.

---

## Derivatives — The Core Idea

The derivative `f'(x)` measures how much `f(x)` changes per unit change in `x`.

```python
import numpy as np
import torch

# Numerical derivative (finite differences)
def numerical_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: f(x) = x^3 + 2x^2 - 5
f = lambda x: x**3 + 2*x**2 - 5
# Analytical: f'(x) = 3x^2 + 4x
x = 2.0
print(f"Numerical:  {numerical_grad(f, x):.6f}")   # 20.000000
print(f"Analytical: {3*x**2 + 4*x:.6f}")           # 20.000000
```

---

## The Chain Rule — Backbone of Backpropagation

If `y = g(f(x))`, then `dy/dx = (dy/dg) × (dg/dx)`.

```python
# Chain rule example: y = sigmoid(wx + b)
# dy/dw = dy/d(sigmoid) × d(sigmoid)/d(z) × dz/dw
#       = 1 × σ(z)(1-σ(z)) × x

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

# Manual chain rule
x, w, b = 2.0, 0.5, 0.1
z = w * x + b              # z = 1.1
y = sigmoid(z)             # y ≈ 0.75
dz_dw = x                  # ∂z/∂w = x
dy_dz = sigmoid_grad(z)    # ∂y/∂z
dy_dw = dy_dz * dz_dw      # ∂y/∂w via chain rule
print(f"dy/dw = {dy_dw:.6f}")
```

---

## PyTorch Autograd — Automatic Differentiation

PyTorch builds a **computational graph** dynamically and applies the chain rule automatically via `.backward()`.

```python
import torch

# Scalar example
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 - 5      # y = 3

y.backward()                 # Compute dy/dx
print(x.grad)                # tensor(20.) — dy/dx = 3x² + 4x = 20

# Reset gradient before next computation
x.grad.zero_()
```

### Neural Network Layer Gradients

```python
import torch
import torch.nn as nn

# Simple 2-layer network
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Forward pass
X = torch.randn(32, 4)       # batch of 32
y = torch.randn(32, 1)
pred = model(X)

# Loss
loss = nn.MSELoss()(pred, y)
print(f"Loss: {loss.item():.4f}")

# Backward — computes all gradients
loss.backward()

# Inspect gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name:20s}: grad shape {param.grad.shape}, "
              f"grad norm {param.grad.norm():.4f}")
```

### Gradient Accumulation
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
accumulation_steps = 4

for i, (X_batch, y_batch) in enumerate(dataloader):
    pred = model(X_batch)
    loss = criterion(pred, y_batch) / accumulation_steps  # scale loss
    loss.backward()   # accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    # update weights
        optimizer.zero_grad()  # clear gradients
```

---

## Partial Derivatives & Gradients

For functions with multiple inputs, the **gradient** is the vector of partial derivatives.

```python
# f(x, y) = x^2 + xy + y^2
# ∂f/∂x = 2x + y
# ∂f/∂y = x + 2y
# ∇f = [2x+y, x+2y]

x = torch.tensor([2.0, 3.0], requires_grad=True)

def f(v):
    return v[0]**2 + v[0]*v[1] + v[1]**2

loss = f(x)
loss.backward()
print(x.grad)   # tensor([7., 8.])  == [2*2+3, 2+2*3]
```

---

## The Jacobian & Hessian

```python
# Jacobian: ∂y_i/∂x_j — for vector-valued functions
# Used in: understanding layer transformations

x = torch.randn(3, requires_grad=True)
y = torch.stack([x[0]**2 + x[1], x[1]*x[2], x[2]**3])

# Compute full Jacobian
J = torch.autograd.functional.jacobian(
    lambda v: torch.stack([v[0]**2 + v[1], v[1]*v[2], v[2]**3]),
    x
)
print("Jacobian shape:", J.shape)   # (3, 3)

# Hessian: second-order derivatives — ∂²f/∂x_i∂x_j
# Used in: second-order optimizers (L-BFGS, Newton's method)
def scalar_f(v):
    return (v**2).sum()

H = torch.autograd.functional.hessian(scalar_f, x)
print("Hessian:\n", H)   # Diagonal matrix with 2s
```

---

## Gradient Flow & Vanishing/Exploding Gradients

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Demonstrate vanishing gradient with sigmoid activation
def deep_sigmoid_network(depth=20):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(16, 16), nn.Sigmoid()])
    return nn.Sequential(*layers)

model = deep_sigmoid_network(depth=20)
x = torch.randn(1, 16)
y = torch.zeros(1, 16)

loss = nn.MSELoss()(model(x), y)
loss.backward()

# Gradient norms by layer — should decrease exponentially in deep sigmoid nets
grad_norms = []
for name, param in model.named_parameters():
    if 'weight' in name and param.grad is not None:
        grad_norms.append(param.grad.norm().item())

print("Gradient norms (first → last layer):")
for i, g in enumerate(grad_norms):
    bar = "█" * max(1, int(g * 100))
    print(f"  Layer {i+1:2d}: {g:.2e} {bar}")
# With sigmoid: gradients near zero by layer 5
# With ReLU: much better gradient flow

# FIX: Use ReLU + He initialization + residual connections
def deep_relu_network(depth=20):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(16, 16), nn.ReLU()])
    return nn.Sequential(*layers)

# Even better: use residual connections (skip connections)
class ResBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))   # x + F(x) — residual
```

---

## Learning Rate & Gradient Descent

```python
import torch

# Manual gradient descent — what PyTorch optimizers do under the hood
def manual_sgd(params, lr=0.01):
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= lr * p.grad
                p.grad.zero_()

# Gradient clipping — prevents exploding gradients
model = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Key Derivatives to Memorize

| Function | Derivative | Used In |
|----------|-----------|--------|
| `σ(x) = 1/(1+e⁻ˣ)` | `σ(x)(1-σ(x))` | Logistic regression, gates |
| `tanh(x)` | `1 - tanh²(x)` | LSTM gates |
| `ReLU(x) = max(0,x)` | `1 if x>0 else 0` | Activation functions |
| `softmax(xᵢ)` | `sᵢ(δᵢⱼ - sⱼ)` | Classification output |
| `log(x)` | `1/x` | Cross-entropy loss |
| `‖x‖²` | `2x` | L2 regularization |
| `xᵀAx` | `2Ax` (if A symmetric) | Quadratic objectives |

---

## Tips & Tricks

| Problem | Solution |
|---------|---------|
| `nan` gradients | Clip gradients: `clip_grad_norm_(..., 1.0)` |
| Slow convergence | Tune learning rate; use `torch.optim.lr_scheduler` |
| Memory during backward | Use `torch.no_grad()` for inference, `checkpoint` for training |
| Check grad correctness | `torch.autograd.gradcheck(fn, inputs)` |
| Profile backward pass | `torch.profiler.profile()` |

---

## Related Topics

- [Linear Algebra](linear-algebra.md)
- [Optimization Theory](optimization.md)
- [Neural Networks](../../deep-learning/neural-networks.md)
- [Gradient Descent](../../optimization/gradient-descent.md)
