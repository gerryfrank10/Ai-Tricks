# Particle Swarm Optimization

PSO simulates the social behavior of birds flocking or fish schooling. Each "particle" explores the search space, influenced by its own best position and the swarm's global best.

---

## Algorithm Mechanics

```
Initialize N particles with random positions and velocities

For each iteration:
  For each particle i:
    v[i] = w * v[i]                    # inertia (keep going)
          + c1 * r1 * (pBest[i] - x[i])  # cognitive (personal best)
          + c2 * r2 * (gBest - x[i])     # social (swarm best)
    x[i] = x[i] + v[i]                # update position

    if f(x[i]) < f(pBest[i]):
        pBest[i] = x[i]               # update personal best
    if f(x[i]) < f(gBest):
        gBest = x[i]                  # update global best
```

**Parameters:**
- `w` (inertia weight) — controls exploration vs exploitation
- `c1` (cognitive) — attraction to personal best (typically 1.5-2.0)
- `c2` (social) — attraction to global best (typically 1.5-2.0)

---

## Python Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

class PSO:
    """Particle Swarm Optimization."""

    def __init__(
        self,
        objective_fn: Callable,
        n_dims: int,
        bounds: list[tuple[float, float]],
        n_particles: int = 50,
        max_iter: int = 200,
        w: float = 0.7,    # Inertia weight
        c1: float = 1.5,   # Cognitive coefficient
        c2: float = 1.5,   # Social coefficient
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.fn = objective_fn
        self.n_dims = n_dims
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2

        # Initialize particles
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.x = lb + np.random.rand(n_particles, n_dims) * (ub - lb)
        self.v = np.random.randn(n_particles, n_dims) * 0.1

        # Personal & global bests
        self.pBest = self.x.copy()
        self.pBest_scores = np.array([self.fn(p) for p in self.x])

        best_idx = self.pBest_scores.argmin()
        self.gBest = self.pBest[best_idx].copy()
        self.gBest_score = self.pBest_scores[best_idx]
        self.history = [self.gBest_score]

    def optimize(self) -> dict:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]

        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            # Update velocities
            self.v = (
                self.w * self.v
                + self.c1 * r1 * (self.pBest - self.x)
                + self.c2 * r2 * (self.gBest - self.x)
            )

            # Update positions
            self.x = np.clip(self.x + self.v, lb, ub)

            # Evaluate all particles
            scores = np.array([self.fn(p) for p in self.x])

            # Update personal bests
            improved = scores < self.pBest_scores
            self.pBest[improved] = self.x[improved]
            self.pBest_scores[improved] = scores[improved]

            # Update global best
            best_idx = self.pBest_scores.argmin()
            if self.pBest_scores[best_idx] < self.gBest_score:
                self.gBest = self.pBest[best_idx].copy()
                self.gBest_score = self.pBest_scores[best_idx]

            self.history.append(self.gBest_score)

            if (iteration + 1) % 50 == 0:
                print(f"Iter {iteration+1:3d} | Best: {self.gBest_score:.6f}")

        return {
            "best_position": self.gBest,
            "best_score": self.gBest_score,
            "history": self.history,
        }

# ── Example: Optimize Rastrigin function ──────────────────────────
def rastrigin(x: np.ndarray, A: float = 10) -> float:
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

pso = PSO(
    objective_fn=rastrigin,
    n_dims=10,
    bounds=[(-5.12, 5.12)] * 10,
    n_particles=100,
    max_iter=300,
)
result = pso.optimize()
print(f"\nGlobal optimum: 0.0 | Found: {result['best_score']:.6f}")
print(f"Best position:  {np.round(result['best_position'], 4)}")
```

---

## Hyperparameter Tuning with PSO

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

def rf_objective(params: np.ndarray) -> float:
    """Minimize negative CV AUC (PSO minimizes)."""
    n_estimators    = int(params[0])
    max_depth       = int(params[1])
    min_samples_leaf = int(params[2])
    max_features    = float(params[3])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return -scores.mean()   # Negate because PSO minimizes

pso = PSO(
    objective_fn=rf_objective,
    n_dims=4,
    bounds=[
        (50, 500),     # n_estimators
        (2, 20),       # max_depth
        (1, 20),       # min_samples_leaf
        (0.1, 1.0),    # max_features
    ],
    n_particles=30,
    max_iter=50,
    seed=42,
)

result = pso.optimize()
best_params = result["best_position"]
print(f"\nBest ROC-AUC: {-result['best_score']:.4f}")
print(f"n_estimators:     {int(best_params[0])}")
print(f"max_depth:        {int(best_params[1])}")
print(f"min_samples_leaf: {int(best_params[2])}")
print(f"max_features:     {best_params[3]:.3f}")
```

---

## Adaptive Inertia Weight (APSO)

```python
class AdaptivePSO(PSO):
    """PSO with linearly decreasing inertia weight.
    Starts exploratory (high w), becomes exploitative (low w).
    """

    def __init__(self, *args, w_start: float = 0.9, w_end: float = 0.4, **kwargs):
        super().__init__(*args, w=w_start, **kwargs)
        self.w_start = w_start
        self.w_end   = w_end

    def optimize(self) -> dict:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]

        for iteration in range(self.max_iter):
            # Linearly decay inertia weight
            self.w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iter

            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            self.v = (
                self.w * self.v
                + self.c1 * r1 * (self.pBest - self.x)
                + self.c2 * r2 * (self.gBest - self.x)
            )
            self.x = np.clip(self.x + self.v, lb, ub)

            scores = np.array([self.fn(p) for p in self.x])
            improved = scores < self.pBest_scores
            self.pBest[improved] = self.x[improved]
            self.pBest_scores[improved] = scores[improved]

            best_idx = self.pBest_scores.argmin()
            if self.pBest_scores[best_idx] < self.gBest_score:
                self.gBest = self.pBest[best_idx].copy()
                self.gBest_score = self.pBest_scores[best_idx]

            self.history.append(self.gBest_score)

        return {"best_position": self.gBest, "best_score": self.gBest_score, "history": self.history}
```

---

## PSO vs Genetic Algorithm

| Aspect | PSO | Genetic Algorithm |
|--------|-----|------------------|
| Mechanism | Velocity + social behavior | Selection + crossover + mutation |
| Parameters | w, c1, c2 | Population size, crossover rate, mutation rate |
| Convergence | Often faster | More thorough exploration |
| Memory | Uses personal + global best | Only current population |
| Discrete problems | Needs adaptation (DPSO) | Natural fit (binary GA) |
| Continuous problems | Excellent | Good with real-valued encoding |
| Implementation | Simpler | More complex |

---

## Tips & Tricks

| Issue | Fix |
|-------|-----|
| Premature convergence | Increase w or c1/c2 |
| Slow convergence | Decrease w, increase n_particles |
| Particles escape bounds | Velocity clamping or reflective boundaries |
| Many local optima | Increase n_particles to 100-200 |
| Mixed integer problems | Round integer dimensions after update |
| High-dimensional (>50D) | Use SPSO-2011 or CMA-ES instead |

---

## Related Topics

- [Genetic Algorithms](genetic-algorithms.md)
- [Multi-Objective Optimization](multi-objective.md)
- [Gradient Descent](gradient-descent.md)
