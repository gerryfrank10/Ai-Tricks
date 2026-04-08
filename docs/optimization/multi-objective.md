# Multi-Objective Optimization

Real ML problems involve trade-offs: accuracy vs latency, precision vs recall, model size vs performance. Multi-objective optimization (MOO) finds all optimal trade-offs simultaneously.

---

## The Pareto Front

Instead of one "best" solution, MOO produces a **Pareto front** — the set of solutions where you can't improve one objective without worsening another.

```
         ↑ Accuracy
         |      ← Pareto Front
     *   *   *
   *         * ← Dominated (another solution is better on both objectives)
         |
         +──────────────→ Latency (lower is better)
```

A solution is **Pareto optimal** (non-dominated) if no other solution is better on ALL objectives simultaneously.

---

## NSGA-II Algorithm

```python
import numpy as np
from typing import Callable

class NSGA2:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II
    The standard algorithm for multi-objective optimization.
    """

    def __init__(
        self,
        objectives: list[Callable],  # List of functions to minimize
        n_dims: int,
        bounds: list[tuple[float, float]],
        pop_size: int = 100,
        n_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.n_dims = n_dims
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all objectives for all individuals."""
        return np.array([[f(ind) for f in self.objectives] for ind in population])

    def non_dominated_sort(self, scores: np.ndarray) -> list[list[int]]:
        """Sort population into Pareto fronts."""
        n = len(scores)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if all(scores[i] <= scores[j]) and any(scores[i] < scores[j]):
                    dominated_solutions[i].append(j)   # i dominates j
                elif all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                    domination_count[i] += 1            # j dominates i

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def crowding_distance(self, scores: np.ndarray, front: list[int]) -> np.ndarray:
        """Compute crowding distance for diversity preservation."""
        n = len(front)
        distances = np.zeros(n)

        for m in range(self.n_objectives):
            sorted_idx = sorted(range(n), key=lambda i: scores[front[i], m])
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf

            obj_range = scores[front[sorted_idx[-1]], m] - scores[front[sorted_idx[0]], m]
            if obj_range == 0:
                continue

            for i in range(1, n-1):
                distances[sorted_idx[i]] += (
                    scores[front[sorted_idx[i+1]], m] - scores[front[sorted_idx[i-1]], m]
                ) / obj_range

        return distances

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        """Simulated Binary Crossover (SBX)."""
        if np.random.rand() > self.cx_prob:
            return p1.copy(), p2.copy()

        eta = 20   # Distribution index
        u = np.random.rand(self.n_dims)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1/(eta+1)),
                        (1 / (2 * (1-u))) ** (1/(eta+1)))

        child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        result = individual.copy()
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        eta_m = 20

        for i in range(self.n_dims):
            if np.random.rand() < self.mut_prob:
                delta1 = (individual[i] - lb[i]) / (ub[i] - lb[i])
                delta2 = (ub[i] - individual[i]) / (ub[i] - lb[i])
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2*u + (1-2*u)*(1-delta1)**(eta_m+1))**(1/(eta_m+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta2)**(eta_m+1))**(1/(eta_m+1))
                result[i] = np.clip(individual[i] + delta_q * (ub[i] - lb[i]), lb[i], ub[i])

        return result

    def optimize(self) -> dict:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        population = lb + np.random.rand(self.pop_size, self.n_dims) * (ub - lb)

        for gen in range(self.n_generations):
            scores = self.evaluate(population)
            fronts = self.non_dominated_sort(scores)

            # Create offspring
            offspring = []
            while len(offspring) < self.pop_size:
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                child1, child2 = self.crossover(population[idx1], population[idx2])
                offspring.extend([self.mutate(child1), self.mutate(child2)])

            # Combine parent + offspring
            combined = np.vstack([population, offspring[:self.pop_size]])
            combined_scores = self.evaluate(combined)
            combined_fronts = self.non_dominated_sort(combined_scores)

            # Select next generation
            next_pop = []
            for front in combined_fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend(front)
                else:
                    remaining = self.pop_size - len(next_pop)
                    crowd_dist = self.crowding_distance(combined_scores, front)
                    sorted_by_crowd = sorted(range(len(front)),
                                             key=lambda i: crowd_dist[i], reverse=True)
                    next_pop.extend([front[i] for i in sorted_by_crowd[:remaining]])
                    break

            population = combined[next_pop]

            if (gen + 1) % 50 == 0:
                pareto_size = len(combined_fronts[0])
                print(f"Gen {gen+1:3d} | Pareto front size: {pareto_size}")

        # Return Pareto-optimal solutions
        final_scores = self.evaluate(population)
        final_fronts = self.non_dominated_sort(final_scores)
        pareto_idx   = final_fronts[0]

        return {
            "pareto_solutions": population[pareto_idx],
            "pareto_scores":    final_scores[pareto_idx],
        }
```

---

## Real ML Use Case: Accuracy vs Latency

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import time

X, y = load_breast_cancer(return_X_y=True)

def accuracy_objective(params: np.ndarray) -> float:
    """Minimize 1 - accuracy."""
    model = RandomForestClassifier(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        n_jobs=-1, random_state=42
    )
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    return 1 - scores.mean()

def latency_objective(params: np.ndarray) -> float:
    """Minimize inference latency (ms per 100 predictions)."""
    model = RandomForestClassifier(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        n_jobs=1, random_state=42
    )
    model.fit(X, y)
    start = time.perf_counter()
    for _ in range(100):
        model.predict(X[:100])
    return (time.perf_counter() - start) * 1000

nsga2 = NSGA2(
    objectives=[accuracy_objective, latency_objective],
    n_dims=2,
    bounds=[(10, 500), (2, 20)],   # [n_estimators, max_depth]
    pop_size=50,
    n_generations=30,
)
result = nsga2.optimize()

print("Pareto Front (Accuracy vs Latency):")
print(f"{'n_est':>6} {'depth':>6} {'1-Acc':>8} {'Latency(ms)':>12}")
for sol, score in zip(result["pareto_solutions"], result["pareto_scores"]):
    print(f"{int(sol[0]):>6} {int(sol[1]):>6} {score[0]:.4f}   {score[1]:>10.2f}")
```

---

## Optuna Multi-Objective (Simpler API)

```python
import optuna

def multi_obj_optuna(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth    = trial.suggest_int("max_depth", 2, 20)
    learning_rate = trial.suggest_float("lr", 1e-3, 0.3, log=True)

    # Objective 1: validation accuracy (maximize → minimize negative)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    accuracy = scores.mean()

    # Objective 2: model complexity (minimize for interpretability)
    complexity = n_estimators * max_depth

    return 1 - accuracy, complexity   # Both minimized

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)
study.optimize(multi_obj_optuna, n_trials=100, n_jobs=4)

# Get Pareto front
pareto_trials = study.best_trials
print(f"\nPareto front: {len(pareto_trials)} solutions")
for t in sorted(pareto_trials, key=lambda t: t.values[0]):
    print(f"  Accuracy: {1-t.values[0]:.4f} | Complexity: {t.values[1]:.0f}")
```

---

## Visualizing the Pareto Front

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_front(pareto_scores: np.ndarray, all_scores: np.ndarray = None):
    fig, ax = plt.subplots(figsize=(8, 6))

    if all_scores is not None:
        ax.scatter(all_scores[:, 0], all_scores[:, 1],
                   c="lightgray", s=30, alpha=0.5, label="All solutions")

    # Sort Pareto front for line plot
    sorted_idx = pareto_scores[:, 0].argsort()
    pareto_sorted = pareto_scores[sorted_idx]

    ax.scatter(pareto_scores[:, 0], pareto_scores[:, 1],
               c="#7c4dff", s=80, zorder=5, label="Pareto optimal")
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
            "--", color="#7c4dff", alpha=0.7)

    ax.set_xlabel("Objective 1 (1 - Accuracy)")
    ax.set_ylabel("Objective 2 (Model Complexity)")
    ax.set_title("Pareto Front: Accuracy vs Complexity")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=150)
```

---

## Tips & Tricks

| Issue | Solution |
|-------|---------|
| Slow convergence | Increase pop_size to 200+ |
| Poor diversity | Add crowding distance selection |
| Many objectives (>3) | Use NSGA-III or MOEA/D |
| Fast evaluation | Use Optuna with TPE sampler |
| Need interpretable trade-offs | Use Optuna visualization tools |

---

## Related Topics

- [Genetic Algorithms](genetic-algorithms.md)
- [Particle Swarm Optimization](particle-swarm.md)
- [Hyperparameter Tuning](../machine-learning/lifecycle.md)
