# 🔍 Search Techniques

Search techniques are fundamental algorithms for navigating large solution spaces in optimization and problem-solving. They can be categorized into **Uninformed**, **Informed**, **Adversarial**, and **Local** search strategies.

---

## 1. **Uninformed Search (Blind Search)**

**Uninformed Search** does not use information about the goals (e.g., heuristic values); it explores the search space systematically.

### 📌 (a) Breadth-First Search (BFS)
- Explores all nodes at the current depth before going deeper.
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(f"Visited: {node}")
            visited.add(node)
            queue.extend(graph[node] - visited)

# Example usage
graph = {'A': {'B', 'C'}, 'B': {'D'}, 'C': {'E', 'F'}, 'D': {}, 'E': {}, 'F': {}}
bfs(graph, 'A')
```

---

### 📌 (b) Depth-First Search (DFS)
- Explores as far as possible along a branch before backtracking.
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(f"Visited: {start}")
    for neighbor in graph[start] - visited:
        dfs(graph, neighbor, visited)
    return visited

# Example usage
graph = {'A': {'B', 'C'}, 'B': {'D'}, 'C': {'E', 'F'}, 'D': {}, 'E': {}, 'F': {}}
dfs(graph, 'A')
```

---

### 📌 (c) Uniform-Cost Search (UCS)
- Chooses the next node to expand based on the lowest cost.
```python
import heapq

def ucs(graph, start, goal):
    queue = [(0, start)]  # Priority queue
    visited = set()

    while queue:
        cost, node = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            print(f"Node: {node}, Cost: {cost}")
            if node == goal:
                return cost
            for neighbor, weight in graph[node]:
                heapq.heappush(queue, (cost + weight, neighbor))

# Example usage (graph with weights)
graph = {'A': [('B', 1), ('C', 4)], 'B': [('D', 2)], 'C': [('E', 3)], 'D': [], 'E': []}
ucs(graph, 'A', 'E')
```

---

## 2. **Informed Search**

### 📌 (a) A* Search
- Combines the benefits of UCS and heuristics.
```python
import heapq

def a_star(graph, start, goal, heuristic):
    queue = [(0, start)]  # Priority queue
    costs = {start: 0}
    visited = set()

    while queue:
        cost, node = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            print(f"Node: {node}, Cost: {cost}")
            if node == goal:
                return cost
            for neighbor, weight in graph[node]:
                new_cost = costs[node] + weight
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(queue, (priority, neighbor))

# Heuristic function and usage
heuristic = lambda n, g: 0  # Replace with actual heuristic for specific problems
graph = {'A': [('B', 1), ('C', 4)], 'B': [('D', 2)], 'C': [('E', 3)], 'D': [], 'E': []}
a_star(graph, 'A', 'E', heuristic)
```

---

## 3. **Adversarial Search**

### 📌 (a) Minimax Algorithm
- Decision-making algorithm for zero-sum games.
```python
def minimax(depth, is_maximizing, scores, alpha=-float('inf'), beta=float('inf')):
    if depth == 0:
        return scores.pop(0)
    
    if is_maximizing:
        max_eval = -float('inf')
        for _ in range(2):  # Two child nodes
            eval = minimax(depth - 1, False, scores, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for _ in range(2):  # Two child nodes
            eval = minimax(depth - 1, True, scores, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Example usage with scores at leaf nodes
scores = [3, 5, 6, 9, 1, 2, 0, -1]
print("Optimal Score:", minimax(3, True, scores))
```

---

## 4. **Local Search**

### 📌 (a) Hill Climbing
- Iteratively improves the solution by moving to a better neighbor.
```python
import random

def hill_climbing(cost_function, neighbors_function, start):
    current = start
    while True:
        neighbors = neighbors_function(current)
        next_candidate = min(neighbors, key=cost_function)
        if cost_function(next_candidate) >= cost_function(current):
            break
        current = next_candidate
    return current

# Example usage with a simple cost function
cost = lambda x: x**2 - 4*x + 4
neighbors = lambda x: [x - 0.1, x + 0.1]
start = random.uniform(-10, 10)
optimal = hill_climbing(cost, neighbors, start)
print("Optimal Solution:", optimal)
```

---

## 5. **Advanced Optimization Techniques**

### 📌 (a) Particle Swarm Optimization
- Inspired by the social behavior of birds and fish.
📄 **Separate File**: [Particle_Swarm_Optimization](Particle_Swarm_Optimization.md)

---

### 📌 (b) Genetic Algorithm
- Mimics the process of natural selection.
📄 **Separate File**: [Genetic_Algorithm](Genetic-Algorithm.md)

---

### 📌 (c) Multi-Objective Optimization
- Simultaneously optimizes multiple conflicting objectives.
📄 **Separate File**: [Multi_Objective_Optimization](Multi_Objective_Optimization.md)

---

# ⚙️ Tricks and Guidelines

1. **Decompose Problems**: Divide problems into smaller abstract subproblems for easier computation.
2. **Parallel and Distributed Computation**: Use multiprocessing or distributed systems for large-scale search and optimization.
3. **Logging and Visualization**: Monitor optimization progress using real-time graphs or logs.
4. **Hybrid Approaches**: Combine techniques (e.g., genetic algorithms with local search refinements).

---

**NOTE**: Check the specified separate files (e.g., `[Particle_Swarm_Optimization](Particle_Swarm_Optimization.md)`) for more advanced techniques and implementations.
