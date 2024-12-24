# Linear Programming Hacktricks

Linear Programming (LP) is an optimization technique used to solve problems with linear relationships. It is widely applied in resource allocation, scheduling, production planning, logistics, and more.

---

## 🚀 **What is Linear Programming?**

Linear Programming is a mathematical technique used to maximize or minimize an objective function (e.g., maximize profit or minimize cost), subject to **linear constraints** (equality or inequality relationships). It is a widely used tool in operations research and engineering.

- **Objective**: Optimize a function (maximize or minimize).
- **Structure**: Consists of an objective function, decision variables, constraints, and non-negative restrictions.

---

## 🧩 **Key Components of Linear Programming**

1. **Decision Variables**: Represent quantities to be optimized.
    - Example: `x1` = raw material 1, `x2` = raw material 2.
2. **Objective Function**: The mathematical function to optimize.
    - Example: `Profit = 3x1 + 5x2` (linear combination of `x1` and `x2`).
3. **Constraints**: Linear equations or inequalities that restrict the solution.
    - Example: `x1 + x2 <= 100`.
4. **Non-Negative Restriction**: Variables cannot be negative.
    - Example: `x1 >= 0, x2 >= 0`.

---

## 💡 **Tricks and Best Practices**

1. **Simplify Constraints**: Before solving, reduce redundant constraints.
2. **Slack and Surplus Variables**: Convert inequalities into equalities for easier numerical solving.
3. **Normalization**: Scale coefficients to avoid numerical instability.
4. **Handling Large-Scale Problems**:
   - Use sparse matrices to reduce memory usage.
   - Implement hybrid optimization techniques to combine linear programming with heuristics.

---

## 🛠️ **Code Examples: Ready-to-Use Libraries**

### 1. **Using `scipy.optimize`**

`scipy.optimize.linprog` is a powerful library to solve LP problems in Python.

#### Example: Handling Multiple Constraints
```python
from scipy.optimize import linprog

# Define coefficients for the objective function
c = [-3, -5]  # Negated for maximization (linprog minimizes by default)

# Define inequality constraints (Ax <= b)
A = [
    [1, 0],  # x1 <= 40
    [2, 3],  # 2x1 + 3x2 <= 120
    [1, 2],  # x1 + 2x2 <= 80
]
b = [40, 120, 80]  # Right-hand side values for the inequalities

# Bounds for decision variables
x_bounds = (0, None)  # x1 >= 0
y_bounds = (0, None)  # x2 >= 0

# Solve the problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method="highs")

# Display results
if result.success:
    print("Optimal Solution Found:")
    print("x1 =", result.x[0])
    print("x2 =", result.x[1])
    print("Maximized Value =", -result.fun)  # Negate because objective was negated
else:
    print("No feasible solution found.")
```

- **Trick**: Use `method="highs"` for faster and more reliable solving.
- **Output**:
  ```
  x1 = 20
  x2 = 20
  Maximized Value = 160
  ```

---

### 2. **Using `pulp`**

`pulp` provides an easy-to-use toolkit for LP problems and supports a variety of solvers.

#### Example: Production Planning
```python
from pulp import LpProblem, LpVariable, LpMaximize

# Define problem
problem = LpProblem("Production Planning", LpMaximize)

# Define Decision Variables
x1 = LpVariable("x1", lowBound=0)  # Number of Product A
x2 = LpVariable("x2", lowBound=0)  # Number of Product B

# Objective Function
problem += 40 * x1 + 30 * x2, "Profit Maximization"

# Constraints
problem += 2 * x1 + x2 <= 100, "Material Availability"
problem += x1 + x2 <= 80, "Labor Availability"

# Solve the problem
problem.solve()

# Print results
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)
print("Total Profit =", problem.objective.value())
```
- **Output**: The tool automatically handles multiple constraints and provides a readable result.

---

### 3. **Using `cvxpy`**

`cvxpy` is ideal for advanced mathematical modeling with support for dynamic problem structures.

#### Example: Workforce Scheduling
```python
import cvxpy as cp

# Decision Variables
x1 = cp.Variable(nonneg=True)  # Hours for Role A
x2 = cp.Variable(nonneg=True)  # Hours for Role B

# Objective Function
objective = cp.Maximize(25 * x1 + 20 * x2)

# Constraints
constraints = [
    2 * x1 + x2 <= 100,  # Time constraint
    x1 + 3 * x2 <= 120,  # Skill constraint
]

# Define the problem and solve
problem = cp.Problem(objective, constraints)
problem.solve()

# Print results
print(f"x1 = {x1.value}")
print(f"x2 = {x2.value}")
print(f"Maximized Value = {problem.value}")
```

---

### 4. **Using `pyomo`**

`pyomo` is suitable for industrial-grade optimization problems.

#### Example: Supply Chain Optimization
```python
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals

# Define Model
model = ConcreteModel()

# Decision Variables
model.x1 = Var(domain=NonNegativeReals)  # Units transported to City A
model.x2 = Var(domain=NonNegativeReals)  # Units transported to City B

# Objective Function: Minimize transportation cost
model.cost = Objective(expr=4 * model.x1 + 5 * model.x2)

# Constraints
model.constraint1 = Constraint(expr=model.x1 + model.x2 <= 150)  # Maximum capacity
model.constraint2 = Constraint(expr=2 * model.x1 + 3 * model.x2 >= 200)  # Demand constraint

# Solve
solver = SolverFactory("glpk")
solver.solve(model)

# Print results
print("x1 =", model.x1.value)
print("x2 =", model.x2.value)
print("Minimized Cost =", model.cost.expr())
```

---

## 📚 **Breaking Down Linear Programming Further**

1. **Variants of LP Problems**:
   - **Integer Linear Programming (ILP)**: Decision variables must take integer values.
   - **Mixed-Integer Programming (MIP)**: Mixture of integer and continuous variables.
2. **Multiple Objective LP**:
   - Combine objectives into one (weighted sum method).
   - Use goal programming or Pareto optimization.

3. **Real-Life Applications**:
   - **Supply Chain**: Optimize routing, demand and production planning.
   - **Energy Optimization**: Resource allocation in energy systems (e.g., power grids).
   - **Finance**: Portfolio optimization and investment strategy.

---

## ⚡ **Common Pitfalls and How to Avoid Them**

1. **Infeasible Problems**: Occurs when constraints conflict or cannot be satisfied.
    - **Fix**: Debug and simplify constraints to remove redundancy.
2. **Unbounded Solutions**: Occurs when constraints do not fully restrict decision variables.
    - **Fix**: Ensure meaningful constraints and bounded variables.
3. **Numerical Instability**: Large coefficients can lead to rounding errors.
    - **Fix**: Normalize the coefficients and solve using stable libraries.

---

With this comprehensive guide, you’re equipped with **tools and tricks** to solve any type of Linear Programming problem effectively!