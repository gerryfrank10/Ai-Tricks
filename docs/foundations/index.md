# Foundations

Everything AI is built on math, statistics, and clean code. These pages give you the minimum effective dose of theory with maximum practical payoff.

---

## Learning Path

```
Linear Algebra → Calculus & Autodiff → Optimization Theory
       ↓
Statistics + Probability
       ↓
Python for AI → ML Frameworks
```

## Topics

| Page | What You'll Learn |
|------|------------------|
| [Linear Algebra](mathematics/linear-algebra.md) | Vectors, matrices, dot products, eigenvalues, SVD |
| [Calculus & Autodiff](mathematics/calculus.md) | Derivatives, chain rule, backprop, PyTorch autograd |
| [Optimization Theory](mathematics/optimization.md) | Convexity, gradient descent, loss landscapes |
| [Statistics](statistics.md) | Distributions, hypothesis testing, regression |
| [Probability](probability.md) | Bayes theorem, random variables, information theory |
| [Python for AI](programming/python.md) | NumPy tricks, pandas, async, profiling |
| [ML Frameworks](programming/frameworks.md) | PyTorch, JAX, Hugging Face, scikit-learn |

## Why Foundations Matter

> "You don't need a PhD to do AI, but you do need to understand what your optimizer is doing." — Every senior ML engineer

- **Debugging**: When your loss explodes or NaNs appear, linear algebra and calculus tell you exactly why
- **Architecture choices**: Knowing eigenvalues helps you understand why batch norm works
- **Reading papers**: Most papers are just applied calculus + linear algebra + probability
- **Interview prep**: FAANG ML interviews test fundamentals heavily
