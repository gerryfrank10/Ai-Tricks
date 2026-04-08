# Linear Algebra for AI

Linear algebra is the language of machine learning. Every neural network forward pass, every PCA decomposition, every attention mechanism is fundamentally matrix math.

---

## Vectors & Matrices

```python
import numpy as np

# Vectors — 1D arrays
v = np.array([1.0, 2.0, 3.0])          # shape (3,)
w = np.array([4.0, 5.0, 6.0])

# Dot product — sum of element-wise products
dot = np.dot(v, w)                       # 1×4 + 2×5 + 3×6 = 32

# Magnitude (L2 norm)
magnitude = np.linalg.norm(v)            # √14 ≈ 3.742

# Cosine similarity — angle between vectors (used in embeddings)
cos_sim = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
print(f"Cosine similarity: {cos_sim:.4f}")  # ~0.9746

# Matrices — 2D arrays
A = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
B = np.array([[7, 8, 9], [10, 11, 12]]) # shape (2, 3)

# Matrix multiplication — CRITICAL for neural networks
C = A @ B                                # shape (3, 3)
print(C)
# Every linear layer: y = xW + b  is just matrix multiply
```

---

## Matrix Operations Every AI Practitioner Needs

### Transpose
```python
# Transpose flips rows and columns
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
A_T = A.T                               # (3, 2)

# In attention: Q @ K.T computes all pairwise scores
Q = np.random.randn(8, 64)   # 8 tokens, 64-dim queries
K = np.random.randn(8, 64)   # 8 tokens, 64-dim keys
scores = Q @ K.T              # (8, 8) attention scores
```

### Broadcasting
```python
# NumPy broadcasts shapes automatically — essential for batch ops
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
mean = X.mean(axis=0)           # (10,) — mean of each feature
std  = X.std(axis=0)            # (10,)

# Normalize all 1000 samples at once — no loop needed
X_norm = (X - mean) / (std + 1e-8)  # (1000, 10) via broadcast
```

### Element-wise vs Matrix Multiply
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise (Hadamard product)
hadamard = A * B       # [[5,12],[21,32]]

# Matrix multiplication
matmul   = A @ B       # [[19,22],[43,50]]
```

---

## Eigenvalues & Eigenvectors

An eigenvector doesn't change direction when multiplied by its matrix. This is the geometric intuition behind PCA, graph convolutions, and spectral methods.

```
Av = λv    (v = eigenvector, λ = eigenvalue)
```

```python
import numpy as np

# Covariance matrix of data
X = np.random.randn(100, 4)
cov = np.cov(X.T)                     # (4, 4)

eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by descending eigenvalue (= explained variance)
idx = eigenvalues.argsort()[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance ratio
explained = eigenvalues / eigenvalues.sum()
print("Explained variance:", np.round(explained, 3))
# e.g. [0.312, 0.268, 0.231, 0.189]
```

### PCA from Scratch (using eigen-decomposition)
```python
def pca_from_scratch(X: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce dimensionality using eigendecomposition."""
    # 1. Center the data
    X_centered = X - X.mean(axis=0)

    # 2. Compute covariance matrix
    cov = np.cov(X_centered.T)

    # 3. Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 4. Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:n_components]]

    # 5. Project
    return X_centered @ components

X_2d = pca_from_scratch(X, n_components=2)  # (100, 2)
```

---

## Singular Value Decomposition (SVD)

SVD factorizes any matrix: `A = U Σ Vᵀ`
Used in: collaborative filtering, LSA, image compression, pseudo-inverse.

```python
# Matrix A = U @ S_diag @ V.T
A = np.random.randn(100, 50)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
# U: (100, 50) — left singular vectors
# s: (50,)     — singular values (non-negative, descending)
# Vt: (50, 50) — right singular vectors transposed

# Low-rank approximation — keep top k singular values
k = 10
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
error = np.linalg.norm(A - A_approx, 'fro')
print(f"Frobenius reconstruction error (rank {k}): {error:.4f}")

# Variance explained
variance_explained = (s[:k]**2).sum() / (s**2).sum()
print(f"Variance explained: {variance_explained:.2%}")
```

### SVD for Recommendation Systems
```python
# Simple collaborative filtering via SVD
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# User-item rating matrix (sparse)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 4, 4],
    [0, 1, 5, 4],
], dtype=float)

R = csr_matrix(ratings)
U, sigma, Vt = svds(R, k=2)  # rank-2 approximation

# Reconstruct — fills in zeros (predicted ratings)
R_pred = U @ np.diag(sigma) @ Vt
print("Predicted ratings:\n", np.round(R_pred, 1))
```

---

## Matrix Norms & Regularization

```python
A = np.array([[1, -2], [3, 4]], dtype=float)

# Frobenius norm — like L2 over all elements (used in weight decay)
frob = np.linalg.norm(A, 'fro')    # √(1+4+9+16) ≈ 5.477

# Spectral norm — largest singular value (used in spectral normalization)
spectral = np.linalg.norm(A, 2)    # σ_max

# Nuclear norm — sum of singular values (used in matrix completion)
nuclear = np.linalg.norm(A, 'nuc')

# L2 regularization in ML = penalizing Frobenius norm of weight matrix
loss_with_reg = cross_entropy_loss + 0.01 * frob**2
```

---

## Solving Linear Systems

```python
# Ax = b  →  x = A⁻¹b
A = np.array([[2., 1.], [5., 7.]])
b = np.array([11., 13.])

# Direct solve (better than explicit inverse — more numerically stable)
x = np.linalg.solve(A, b)
print(f"Solution: {x}")          # [7.1818, -3.3636]
print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")  # ~1e-15

# Least squares for overdetermined systems (regression!)
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

# Analytical solution: w = (X^T X)^{-1} X^T y
w_exact, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
```

---

## Key Identities for Neural Networks

| Identity | Where Used |
|---------|-----------|
| `(AB)ᵀ = BᵀAᵀ` | Backprop gradient derivation |
| `d/dx(xᵀAx) = 2Ax` | Quadratic loss gradient |
| `tr(AB) = tr(BA)` | Covariance / Fisher information |
| `det(AB) = det(A)det(B)` | Normalizing flows |
| `(A⁻¹)ᵀ = (Aᵀ)⁻¹` | Weight matrix updates |

---

## Tips & Tricks

| Situation | Action |
|-----------|--------|
| `np.linalg.inv(A)` is unstable | Use `np.linalg.solve(A, b)` instead |
| Need fast batched matmul | Use `np.einsum` or `torch.bmm` |
| Large sparse matrices | Use `scipy.sparse` + `svds` |
| Checking numerical stability | Print condition number: `np.linalg.cond(A)` |
| Ill-conditioned matrix | Add ridge: `A + λI` before inversion |

---

## Related Topics

- [Calculus & Autodiff](calculus.md)
- [PCA & Dimensionality Reduction](../../machine-learning/unsupervised.md)
- [Transformers (Attention = QKᵀ)](../../deep-learning/transformers.md)
- [Neural Networks](../../deep-learning/neural-networks.md)
