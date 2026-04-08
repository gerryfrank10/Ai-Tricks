# Logistic Regression

Logistic Regression is a **supervised classification algorithm** that predicts the probability that an input belongs to a given class. Despite its name, it is a classification algorithm, not a regression one. It is the go-to baseline for binary and multiclass problems before reaching for more complex models.

---

## 📐 1. The Math Behind Logistic Regression

### Sigmoid Function

Logistic Regression maps any real-valued input to a probability between 0 and 1 using the **sigmoid (logistic) function**:

```txt
σ(z) = 1 / (1 + e^(-z))
```

Where `z = w · x + b` (the linear combination of weights and features).

- When `z → +∞`, `σ(z) → 1`
- When `z → -∞`, `σ(z) → 0`
- When `z = 0`, `σ(z) = 0.5`

### Decision Boundary

```txt
Predict class 1 if σ(z) ≥ 0.5  →  z ≥ 0
Predict class 0 if σ(z) < 0.5  →  z < 0
```

### Log Loss (Binary Cross-Entropy)

The model is trained by minimizing log loss:

```txt
Loss = -(1/n) * Σ [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]
```

Where `ŷ_i` is the predicted probability and `y_i` is the true label.

---

## 📦 2. Required Libraries

```bash
pip install scikit-learn numpy pandas matplotlib seaborn imbalanced-learn
```

---

## 🔬 3. Logistic Regression from Scratch

Understanding the internals before using sklearn:

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Gradients of log loss
            dw = (1 / n_samples) * X.T @ (y_hat - y)
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# Quick demo
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegressionScratch(lr=0.1, n_iter=500)
model.fit(X_train_s, y_train)
print(f"Scratch accuracy: {accuracy_score(y_test, model.predict(X_test_s)):.4f}")
```

---

## ⚡ 4. Binary Classification with scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, log_loss, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling is critical for logistic regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train — solver='lbfgs' is the modern default for small/medium datasets
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_proba     = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Log Loss : {log_loss(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=data.target_names,
    ax=ax
)
plt.title("Confusion Matrix — Breast Cancer")
plt.tight_layout()
plt.show()
```

---

## 📈 5. ROC Curve & AUC

```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.legend()
plt.show()
```

---

## 🔢 6. Multiclass Classification (One-vs-Rest / Softmax)

scikit-learn handles multiclass automatically. For more than 2 classes use `multi_class="multinomial"` with `solver="lbfgs"` or `"saga"`:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Multinomial logistic regression (softmax)
multi_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=1.0,
    max_iter=500,
    random_state=42
)
multi_model.fit(X_train, y_train)

cv_scores = cross_val_score(multi_model, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test Accuracy: {multi_model.score(X_test, y_test):.4f}")

# Predicted probabilities — one column per class
print("Predicted class probabilities (first 3 rows):")
print(multi_model.predict_proba(X_test[:3]).round(3))
```

---

## 🔧 7. Regularization: L1 (Lasso) vs L2 (Ridge)

Regularization prevents overfitting by penalizing large weights. The parameter `C` is the **inverse of regularization strength** — smaller `C` = stronger regularization.

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# L1 (Lasso) — produces sparse models; use 'saga' or 'liblinear' solver
model_l1 = LogisticRegressionCV(
    Cs=10, cv=5, penalty="l1", solver="saga",
    max_iter=2000, random_state=42
)
model_l1.fit(X_train, y_train)
n_nonzero = (model_l1.coef_[0] != 0).sum()
print(f"L1 — Best C: {model_l1.C_[0]:.4f} | Non-zero features: {n_nonzero}")

# L2 (Ridge) — shrinks all weights; good default
model_l2 = LogisticRegressionCV(
    Cs=10, cv=5, penalty="l2", solver="lbfgs",
    max_iter=2000, random_state=42
)
model_l2.fit(X_train, y_train)
print(f"L2 — Best C: {model_l2.C_[0]:.4f} | Test Acc: {model_l2.score(X_test, y_test):.4f}")
```

| Penalty | Effect | Good When |
|---------|--------|-----------|
| L1 | Drives some weights to zero (feature selection) | Many irrelevant features |
| L2 | Shrinks all weights proportionally | Correlated features |
| ElasticNet | Combines L1 + L2 | Both cases above |

---

## ⚖️ 8. Handling Class Imbalance

When one class heavily outnumbers the other, accuracy can be misleading. Three practical strategies:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate imbalanced dataset (5% positive class)
X, y = make_classification(
    n_samples=5000, n_features=20,
    weights=[0.95, 0.05], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Strategy 1: class_weight='balanced' — fast and effective
model_balanced = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model_balanced.fit(X_train_s, y_train)
print("Balanced weights:")
print(classification_report(y_test, model_balanced.predict(X_test_s)))

# Strategy 2: SMOTE — synthetic oversampling (applied BEFORE fitting)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_s, y_train)
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_res, y_res)
y_proba = model_smote.predict_proba(X_test_s)[:, 1]
print(f"\nSMOTE ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Strategy 3: Adjust threshold (instead of default 0.5)
import numpy as np
threshold = 0.3  # Lower threshold to catch more positives
y_pred_custom = (y_proba >= threshold).astype(int)
print(f"Custom threshold ({threshold}) classification report:")
print(classification_report(y_test, y_pred_custom))
```

---

## 📊 9. Full Evaluation Pipeline

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# Build a pipeline to avoid data leakage
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    pipe, X, y, cv=cv,
    scoring=["accuracy", "roc_auc", "neg_log_loss"],
    return_train_score=True
)

for metric in ["test_accuracy", "test_roc_auc", "test_neg_log_loss"]:
    scores = results[metric]
    label  = metric.replace("test_", "").replace("neg_", "-")
    print(f"{label:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 💡 Tips & Tricks

| Tip | Why It Matters |
|-----|---------------|
| Always scale features | LR uses gradient-based optimization; unscaled features slow convergence or cause divergence |
| Start with `C=1.0`, tune via `LogisticRegressionCV` | Avoids manual grid search; cross-validates `C` automatically |
| Use `stratify=y` in train/test split | Preserves class ratios — essential with imbalanced data |
| Prefer `solver="lbfgs"` for small/medium data, `"saga"` for large or L1 | Different solvers support different penalties |
| Check `model.coef_` for interpretability | Coefficients reveal direction and magnitude of feature influence |
| Use `predict_proba` over `predict` for ranking / threshold tuning | Raw probabilities give more flexibility than hard labels |
| Log loss is a better training signal than accuracy | Accuracy ignores how confident wrong predictions are |
| `class_weight="balanced"` is the fastest fix for imbalance | Weights samples inversely proportional to class frequency |

---

## 🔗 Related Links

- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Linear Regression](linear-regression.md)
- [Decision Trees](decision-trees.md)
- [Random Forest & Gradient Boosting](random-forest.md)
- [ML Lifecycle](../lifecycle.md)
