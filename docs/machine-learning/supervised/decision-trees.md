# Decision Trees

A **Decision Tree** is a supervised learning algorithm that makes predictions by learning a hierarchy of if-else rules from data. It splits the feature space recursively, making it highly interpretable and a strong baseline for both classification and regression tasks.

---

## 🌳 1. How Decision Trees Work

A tree is built top-down by repeatedly asking: *"Which feature and threshold best separates the data at this node?"*

Each internal node is a decision based on a feature. Each leaf node holds a predicted class (classification) or value (regression). The best split is chosen by maximizing **information gain** or minimizing **Gini impurity**.

---

## 📐 2. Splitting Criteria

### Gini Impurity

Measures the probability of incorrectly classifying a randomly chosen element:

```txt
Gini(t) = 1 - Σ p(c|t)²
```

- `p(c|t)` = fraction of samples at node `t` belonging to class `c`
- Gini = 0 → perfect purity (all one class)
- Gini = 0.5 → maximum impurity (two equal classes)

### Entropy (Information Gain)

```txt
Entropy(t) = -Σ p(c|t) * log2(p(c|t))
```

Information Gain of a split:

```txt
IG = Entropy(parent) - Σ (|child_k| / |parent|) * Entropy(child_k)
```

Higher IG means the split is more informative. scikit-learn defaults to Gini; both usually produce very similar trees.

---

## 📦 3. Required Libraries

```bash
pip install scikit-learn numpy pandas matplotlib
```

---

## 🔬 4. Decision Tree from Scratch (Concept)

A minimal recursive implementation to understand the mechanics:

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature    # Feature index used for split
        self.threshold = threshold  # Split threshold
        self.left      = left       # Left subtree
        self.right     = right      # Right subtree
        self.value     = value      # Leaf prediction value

class DecisionTreeScratch:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.root             = None

    def _gini(self, y: np.ndarray) -> float:
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask  = X[:, feat_idx] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                child_gini = (
                    left_mask.sum()  / n * self._gini(y[left_mask]) +
                    right_mask.sum() / n * self._gini(y[right_mask])
                )
                gain = parent_gini - child_gini
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, thresh
        return best_feat, best_thresh

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # Stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        mask  = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build(X, y, depth=0)

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.root) for row in X])


# Verify against sklearn on iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scratch_tree = DecisionTreeScratch(max_depth=5)
scratch_tree.fit(X_tr, y_tr)
print(f"Scratch accuracy: {accuracy_score(y_te, scratch_tree.predict(X_te)):.4f}")
```

---

## ⚡ 5. Decision Tree with scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Start with a reasonable depth — shallow trees are more interpretable
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Print text representation of the tree
print("\nTree structure:")
print(export_text(clf, feature_names=list(data.feature_names), max_depth=3))
```

---

## 🎨 6. Visualizing the Tree with `plot_tree`

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,       # Color nodes by majority class
    rounded=True,
    fontsize=9,
    ax=ax
)
plt.title("Decision Tree — Breast Cancer (max_depth=4)")
plt.tight_layout()
plt.show()
```

---

## 📊 7. Feature Importance

Decision trees provide built-in feature importances based on the total reduction in impurity weighted by the number of samples reaching each node (mean decrease in impurity):

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(clf.feature_importances_, index=data.feature_names)
importances_sorted = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
importances_sorted.tail(15).plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Feature Importances (Mean Decrease Impurity)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()

# Top 5 features
print("Top 5 features:")
print(importances.nlargest(5))
```

> **Note:** MDI feature importance can be biased toward high-cardinality features. For a more reliable alternative, use permutation importance from scikit-learn 1.5+.

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=42)
perm_series = pd.Series(perm.importances_mean, index=data.feature_names)
print("Top 5 by permutation importance:")
print(perm_series.nlargest(5))
```

---

## 🔧 8. Hyperparameter Tuning

Key hyperparameters that control tree complexity:

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `max_depth` | Limits tree depth — primary overfitting control | 3–15 |
| `min_samples_split` | Min samples to split an internal node | 2–50 |
| `min_samples_leaf` | Min samples required at a leaf | 1–20 |
| `max_features` | Features considered at each split (also controls overfitting) | `"sqrt"`, `"log2"`, float |
| `criterion` | Split quality measure | `"gini"`, `"entropy"`, `"log_loss"` |

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "max_depth":         [3, 5, 7, 10, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf":  [1, 5, 10],
    "criterion":         ["gini", "entropy"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params : {grid.best_params_}")
print(f"Best ROC-AUC: {grid.best_score_:.4f}")
best_clf = grid.best_estimator_
```

---

## ✂️ 9. Pruning: Preventing Overfitting

### Pre-pruning (Structural Constraints)

Setting `max_depth`, `min_samples_split`, etc., during fitting.

### Post-pruning: Cost-Complexity Pruning (sklearn 1.5+)

sklearn implements **minimal cost-complexity pruning** via the `ccp_alpha` parameter. Larger `ccp_alpha` prunes more aggressively:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Find the optimal alpha via cross-validation path
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Exclude final trivial tree

train_scores, test_scores = [], []
for alpha in ccp_alphas:
    clf_a = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf_a.fit(X_train, y_train)
    train_scores.append(clf_a.score(X_train, y_train))
    test_scores.append(clf_a.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ccp_alphas, train_scores, "o-", label="Train accuracy")
ax.plot(ccp_alphas, test_scores,  "o-", label="Test accuracy")
ax.set_xlabel("ccp_alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs ccp_alpha (Pruning)")
ax.legend()
plt.tight_layout()
plt.show()

# Pick alpha with best test accuracy
best_alpha = ccp_alphas[np.argmax(test_scores)]
pruned_clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
pruned_clf.fit(X_train, y_train)
print(f"Best alpha: {best_alpha:.5f}")
print(f"Pruned tree depth: {pruned_clf.get_depth()}")
print(f"Pruned test accuracy: {pruned_clf.score(X_test, y_test):.4f}")
```

---

## ⚠️ 10. Overfitting and When to Move On

Decision trees overfit easily when grown deep. Signs:
- Train accuracy >> Test accuracy
- Tree depth is very large (50+ nodes)

**Solutions in order of preference:**
1. Limit `max_depth` (quickest fix)
2. Apply cost-complexity pruning
3. Move to [Random Forest](random-forest.md) (ensemble of trees — almost always better)

```python
# Overfitting demonstration
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy"
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, val_scores.mean(axis=1),   label="Validation")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.title("Learning Curve — Unlimited Decision Tree")
plt.legend()
plt.show()
```

---

## 💡 Tips & Tricks

| Tip | Why It Matters |
|-----|---------------|
| Start with `max_depth=4` or `5` | Deep trees almost always overfit; start shallow and go deeper only if validation improves |
| Use `criterion="entropy"` for noisy data | Entropy can be more sensitive to subtle class structure than Gini |
| Inspect `export_text` before `plot_tree` | Text representation is faster to read for quick sanity-checks |
| Feature scaling is NOT needed | Trees split on thresholds, making them invariant to monotonic transformations |
| Prefer `permutation_importance` over `feature_importances_` | MDI is biased; permutation importance is more reliable for correlated features |
| Use `cost_complexity_pruning_path` post-training | Gives a full regularization path; better than guessing `max_depth` |
| Stratify train/test splits | Class ratios matter for reproducible evaluation |
| Single trees are interpretable — use them for explanation even when RF is used for prediction | A single shallow tree on the most important features communicates decisions to stakeholders |

---

## 🔗 Related Links

- [scikit-learn Decision Tree User Guide](https://scikit-learn.org/stable/modules/tree.html)
- [Logistic Regression](logistic-regression.md)
- [Random Forest & Gradient Boosting](random-forest.md)
- [Supervised Learning Overview](supervised-overview.md)
- [ML Lifecycle](../lifecycle.md)
