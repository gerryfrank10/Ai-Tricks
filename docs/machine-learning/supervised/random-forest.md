# Random Forest & Gradient Boosting

Ensemble methods combine multiple weak learners to build a stronger model. **Random Forest** uses **bagging** (parallel independent trees), while **Gradient Boosting** (XGBoost, LightGBM) uses **boosting** (sequential correction of errors). For tabular data in 2025, LightGBM and XGBoost remain the top choices in competitions and production.

---

## 🧩 1. Bagging vs Boosting

| Aspect | Bagging (Random Forest) | Boosting (XGBoost / LightGBM) |
|--------|------------------------|-------------------------------|
| Training | Trees built in **parallel** on random bootstrap samples | Trees built **sequentially**; each corrects previous errors |
| Bias-variance | Reduces variance | Reduces bias |
| Overfitting risk | Low | Moderate (requires tuning) |
| Speed | Fast to train | Faster with LightGBM's histogram algorithm |
| Interpretability | Feature importances, OOB score | SHAP values, gain-based importance |
| Typical sweet spot | Quick baselines, noisy data | Competition-grade accuracy on tabular data |

---

## 📦 2. Required Libraries

```bash
pip install scikit-learn xgboost lightgbm optuna shap
```

---

## 🌲 3. Random Forest with scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,       # More trees → more stable, diminishing returns after ~300
    max_features="sqrt",    # sqrt(n_features) features considered at each split
    max_depth=None,         # Fully grown trees; bagging handles variance
    min_samples_leaf=1,
    oob_score=True,         # Free validation estimate using out-of-bag samples
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"OOB Score  : {rf.oob_score_:.4f}")
print(f"Test Acc   : {rf.score(X_test, y_test):.4f}")
print(f"ROC-AUC    : {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### Out-of-Bag (OOB) Score

Each tree is trained on ~63% of the data (bootstrap sample). The remaining ~37% (**out-of-bag** samples) act as a free built-in validation set. `oob_score=True` leverages this for a nearly unbiased accuracy estimate without a separate validation split.

---

## 📊 4. Feature Importance in Random Forest

```python
import matplotlib.pyplot as plt

# MDI importance (fast, built-in)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
top_features.plot(kind="barh", ax=ax, color="forestgreen")
ax.set_title("Random Forest — Top 10 Feature Importances (MDI)")
ax.set_xlabel("Mean Decrease Impurity")
plt.tight_layout()
plt.show()

# Permutation importance — more reliable with correlated features (sklearn 1.5+)
from sklearn.inspection import permutation_importance

perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=feature_names)
print("Top 5 by permutation importance:")
print(perm_series.nlargest(5).round(4))
```

---

## ⚡ 5. XGBoost

XGBoost 2.x brings improved multi-core performance and a new `device="cuda"` parameter for GPU training. The interface is fully sklearn-compatible.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.datasets import load_breast_cancer

print(f"XGBoost version: {xgb.__version__}")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    eval_metric="logloss",
    early_stopping_rounds=30,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
    # device="cuda"  # Uncomment for GPU training
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_proba = xgb_model.predict_proba(X_test)[:, 1]
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Best iteration : {xgb_model.best_iteration}")
```

---

## 🚀 6. LightGBM — 2025 Best Practice for Tabular Data

LightGBM 4.x uses a **histogram-based** algorithm and **leaf-wise** tree growth (vs. level-wise in XGBoost), making it faster and often more accurate on large datasets. It is the dominant choice in Kaggle competitions for structured data.

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print(f"LightGBM version: {lgb.__version__}")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,            # Key hyperparameter — controls model complexity
    max_depth=-1,             # -1 means no limit; num_leaves controls depth
    min_child_samples=20,     # Regularization: min data per leaf
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

y_proba = lgb_model.predict_proba(X_test)[:, 1]
print(f"LightGBM ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Best iteration   : {lgb_model.best_iteration_}")

# Native API for even more control
lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbosity": -1,
}
booster = lgb.train(
    params, lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
```

---

## 🔍 7. Hyperparameter Tuning with Optuna

Optuna is the 2025 standard for Bayesian hyperparameter optimization — faster and smarter than GridSearchCV for large search spaces.

```python
import optuna
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest ROC-AUC : {study.best_value:.4f}")
print(f"Best params  : {study.best_params}")

# Retrain best model
best_model = lgb.LGBMClassifier(**study.best_params, verbose=-1)
best_model.fit(X, y)
```

---

## 🔎 8. SHAP Values for Interpretability

SHAP (SHapley Additive exPlanations) explains individual predictions by attributing each feature's contribution. It is the gold standard for model interpretability in 2025.

```python
import shap
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# TreeExplainer is fast and exact for tree-based models
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class_0, class_1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary plot — global feature importance + direction
shap.summary_plot(sv, X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP — Global Feature Importance")
plt.tight_layout()
plt.show()

# Beeswarm — shows distribution of SHAP values per feature
shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
plt.title("SHAP Beeswarm")
plt.tight_layout()
plt.show()

# Explain a single prediction
idx = 5
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    sv[idx],
    X_test[idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot — Sample {idx}")
plt.tight_layout()
plt.show()
```

---

## 🏆 9. Putting It All Together: Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
import lightgbm as lgb
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, oob_score=False, n_jobs=-1, random_state=42),
    "XGBoost":       xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0),
    "LightGBM":      lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1),
}

print(f"{'Model':<20} {'ROC-AUC Mean':>14} {'Std':>8}")
print("-" * 44)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name:<20} {scores.mean():>14.4f} {scores.std():>8.4f}")
```

---

## 💡 Tips & Tricks

| Tip | Why It Matters |
|-----|---------------|
| Use `oob_score=True` with Random Forest | Free validation estimate without a holdout split |
| LightGBM is the default choice for tabular data in 2025 | Faster training, often better accuracy; native categorical feature support |
| Always use early stopping with XGBoost/LightGBM | Prevents overfitting without manual `n_estimators` guessing |
| Start `num_leaves=31` for LightGBM | Equivalent to `max_depth=5` level-wise; increase slowly |
| Use Optuna over GridSearchCV for > 5 hyperparameters | Bayesian optimization finds good params in far fewer trials |
| SHAP `TreeExplainer` is exact and fast for tree models | Preferred over `KernelExplainer` which is slow and approximate |
| `colsample_bytree` + `subsample` together provide strong regularization | Both add randomness independently — column and row subsampling |
| XGBoost `device="cuda"` and LightGBM `device="gpu"` for large datasets | 10–50x speedup on GPU; same API |
| Keep a Random Forest baseline | RFs are robust and rarely need much tuning — a good sanity check |

---

## 🔗 Related Links

- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost 2.x Documentation](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM 4.x Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [SHAP Library](https://shap.readthedocs.io/en/latest/)
- [Decision Trees](decision-trees.md)
- [ML Lifecycle](../lifecycle.md)
