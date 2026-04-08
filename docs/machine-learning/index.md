# Machine Learning

Machine Learning (ML) is the discipline of building systems that learn from data — improving their performance on tasks without being explicitly programmed with rules. It is a subset of artificial intelligence and the backbone of modern data products, from recommendation engines to fraud detection to medical diagnostics.

---

## 🤖 1. What is Machine Learning?

At its core, ML is about finding patterns in data and using those patterns to make predictions or decisions. A model is trained by showing it many examples (`X → y`), adjusting internal parameters until it generalizes well to unseen data.

### The ML Flywheel

```txt
Data → Features → Model → Predictions → Feedback → Better Data
```

Three fundamental ingredients:
1. **Data** — labeled or unlabeled examples
2. **Algorithm** — the learning procedure
3. **Evaluation** — a metric that defines what "good" means

---

## 🗺️ 2. Taxonomy of Machine Learning

### Supervised Learning

The training data includes labels (ground truth outputs). The model learns to map inputs to outputs.

- **Classification**: Predicts a discrete category
  - Binary: spam vs. not spam
  - Multiclass: image class out of 1000 categories
- **Regression**: Predicts a continuous value
  - House prices, demand forecasting, stock returns

### Unsupervised Learning

No labels. The model discovers structure in the data on its own.

- **Clustering**: Groups similar samples (K-Means, DBSCAN, Hierarchical)
- **Dimensionality Reduction**: Compresses features while retaining information (PCA, UMAP, t-SNE)
- **Anomaly Detection**: Identifies outliers (Isolation Forest, One-Class SVM)
- **Association Rules**: Market basket analysis (Apriori)

### Reinforcement Learning (RL)

An agent learns a policy by interacting with an environment, receiving rewards or penalties for actions.

- Applications: game playing (AlphaGo), robotics, recommendation systems
- Key concepts: state, action, reward, policy, Q-learning, PPO

### Self-Supervised & Foundation Models (2025)

Large models trained on massive unlabeled corpora with proxy tasks (e.g., predicting the next token). The resulting representations transfer to downstream tasks with little labeled data.

---

## 📊 3. Algorithm Selection Guide

Use this table as a starting checklist. Always establish a simple baseline first, then escalate complexity if needed.

| Task | Dataset Size | Recommended Starting Point | Notes |
|------|-------------|---------------------------|-------|
| Binary classification | Small–Medium (< 100k) | Logistic Regression → Random Forest | LR is interpretable; RF is robust |
| Binary classification | Large (> 100k) | LightGBM / XGBoost | Tune with Optuna; add SHAP for explainability |
| Multiclass classification | Any | LightGBM / RandomForest | native multiclass support |
| Regression | Small–Medium | Linear Regression → Ridge/Lasso | Add polynomial features if nonlinear |
| Regression | Large | LightGBM / XGBoost | Use `objective="regression"` |
| Text classification | Any | TF-IDF + LogReg → fine-tune BERT | Fine-tuning often outperforms by 5–15% |
| Image classification | Any | Pre-trained CNN (ResNet, EfficientNet) | Transfer learning from ImageNet |
| Tabular data (competition) | Any | LightGBM + Optuna + SHAP | Dominant approach in 2024–2025 |
| Clustering | Any | K-Means (known k), HDBSCAN (unknown k) | Visualize with UMAP first |
| Anomaly detection | Any | Isolation Forest → Autoencoder | Autoencoder for high-dim data |
| Time series forecast | Any | Prophet → LSTM → PatchTST (2025) | PatchTST is state-of-the-art for long horizon |
| Reinforcement learning | Any | Stable-Baselines3 (PPO / SAC) | Start in simulation, then sim-to-real |

---

## 🛠️ 4. The Standard ML Workflow

```python
# Minimal reproducible workflow template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

# 1. Load data
df = pd.read_csv("data.csv")
X  = df.drop("target", axis=1)
y  = df["target"]

# 2. Split — always stratify for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipeline (prevents data leakage)
model = lgb.LGBMClassifier(n_estimators=300, verbose=-1, random_state=42)

# 4. Cross-validate on training data only
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# 5. Final evaluation on held-out test set
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, model.predict(X_test)))
```

---

## ⚠️ 5. Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Data leakage** | Test data information used during training (e.g., scaling on full dataset) | Always fit transformers on train set only; use `Pipeline` |
| **Not stratifying splits** | Class ratios differ between train/test | Use `stratify=y` in `train_test_split` |
| **Evaluating on training data** | Model memorizes training labels; inflated metrics | Always evaluate on a held-out test set or use cross-validation |
| **Ignoring class imbalance** | Accuracy can be 95% while minority class recall is 0% | Use `class_weight="balanced"`, SMOTE, or ROC-AUC as metric |
| **Wrong metric for the task** | Accuracy is misleading for imbalanced data | Use ROC-AUC, F1, or PR-AUC for imbalanced; RMSE for regression |
| **Feature scaling omission** | LR, SVM, KNN degrade without scaling | Use `StandardScaler` or `MinMaxScaler` inside a pipeline |
| **Overfitting** | Model performs well on train but not test | Regularization, cross-validation, simpler model, more data |
| **P-hacking / multiple comparisons** | Tuning on test set causes overfitting to test | Keep test set locked; tune on validation set only |
| **Ignoring temporal structure** | Random splits leak future into past for time-series | Use `TimeSeriesSplit` for temporal data |
| **Treating missing values carelessly** | Imputing with test stats before split | Fit imputer on train, transform both train and test |

---

## 📐 6. Key Evaluation Metrics Quick Reference

```python
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error,  # sklearn 1.5+
    r2_score, average_precision_score
)

# Classification
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
roc_auc   = roc_auc_score(y_test, y_proba)
pr_auc    = average_precision_score(y_test, y_proba)  # Better than ROC for imbalanced

# Regression
mae   = mean_absolute_error(y_test, y_pred)
rmse  = root_mean_squared_error(y_test, y_pred)  # Direct function in sklearn 1.5+
r2    = r2_score(y_test, y_pred)
```

---

## 📖 7. Sub-Pages: Supervised Learning

Explore the individual algorithm pages for full implementations, math, and advanced techniques:

| Algorithm | Best For | Page |
|-----------|----------|------|
| Logistic Regression | Binary/multiclass classification, interpretable baseline | [logistic-regression](supervised/logistic-regression.md) |
| Linear Regression | Continuous target prediction | [linear-regression](supervised/linear-regression.md) |
| Decision Trees | Interpretable rules, small datasets | [decision-trees](supervised/decision-trees.md) |
| Random Forest & Gradient Boosting | Tabular data, production models | [random-forest](supervised/random-forest.md) |
| Support Vector Machines | High-dimensional, small-medium datasets | [svm](supervised/svm.md) |
| Supervised Learning Overview | Workflow, evaluation, cross-validation | [supervised-overview](supervised/supervised-overview.md) |

---

## 🔄 8. Further Sections in This Guide

| Section | Description | Link |
|---------|-------------|------|
| ML Lifecycle | End-to-end process: data → deployment → monitoring, AutoML, MLflow | [lifecycle](lifecycle.md) |
| Unsupervised Learning | Clustering, dimensionality reduction, anomaly detection | [unsupervised](unsupervised.md) |
| Deep Learning | Neural networks, CNNs, optimization | [deep-learning](../deep-learning/neural-networks.md) |
| NLP | Text preprocessing, sentiment analysis | [nlp](../nlp/text-preprocessing.md) |
| LLMs | Prompt engineering, RAG, fine-tuning, agents | [llm](../llm/prompt-engineering.md) |
| MLOps | Model deployment, monitoring, CI/CD for ML | [mlops](../mlops/index.md) |
| Data | EDA, feature engineering, visualization | [data](../data/eda.md) |
| Foundations | Statistics and probability for ML | [foundations](../foundations/statistics.md) |

---

## 💡 Tips & Tricks

| Tip | Why It Matters |
|-----|---------------|
| Build the simplest possible baseline first | Establishes a floor; prevents over-engineering before you know the difficulty of the task |
| Use `Pipeline` for every workflow | Prevents leakage and makes deployment trivial |
| `LightGBM` is the 2025 default for tabular data | Handles missing values natively, fast, state-of-the-art accuracy |
| Cross-validate, never evaluate on train | Cross-validation gives reliable generalization estimates |
| ROC-AUC is a better default than accuracy | Threshold-invariant; meaningful even with class imbalance |
| Tune with Optuna not GridSearchCV | Bayesian search finds good hyperparameters in fewer trials |
| SHAP values explain any model | Global and local explanations; works with tree models, neural networks, and linear models |
| Lock the test set; never tune on it | Looking at the test set — even once — causes overfitting |
| Log experiments with MLflow | Reproducibility and comparison across runs |

---

## 🔗 Related Links

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [Optuna](https://optuna.org/)
- [SHAP](https://shap.readthedocs.io/en/latest/)
- [MLflow](https://mlflow.org/)
- [Getting Started](../getting-started.md)
