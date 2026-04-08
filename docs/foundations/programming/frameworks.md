# ML Frameworks Guide (2025)

The AI/ML framework landscape has stabilized around a clear hierarchy. Here's an opinionated guide to picking and using the right tool.

---

## Framework Selection Guide

| Use Case | Recommended | Why |
|----------|------------|-----|
| Deep learning research | **PyTorch 2.x** | Dynamic graphs, Pythonic, ecosystem |
| Production inference | **ONNX + TensorRT** | Hardware optimization, portability |
| TPU / JAX research | **JAX + Flax/NNX** | Functional, XLA compilation, speed |
| Classical ML | **scikit-learn** | Stable API, comprehensive algorithms |
| Tabular data | **LightGBM / XGBoost** | Best-in-class for structured data |
| LLM inference | **vLLM / llama.cpp** | PagedAttention, quantization |
| Data processing | **Polars** | Rust-backed, 10-100x pandas |
| Distributed ML | **Ray / DeepSpeed** | Multi-GPU/node training |
| LLM applications | **Hugging Face ecosystem** | Models, datasets, training |

---

## PyTorch 2.x — The Research Standard

```python
import torch
import torch.nn as nn
import torch.compile  # NEW in 2.0

# torch.compile() — 2x speedup with one line
model = MyModel()
model = torch.compile(model)      # Compiles to optimized kernels

# Device-agnostic code
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()  # Apple Silicon
    else "cpu"
)
model = model.to(device)

# Mixed precision (AMP) — 2x memory, 1.5-2x speed
from torch.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Flash Attention 2 (included in PyTorch 2.2+)
# Enable via: model = nn.MultiheadAttention(..., batch_first=True)
# With scaled_dot_product_attention using Flash under the hood:
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = nn.functional.scaled_dot_product_attention(q, k, v)
```

---

## Hugging Face Ecosystem

The de-facto standard for working with LLMs and NLP.

```python
# transformers — load any model in 3 lines
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",             # Auto-shard across GPUs
    attn_implementation="flash_attention_2",
)

# pipeline — highest-level API
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
result = pipe("Explain transformers in one sentence:", max_new_tokens=100)

# datasets — efficient data loading
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
ds = ds.map(tokenize_fn, batched=True, num_proc=8)
ds = ds.filter(lambda x: len(x["input_ids"]) > 128)

# Trainer API — full-featured training loop
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## scikit-learn Pipelines — The Right Way

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

# Define feature groups
numeric_features = ["age", "income", "credit_score"]
categorical_features = ["occupation", "city", "marital_status"]

# Sub-pipelines for each type
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Full pipeline: preprocess → model
clf = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200, max_depth=5)),
])

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Fit and persist
clf.fit(X_train, y_train)
import joblib
joblib.dump(clf, "model.joblib")
```

---

## LightGBM & XGBoost (2025 Best for Tabular)

```python
import lightgbm as lgb
import xgboost as xgb
import optuna

# LightGBM — faster training, better on large data
dtrain_lgb = lgb.Dataset(X_train, y_train)
dval_lgb   = lgb.Dataset(X_val,   y_val, reference=dtrain_lgb)

params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "device_type": "gpu",     # GPU acceleration
}

model_lgb = lgb.train(
    params,
    dtrain_lgb,
    num_boost_round=2000,
    valid_sets=[dval_lgb],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# XGBoost 2.x with GPU
model_xgb = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="auc",
    device="cuda",            # GPU
    tree_method="hist",       # Fastest
)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = lgb.LGBMClassifier(**params, early_stopping_rounds=50)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=4)
print(f"Best AUC: {study.best_value:.4f}")
```

---

## JAX — Functional ML at Scale

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap

# JAX: functional transformations on NumPy-like arrays
def model(params, x):
    W, b = params
    return jnp.dot(x, W) + b

def loss(params, x, y):
    pred = model(params, x)
    return jnp.mean((pred - y)**2)

# grad: automatic differentiation
grad_fn = grad(loss)       # Gradient w.r.t. params

# jit: compile to XLA (massive speedup)
loss_jit = jit(loss)
grad_jit  = jit(grad_fn)

# vmap: vectorize over batch dimension
batched_predict = vmap(lambda p, x: model(p, x), in_axes=(None, 0))

# pmap: parallelize across devices (multi-GPU/TPU)
parallel_train_step = pmap(train_step, axis_name="devices")

# Example: training loop
import optax   # Optimizers for JAX

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, x, y):
    loss_val, grads = jax.value_and_grad(loss)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val

for X_batch, y_batch in loader:
    params, opt_state, loss_val = train_step(params, opt_state, X_batch, y_batch)
```

---

## Ray — Distributed ML

```python
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

ray.init()

# Distributed hyperparameter search
def train_fn(config):
    model = build_model(config["hidden_size"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(10):
        loss = train_epoch(model, optimizer)
        tune.report(loss=loss)

analysis = tune.run(
    train_fn,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_size": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
    },
    num_samples=50,
    resources_per_trial={"cpu": 4, "gpu": 1},
    scheduler=tune.schedulers.ASHAScheduler(metric="loss", mode="min"),
)
```

---

## Tips & Tricks

| Scenario | Recommendation |
|----------|---------------|
| Starting a new project | PyTorch + uv + Hugging Face |
| Tabular competition | LightGBM + Optuna |
| LLM fine-tuning | Hugging Face TRL + PEFT |
| Serving LLMs | vLLM (open) or Anthropic API (managed) |
| Multi-GPU training | DeepSpeed ZeRO-3 or FSDP |
| Reproducibility | Pin all versions in `pyproject.toml` |

---

## Related Topics

- [Python for AI](python.md)
- [Deep Learning](../../deep-learning/index.md)
- [Fine-Tuning LLMs](../../llm/fine-tuning.md)
- [MLOps & Deployment](../../mlops/index.md)
