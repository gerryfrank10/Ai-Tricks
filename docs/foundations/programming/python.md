# Python for AI — Practical Guide

Python is the lingua franca of AI. This page covers the idioms, libraries, and tricks that separate a productive AI practitioner from a slow one.

---

## Environment Setup (2025 Best Practices)

```bash
# uv — the fastest Python package manager (replaces pip + venv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init my-ai-project && cd my-ai-project

# Add dependencies
uv add torch torchvision transformers scikit-learn pandas numpy

# Run scripts in project venv
uv run python train.py

# Or use traditional conda for CUDA management
conda create -n ai python=3.12 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate ai
pip install transformers datasets accelerate
```

---

## NumPy Vectorization — Never Write Loops

```python
import numpy as np

# ❌ Slow Python loop
def slow_normalize(X):
    result = []
    for row in X:
        norm = sum(x**2 for x in row) ** 0.5
        result.append([x/norm for x in row])
    return result

# ✅ Fast vectorized
def fast_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)

X = np.random.randn(10000, 128)
# fast_normalize: ~0.5ms | slow: ~900ms (1800x faster!)

# Broadcasting tricks
batch_mean = X.mean(axis=0)       # (128,)
batch_std  = X.std(axis=0)        # (128,)
X_norm = (X - batch_mean) / (batch_std + 1e-8)  # (10000, 128) — auto-broadcast

# Fancy indexing
indices = np.array([0, 5, 99, 200])
subset = X[indices]               # Select specific rows

# Boolean masking
positive_mask = X[:, 0] > 0
positive_rows = X[positive_mask]  # Rows where first feature > 0

# np.einsum — expressive batched operations
# Batch matrix multiply: (B, M, K) @ (B, K, N) → (B, M, N)
A = np.random.randn(32, 8, 64)
B = np.random.randn(32, 64, 8)
C = np.einsum('bmk,bkn->bmn', A, B)   # Much clearer than reshape tricks
```

---

## Pandas Performance Tips

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("large_dataset.parquet")

# ✅ Use vectorized operations — never use .apply() for math
df["log_price"] = np.log1p(df["price"])          # fast
# df["log_price"] = df["price"].apply(np.log1p)  # slow

# ✅ Category dtype for string columns — saves 10-50x memory
df["category"] = df["category"].astype("category")
df["country"]  = df["country"].astype("category")

# ✅ Downcast numeric types
df["age"] = pd.to_numeric(df["age"], downcast="integer")   # int64 → int8
df["score"] = pd.to_numeric(df["score"], downcast="float") # float64 → float32

print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ✅ Use query() for filtering (compiled, faster on large DFs)
result = df.query("age > 25 and score > 0.8 and category == 'premium'")

# ✅ Polars for large datasets (10-100x faster than pandas)
import polars as pl

lf = pl.scan_parquet("large_dataset.parquet")   # Lazy evaluation
result = (
    lf
    .filter(pl.col("age") > 25)
    .with_columns(pl.col("price").log1p().alias("log_price"))
    .group_by("category")
    .agg(pl.col("score").mean().alias("avg_score"))
    .sort("avg_score", descending=True)
    .collect()
)
```

---

## Type Hints for ML Code

```python
from typing import Optional
import numpy as np
import torch
from pathlib import Path

def preprocess(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features and return stats for later use."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    return (X - mean) / (std + eps), mean, std


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["metadata"]
```

---

## Async for AI APIs

```python
import asyncio
import anthropic

client = anthropic.AsyncAnthropic()

async def classify_single(text: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": f"Classify sentiment: {text}\nAnswer: positive/negative/neutral"}]
    )
    return response.content[0].text.strip()

async def classify_batch(texts: list[str], concurrency: int = 10) -> list[str]:
    """Process batch with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(text):
        async with semaphore:
            return await classify_single(text)

    return await asyncio.gather(*[bounded(t) for t in texts])

# Run
texts = ["Great product!", "Terrible experience.", "It's okay."] * 100
results = asyncio.run(classify_batch(texts, concurrency=10))
```

---

## Generators & Memory Efficiency

```python
# For large datasets that don't fit in RAM
def data_generator(file_path: str, batch_size: int = 32):
    """Yield batches from a large CSV without loading all into memory."""
    chunk_iter = pd.read_csv(file_path, chunksize=batch_size)
    for chunk in chunk_iter:
        X = chunk.drop("target", axis=1).values.astype(np.float32)
        y = chunk["target"].values
        yield torch.FloatTensor(X), torch.LongTensor(y)

# PyTorch IterableDataset for streaming
from torch.utils.data import IterableDataset, DataLoader

class StreamDataset(IterableDataset):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as f:
            for line in f:
                row = line.strip().split(",")
                x = torch.tensor([float(v) for v in row[:-1]])
                y = torch.tensor(int(row[-1]))
                yield x, y

loader = DataLoader(StreamDataset("huge_file.csv"), batch_size=256, num_workers=4)
```

---

## Profiling — Find the Bottleneck

```python
# 1. Line profiler (pip install line_profiler)
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(my_training_function)
lp_wrapper = lp(my_training_function)
lp_wrapper(model, loader, optimizer)
lp.print_stats()

# 2. PyTorch profiler
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        with torch.profiler.record_function("forward"):
            pred = model(X)
        with torch.profiler.record_function("backward"):
            loss = criterion(pred, y)
            loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")  # View in chrome://tracing

# 3. Simple timing decorator
import time, functools

def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"{fn.__name__}: {(time.perf_counter()-t)*1000:.2f}ms")
        return result
    return wrapper

@timeit
def train_epoch(model, loader):
    ...
```

---

## Tips & Tricks

| Pattern | Tool | Speedup |
|---------|------|---------|
| Package management | `uv` over `pip` | 10-100x install time |
| Numeric ops | NumPy broadcasting | 100-1000x vs loops |
| Large DataFrames | Polars | 10-100x vs pandas |
| API batching | `asyncio.gather` | Linear with concurrency |
| Profiling | `torch.profiler` | Find GPU vs CPU bottlenecks |
| Reproducibility | Set all seeds | Consistent results |

```python
# Reproducibility boilerplate
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## Related Topics

- [ML Frameworks Guide](frameworks.md)
- [Data Engineering](../../data/engineering.md)
- [MLOps](../../mlops/index.md)
