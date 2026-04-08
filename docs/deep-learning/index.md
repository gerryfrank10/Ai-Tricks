# 🧠 Deep Learning: Introduction & Overview

Deep learning is the subfield of machine learning that uses **multi-layered neural networks** to learn representations directly from raw data. Rather than hand-crafting features, deep networks learn hierarchical abstractions layer by layer — edges → textures → shapes → objects in vision; tokens → phrases → semantics in language.

---

## 📚 Table of Contents

- [What Is Deep Learning?](#what-is-deep-learning)
- [Historical Timeline: 2012 → 2025](#historical-timeline-2012--2025)
- [Why Deep Learning Works](#why-deep-learning-works)
- [Hardware Requirements](#hardware-requirements)
- [PyTorch 2.x Fundamentals](#pytorch-2x-fundamentals)
- [torch.compile() for Speed](#torchcompile-for-speed)
- [Common Architectures Overview](#common-architectures-overview)
- [When to Use DL vs Classical ML](#when-to-use-dl-vs-classical-ml)

---

## 🤔 What Is Deep Learning?

Deep learning (DL) is a class of machine learning algorithms that:

1. Use **artificial neural networks** with many layers (hence "deep").
2. Learn feature representations **automatically** from data via backpropagation.
3. Scale in performance with **more data** and **more compute** (the scaling laws).
4. Excel at unstructured data: images, audio, text, video, point clouds.

### The Universal Approximation Theorem

A neural network with even a **single hidden layer** of sufficient width can approximate any continuous function on a compact domain. Depth makes this practically efficient — deep networks can represent exponentially more functions with polynomially more parameters than shallow ones.

```python
# Conceptual: a 2-layer network as a function approximator
import torch
import torch.nn as nn

class UniversalApproximator(nn.Module):
    """A simple 2-layer MLP demonstrating the concept."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = UniversalApproximator(input_dim=10, hidden_dim=256, output_dim=1)
x = torch.randn(32, 10)   # batch of 32
print(model(x).shape)      # torch.Size([32, 1])
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🕰️ Historical Timeline: 2012 → 2025

| Year | Milestone | Impact |
|------|-----------|--------|
| 2012 | **AlexNet** wins ImageNet (16.4% → 10.8% top-5 error) | Proved deep CNNs work at scale; GPU training era begins |
| 2014 | **GANs** (Goodfellow et al.), VGGNet, GoogLeNet | Generative modelling; deeper architectures |
| 2015 | **ResNets** (He et al.) — skip connections | Trained 152-layer networks; solved vanishing gradients |
| 2017 | **Transformer** ("Attention Is All You Need") | Replaced RNNs for sequences; foundation of modern AI |
| 2018 | **BERT**, **GPT-1** | Pretraining + fine-tuning paradigm for NLP |
| 2019 | **GPT-2**, EfficientNet | Large language models; neural scaling laws |
| 2020 | **GPT-3** (175B params), **ViT** | Few-shot learning; vision transformers rival CNNs |
| 2021 | **CLIP**, **DALL-E**, **Codex** | Multimodal and code generation models |
| 2022 | **ChatGPT**, **Stable Diffusion**, **AlphaCode** | Consumer AI; diffusion models; code synthesis |
| 2023 | **LLaMA**, **GPT-4**, **SAM**, **Mamba** | Open weights; multimodal GPT-4; state-space models |
| 2024 | **Gemini**, **Llama 3**, **Sora**, **FlashAttention 3** | Video generation; 70B+ open models; efficiency gains |
| 2025 | **DeepSeek-R1**, **Qwen3**, mixture-of-experts at scale | Reasoning models; efficient MoE; on-device inference |

---

## 💡 Why Deep Learning Works

### 1. Hierarchical Feature Learning
Each layer learns increasingly abstract representations. In a CNN:
- Layer 1: edges, colours
- Layer 2: textures, corners
- Layer 3: object parts
- Layer N: semantic concepts

### 2. The Scaling Laws
OpenAI's 2020 scaling laws paper showed that loss decreases **predictably** as a power law with:
- Model parameters (N)
- Training tokens (D)
- Compute budget (C ≈ 6ND)

This gave practitioners a principled way to allocate resources.

### 3. Inductive Biases
Different architectures bake in useful priors:
- **CNNs**: translation equivariance, local connectivity
- **RNNs**: sequential order, shared weights over time
- **Transformers**: permutation-equivariant, global attention
- **GNNs**: permutation invariance over graph nodes

### 4. Regularisation at Scale
Techniques like dropout, batch normalisation, weight decay, and data augmentation prevent overfitting even with billions of parameters.

---

## 💻 Hardware Requirements

### GPU (NVIDIA CUDA)

```python
import torch

# Check CUDA availability
print(torch.cuda.is_available())           # True if NVIDIA GPU present
print(torch.cuda.get_device_name(0))       # e.g. "NVIDIA A100-SXM4-80GB"
print(torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model and data to GPU
model = UniversalApproximator(10, 256, 1).to(device)
x = torch.randn(64, 10, device=device)
out = model(x)
```

### Apple Silicon MPS (M1/M2/M3/M4)

```python
import torch

# Metal Performance Shaders backend (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# MPS supports most PyTorch operations as of PyTorch 2.x
model = UniversalApproximator(10, 256, 1).to(device)
x = torch.randn(64, 10, device=device)
```

### TPU (Google Cloud)

```python
# TPU via torch_xla (Google Cloud / Colab TPU runtime)
# pip install torch_xla
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = UniversalApproximator(10, 256, 1).to(device)
```

### Hardware Tiers for Practitioners

| Task | Minimum | Recommended | Production |
|------|---------|-------------|------------|
| Learning DL | CPU / MPS (M1) | RTX 3090 (24 GB) | — |
| Fine-tuning (7B LLM) | RTX 3090 (24 GB) | A100 40 GB | 2× A100 80 GB |
| Training from scratch | A100 80 GB | 8× A100 | H100 cluster |
| Inference (serving) | T4 (16 GB) | A10G (24 GB) | H100 NVL |

> **Tip:** Colab T4 (free) and Kaggle P100 (free) are great starting points. For serious work, consider Lambda Labs, RunPod, or Vast.ai for cheap hourly GPU rentals.

---

## 🔥 PyTorch 2.x Fundamentals

PyTorch 2.0 (released March 2023) introduced a new compilation stack. The API is fully backward-compatible.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── 1. Tensors ──────────────────────────────────────────────────────────────
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype)     # torch.Size([2, 2]) torch.float32

# Common creation methods
zeros = torch.zeros(3, 4)
ones  = torch.ones(3, 4)
rand  = torch.rand(3, 4)     # uniform [0, 1)
randn = torch.randn(3, 4)    # standard normal

# ── 2. Autograd ─────────────────────────────────────────────────────────────
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x            # y = x³ + 2x
y.backward()
print(x.grad)                 # dy/dx = 3x² + 2 = 14.0

# ── 3. A full training loop ─────────────────────────────────────────────────
torch.manual_seed(42)

# Synthetic regression dataset
N, D_in, D_out = 1000, 20, 1
X = torch.randn(N, D_in)
y = torch.randn(N, D_out)

dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

model     = nn.Sequential(nn.Linear(D_in, 128), nn.ReLU(), nn.Linear(128, D_out))
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.MSELoss()

model.train()
for epoch in range(5):
    total_loss = 0.0
    for xb, yb in loader:
        optimiser.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

# ── 4. Mixed precision (AMP) ────────────────────────────────────────────────
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")  # or "cpu" / "mps"

# Inside training loop:
# with autocast("cuda"):
#     pred = model(xb)
#     loss = criterion(pred, yb)
# scaler.scale(loss).backward()
# scaler.step(optimiser)
# scaler.update()
```

---

## ⚡ torch.compile() for Speed

`torch.compile()` is PyTorch 2.x's JIT compiler. It uses **TorchDynamo** (graph capture) + **TorchInductor** (code generation) to produce fused CUDA kernels automatically.

```python
import torch
import torch.nn as nn
import time

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SimpleMLP().to(device)
x      = torch.randn(256, 512, device=device)

# Benchmark eager mode
model.eval()
with torch.no_grad():
    _ = model(x)  # warm-up
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model(x)
    eager_time = time.perf_counter() - t0

# Compile the model
compiled_model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
with torch.no_grad():
    _ = compiled_model(x)  # warm-up (triggers compilation)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = compiled_model(x)
    compiled_time = time.perf_counter() - t0

print(f"Eager:    {eager_time*1000:.1f} ms")
print(f"Compiled: {compiled_time*1000:.1f} ms")
print(f"Speedup:  {eager_time/compiled_time:.2f}x")
```

### torch.compile() Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `"default"` | Balanced compile time vs speedup | Most use cases |
| `"reduce-overhead"` | Minimises Python overhead via CUDA graphs | Small models, inference |
| `"max-autotune"` | Exhaustive kernel search (slow compile) | Production inference |

> **Tip:** `torch.compile()` gives 20–40% speedups on modern GPUs with almost zero code changes. It works with `autocast` and `GradScaler` too. On MPS (Apple Silicon), compile support is improving across PyTorch 2.x releases.

---

## 🏛️ Common Architectures Overview

| Architecture | Domain | Key Idea | 2025 Relevance |
|---|---|---|---|
| **MLP / FNN** | Tabular, general | Fully connected layers | Baselines, small datasets |
| **CNN** | Vision | Local receptive fields, weight sharing | Still dominant for edge/mobile |
| **ResNet / EfficientNet** | Vision | Skip connections, compound scaling | Transfer learning backbone |
| **RNN / LSTM / GRU** | Sequences | Hidden state, gates | Legacy NLP, time series |
| **Transformer** | NLP, Vision, Multi | Self-attention, positional encoding | Foundation of all modern LLMs |
| **ViT** | Vision | Image patches as tokens | Large-scale vision tasks |
| **Diffusion Model** | Generation | Iterative denoising | Image/video/audio generation |
| **GNN** | Graphs | Message passing | Molecular, social networks |
| **Mamba / SSM** | Sequences | Selective state spaces | Long-context, linear-time alt to Transformers |
| **MoE** | General | Sparse expert routing | Scale efficiently (DeepSeek, Mixtral) |

---

## 🆚 When to Use DL vs Classical ML

```python
# Decision heuristic (not a real classifier — illustrative)
def should_use_deep_learning(
    n_samples: int,
    n_features: int,
    data_type: str,          # "tabular", "image", "text", "audio", "graph"
    interpretability_needed: bool,
    compute_budget: str,     # "low", "medium", "high"
) -> str:

    if data_type in ("image", "text", "audio"):
        return "Deep Learning (CNN/Transformer/Diffusion)"

    if n_samples < 1_000:
        return "Classical ML (SVM, Random Forest, XGBoost) — insufficient data for DL"

    if interpretability_needed and data_type == "tabular":
        return "Classical ML (Decision Tree, Logistic Regression, SHAP-explained XGBoost)"

    if data_type == "tabular" and n_samples < 100_000:
        return "XGBoost / LightGBM — usually beats DL on tabular data at this scale"

    if data_type == "tabular" and n_samples >= 100_000:
        return "Try both: TabNet / FT-Transformer vs XGBoost (benchmark!)"

    if compute_budget == "low":
        return "Classical ML or lightweight DL (MobileNet, DistilBERT)"

    return "Deep Learning — large data, complex patterns, sufficient compute"
```

### Rule-of-Thumb Comparison

| Criterion | Classical ML | Deep Learning |
|---|---|---|
| Data size | Works well with < 10K samples | Shines with > 100K samples |
| Tabular data | XGBoost usually wins | FT-Transformer competitive at scale |
| Images/audio/text | Feature eng. required | Learns features automatically |
| Training time | Minutes on CPU | Hours to weeks on GPU |
| Interpretability | High (trees, linear models) | Lower (black box by default) |
| Deployment size | KB–MB | MB–GB |
| Hyperparameter tuning | Relatively easy | Requires care (LR, scheduler, etc.) |

---

## 🔗 Cross-References

- [Neural Networks Fundamentals](./neural-networks.md) — backpropagation, activations, initialisation
- [CNNs](./cnn.md) — convolutional layers, pooling, vision architectures
- [RNNs & LSTMs](./rnn.md) — sequence modelling, gating mechanisms
- [Transformers](./transformers.md) — self-attention, BERT, GPT, ViT
- [Transfer Learning](./transfer-learning.md) — pretrained models, fine-tuning, PEFT
- [Optimisation](./optimization.md) — gradient descent, Adam, learning rate schedules
- [Linear Algebra](../foundations/mathematics/linear-algebra.md) — the maths behind matrix ops in NNs
- [Calculus & Autograd](../foundations/mathematics/calculus.md) — backpropagation theory

---

## 💎 Tips & Tricks

> **Start with a pretrained model.** For 90% of tasks, fine-tuning a pretrained model beats training from scratch. See [Transfer Learning](./transfer-learning.md).

> **Normalise your inputs.** Zero-mean, unit-variance inputs (or [0,1] for images) drastically stabilise training.

> **Use AdamW, not Adam.** AdamW decouples weight decay from the gradient update, leading to better generalisation.

> **`torch.compile()` is free speedup.** Add `model = torch.compile(model)` after model definition — it's backward-compatible and gives real gains.

> **Profile before optimising.** Use `torch.profiler` to find actual bottlenecks before rewriting anything.

> **Mixed precision is almost always worth it.** `autocast("cuda")` halves memory and speeds up training by 1.5–2× on Ampere GPUs.
