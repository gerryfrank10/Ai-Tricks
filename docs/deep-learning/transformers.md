# 🤖 Transformer Architecture

The Transformer (Vaswani et al., 2017 — "Attention Is All You Need") replaced recurrence with **self-attention**, enabling full parallelisation during training and learning long-range dependencies in a single layer. It is the foundation of every major AI system in 2025: GPT, BERT, Llama, Gemini, CLIP, ViT, Stable Diffusion, AlphaFold, and more.

---

## 📚 Table of Contents

- [Self-Attention Mechanism](#self-attention-mechanism)
- [Multi-Head Attention](#multi-head-attention)
- [Positional Encoding](#positional-encoding)
- [Transformer Block](#transformer-block)
- [Encoder-Decoder Architecture](#encoder-decoder-architecture)
- [BERT vs GPT Architectures](#bert-vs-gpt-architectures)
- [Vision Transformers (ViT)](#vision-transformers-vit)
- [FlashAttention 2](#flashattention-2)
- [PyTorch: Mini Transformer from Scratch](#pytorch-mini-transformer-from-scratch)
- [Modern Variants: Mamba / SSM (2024)](#modern-variants-mamba--ssm-2024)

---

## 🎯 Self-Attention Mechanism

Self-attention allows each token to **attend to every other token** in the sequence. Given an input matrix $X \in \mathbb{R}^{T \times d}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

- $Q$ (queries): "what am I looking for?"
- $K$ (keys): "what do I contain?"
- $V$ (values): "what do I return if selected?"
- $\sqrt{d_k}$ scaling: prevents dot products from growing too large (keeps softmax in a useful range)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,      # (batch, heads, seq, d_k)
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention — the core operation."""
    d_k     = Q.size(-1)
    scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)                               # (B, H, T, T)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    output  = torch.matmul(weights, V)                                # (B, H, T, d_v)
    return output, weights


# Quick demo
B, H, T, d_k = 2, 8, 16, 64
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)

out, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output: {out.shape}")      # (2, 8, 16, 64)
print(f"Attention weights: {attn_weights.shape}")  # (2, 8, 16, 16)

# PyTorch 2.x built-in (uses FlashAttention where available)
out_builtin = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
print(f"PyTorch built-in: {out_builtin.shape}")
```

---

## 🔀 Multi-Head Attention

Instead of one attention pass, Multi-Head Attention runs **H parallel attention heads** then concatenates:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$
$$\text{head}_i = \text{Attention}(Q W_{Q_i},\ K W_{K_i},\ V W_{V_i})$$

Each head can attend to different positions / representation subspaces.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        self.W_q      = nn.Linear(d_model, d_model, bias=False)
        self.W_k      = nn.Linear(d_model, d_model, bias=False)
        self.W_v      = nn.Linear(d_model, d_model, bias=False)
        self.W_o      = nn.Linear(d_model, d_model)
        self.dropout  = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,             # (B, T_q, d_model)
        key:   torch.Tensor,             # (B, T_k, d_model)
        value: torch.Tensor,             # (B, T_v, d_model)
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.size(0)

        Q = self._split_heads(self.W_q(query))   # (B, H, T_q, d_k)
        K = self._split_heads(self.W_k(key))     # (B, H, T_k, d_k)
        V = self._split_heads(self.W_v(value))   # (B, H, T_v, d_k)

        # Use PyTorch 2.x built-in (automatically uses FlashAttention 2 on CUDA)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T_q, d_k)

        # Merge heads: (B, H, T, d_k) → (B, T, d_model)
        attn_out  = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out)
```

---

## 📍 Positional Encoding

Transformers are **permutation-equivariant** — self-attention has no notion of order. We inject position information via:

### Sinusoidal (Original, Absolute)
$$PE_{(pos, 2i)}   = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)            # (max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)      # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                              # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# Rotary Positional Encoding (RoPE) — used by Llama, GPT-NeoX, 2023+
# RoPE applies a rotation matrix to Q and K, encoding relative positions
# It generalises better to lengths longer than seen during training
# Available via: from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
```

---

## 🏗️ Transformer Block

```python
class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),        # GELU preferred over ReLU in modern transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One encoder block: MHA + FFN with Pre-LN (modern default)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn    = FeedForward(d_model, d_ff, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN (GPT-2 style) — more stable than original Post-LN
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 🔧 Encoder-Decoder Architecture

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe    = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        x = self.pe(self.embed(src) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)   # (B, T, d_model)
```

---

## 🆚 BERT vs GPT Architectures

| Feature | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional (sees full context) | Causal / autoregressive (left-to-right only) |
| Pretraining | Masked Language Modelling (MLM) | Causal Language Modelling (CLM) / next-token prediction |
| Fine-tuning tasks | Classification, QA, NER | Generation, chat, reasoning |
| Positional encoding | Learned absolute | Learned absolute (GPT-2) / RoPE (GPT-NeoX, Llama) |
| Tokenisation | WordPiece | BPE (Byte-Pair Encoding) |
| Modern examples | BERT, RoBERTa, DeBERTa, ALBERT | GPT-4, Llama 3, Mistral, DeepSeek |

```python
# Causal (decoder-only) self-attention mask — used in GPT-style models
def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Upper-triangular mask: token i can only attend to tokens 0..i."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask   # 1 = attend, 0 = mask out


class GPTBlock(nn.Module):
    """Decoder-only transformer block with causal masking."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T    = x.size(1)
        mask = make_causal_mask(T, device=x.device)
        x    = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x    = x + self.ffn(self.norm2(x))
        return x
```

---

## 🖼️ Vision Transformers (ViT)

ViT (Dosovitskiy et al., 2020) applies the Transformer directly to image patches:

1. Split image into $P \times P$ patches (e.g., 16×16)
2. Linearly project each patch to $d_{\text{model}}$
3. Prepend a `[CLS]` token
4. Add positional embeddings
5. Feed through Transformer encoder
6. Use `[CLS]` representation for classification

```python
class PatchEmbedding(nn.Module):
    """Split image into patches and project to d_model."""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2          # 196 for 224/16
        # Conv2d with stride=patch_size is an efficient patch embed
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, d_model, H/P, W/P) → (B, n_patches, d_model)
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 n_classes=1000, d_model=768, n_heads=12, n_layers=12,
                 d_ff=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches        = (img_size // patch_size) ** 2

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.dropout     = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B  = x.size(0)
        x  = self.patch_embed(x)                              # (B, n_patches, d)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d)
        x  = torch.cat([cls, x], dim=1)                       # (B, n_patches+1, d)
        x  = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x  = self.norm(x[:, 0])      # [CLS] token representation
        return self.head(x)           # (B, n_classes)


# ViT-Base/16
vit  = ViT(img_size=224, patch_size=16, n_classes=1000, d_model=768)
imgs = torch.randn(4, 3, 224, 224)
logits = vit(imgs)
print(f"ViT output: {logits.shape}")   # (4, 1000)
print(f"ViT params: {sum(p.numel() for p in vit.parameters()):,}")
```

---

## ⚡ FlashAttention 2

Standard attention has $O(T^2)$ memory complexity (the attention matrix). **FlashAttention 2** (Dao, 2023) uses **tiling** and **kernel fusion** to compute attention in $O(T)$ memory while achieving the same output — 2–4× faster on A100s.

```python
import torch
import torch.nn.functional as F

# PyTorch 2.x automatically uses FlashAttention 2 via
# F.scaled_dot_product_attention when:
#   - Running on CUDA with PyTorch >= 2.0
#   - Head dimension is 16, 32, 64, or 128
#   - dtype is float16 or bfloat16

device = "cuda" if torch.cuda.is_available() else "cpu"
B, H, T, d_k = 2, 12, 2048, 64

Q = torch.randn(B, H, T, d_k, device=device, dtype=torch.bfloat16)
K = torch.randn(B, H, T, d_k, device=device, dtype=torch.bfloat16)
V = torch.randn(B, H, T, d_k, device=device, dtype=torch.bfloat16)

# This will use FlashAttention 2 automatically on compatible hardware
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False,
):
    out = F.scaled_dot_product_attention(Q, K, V)

print(f"FlashAttention output: {out.shape}")   # (2, 12, 2048, 64)

# Check which backend was selected
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    print("Flash attention available:", torch.backends.cuda.flash_sdp_enabled())
```

---

## 🧩 PyTorch: Mini Transformer from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MiniGPT(nn.Module):
    """A minimal GPT-style causal language model."""

    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 4    # standard ratio

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.dropout     = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_model)
        self.lm_head     = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (token embed and lm_head share weights)
        self.token_embed.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device)

        x    = self.dropout(self.token_embed(tokens) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)

        x       = self.norm(x)
        logits  = self.lm_head(x)    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: shift so we predict token[i+1] from token[i]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=0,   # ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(prompt[:, -512:])           # trim to max_len
            logits     = logits[:, -1, :] / temperature  # last-token logits

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt


# Instantiate and test
VOCAB = 50_257   # GPT-2 tokenizer vocab size
model  = MiniGPT(vocab_size=VOCAB, d_model=256, n_heads=8, n_layers=6)
tokens = torch.randint(0, VOCAB, (2, 128))
targets = tokens.clone()

logits, loss = model(tokens, targets)
print(f"Logits: {logits.shape}")  # (2, 128, 50257)
print(f"Loss:   {loss.item():.4f}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Compile for speed
compiled_model = torch.compile(model)
```

---

## 🌊 Modern Variants: Mamba / SSM (2024)

**Mamba** (Gu & Dao, 2023) uses **Selective State Space Models (SSMs)** as an alternative to attention:

- **Linear time/memory** complexity O(T) — no quadratic attention matrix
- Matches or beats Transformers on language tasks up to 1M+ tokens
- Hardware-efficient: recurrent at inference (fast), parallel at training (fast)
- Key innovation: **selective** input-dependent SSM parameters (unlike fixed HIPPO/S4)

```python
# Mamba is available via the `mamba-ssm` package (CUDA required)
# pip install mamba-ssm causal-conv1d

# Conceptual implementation of the Mamba recurrence
# (simplified; real implementation uses custom CUDA kernels)

class MambaBlock(nn.Module):
    """Simplified Mamba-style SSM block (pedagogical, not production)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                  padding=d_conv - 1, groups=d_inner)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.dt_proj  = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.A = nn.Parameter(-torch.ones(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x        = self.norm(x)

        xz       = self.in_proj(x)                   # (B, T, d_inner*2)
        x_part, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM parameters (input-dependent — "selective")
        ssm_params = self.x_proj(x_conv)
        # (Simplified — real Mamba uses ZOH discretisation)
        out = self.out_proj(x_conv * F.silu(z))

        return out + residual


# Real Mamba usage (requires mamba-ssm on CUDA):
# from mamba_ssm import Mamba
# layer = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
# y = layer(x)   # x: (B, T, d_model)
```

### Comparison: Transformer vs Mamba vs RWKV

| Property | Transformer | Mamba (SSM) | RWKV |
|---|---|---|---|
| Attention complexity | O(T²) | O(T) | O(T) |
| Infinite context | No (quadratic) | Yes (recurrent) | Yes (recurrent) |
| Training parallelism | Full | Full (parallel scan) | Partial |
| In-context learning | Excellent | Good | Good |
| GPU efficiency | FlashAttn helps | Custom CUDA kernels | Efficient |
| 2025 maturity | Dominant | Growing fast | Niche |

---

## 💎 Tips & Tricks

> **Use `F.scaled_dot_product_attention`.** PyTorch 2.x dispatches to FlashAttention 2 automatically on CUDA with float16/bfloat16. Never implement vanilla attention in production.

> **Pre-LN > Post-LN.** Layer normalisation *before* the sublayer (Pre-LN) is more training-stable. All modern models (GPT-2 onwards) use it.

> **RoPE for long contexts.** Rotary Positional Encoding generalises to longer sequences than seen during training. Prefer it for new models over sinusoidal or learned absolute PE.

> **Weight tying saves parameters.** Share input embedding weights with the output projection (`lm_head.weight = token_embed.weight`). Used by GPT-2 and most modern LLMs.

> **Gradient checkpointing for large models.** `torch.utils.checkpoint.checkpoint_sequential` recomputes activations during backward instead of storing them, trading compute for memory.

> **KV-cache for fast inference.** During autoregressive generation, cache the K and V projections for already-processed tokens to avoid recomputation. Every production serving framework does this.

---

## 🔗 Cross-References

- [Deep Learning Overview](./index.md) — architecture comparison and hardware guide
- [RNNs & LSTMs](./rnn.md) — what Transformers replaced and why
- [Transfer Learning](./transfer-learning.md) — using pretrained Transformers (BERT, Llama)
- [CNNs](./cnn.md) — compare with ViT for vision tasks
- [Linear Algebra](../foundations/mathematics/linear-algebra.md) — matrix multiplication and eigenvectors
- [Calculus & Autograd](../foundations/mathematics/calculus.md) — backprop through attention
