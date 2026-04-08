# 🔁 Recurrent Neural Networks, LSTMs, GRUs & Modern Sequence Models

Recurrent Neural Networks (RNNs) are architectures designed for **sequential data** — they maintain a hidden state that carries information across time steps, making them natural for text, time series, speech, and video.

---

## 📚 Table of Contents

- [The Vanilla RNN Cell](#the-vanilla-rnn-cell)
- [Vanishing & Exploding Gradients](#vanishing--exploding-gradients)
- [LSTM: Long Short-Term Memory](#lstm-long-short-term-memory)
- [GRU: Gated Recurrent Unit](#gru-gated-recurrent-unit)
- [Bidirectional RNNs](#bidirectional-rnns)
- [Sequence-to-Sequence (Seq2Seq)](#sequence-to-sequence-seq2seq)
- [Attention Mechanism (Pre-Transformer)](#attention-mechanism-pre-transformer)
- [PyTorch Implementation](#pytorch-implementation)
- [When to Use RNNs vs Transformers](#when-to-use-rnns-vs-transformers)

---

## 📐 The Vanilla RNN Cell

An RNN processes a sequence $x_1, x_2, \ldots, x_T$ by updating a hidden state at each step:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$
$$y_t = W_{hy} \cdot h_t + b_y$$

- $h_t \in \mathbb{R}^H$ — hidden state (memory)
- $x_t \in \mathbb{R}^D$ — input at step $t$
- $W_{hh}, W_{xh}, W_{hy}$ — weight matrices shared across all time steps

```python
import torch
import torch.nn as nn

# Manual RNN cell to understand the math
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined weight matrix [W_xh | W_hh] for efficiency
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_size), h: (batch, hidden_size)
        return torch.tanh(self.W_xh(x) + self.W_hh(h))

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)


# Process a sequence manually
cell   = VanillaRNNCell(input_size=10, hidden_size=64)
seq    = torch.randn(5, 32, 10)  # (T, batch, features)
h      = cell.init_hidden(32)

hidden_states = []
for t in range(seq.size(0)):
    h = cell(seq[t], h)
    hidden_states.append(h)

print(f"Final hidden state shape: {h.shape}")  # (32, 64)

# PyTorch built-in — much faster
rnn = nn.RNN(input_size=10, hidden_size=64, num_layers=2,
             batch_first=True, dropout=0.1)
x   = torch.randn(32, 5, 10)   # (batch, seq_len, features)
out, h_n = rnn(x)
print(f"Output: {out.shape}")   # (32, 5, 64)
print(f"h_n:    {h_n.shape}")   # (num_layers, batch, hidden)
```

---

## ⚠️ Vanishing & Exploding Gradients

During backpropagation through time (BPTT), gradients flow through repeated matrix multiplications:

$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each factor is $W_{hh}^T \cdot \text{diag}(\tanh'(h_{t-1}))$.

- If $\|W_{hh}\| < 1$: gradients **vanish** (network forgets long-range dependencies)
- If $\|W_{hh}\| > 1$: gradients **explode** (training diverges)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Visualise vanishing gradients
def simulate_gradient_flow(W_norm: float, T: int = 50) -> list:
    """Simulate how gradient magnitude changes over T timesteps."""
    grad_magnitude = 1.0
    magnitudes = [grad_magnitude]
    for _ in range(T):
        # Each step multiplies by W_norm * tanh_derivative (≈ 0.5 on average)
        grad_magnitude *= W_norm * 0.5
        magnitudes.append(grad_magnitude)
    return magnitudes

T = 50
vanishing = simulate_gradient_flow(0.9, T)
exploding = simulate_gradient_flow(2.0, T)

# Exploding gradient fix: gradient clipping
def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Standard practice: clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# In training loop:
# loss.backward()
# clip_gradients(model, max_norm=1.0)
# optimiser.step()
```

---

## 🔒 LSTM: Long Short-Term Memory

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem with **gating mechanisms** and a separate **cell state** that flows with minimal modification.

### Gate Equations

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$   ← **Forget gate**: what to erase from cell
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$   ← **Input gate**: what new info to write
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$   ← **Candidate cell**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$   ← **Cell state update**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$   ← **Output gate**
$$h_t = o_t \odot \tanh(c_t)$$   ← **Hidden state**

The key insight: gradients flow through $c_t$ without passing through nonlinearities repeatedly (the "highway" through time).

```python
# Manual LSTM cell for understanding
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Pack all gates into one matrix multiply for efficiency
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        combined = torch.cat([x, h], dim=1)       # (batch, input+hidden)
        all_gates = self.gates(combined)            # (batch, 4*hidden)

        # Split into 4 gates
        f, i, g, o = all_gates.chunk(4, dim=1)
        f = torch.sigmoid(f)                       # forget gate
        i = torch.sigmoid(i)                       # input gate
        g = torch.tanh(g)                          # candidate
        o = torch.sigmoid(o)                       # output gate

        c_new = f * c + i * g                      # cell state
        h_new = o * torch.tanh(c_new)              # hidden state
        return h_new, (h_new, c_new)

    def init_state(self, batch_size: int, device=None):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h, c)


# PyTorch built-in LSTM
lstm = nn.LSTM(
    input_size=64,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=False,
)

x          = torch.randn(32, 100, 64)   # (batch, seq_len, features)
out, (h_n, c_n) = lstm(x)
print(f"Output:     {out.shape}")       # (32, 100, 256)
print(f"h_n (final hidden):  {h_n.shape}")  # (num_layers, batch, hidden)
print(f"c_n (final cell):    {c_n.shape}")  # (num_layers, batch, hidden)
```

---

## 🚪 GRU: Gated Recurrent Unit

GRUs (Cho et al., 2014) simplify LSTMs by **merging** the cell and hidden states and using only two gates:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$   ← **Update gate** (like forget + input combined)
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$   ← **Reset gate**
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$   ← **Candidate**
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$   ← **Output**

```python
# GRU vs LSTM: parameter comparison
def count_params(module):
    return sum(p.numel() for p in module.parameters())

lstm_model = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
gru_model  = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

print(f"LSTM params: {count_params(lstm_model):,}")   # ~1.3M (4 gates)
print(f"GRU  params: {count_params(gru_model):,}")    # ~985K  (3 gate matrices)

# GRU is ~75% the size of LSTM — often similar performance
gru = nn.GRU(input_size=64, hidden_size=256, num_layers=1, batch_first=True)
x   = torch.randn(32, 50, 64)
out, h_n = gru(x)
print(f"GRU output: {out.shape}")   # (32, 50, 256)
```

---

## ↔️ Bidirectional RNNs

Standard RNNs only see past context. **Bidirectional** RNNs run two RNNs — one forward, one backward — and concatenate their hidden states:

```python
# Bidirectional LSTM for NLP tasks (e.g., NER, sentiment)
bi_lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True,   # <-- key flag
)

x           = torch.randn(32, 50, 128)
out, (h, c) = bi_lstm(x)
# Output is (batch, seq, 2 * hidden) — forward + backward concatenated
print(f"BiLSTM output: {out.shape}")  # (32, 50, 512)
print(f"h_n:           {h.shape}")   # (2*num_layers, batch, hidden) = (4, 32, 256)

# Extract forward and backward final hidden states
h_forward  = h[-2]   # last layer, forward
h_backward = h[-1]   # last layer, backward
sentence_repr = torch.cat([h_forward, h_backward], dim=-1)  # (32, 512)
```

---

## 🔄 Sequence-to-Sequence (Seq2Seq)

Seq2Seq (Sutskever et al., 2014) uses an **encoder** RNN to compress input into a context vector, then a **decoder** RNN to generate output step-by-step.

```python
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.embedding(src)             # (batch, src_len, embed)
        outputs, (h, c) = self.lstm(embedded)
        return outputs, h, c                       # context: (h, c)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc_out    = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_token: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # tgt_token: (batch,) → unsqueeze to (batch, 1)
        embedded        = self.embedding(tgt_token.unsqueeze(1))
        output, (h, c)  = self.lstm(embedded, (h, c))
        logits          = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        return logits, h, c


# Usage
VOCAB, EMBED, HIDDEN = 10_000, 256, 512
encoder = Seq2SeqEncoder(VOCAB, EMBED, HIDDEN)
decoder = Seq2SeqDecoder(VOCAB, EMBED, HIDDEN)

src   = torch.randint(1, VOCAB, (32, 20))   # source sequence
enc_out, h, c = encoder(src)

# Decode one step at a time (teacher forcing / autoregressive)
tgt_token = torch.zeros(32, dtype=torch.long)  # <BOS> token = 0
for step in range(15):
    logits, h, c = decoder(tgt_token, h, c)
    tgt_token    = logits.argmax(dim=-1)         # greedy decode
```

---

## 🎯 Attention Mechanism (Pre-Transformer)

Bahdanau attention (2015) was the breakthrough before the Transformer. Instead of a single context vector, the decoder **attends** to all encoder outputs:

$$e_{tj} = \text{score}(h_{t-1}^{\text{dec}}, h_j^{\text{enc}})$$
$$\alpha_{tj} = \text{softmax}(e_{tj})$$
$$c_t = \sum_j \alpha_{tj} \cdot h_j^{\text{enc}}$$

```python
class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v     = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Expand decoder hidden: (batch, 1, hidden) → broadcast over src_len
        dec_h   = self.W_dec(decoder_hidden).unsqueeze(1)
        enc_out = self.W_enc(encoder_outputs)               # (batch, src_len, hidden)

        energy  = torch.tanh(dec_h + enc_out)               # (batch, src_len, hidden)
        scores  = self.v(energy).squeeze(-1)                # (batch, src_len)
        weights = torch.softmax(scores, dim=-1)             # (batch, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        return context.squeeze(1), weights


attn       = BahdanauAttention(hidden_size=512)
dec_hidden = torch.randn(32, 512)
enc_out    = torch.randn(32, 20, 512)
context, weights = attn(dec_hidden, enc_out)
print(f"Context:  {context.shape}")   # (32, 512)
print(f"Weights:  {weights.shape}")   # (32, 20)  — interpretable alignment
```

> **Attention → Transformer connection:** The Transformer (2017) generalised this: it dropped the RNN entirely and made attention the *primary* operation, adding scaled dot-product attention and multi-head variants. See [Transformers](./transformers.md).

---

## 🛠️ PyTorch Implementation: Sentiment Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SentimentLSTM(nn.Module):
    """Binary sentiment classifier using bidirectional LSTM."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_size, n_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True,
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq_len)
        embedded  = self.dropout(self.embedding(tokens))
        out, (h, _) = self.lstm(embedded)

        # Use final forward and backward hidden states
        h = self.dropout(torch.cat([h[-2], h[-1]], dim=1))  # (batch, hidden*2)
        return self.fc(h).squeeze(1)   # (batch,)


# Quick training demo (random data)
VOCAB_SIZE = 25_000
model     = SentimentLSTM(vocab_size=VOCAB_SIZE)
optimiser = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Dummy batch: (batch=32, seq_len=128) token ids
tokens = torch.randint(0, VOCAB_SIZE, (32, 128))
labels = torch.randint(0, 2, (32,)).float()

model.train()
optimiser.zero_grad()
logits = model(tokens)
loss   = criterion(logits, labels)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimiser.step()

preds    = (torch.sigmoid(logits) > 0.5).float()
accuracy = (preds == labels).float().mean()
print(f"Loss: {loss.item():.4f}  |  Acc: {accuracy.item():.4f}")
```

---

## 🔁 Packed Sequences for Variable-Length Inputs

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# When sequences in a batch have different lengths, use PackedSequence
# to avoid wasting computation on padding tokens

batch_size = 4
tokens  = torch.randint(1, 1000, (batch_size, 20))   # padded batch
lengths = torch.tensor([20, 15, 10, 5])               # actual lengths (sorted desc)

embedding = nn.Embedding(1000, 64)
lstm      = nn.LSTM(64, 128, batch_first=True)

embedded = embedding(tokens)

packed   = pack_padded_sequence(embedded, lengths, batch_first=True,
                                enforce_sorted=True)
out_packed, (h, c) = lstm(packed)
out, lens = pad_packed_sequence(out_packed, batch_first=True)

print(f"Output shape: {out.shape}")   # (4, 20, 128) — padded back
```

---

## ⚖️ When to Use RNNs vs Transformers

| Criterion | RNN/LSTM/GRU | Transformer |
|---|---|---|
| Sequence length | Good for short–medium (< 512) | Excels at any length with FlashAttention |
| Causal (streaming) inference | Natural (left-to-right) | Requires causal masking |
| Memory usage | O(T) — constant per step | O(T²) — quadratic attention |
| Parallelism in training | Sequential (slow) | Fully parallel (fast) |
| Long-range dependencies | Struggles (even with LSTM) | Excellent via direct attention |
| Small datasets | Often competitive | Needs more data or pretraining |
| On-device / edge | Very efficient (small state) | Heavier; use distilled models |
| Time series (short) | LSTM/GRU often excellent | Transformer viable with patching |

**2025 Recommendation:**

- New NLP / LM projects → **Transformer** (always)
- Streaming inference, strict latency constraints → **LSTM/GRU** or **Mamba (SSM)**
- Time series with < 1K steps → **LSTM, Temporal Fusion Transformer, or PatchTST**
- Legacy models / existing LSTM codebases → maintain, but plan migration

---

## 💎 Tips & Tricks

> **Always clip gradients** with `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` when training RNNs. Exploding gradients are the most common failure mode.

> **Use `batch_first=True`** in PyTorch's RNN modules — it's more intuitive and matches the convention for most downstream layers.

> **Packed sequences matter.** Without `pack_padded_sequence`, your LSTM wastes compute on padding tokens and might learn spurious patterns from them.

> **Bidirectional LSTMs need careful handling.** Remember the output is 2× hidden size and `h_n` has `2 × num_layers` entries along dimension 0.

> **For time series regression,** use only the last hidden state (or pool over outputs) rather than all time step outputs.

---

## 🔗 Cross-References

- [Deep Learning Overview](./index.md) — architecture comparison table
- [Transformers](./transformers.md) — the architecture that superseded RNNs for most tasks
- [Neural Networks](./neural-networks.md) — backpropagation fundamentals
- [NLP section](../../nlp/) — practical NLP with modern libraries
- [Calculus & Autograd](../foundations/mathematics/calculus.md) — BPTT derivation
