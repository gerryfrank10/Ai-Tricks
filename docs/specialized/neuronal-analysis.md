# 🧠 Neuronal Analysis & Computational Neuroscience

Computational neuroscience bridges neurobiology and machine learning — studying how neurons encode, transmit, and process information. Modern tools let us decode brain activity, simulate spiking dynamics, and build brain-inspired AI systems.

---

## ⚡ Leaky Integrate-and-Fire (LIF) Model

The LIF neuron is the workhorse model of computational neuroscience. It captures the essential dynamics of a real neuron: membrane potential charges up, and a spike fires when a threshold is crossed.

**The differential equation:**
```
τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
```

| Parameter | Symbol | Typical Value | Meaning |
|-----------|--------|--------------|---------|
| Membrane time constant | τ_m | 20 ms | RC time constant |
| Resting potential | V_rest | -70 mV | Equilibrium potential |
| Threshold | V_thresh | -55 mV | Spike trigger |
| Reset potential | V_reset | -75 mV | After-spike reset |
| Refractory period | t_ref | 2 ms | Silent period post-spike |

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_lif(I_ext, dt=0.1, T=200.0,
                 tau_m=20.0, V_rest=-70.0, V_thresh=-55.0,
                 V_reset=-75.0, R_m=10.0, t_ref=2.0):
    """
    Simulate a Leaky Integrate-and-Fire neuron.

    Args:
        I_ext: External current in nA (scalar or array of length T/dt)
        dt:    Time step in ms
        T:     Total simulation time in ms
    Returns:
        t:      Time array
        V:      Membrane potential trace (mV)
        spikes: Boolean spike train
    """
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    V = np.full(n_steps, V_rest)
    spikes = np.zeros(n_steps, dtype=bool)

    if np.isscalar(I_ext):
        I_ext = np.full(n_steps, I_ext)

    ref_countdown = 0.0

    for i in range(1, n_steps):
        if ref_countdown > 0:
            V[i] = V_reset
            ref_countdown -= dt
            continue

        dV = (-(V[i-1] - V_rest) + R_m * I_ext[i-1]) * dt / tau_m
        V[i] = V[i-1] + dV

        if V[i] >= V_thresh:
            V[i] = 40.0              # spike peak (cosmetic)
            spikes[i] = True
            ref_countdown = t_ref

    return t, V, spikes

# Run simulation with constant current
t, V, spikes = simulate_lif(I_ext=1.8)
spike_times = t[spikes]
isi = np.diff(spike_times)

print(f"Number of spikes: {spikes.sum()}")
print(f"Mean firing rate: {spikes.sum() / 0.2:.1f} Hz")
print(f"Mean ISI: {isi.mean():.2f} ms  (CV={isi.std()/isi.mean():.3f})")
```

---

## 🔥 Spiking Neural Networks (SNNs)

SNNs use discrete spike events rather than continuous activations — more biologically plausible and energy-efficient on neuromorphic hardware.

```python
import torch
import torch.nn as nn

# SpikingJelly-style LIF layer (install: pip install spikingjelly)
# Here we show a self-contained implementation

class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire layer with surrogate gradient."""

    def __init__(self, in_features, out_features,
                 tau=2.0, threshold=1.0, surrogate_slope=25.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tau = tau
        self.threshold = threshold
        self.slope = surrogate_slope
        self.membrane = None

    def reset_state(self, batch_size, device):
        self.membrane = torch.zeros(batch_size, self.linear.out_features,
                                    device=device)

    def forward(self, x):
        if self.membrane is None:
            self.reset_state(x.size(0), x.device)

        # Membrane update: leak + input
        self.membrane = self.membrane / self.tau + self.linear(x)

        # Surrogate gradient: sigmoid approximation to Heaviside
        spike = SurrogateSpike.apply(self.membrane - self.threshold, self.slope)
        self.membrane = self.membrane * (1 - spike.detach())  # reset after spike
        return spike

class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, sigmoid-derivative backward."""
    @staticmethod
    def forward(ctx, u, slope):
        ctx.save_for_backward(u)
        ctx.slope = slope
        return (u >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        sigmoid_u = torch.sigmoid(ctx.slope * u)
        grad = ctx.slope * sigmoid_u * (1 - sigmoid_u)
        return grad_output * grad, None

# Multi-layer SNN for temporal classification
class SpikingMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, timesteps=10):
        super().__init__()
        self.timesteps = timesteps
        self.layer1 = LIFLayer(input_dim, hidden_dim)
        self.layer2 = LIFLayer(hidden_dim, output_dim)

    def forward(self, x_seq):
        # x_seq: (timesteps, batch, features)
        self.layer1.membrane = None
        self.layer2.membrane = None
        outputs = []
        for t in range(self.timesteps):
            s1 = self.layer1(x_seq[t])
            s2 = self.layer2(s1)
            outputs.append(s2)
        # Rate coding: mean spike count over time
        return torch.stack(outputs).mean(dim=0)

model = SpikingMLP()
dummy = torch.randn(10, 32, 784)   # (timesteps, batch, features)
output = model(dummy)
print(f"SNN output shape: {output.shape}")  # (32, 10)
```

---

## 📍 Place Cells & Spatial Encoding

Place cells in the hippocampus fire when an animal occupies a specific location — a spatial GPS system.

```python
import numpy as np
import matplotlib.pyplot as plt

def place_cell_tuning(x, y, x_center, y_center, sigma=0.15):
    """2D Gaussian place field."""
    dist_sq = (x - x_center)**2 + (y - y_center)**2
    return np.exp(-dist_sq / (2 * sigma**2))

# Simulate 20 place cells on a 1x1 arena
np.random.seed(42)
n_cells = 20
centers = np.random.rand(n_cells, 2)   # random place field centers

# Generate trajectory
t = np.linspace(0, 4 * np.pi, 500)
traj_x = 0.5 + 0.4 * np.sin(t)
traj_y = 0.5 + 0.4 * np.cos(t)

# Compute population activity along trajectory
activity = np.array([
    place_cell_tuning(traj_x, traj_y, cx, cy)
    for cx, cy in centers
])  # shape: (n_cells, n_timepoints)

# Decode position from population activity (linear regression)
from sklearn.linear_model import Ridge

X = activity.T              # (n_timepoints, n_cells)
y_pos = np.column_stack([traj_x, traj_y])

split = 400
decoder = Ridge(alpha=0.1)
decoder.fit(X[:split], y_pos[:split])
decoded = decoder.predict(X[split:])

decode_error = np.sqrt(np.mean(np.sum((decoded - y_pos[split:])**2, axis=1)))
print(f"Mean decode error: {decode_error:.4f} arena units")
print(f"Max activity per cell: {activity.max(axis=1).round(3)}")
```

---

## 👥 Population Coding

Populations of neurons collectively encode more information than any single cell.

```python
# Fisher Information & population decoding
def von_mises_tuning(theta, theta_pref, kappa=4.0, r_max=50.0):
    """Von Mises tuning curve for a direction-selective neuron (Hz)."""
    return r_max * np.exp(kappa * (np.cos(theta - theta_pref) - 1))

n_neurons = 32
theta_prefs = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
test_theta = np.pi / 4    # 45 degrees

# Population response vector
responses = np.array([von_mises_tuning(test_theta, pref)
                      for pref in theta_prefs])

# Add Poisson noise (variance = mean for Poisson)
noisy_responses = np.random.poisson(responses.clip(0))

# Maximum Likelihood Estimate via population vector
pref_complex = np.exp(1j * theta_prefs)
population_vector = np.sum(noisy_responses * pref_complex)
theta_decoded = np.angle(population_vector)

error_deg = np.abs(np.degrees(theta_decoded - test_theta))
print(f"True angle:    {np.degrees(test_theta):.1f}°")
print(f"Decoded angle: {np.degrees(theta_decoded):.1f}°")
print(f"Decode error:  {min(error_deg, 360-error_deg):.1f}°")
```

---

## 🔬 Calcium Imaging Analysis

Calcium imaging records neural activity via fluorescent indicators (GCaMP). The raw signal is ΔF/F.

```python
import numpy as np
from scipy import signal

def compute_dff(F_raw, baseline_percentile=8, window_size=200):
    """
    Compute ΔF/F from raw fluorescence.
    baseline_percentile: rolling percentile as estimate of F0
    """
    from scipy.ndimage import percentile_filter
    F0 = percentile_filter(F_raw, percentile=baseline_percentile,
                           size=window_size)
    F0 = np.maximum(F0, 1e-6)    # avoid division by zero
    dff = (F_raw - F0) / F0
    return dff

def detect_transients(dff, threshold=2.5, min_duration=3):
    """
    Detect calcium transients above threshold * std.
    Returns start/end indices of each transient.
    """
    baseline_std = np.std(dff[dff < np.percentile(dff, 25)])
    binary = (dff > threshold * baseline_std).astype(int)
    # Label connected components
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    transients = [(s, e) for s, e in zip(starts, ends)
                  if (e - s) >= min_duration]
    return transients

# Simulate fluorescence trace
np.random.seed(0)
T = 1000
F_true = np.random.exponential(1.0, T)   # sparse activity
F_raw = F_true + 0.5 * np.random.randn(T) + 100  # noise + baseline

dff = compute_dff(F_raw)
transients = detect_transients(dff)
print(f"Detected {len(transients)} calcium transients")
print(f"ΔF/F range: [{dff.min():.2f}, {dff.max():.2f}]")
```

---

## 🔍 Neural Decoding — Spikes to Behavior

```python
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Synthetic spike data: 50 neurons, 200 trials
np.random.seed(42)
n_neurons, n_trials = 50, 200

# Behavioral variable: movement speed (0-1)
speed = np.random.rand(n_trials)

# Neurons have different preferred speeds (Gaussian tuning)
preferred_speeds = np.random.rand(n_neurons)
tuning_width = 0.2

# Generate spike counts (Poisson)
rate = 20 * np.exp(-((speed[:, None] - preferred_speeds[None, :]) ** 2)
                   / (2 * tuning_width**2))
spike_counts = np.random.poisson(rate)   # shape: (n_trials, n_neurons)

# Decode speed from population activity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spike_counts)

from sklearn.linear_model import Ridge
decoder = Ridge(alpha=1.0)
r2_scores = cross_val_score(decoder, X_scaled, speed, cv=5, scoring="r2")
print(f"Decoding R² = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

# Mutual information between neural activity and behavior
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(spike_counts, speed)
top_cells = np.argsort(mi)[::-1][:5]
print(f"Top informative neurons: {top_cells}")
print(f"Their MI values: {mi[top_cells].round(3)}")
```

---

## 🛠️ Tools & Libraries

| Tool | Purpose | Install |
|------|---------|---------|
| **MNE-Python** | EEG/MEG analysis, time-frequency | `pip install mne` |
| **PySpike** | Spike train similarity metrics (SPIKE) | `pip install pyspike` |
| **Brian2** | Full neural simulation | `pip install brian2` |
| **SpikingJelly** | SNN deep learning | `pip install spikingjelly` |
| **Neo** | Electrophysiology data I/O | `pip install neo` |
| **Elephant** | Spike train statistics | `pip install elephant` |
| **CaImAn** | Calcium imaging processing | conda install |
| **Suite2p** | Two-photon pipeline | `pip install suite2p` |

```python
# MNE-Python: EEG preprocessing pipeline
import mne

# Load sample data
sample_data_raw_file = mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

# Standard preprocessing pipeline
raw.filter(l_freq=1.0, h_freq=40.0)              # bandpass filter
raw.notch_filter(freqs=60)                        # remove line noise
raw.set_eeg_reference("average", projection=True) # average reference

# Epoch around events
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0), preload=True)

# Compute ERP
evoked = epochs["auditory/left"].average()
print(f"ERP shape: {evoked.data.shape}")  # (n_channels, n_times)

# PySpike: spike train distance
import pyspike as spk
st1 = spk.SpikeTrain([0.1, 0.3, 0.7, 0.9], edges=(0, 1.0))
st2 = spk.SpikeTrain([0.1, 0.31, 0.68, 0.92], edges=(0, 1.0))
isi_distance = spk.isi_distance(st1, st2)
spike_distance = spk.spike_distance(st1, st2)
print(f"ISI distance:   {isi_distance:.4f}")
print(f"SPIKE distance: {spike_distance:.4f}")
```

---

## 🤖 Brain-Inspired AI Connections

| Neuroscience Concept | AI Equivalent | Connection |
|----------------------|--------------|------------|
| Hebbian learning | Contrastive Hebbian / local rules | "Neurons that fire together, wire together" |
| Sparse coding | Sparse autoencoders | Efficient representation hypothesis |
| Predictive coding | Prediction error networks | Free-energy principle → active inference |
| Hippocampal replay | Experience replay (DQN) | Offline memory consolidation |
| Attention (thalamus) | Transformer self-attention | Selective information routing |
| Neuromodulators | Adaptive learning rates | Dopamine ≈ TD error signal |
| Cortical columns | Capsule networks | Hierarchical feature detectors |
| Oscillations (theta/gamma) | Positional encoding | Temporal multiplexing of information |

```python
# Predictive Coding — simple implementation
import torch
import torch.nn as nn

class PredictiveCodingLayer(nn.Module):
    """
    A single predictive coding layer.
    Maintains a representation that minimizes prediction error.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.predict = nn.Linear(hidden_dim, input_dim)  # top-down prediction
        self.encode  = nn.Linear(input_dim, hidden_dim)  # bottom-up error

    def forward(self, x_input, r_higher=None, n_iters=10, lr_r=0.1):
        batch = x_input.size(0)
        # Initialize representation
        r = torch.zeros(batch, self.encode.out_features, requires_grad=False)

        for _ in range(n_iters):
            # Prediction (generative)
            x_pred = self.predict(r)
            # Prediction error (bottom-up)
            error = x_input - x_pred
            # Update representation
            r = r + lr_r * (self.encode(error))
            r = torch.relu(r)

        return r, error

pc_layer = PredictiveCodingLayer(784, 256)
x = torch.randn(32, 784)
representation, pred_error = pc_layer(x)
print(f"Representation shape: {representation.shape}")
print(f"Prediction error MSE: {(pred_error**2).mean().item():.4f}")
```

---

## 💡 Practical Tips

| Tip | Detail |
|-----|--------|
| **Always bandpass filter** | Remove slow drift (< 0.5 Hz) and high-freq noise before analysis |
| **Visualize raw traces first** | Artifacts are obvious in raw data; always sanity-check |
| **Use z-score for comparisons** | Normalize firing rates across neurons/sessions before comparing |
| **Poisson assumptions** | Spike counts are approximately Poisson; use GLMs not OLS |
| **Cross-validate decoders** | Never evaluate decode accuracy on training trials |
| **Spike sorting caution** | Single-unit isolation has errors; consider multi-unit for robustness |
| **SNN training instability** | Start with smaller thresholds and larger surrogate slopes |
| **Time constants matter** | τ_m = 20ms is typical cortex; adjust for other brain regions |

---

## 📚 Further Reading

- [Theoretical Neuroscience — Dayan & Abbott](http://www.gatsby.ucl.ac.uk/~lmate/biblio/dayanabbott.pdf)
- [MNE-Python Documentation](https://mne.tools/)
- [Brian2 Documentation](https://brian2.readthedocs.io/)
- [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/)
- [Neuromatch Academy — Computational Neuroscience](https://neuromatch.io/neuroscience/)
