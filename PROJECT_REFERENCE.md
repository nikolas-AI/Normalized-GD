# Project Reference: Normalized Gradient Descent on Two-Layer Neural Networks

---

## 1. Project Overview

This project implements and empirically evaluates **Normalized Gradient Descent (NGD)** and its stochastic variant (SNGD) applied to binary classification with two-layer neural networks. It reproduces three figures from a research paper:

- **Figure 1** — MNIST 0-vs-1 classification: GD vs NGD on a linear model
- **Figure 2** — Synthetic X-shaped data: GD vs NGD on a two-layer NN (d=2 and d=5)
- **Figure 3** — Stochastic NGD: varying batch sizes on a linear model and two-layer NN

**Key idea:** Standard gradient descent slows as the gradient norm shrinks near a solution. NGD normalizes the gradient by its global L2 norm before each update, maintaining a fixed step size throughout training and converging significantly faster on separable problems.

**High-level pipeline:**

```
Dataset → Model (TwoLayerFixedA or LinearBinary)
       → Forward pass → Exponential Loss
       → Backward pass → Gradient
       → Optimizer (GD / NGD / SNGD) → Weight update
       → Metrics (loss, error, weight norm) → Saved CSV/JSON → Plot
```

---

## 2. File Tree

```
pj3/
├── PROJECT_REFERENCE.md          ← this document
│
├── src/
│   ├── __init__.py
│   ├── sanity.py                 ← quick smoke test
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py             ← frozen dataclass config schemas
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mnist01.py            ← MNIST 0/1 loader → {+1, -1} labels
│   │   ├── synthetic.py          ← X-shaped, XOR, and linear synthetic datasets
│   │   └── sanity.py             ← quick checks on data tensors
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── exp_loss.py           ← elementwise exp loss: exp(-t)
│   │   ├── objectives.py         ← training_loss_exp: mean exp loss over dataset
│   │   └── metrics.py            ← classification_error, weight_norm
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── activations.py        ← leaky_relu (forward + derivative)
│   │   ├── functional.py         ← stateless forward pass helper
│   │   ├── init.py               ← weight initialization utilities
│   │   ├── linear.py             ← LinearBinary: single-layer linear model
│   │   └── two_layer_fixed_a.py  ← TwoLayerFixedA: 2-layer NN, fixed second layer
│   │
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── checks.py             ← descent condition verification
│   │   ├── gd.py                 ← standard gradient descent step
│   │   ├── ngd.py                ← normalized GD (divide by global gradient norm)
│   │   └── sngd.py               ← stochastic NGD (divide batch grad by its norm)
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── autograd.py           ← finite-difference gradient checker
│   │   ├── batching.py           ← minibatch iterator with optional shuffle
│   │   └── engine.py             ← unified training loop (GD / NGD / SNGD)
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── fig1_mnist01.py       ← Figure 1: MNIST GD vs NGD
│   │   ├── fig2_synthetic.py     ← Figure 2: X-shaped data GD vs NGD
│   │   └── fig3_sngd.py          ← Figure 3: SNGD batch-size sweep
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                 ← run directory creation, CSV/JSON saving
│       └── seed.py               ← global seed setter (torch + numpy + random)
│
├── tests/
│   ├── __init__.py
│   ├── test_activation.py
│   ├── test_batching.py
│   ├── test_data_mnist01.py
│   ├── test_data_synthetic.py
│   ├── test_descent_check.py
│   ├── test_forward.py
│   ├── test_gradcheck.py
│   ├── test_loss_metrics.py
│   ├── test_optimizers.py
│   └── test_training_engine.py
│
└── runs/                         ← generated at runtime, one subdir per experiment
    ├── fig1_mnist01_seed0/
    ├── fig2_synth_seed0/
    └── fig3_sngd_seed0/
```

---

## 3. File-by-File Breakdown

---

### `src/config/schema.py`

**Purpose:** Defines frozen dataclasses that describe a complete experiment configuration. Serves as a type-safe schema for future config-driven experiment runners.

| Name | Type | Description |
|---|---|---|
| `ModelConfig` | dataclass | `d`, `m`, `alpha`, `ell`, `init_type`, `init_norm` |
| `LossConfig` | dataclass | `type` — only `"exp"` supported |
| `OptimizerConfig` | dataclass | `type`, `eta`, optional `batch_size` |
| `DataConfig` | dataclass | `dataset`, `root`, `n_train`, `seed` |
| `TrainConfig` | dataclass | `steps`, `eval_every`, `device` |
| `ExperimentConfig` | dataclass | Composes all of the above; has `.validate()` |

**Key logic:** `ExperimentConfig.validate()` raises `ValueError` on illegal parameter combinations (non-positive `m`, `d`, `eta`, `steps`; missing `batch_size` for SNGD).

---

### `src/data/mnist01.py`

**Purpose:** Loads MNIST digits 0 and 1, maps labels to `{+1, −1}`, and returns flat tensors ready for the training engine.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `load_mnist01` | function | `root`, `n_train`, `n_test`, `seed`, `device` | `(SyntheticDataset_train, SyntheticDataset_test)` | Downloads MNIST if absent, filters to classes {0,1}, relabels 0→−1 and 1→+1, flattens 28×28→784, optionally subsamples |

**Key logic:** Uses `torchvision.datasets.MNIST` with `download=True`. Labels: digit 1 → +1, digit 0 → −1. Returns standard `SyntheticDataset` namedtuples reused across all datasets.

---

### `src/data/synthetic.py`

**Purpose:** All synthetic dataset generators. Returns `SyntheticDataset(X, y)` with `y ∈ {+1, −1}`.

| Name | Type | Description |
|---|---|---|
| `SyntheticDataset` | frozen dataclass | Holds `X: (n,d) float32`, `y: (n,) float32 ∈ {±1}` |
| `_labels_pm1` | function | Creates label vector: first `n0` entries are −1, next `n1` are +1 |
| `gaussian_mixture_zero_mean` | function | Zero-mean GMM; class covariances Σ₀, Σ₁ supplied explicitly |
| `gaussian_mixture_d2_fig2_top` | function | d=2 GMM with correlated covariances (legacy, kept for tests) |
| `gaussian_mixture_d5_fig2_bottom` | function | d=5 GMM with Σ₀=I, Σ₁=0.25·I (legacy) |
| `x_shaped_d2_fig2_top` | function | **Active fig2 dataset.** 20 pts along t·[1,1]+noise (class +1), 20 along t·[1,−1]+noise (class −1); `noise_std=0.2`, `t_scale=4.0` |
| `x_shaped_d5_fig2_bottom` | function | X-shaped in first 2 dims, 3 extra Gaussian noise dims appended; `noise_std=0.1`, `t_scale=1.5` |
| `xor_d2_fig3_bottom` | function | XOR pattern: class +1 in Q1&Q3, class −1 in Q2&Q4 (legacy, superseded by x_shaped for fig3) |
| `signed_linear_measurements` | function | Fig3 linear data: `y = sign(Xw*)`, d=50, n=100; returns dataset + ground-truth `w*` |

**Key logic for X-shaped data:**
```
t ~ Uniform[-t_scale, t_scale]
Class +1: x = t·[1, 1] + N(0, noise_std²·I)
Class -1: x = t·[1,-1] + N(0, noise_std²·I)
```
The classes are **not linearly separable** but are separable by a two-layer NN with enough neurons.

---

### `src/losses/exp_loss.py`

**Purpose:** Elementwise exponential loss primitive.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `exp_loss` | function | `t: Tensor` | `Tensor` | Returns `exp(-t)` elementwise |

---

### `src/losses/objectives.py`

**Purpose:** Wraps `exp_loss` into the training objective used everywhere.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `training_loss_exp` | function | `phi: (n,)`, `y: (n,)` | scalar `Tensor` | Computes `mean(exp(-y * phi))` — the exponential surrogate loss |

**Key logic:** `t = y * phi` (margin), then `mean(exp(-t))`. At initialization (phi≈0), loss ≈ 1.0 per sample.

---

### `src/losses/metrics.py`

**Purpose:** Evaluation metrics, not used in the gradient computation.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `classification_error` | function | `phi: (n,)`, `y: (n,)` | `float` | Fraction of samples where `sign(phi) ≠ y`; ties (`phi=0`) count as correct |
| `weight_norm` | function | `W: Tensor` | `float` | `‖W‖_F` — Frobenius norm of the weight matrix |

---

### `src/models/activations.py`

**Purpose:** Leaky ReLU activation and its derivative, used by the two-layer model.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `leaky_relu` | function | `z`, `alpha=0.2`, `ell=1.0` | Tensor | `ell·z` if `z≥0`, else `ell·alpha·z` |
| `leaky_relu_prime` | function | `z`, `alpha=0.2`, `ell=1.0` | Tensor | `ell` if `z≥0`, else `ell·alpha` |

**Key logic:** `ell` scales both branches uniformly. Default `alpha=0.2` means the negative slope is 20% of the positive slope.

---

### `src/models/functional.py`

**Purpose:** Stateless forward pass that takes W and a as plain tensors (not `nn.Parameter`). Used in gradient checking.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `two_layer_forward` | function | `x: (b,d)`, `W: (m,d)`, `a: (m,)`, `alpha`, `ell` | `(b,)` | Computes `a · σ(W x)` without a module wrapper |

---

### `src/models/init.py`

**Purpose:** Weight initialization for the two-layer model.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `init_first_layer` | function | `m, d`, `normalize=True` | `(m,d) Tensor` | Gaussian random W; if `normalize=True`, each row divided by its L2 norm |
| `init_second_layer` | function | `m` | `(m,) Tensor` | Each entry is `±1/m` (random sign, fixed magnitude) |

**Key logic:** Row normalization in `init_first_layer` ensures each hidden neuron's weight vector starts as a unit vector on the sphere.

---

### `src/models/linear.py`

**Purpose:** Single-layer linear binary classifier for Fig 3 linear experiments.

| Name | Type | Description |
|---|---|---|
| `LinearBinary` | `nn.Module` | `W ∈ ℝᵈ` (1D parameter); forward computes `X @ W`; initialized to zero by default |

**Key logic:** `init_zero=True` matches the paper ("w₀ = 0_d"). The `.W` attribute name is shared with `TwoLayerFixedA` so the training engine works uniformly on both.

---

### `src/models/two_layer_fixed_a.py`

**Purpose:** The main model — a two-layer neural network where the second layer `a` is **fixed** (frozen) after initialization. Only `W` (first layer) is trained.

| Name | Type | Description |
|---|---|---|
| `TwoLayerFixedA` | `nn.Module` | `W ∈ ℝ^{m×d}` (trainable), `a ∈ ℝᵐ` (buffer, frozen) |

**Forward pass:**
```
z = x @ W.T          # (batch, m) — pre-activations
h = leaky_relu(z)    # (batch, m) — hidden layer output
phi = h @ a          # (batch,)   — scalar prediction per sample
```

**Parameters:**
- `alpha=0.2` — leaky ReLU negative slope
- `ell=1.0` — leaky ReLU scale factor
- `m` — number of hidden neurons
- `d` — input dimension (derived from W)

---

### `src/optim/gd.py`

**Purpose:** Standard gradient descent primitives.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `gd_step` | function | `W`, `gradW`, `eta` | new `W` | Returns `W - eta * gradW`; validates `eta > 0` and shape match |
| `apply_update_` | function | `param: nn.Parameter`, `new_value` | — | In-place copy of `new_value` into `param`; clears `.grad` |

---

### `src/optim/ngd.py`

**Purpose:** Normalized Gradient Descent — divides the step by the **global gradient norm**.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `ngd_stepsize` | function | `eta_base`, `F_scalar`, `eps=0` | `float` | Returns `eta_base / (F_scalar + eps)` |
| `ngd_step` | function | `W`, `gradW`, `eta_base`, `F_scalar`, `eps` | new `W` | Computes `eta_t = eta_base / F_scalar`, then calls `gd_step(W, gradW, eta_t)` |

**Usage in engine:** `F_scalar = grad.norm()` → effective update is `W - eta · (gradW / ‖gradW‖)`.

---

### `src/optim/sngd.py`

**Purpose:** Stochastic Normalized Gradient Descent — normalizes the **mini-batch gradient** by its own L2 norm.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `sngd_step` | function | `W`, `grad_batch`, `eta_base`, `eps=1e-12` | new `W` | `eta_t = eta_base / (‖grad_batch‖ + eps)`, returns `W - eta_t * grad_batch` |

**Key distinction from NGD:** NGD normalizes the full-batch gradient; SNGD normalizes the mini-batch gradient. Both produce a step of fixed magnitude `eta_base`, but in different directions (full-gradient direction vs mini-batch gradient direction).

---

### `src/optim/checks.py`

**Purpose:** Verifies that a proposed weight update actually decreases the loss (descent condition check), used in tests.

| Name | Type | Description |
|---|---|---|
| `check_descent` | function | Runs one optimizer step, checks that `L(W_new) < L(W_old)` |

---

### `src/train/batching.py`

**Purpose:** Mini-batch iterator for SNGD.

| Name | Type | Inputs | Outputs | Description |
|---|---|---|---|---|
| `iterate_minibatches` | function | `X`, `y`, `batch_size`, `shuffle`, `generator` | iterator of `(Xb, yb)` pairs | Yields consecutive slices of size `batch_size`; last batch may be smaller |

**Key logic:** If `shuffle=True`, draws a random permutation via the supplied generator before slicing — ensures reproducibility across runs with the same seed.

---

### `src/train/autograd.py`

**Purpose:** Finite-difference gradient checker for verifying autograd correctness.

| Name | Type | Description |
|---|---|---|
| `finite_diff_grad` | function | Perturbs each weight by `±eps` and estimates gradient numerically |
| `check_grad` | function | Compares autograd gradient to finite-difference estimate; raises on mismatch |

---

### `src/train/engine.py`

**Purpose:** Unified training loop supporting all three optimizers (GD, NGD, SNGD). Returns a list of per-step metric dicts.

**Key dataclass:**

| Name | Type | Fields | Description |
|---|---|---|---|
| `TrainParams` | frozen dataclass | `optim`, `eta`, `steps`, `eval_every`, `batch_size`, `shuffle`, `seed` | All parameters needed to run a training job |

**Main function:**

| Name | Type | Inputs | Outputs |
|---|---|---|---|
| `run_training` | function | `model`, `X_train`, `y_train`, `params`, optional `X_test/y_test`, `run_dir`, `metadata` | `list[dict]` — one dict per eval step |

**Training loop logic (per step t):**

```
if t % eval_every == 0:
    record train_loss, weight_norm, (test_error if test set provided)

if optim == "gd":
    loss = training_loss_exp(model(X_train), y_train)
    loss.backward()
    grad = model.W.grad
    W_new = W - eta * grad                          ← standard GD

elif optim == "ngd":
    loss.backward()
    grad = model.W.grad
    grad_norm = grad.norm()
    W_new = W - (eta / grad_norm) * grad            ← normalized GD

elif optim == "sngd":
    sample mini-batch (Xb, yb) from iterator
    batch_loss = training_loss_exp(model(Xb), yb)
    batch_loss.backward()
    grad_b = model.W.grad
    grad_norm_b = grad_b.norm()
    W_new = W - (eta / grad_norm_b) * grad_b        ← stochastic NGD
```

**Outputs saved when `run_dir` provided:**
- `metrics.csv` — full step-by-step metrics
- `meta.json` — hyperparameters and dataset info

---

### `src/utils/io.py`

**Purpose:** File I/O helpers for experiment outputs.

| Name | Type | Description |
|---|---|---|
| `make_run_dir` | function | Creates `runs/<name>_<timestamp>/` and returns the `Path` |
| `save_csv` | function | Writes list of dicts to CSV |
| `save_json` | function | Writes dict to JSON with indent |

---

### `src/utils/seed.py`

**Purpose:** Single function to fix all sources of randomness.

| Name | Type | Description |
|---|---|---|
| `set_seed` | function | Sets `torch`, `numpy`, and `random` seeds; optionally sets `torch.use_deterministic_algorithms(True)` |

---

### `src/experiments/fig1_mnist01.py`

**Purpose:** Generates Figure 1 — MNIST 0-vs-1 classification comparing GD and NGD on a **linear model**.

**Key parameters:**
- Dataset: MNIST digits {0,1}, `n_train=1000`, `n_test=500`, d=784
- Model: `LinearBinary(d=784, init_zero=True)`
- Optimizers: GD (`eta=0.01`) and NGD (`eta=0.01`)
- Steps: 500

**Output:** `runs/fig1_mnist01_seed*/figure1_like.png` + `summary.json`

**Plot:** Two subplots — training loss (log scale) and test error over iterations, both GD and NGD on each.

---

### `src/experiments/fig2_synthetic.py`

**Purpose:** Generates Figure 2 — X-shaped synthetic data comparing GD vs NGD on a **two-layer NN**.

**Key parameters:**

| Case | Dataset | m | eta_gd | eta_ngd | Steps |
|---|---|---|---|---|---|
| Top | `x_shaped_d2_fig2_top` (d=2) | 50 | 3.0 | 3.0 | 1000 |
| Bottom | `x_shaped_d5_fig2_bottom` (d=5) | 100 | 5.0 | 5.0 | 2000 |

**Output:** `runs/fig2_synth_seed*/figure2_like.png` + `summary.json`

**Plot:** 2×2 grid — scatter plots (left) and training loss in log scale (right) for each case.

---

### `src/experiments/fig3_sngd.py`

**Purpose:** Generates Figure 3 — SNGD with varying batch sizes `b`.

**Top experiment — linear model, signed measurements (d=50, n=100):**

| b | eta |
|---|---|
| 1 | 0.35 |
| 4 | 0.2 |
| 10 | 0.2 |
| 50 | 0.15 |
| 100 | 0.1 |

**Bottom experiment — two-layer NN, X-shaped data (d=2, n=40, m=50):**

| b | eta |
|---|---|
| 1 | 9 |
| 4 | 6 |
| 10 | 5 |
| 20 | 5 |
| 40 | 4 |

**Output:** `runs/fig3_sngd_seed*/figure3_like.png` + `summary.json`

**Plot:** 2×2 grid — linear training loss (top-left), test error (top-right), XOR scatter (bottom-left), NN training loss (bottom-right).

---

## 4. Core System Components

### 4.1 Model

**Two-layer network with fixed second layer:**

```
Input x ∈ ℝᵈ
  → First layer: z = W x,   W ∈ ℝ^{m×d}   (trainable)
  → Activation:  h = σ(z),  σ = leaky ReLU
  → Second layer: φ = a·h,  a ∈ ℝᵐ        (fixed, ±1/m)
Output φ ∈ ℝ  (scalar prediction)
```

**Why the second layer is fixed:** The paper studies the optimization of W alone under fixed random `a`. This isolates the effect of the gradient normalization on the first-layer optimization landscape, and mirrors a specific theoretical setting analyzed in the paper.

**Parameter count:** Only `W` (shape `m×d`) is trained. `a` is a buffer, not a `nn.Parameter`.

---

### 4.2 Loss Function

**Exponential loss:**
```
L(W) = (1/n) Σᵢ exp(−yᵢ · φ(xᵢ; W))
```

- `yᵢ ∈ {+1, −1}` — binary label
- `φ(xᵢ; W)` — model output (scalar)
- `yᵢ · φ` = margin; loss decays to 0 as margin → +∞
- Driving `L → 0` requires `‖W‖ → ∞` on separable data

At initialization (φ≈0): `L ≈ 1.0`. On separable data, `L → 0` as training progresses.

---

### 4.3 Optimizers

#### Standard GD
```
W_{t+1} = W_t − η · ∇L(W_t)
```
Step size scales with gradient norm → slows dramatically as `‖∇L‖ → 0` near convergence.

#### Normalized GD (NGD)
```
W_{t+1} = W_t − η · ∇L(W_t) / ‖∇L(W_t)‖
```
Step size is always exactly `η`, independent of gradient magnitude → maintains constant convergence speed even when gradient norm is tiny.

#### Stochastic NGD (SNGD)
```
W_{t+1} = W_t − η · ∇L_b(W_t) / ‖∇L_b(W_t)‖
```
Same as NGD but the gradient is computed on a random mini-batch of size `b`. Gradient direction is noisy but normalization keeps step size fixed at `η`.

---

### 4.4 Data Pipeline

**MNIST:** `torchvision` download → filter {0,1} → relabel 0↔−1, 1↔+1 → flatten → optional subsample → `SyntheticDataset`

**X-shaped synthetic (Fig 2):**
- Draw scalars `t ~ Uniform[−t_scale, t_scale]`
- Class +1: `x = t·[1,1] + ε`, class −1: `x = t·[1,−1] + ε`, where `ε ~ N(0, noise_std²)`
- For d=5: append 3 independent noise dimensions

**Signed linear measurements (Fig 3 top):**
- Draw `X ~ N(0, I)`, draw ground-truth `w*`
- Labels: `y = sign(X w*)`

All datasets return `SyntheticDataset(X: float32, y: float32)` — a unified container used by the engine.

---

### 4.5 Training Engine

`run_training` is the single entry point for all training jobs:

1. Validates params (batch_size required for SNGD)
2. Iterates `t = 0, 1, ..., steps`:
   - **Eval:** every `eval_every` steps, record full-batch train loss, weight norm, optional test error
   - **Update:** compute gradient (full-batch for GD/NGD, mini-batch for SNGD), apply optimizer step via `apply_update_`
3. Optionally saves `metrics.csv` and `meta.json` to `run_dir`
4. Returns list of metric dicts

**The engine is model-agnostic** — it accesses `model.W` (the parameter to update) and calls `model(X)` for the forward pass. Both `LinearBinary` and `TwoLayerFixedA` expose `.W`.

---

## 5. End-to-End Data Flow

### Full training flow (NGD example, Fig 2)

```
1. DATASET CONSTRUCTION
   x_shaped_d2_fig2_top(seed=0)
   → X: (40, 2) float32, y: (40,) ∈ {±1}

2. MODEL INIT
   W0 = init_first_layer(m=50, d=2)   # (50,2), row-normalized
   a0 = init_second_layer(m=50)       # (50,), entries ±1/50
   model = TwoLayerFixedA(W0, a0)     # W is nn.Parameter, a is buffer

3. FORWARD PASS  [inside engine at step t]
   z = X @ model.W.T                  # (40, 50) pre-activations
   h = leaky_relu(z, alpha=0.2)       # (40, 50) hidden outputs
   phi = h @ model.a                  # (40,)    predictions

4. LOSS
   t_margin = y * phi                 # (40,) margins
   L = mean(exp(-t_margin))           # scalar

5. BACKWARD PASS
   L.backward()
   grad = model.W.grad                # (50, 2)

6. NGD UPDATE
   grad_norm = grad.norm()            # scalar ‖∇L‖
   eta_t = eta / grad_norm            # effective step size
   W_new = model.W - eta_t * grad    # (50, 2)
   apply_update_(model.W, W_new)     # in-place, clears .grad

7. METRICS  [recorded every eval_every steps]
   train_loss  = L.item()
   weight_norm = ‖model.W‖_F
   → appended to metrics list

8. OUTPUT
   metrics list → saved to metrics.csv
   summary dict → saved to summary.json
   plot → saved to figure2_like.png
```

### Where normalization happens

| Optimizer | Where | What is divided |
|---|---|---|
| GD | `gd.py: gd_step` | nothing — raw gradient × eta |
| NGD | `engine.py` (NGD branch) | `grad / grad.norm()` before calling `ngd_step` |
| SNGD | `sngd.py: sngd_step` | `grad_batch / grad_batch.norm()` inside the function |

### What is frozen vs trained

| Parameter | Module | Trained? |
|---|---|---|
| `W` (first layer) | `TwoLayerFixedA`, `LinearBinary` | **Yes** — `nn.Parameter` |
| `a` (second layer) | `TwoLayerFixedA` | **No** — registered as buffer |

---

## 6. Experiment Pipeline

### Figure 1 (MNIST)

- **Script:** `python -m src.experiments.fig1_mnist01`
- **Dataset:** MNIST {0,1}, n_train=1000, n_test=500, d=784
- **Model:** `LinearBinary(d=784)`, initialized to zero
- **Comparison:** GD (`eta=0.01`, 500 steps) vs NGD (`eta=0.01`, 500 steps)
- **Outputs:** `runs/fig1_mnist01_seed<N>/figure1_like.png`, `summary.json`
- **Plot:** 2 panels — training loss (log scale) and test error vs iteration

### Figure 2 (Synthetic X-shaped)

- **Script:** `python -m src.experiments.fig2_synthetic [--seed N] [--steps_top 1000] [--steps_bottom 2000]`
- **Datasets:** `x_shaped_d2_fig2_top` (d=2) and `x_shaped_d5_fig2_bottom` (d=5)
- **Model:** `TwoLayerFixedA`, m=50 (top) / m=100 (bottom)
- **Comparison:** GD vs NGD, both at `eta=3.0` (top) / `eta=5.0` (bottom)
- **Outputs:** `runs/fig2_synth_seed<N>/figure2_like.png`, `summary.json`
- **Plot:** 2×2 — scatter plots (left) and training loss log-scale curves (right)

### Figure 3 (Stochastic NGD)

- **Script:** `python -m src.experiments.fig3_sngd [--seed N] [--steps_linear 500] [--steps_nn 1000]`
- **Top:** Linear model on signed measurements, 5 batch sizes with paper-specified eta
- **Bottom:** TwoLayerFixedA (m=50) on X-shaped data (d=2, n=40), 5 batch sizes
- **Outputs:** `runs/fig3_sngd_seed<N>/figure3_like.png`, `summary.json`, per-run `metrics.csv`
- **Plot:** 2×2 — linear loss, test error, scatter plot, NN loss

---

## 7. Key Design Decisions

**Fixed second layer (`a` as buffer, not `nn.Parameter`)**
Training only W under fixed random `a` mirrors the theoretical setup analyzed in the paper. It makes the optimization landscape tractable for analysis — adding second-layer training would introduce coupling between layers that complicates the theory.

**Exponential loss (not cross-entropy)**
The exponential loss `exp(−y·φ)` has a gradient that decays exponentially as the margin grows. This makes the gradient norm → 0 very rapidly near convergence, which dramatically highlights the difference between GD (which effectively stops) and NGD (which maintains its fixed step size). Cross-entropy decays only logarithmically and would produce a less stark comparison.

**NGD step size: `eta / ‖∇L‖` (global norm, not per-parameter)**
The normalization is by the total Frobenius norm of the gradient matrix W, not element-wise or layer-wise. This preserves the gradient direction exactly and matches the theoretical update rule in the paper.

**Separation of `gd.py` / `ngd.py` / `sngd.py`**
Each optimizer is a pure function with no side effects — they take tensors in, return new tensors. The engine handles `nn.Parameter` bookkeeping (`apply_update_`). This makes optimizers independently testable and swappable.

**`run_training` as unified engine**
A single training loop for all three optimizers avoids code duplication and makes it easy to run controlled experiments where only the optimizer changes. The engine is model-agnostic through the `.W` / `model(X)` interface convention.

---

## 8. Extension Guide

### Add a new optimizer

1. Create `src/optim/myoptim.py` with a pure function:
   ```python
   def myoptim_step(W: Tensor, grad: Tensor, **kwargs) -> Tensor:
       ...
       return W_new
   ```
2. Add `"myoptim"` to the `OptimType` literal in `engine.py`
3. Add an `elif params.optim == "myoptim":` branch in `run_training`
4. Add to `OptimizerType` in `config/schema.py`

### Change activation function

1. Add your function to `src/models/activations.py`
2. Replace the `leaky_relu` call in `TwoLayerFixedA.forward` and `functional.py`
3. If the activation has parameters (e.g., a learnable slope), add them as buffers or constructor arguments

### Plug in a new dataset

1. Add a generator function to `src/data/synthetic.py` (or a new file) that returns `SyntheticDataset(X, y)` with `y ∈ {±1}` and `X` as `float32`
2. Import and call it in the relevant experiment script
3. Add the dataset name to `DatasetType` in `config/schema.py`

### Extend to deeper networks

1. Create a new module (e.g., `src/models/three_layer.py`) as an `nn.Module` exposing `.W` as the parameter to train (or change the engine to iterate over `model.parameters()`)
2. The engine's `.W` convention would need to be generalized — the simplest approach is to replace `model.W.grad` with `[p.grad for p in model.parameters()]` and apply updates per-parameter, keeping the normalization global across all parameters

---

## 9. Test Coverage Summary

| Test file | What it covers |
|---|---|
| `test_activation.py` | `leaky_relu` values and derivative at positive/negative/zero inputs |
| `test_batching.py` | Full coverage of dataset, shuffle reproducibility |
| `test_data_mnist01.py` | Label mapping, tensor shapes |
| `test_data_synthetic.py` | Shapes and label sets for all dataset generators |
| `test_descent_check.py` | One NGD step produces lower loss than initial |
| `test_forward.py` | Shape correctness, `requires_grad`, match with `functional.py` |
| `test_gradcheck.py` | Autograd vs finite-difference agreement |
| `test_loss_metrics.py` | `exp_loss`, `training_loss_exp`, `classification_error`, `weight_norm` |
| `test_optimizers.py` | GD formula, NGD stepsize scaling, SNGD batch-norm normalization |
| `test_training_engine.py` | Full GD and SNGD end-to-end runs without errors |

Run all tests:
```bash
python -m pytest tests/ -v
```
