# NGD Two-Layer NN (Taheri & Thrampoulidis, 2023) — Reproduction

This repo reproduces the experiments in `2305.13471v2`:
**Fast Convergence in Learning Two-Layer Neural Networks with Separable Data**.

## Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/pip install -r requirements.txt
```

Quick sanity check (creates a run directory and prints a deterministic check):

```bash
./.venv/Scripts/python -m src.sanity
```

## Reproducing figures (added in later phases)

- Figure 1 (MNIST 0 vs 1): `python -m src.experiments.fig1_mnist01`
- Figure 2 (synthetic GMM): `python -m src.experiments.fig2_synthetic`
- Figure 3 (stochastic NGD): `python -m src.experiments.fig3_sngd`

Quick smoke runs:

```bash
python -m src.experiments.fig1_mnist01 --seed 0 --steps 5 --mnist_root data
python -m src.experiments.fig2_synthetic --seed 0 --steps_top 50 --steps_bottom 100
python -m src.experiments.fig3_sngd --seed 0 --steps_linear 40 --steps_nn 80
```

Expected qualitative behavior:
- NGD training loss decays ~exponentially vs GD much slower.
- NGD weight norm grows roughly linearly in iteration.
- Stochastic NGD approaches full-batch NGD behavior as batch size increases.

