<div align="center">

# BIAM: Bilevel Interactive Additive Model

**A PyTorch implementation of the Bilevel Interactive Additive Model for learning under missing values, noisy labels, and class imbalance**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-63%20passed-brightgreen)](#-testing)

**English** | [简体中文](./README_zh.md)

</div>

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [🔧 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [🧪 Data Corruption Scenarios](#-data-corruption-scenarios)
- [🔬 Ablation Variants](#-ablation-variants)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing](#-testing)
- [☑️ Todo List](#️-todo-list)
- [🪪 License](#-license)

## ✨ Highlights

Real-world tabular data is rarely clean. **BIAM** tackles the three most common corruptions in a single, interpretable framework:

1. **Missing Values** — explicit missing-indicator intercepts plus missing × feature *interactions*, so the model learns how missingness itself carries signal (supports MCAR / MAR / MNAR).
2. **Noisy Labels** — a **bilevel optimization** scheme learns per-sample weights ν(·;θ) on a clean validation set, automatically down-weighting corrupted training samples.
3. **Class Imbalance** — the same reweighting mechanism dynamically compensates for 1:N imbalanced training distributions, while the test set keeps its natural distribution.

Key design points:

- **Interpretable additive structure**: each feature contributes through a learnable shape function built on hinge bases located at data quantiles.
- **First-order bilevel optimization** (Eq. 4–6): a differentiable *virtual step* makes the upper-level hypergradient cheap — no second-order derivatives.
- **Strong extrapolation**: piecewise-linear hinge shape functions extrapolate far better than piecewise-constant alternatives.
- **BIAM² second-order extension**: gated rank-R pairwise interaction surfaces `F_jk(x_j,x_k) = Σ_r u_jr(x_j)·u_kr(x_k)` built on the *same* hinge basis. Zero-initialized gates with an ℓ0 penalty keep pairs switched off unless the data support them, so the model degrades gracefully to BIAM when no interaction exists, and each pair reports an interpretable strength `s_jk = |γ̄_jk|·‖F_jk‖_F` for interaction discovery (interaction-ranking AUPRC).

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zxlml/BIAM.git
cd BIAM

# (Optional) create a virtual environment
conda create -n biam python=3.10 -y
conda activate biam

# Install dependencies
pip install -r requirements.txt
```

> **Note (Windows)**: if you hit OpenMP duplicate-library errors, set the environment variable before running:
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK='TRUE'
> ```

**Minimal requirements**: Python 3.8+, PyTorch 2.0+, NumPy, scikit-learn, pandas, matplotlib. CPU-only execution is fully supported.

## ⚡ Quick Start

### Command Line

```bash
# Regression with 30% label noise and 20% MCAR missing values
python biam_main.py --task regression --noise_ratio 0.3 --missing_ratio 0.2

# Classification with label flipping, class imbalance and MNAR missingness
python biam_main.py --task classification \
    --noise_ratio 0.2 --imbalance_ratio 0.15 \
    --missing_ratio 0.2 --missing_mechanism MNAR

# Fine-grained control over the noise model (paper Eq.: y <- y + N(mu_e, sigma_e))
python biam_main.py --task regression --noise_mean 1.0 --noise_std 0.5 --seed 42

# Ablation: disable bilevel reweighting (BIAM-B variant)
python biam_main.py --task regression --use_bilevel False
```

### Python API

```python
from utils.biam_config import BIAMConfig
from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer

# 1. Configure
config = BIAMConfig()
config.task = 'regression'
config.noise_ratio, config.missing_ratio = 0.3, 0.2

# 2. Generate corrupted data (train / val / test splits, test stays clean)
generator = BIAMDataGenerator(config)
train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()

# 3. Build model + weighting network; place hinge knots at training quantiles
biam_model = BIAMModel(config, config.device)
weighting_network = BIAMWeightingNetwork(config, config.device)
biam_model.additive_model.set_knots(torch.tensor(train_data[0], dtype=torch.float32))

# 4. Bilevel optimization
optimizer = BIAMOptimizer(config, biam_model, weighting_network)
for epoch in range(config.epochs):
    optimizer.train_epoch(train_loader, val_loader, epoch)
    if epoch % 20 == 0:
        print(optimizer.evaluate(test_data))   # {'mse': ..., 'rmse': ..., 'mae': ...}
```

A full end-to-end demo (training, evaluation, interpretation, visualization) is available:

```bash
python demo_biam.py
```

## 🏗️ Architecture

### Core Modules Overview

```
BIAM/
├── biam_main.py                  # Main entry point (CLI)
├── demo_biam.py                  # End-to-end demo
├── run_tests.py                  # Test-suite runner
├── data/                         # Data processing
│   ├── biam_data_generator.py    # Synthetic data + corruption pipeline (noise/imbalance/missing)
│   ├── biam_binarizer.py         # Missing-value indicators & feature binarization
│   ├── biam_dataset_loader.py    # Real-dataset loading utilities
│   └── biam_data_utils.py        # Data helpers
├── models/                       # Model components
│   ├── biam_additive_model.py    # Additive model: hinge bases, missing intercepts & interactions (Eq. 1)
│   ├── biam_weighting_network.py # Sample-weighting network ν(·;θ)
│   └── biam_model.py             # Top-level model wrapper
├── gradients/                    # Optimization
│   ├── biam_optimizer.py         # First-order bilevel optimizer (Eq. 4–6)
│   ├── biam_bilevel_optimizer.py # Alternative bilevel implementation
│   └── biam_gradient_methods.py  # Gradient utilities
├── utils/                        # Configuration, logging, performance tools
│   ├── biam_config.py            # Central configuration + validation
│   └── biam_logger.py            # Structured logging
├── visualization/                # Shape functions, feature importance, training curves
├── experiments/                  # Paper experiment scripts (stress tests, plots, interpretability)
│   ├── run_new_experiments.py    # Synthetic stress-test protocol (noise families, extremes)
│   ├── run_extension_experiments.py  # Extended regimes & robust baselines
│   ├── run_biam2_experiments.py  # BIAM² pair-interaction protocol (planted interactions, AUPRC, EBM/CatBoost)
│   ├── plot_new_results.py / plot_extension_results.py  # Result figures
│   ├── plot_biam2_analysis.py    # BIAM² interaction figures (pair MSE, recovery, surfaces)
│   ├── interpretability_analysis.py  # Interpretability figures (main effects, missingness offsets, surfaces)
│   └── generate_summary_csv.py   # CSV summary of all runs
├── tests/                        # Unit / functional / simulation / performance tests
```


### The Additive Model (Eq. 1)

Each feature contributes through a shape function; missingness enters both as an intercept and as an interaction with other features' shape functions:

$$
f(x, m; w) = \beta_0 + \sum_{j} f_j(x_j) + \sum_{j} \beta^{\text{miss}}_j \, m_j + \sum_{j,k,\tau} \alpha_{j,k,\tau} \, \mathbb{I}(m_j = 1) \, h(x_k; \eta_\tau)
$$

where $h(x;\eta) = \max(0, x - \eta)$ is a hinge basis whose knot $\eta_\tau$ is placed at a training-data quantile, and $m_j \in \{0,1\}$ indicates missingness of feature $j$.

### First-Order Bilevel Optimization (Eq. 4–6)

| Step | Update | Purpose |
|------|--------|---------|
| **Step 1** | $w^{(t)} = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t-1)}\big)$ | Real lower-level update of model parameters with ν-weighted loss |
| **Step 2** | $\hat{w}(\theta) = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t)}\big)$ | Differentiable *virtual step* (kept in the autograd graph) |
| **Step 3** | $\theta^{(t+1)} = \theta^{(t)} - \gamma_\theta \nabla_\theta L_{\text{val}}\big(\hat{w}(\theta)\big)$ | Upper-level update via hypergradient through the virtual step |

The weighting network ν takes the **batch-standardized per-sample loss** as input and outputs sample weights; weights are mean-normalized so that reweighting preserves the average gradient scale.

## 🧪 Data Corruption Scenarios

Following the simulation protocol of the paper (§4.1), `BIAMDataGenerator` reproduces all three corruption types — **applied to the training set only unless stated otherwise**, with the test set kept clean:

| Scenario | Mechanism | Applied to |
|----------|-----------|------------|
| **Label noise (regression)** | $y \leftarrow y + \epsilon,\ \epsilon \sim \mathcal{N}(\mu_e, \sigma_e)$ for a fraction $r_1$ of samples | Train |
| **Label noise (classification)** | Random label flipping for a fraction $r_1$ of samples | Train |
| **Class imbalance** | Majority : minority resampled to `imbalance_ratio` | Train |
| **Missing values** | MCAR / MAR / MNAR masking at ratio `missing_ratio` | Train / Val / Test |
| **Clean test labels** | Natural class distribution, no flipped labels | Test |

## 📊 Interpretability Analysis

`experiments/interpretability_analysis.py` produces the qualitative figures of the paper: because BIAM is additive, every prediction decomposes into interpretable parts — per-feature shape functions, missing-indicator offsets, and missingness–feature interaction surfaces. The script extracts each part directly from the trained model and compares it against the ground truth on synthetic data.

```bash
# All four figures (synthetic + real), PDF for the paper and PNG for preview
python experiments/interpretability_analysis.py

# Only the real-data figure; only the synthetic figures
python experiments/interpretability_analysis.py --skip-synth
python experiments/interpretability_analysis.py --skip-real
```

| Figure | What it shows |
|--------|---------------|
| `fig_interp_main_clean` | Ground-truth vs. BIAM vs. NAM shape functions on clean synthetic data (8 informative features) |
| `fig_interp_main_outlier` | Same comparison under 10% label outliers — shape recovery survives the corruption |
| `fig_interp_missing` | Recovery of the MNAR missing-indicator offsets $\phi_j$ and a missingness–feature conditional response surface vs. the Bayes surface |
| `fig_interp_real` | Contribution curves of the top continuous features on the UCI Adult census data (logit scale) |

Outputs are written to `exp/interpretability/` (each figure as `.pdf` + `.png`), together with a `diagnostics.json` that records the test MSE / accuracy and the per-feature correlation between fitted and true curves.

## 🧩 BIAM² Interaction Experiments

`experiments/run_biam2_experiments.py` implements the pair-interaction benchmark inspired by the AG-NAM protocol: planted pairwise interactions (product, sine-product, tanh-product) on top of additive components, null (no-interaction) controls, MCAR/MAR/MNAR corruption, and an interaction-discovery metric (ranking AUPRC over the 45 candidate pairs). EBM and CatBoost are included as interaction-aware baselines.

```bash
# Regression (E6) + classification (E7) blocks; appends to exp/results_biam2_E{6,7}.csv
python experiments/run_biam2_experiments.py

# Interaction analysis figures: pair-wise MSE, recovery heatmap + AUPRC, fitted surfaces
python experiments/plot_biam2_analysis.py
```

Figures are written to `exp/biam2/` and copied to the paper's `images/` directory (`.pdf` + `.png`).

#### Two gate schedules

The gated extension ships with **two operating points of the same algorithm** (see §3.4/§7 of the paper and the comment block in `utils/biam_config.py`):

| Schedule | Settings | Used for | Rationale |
|----------|----------|----------|-----------|
| **Robust** (default) | `pair_warmup_epochs=10`, `pair_gate_lr_scale=0.3`, `lambda_pair_l0=5e-2`, `lambda_pair_l2=1e-2` | Additive stress tests (E1–E5): corrupted labels, no interaction guaranteed | Reluctant opening keeps gates closed on corruption; calibrated via `experiments/calibrate_pair_reg.py` (E1/outlier30 MSE 0.36 vs BIAM 1.57) |
| **Discovery** | `pair_warmup_epochs=0`, `pair_gate_lr_scale=1.0`, `lambda_pair_l0=l2=0` | Planted-interaction benchmarks (E6/E7) | The protocol asserts informative pairs exist, so gates are free to lock on; recovers pairs at AUPRC 0.54–0.87 |

The two schedules are **not interchangeable**: `experiments/diag_eager_gates.py` shows that the discovery schedule opens gates on the corrupted minority under extreme label corruption (outlier30 MSE ≈ 52), which is precisely the failure mode the robust schedule prevents.

## 🔬 Ablation Variants

| Variant | Switch | What is removed |
|---------|--------|-----------------|
| **BIAM** (full) | — | — |
| **BIAM-B** | `use_bilevel = False` | Bilevel reweighting → uniform sample weights |
| **BIAM-I** | `use_missing_interactions = False` | Missing × feature interaction terms |
| **BIAM-H** | `basis_type = 'piecewise_constant'` | Piecewise-linear hinge bases → piecewise-constant bases |

## ⚙️ Configuration

All behaviour is controlled via `utils/biam_config.py` (overridable through CLI flags). Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `task` | `'regression'` | `'regression'` or `'classification'` |
| `n_samples` / `n_features` | 1200 / 10 | Synthetic dataset size and dimensionality |
| `noise_ratio` | 0.3 | Fraction of training samples corrupted |
| `noise_mean` / `noise_std` | 1.0 / 0.5 | $\mathcal{N}(\mu_e, \sigma_e)$ of the additive regression noise |
| `imbalance_ratio` | 0.2 | Minority : majority ratio for training-set resampling |
| `missing_ratio` | 0.2 | Fraction of masked entries |
| `missing_mechanism` | `'MCAR'` | `'MCAR'`, `'MAR'`, or `'MNAR'` |
| `n_knots` | 8 | Number of hinge knots per feature (placed at quantiles) |
| `basis_type` | `'piecewise_linear'` | `'piecewise_linear'` or `'piecewise_constant'` (BIAM-H) |
| `use_bilevel` | `True` | `False` disables bilevel reweighting (BIAM-B) |
| `use_missing_interactions` | `True` | `False` removes missing × feature interactions (BIAM-I) |
| `lambda_l2` / `lambda_l0` | 1e-3 / 1e-4 | Smoothness / sparsity penalties on shape functions |
| `lambda_pair_l2` / `lambda_pair_l0` | 1e-2 / 5e-2 | Penalties on the BIAM² pair gates γ (robust schedule) |
| `pair_warmup_epochs` / `pair_gate_lr_scale` | 10 / 0.3 | Robust gate schedule: warm-up freeze and gate learning-rate scale |
| `upper_lr` / `lower_lr` | 0.1 / 0.05 | Upper- (θ) and lower-level (w) learning rates |
| `epochs` / `batch_size` | 200 / 64 | Training schedule |
| `seed` | 42 | Global random seed |

## 🧪 Testing

The project ships with a 63-case test-suite covering unit, functional, simulation and performance levels:

```bash
# Full suite
python -m pytest tests/ -q

# By level
python -m pytest tests/test_biam_models.py -v        # model internals (hinge bases, missing handling, penalties)
python -m pytest tests/test_biam_data.py -v          # corruption pipeline (noise / imbalance / missing)
python -m pytest tests/test_biam_optimizer.py -v     # bilevel optimization correctness
python -m pytest tests/test_biam_simulation.py -v    # end-to-end behaviour vs. paper claims
python -m pytest tests/test_biam_performance.py -v   # scalability

# Or use the bundled runner
python run_tests.py
```

## ☑️ Todo List

- [x] Shape-function visualization for the interaction terms (`experiments/interpretability_analysis.py`)
- [ ] GPU acceleration & mixed-precision training
- [ ] Automatic hyperparameter search for the bilevel learning rates

## 🪪 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
