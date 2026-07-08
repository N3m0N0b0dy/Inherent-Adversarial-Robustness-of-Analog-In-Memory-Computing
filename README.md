# Inherent Adversarial Robustness of Analog In-Memory Computing

A simulation framework replicating and extending the results of:

> **"The inherent adversarial robustness of analog in-memory computing"**  
> Lammie et al., *Nature Communications* 16, 1756 (2025)  
> https://doi.org/10.1038/s41467-025-56595-2

This repository uses the [IBM Analog Hardware Acceleration Kit (aihwkit)](https://github.com/IBM/aihwkit) to simulate PCM-based analog in-memory computing (AIMC) inference and evaluate adversarial robustness on MNIST and CIFAR-10.

---

## What This Repo Does

Standard digital neural networks are vulnerable to adversarial attacks — small, carefully crafted input perturbations that cause misclassification. Analog in-memory computing chips introduce intrinsic stochastic noise (from device physics), which has been shown to act as a natural defence.

This project:
- Trains FP32 baseline models on MNIST and CIFAR-10
- Applies Hardware-Aware (HWA) retraining with PCM noise injection
- Simulates analog inference at multiple time points after programming (accounting for conductance drift)
- Evaluates adversarial robustness using PGD, Square, and OnePixel attacks
- Reports the **Adversarial Success Rate (ASR)** metric from Lammie et al.

---

## Repository Structure

```
├── aihwkit_sim.py        # MNIST simulation (FP32 → HWA → Analog inference)
├── cifar10_sim.py        # CIFAR-10 simulation with ResNet-9
├── data/                 # Auto-downloaded datasets (MNIST, CIFAR-10)
├── results/              # Saved model checkpoints (.pt files)
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+, CUDA-capable GPU recommended.

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install aihwkit (GPU build recommended — see aihwkit docs)
pip install aihwkit

# 3. Install remaining dependencies
pip install torch torchvision adversarial-robustness-toolbox numpy
```

For GPU-accelerated aihwkit installation, follow the [official guide](https://aihwkit.readthedocs.io/en/latest/install.html).

---

## Usage

### MNIST

```bash
# Full pipeline (train + evaluate)
python aihwkit_sim.py

# Skip training, reuse saved checkpoints
python aihwkit_sim.py --skip_train
```

### CIFAR-10

```bash
# Full pipeline
python cifar10_sim.py

# Skip training
python cifar10_sim.py --skip_train

# Train longer for higher baseline accuracy
python cifar10_sim.py --epochs_fp 100
```

---

## Pipeline

Each script runs three phases:

```
Phase 1: FP32 Baseline
    Train a standard digital model to convergence.
    Evaluate clean accuracy + ASR (adversarial baseline).

Phase 2: HWA Retraining
    Fine-tune with PCMLikeNoiseModel injected during forward pass.
    Simulates the noise the model will see on real hardware.

Phase 3: Analog Inference
    Deploy HWA-trained weights onto a simulated PCM chip.
    Apply drift compensation (GlobalDriftCompensation).
    Evaluate at t = {1s, 1min, 1hr, 1day} after programming.
    Report clean accuracy + ASR for all three attacks.
```

---

## Models

| Dataset  | Architecture | FP32 Target Accuracy |
|----------|-------------|----------------------|
| MNIST    | MLP (784→256→128→10) | ~99% |
| CIFAR-10 | ResNet-9    | ~92% |

---

## Noise Model

The PCM noise model (`PCMLikeNoiseModel`, calibrated to IBM HERMES chip) injects three noise sources during inference:

| Source | Recurrence | Location | Input-Dependent |
|--------|-----------|----------|----------------|
| Programming noise | Non-recurrent | Weight | Yes |
| Drift (1/f + RTN) | Non-recurrent | Weight | Yes |
| Read noise | Recurrent | Weight | Yes |
| Output noise | Recurrent | Output | No |

Conductance drift follows:
```
g_drift(t) = g_prog × (t / t₀)^{−ν},   ν ~ N(μ_ν(g), σ_ν(g))
```

---

## Adversarial Attacks

| Attack | Type | Norm | Description |
|--------|------|------|-------------|
| PGD | White-box | L∞ | Iterative gradient-based (strongest baseline) |
| Square | Black-box | L∞ | Query-efficient random search |
| OnePixel | Black-box | L0 | Perturbs a single pixel |

**Adversarial Success Rate (ASR):** fraction of correctly-classified clean samples that are misclassified after attack. Lower = more robust.

---

## Expected Results

Reproducing the trend from Lammie et al. Fig. 2d–f:

| Platform | Clean Acc | PGD ASR |
|----------|-----------|---------|
| FP32 Original | high | high |
| HWA Retrained (FP) | ↓ slightly | ↓ |
| Analog (t=1s) | ↓ | ↓↓ |
| Analog (t=1hr) | ↓↓ | ↓↓ |

Analog inference shows lower ASR than digital at the cost of a small accuracy drop, which `GlobalDriftCompensation` partially recovers.

---

## References

- Lammie et al. (2025). *The inherent adversarial robustness of analog in-memory computing.* Nature Communications. https://doi.org/10.1038/s41467-025-56595-2
- Le Gallo et al. (2023). *A 64-core mixed-signal in-memory compute chip based on phase-change memory.* Nature Electronics.
- IBM aihwkit documentation: https://aihwkit.readthedocs.io
- Adversarial Robustness Toolbox: https://adversarial-robustness-toolbox.readthedocs.io

---

## License

MIT License. See `LICENSE` for details.
