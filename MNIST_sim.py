"""
aihwkit_sim.py
==============
Replicates the Lammie et al. (Nat. Commun. 2025) pipeline using aihwkit:

  1. Train a digital FP32 baseline model
  2. Hardware-Aware (HWA) retrain with PCMLikeNoiseModel injected
  3. Evaluate analog inference at multiple t_inference values
  4. Report clean accuracy + Adversarial Success Rate (ASR) vs PGD

Designed to be dataset-agnostic: swap DATASET = "CIFAR10" to extend.

Usage
-----
  pip install aihwkit torch torchvision adversarial-robustness-toolbox
  python aihwkit_sim.py
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# aihwkit imports
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn import AnalogSequential
from aihwkit.nn.conversion import convert_to_analog

# ART for adversarial attacks
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# ── Config ────────────────────────────────────────────────────────────────────

DATASET       = "MNIST"      # swap to "CIFAR10" to extend
BATCH_SIZE    = 128
EPOCHS_FP     = 10           # FP32 baseline training epochs
EPOCHS_HWA    = 5            # HWA retraining epochs on top of FP32
LR            = 1e-3
T_INFERENCES  = [1.0, 60.0, 3600.0, 86400.0]   # seconds: 1s, 1min, 1hr, 1day
PGD_EPS       = 0.3          # L-inf budget (MNIST pixel range [0,1])
PGD_STEPS     = 40
PGD_ALPHA     = 0.01
N_ADV_SAMPLES = 1000         # subset for ASR evaluation
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR   = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────

def get_dataloaders(dataset_name: str):
    if dataset_name == "MNIST":
        tf = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
        test_ds  = datasets.MNIST("data", train=False, download=True, transform=tf)
        in_channels, img_size, n_classes = 1, 28, 10

    elif dataset_name == "CIFAR10":
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10("data", train=True,  download=True, transform=tf_train)
        test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=tf_test)
        in_channels, img_size, n_classes = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader, in_channels, img_size, n_classes

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(in_channels: int, img_size: int, n_classes: int) -> nn.Sequential:
    """
    Simple CNN suitable for both MNIST and CIFAR-10.
    For CIFAR-10, add more channels / blocks as needed.
    """
    flat = in_channels * img_size * img_size
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )

# ── RPU Config ────────────────────────────────────────────────────────────────

def build_rpu_config(g_max: float = 25.0) -> InferenceRPUConfig:
    """
    PCM inference config matching IBM HERMES chip defaults from Lammie et al.
    """
    rpu = InferenceRPUConfig()
    # DAC / ADC discretization
    rpu.forward.inp_res  = 1 / 64.     # 6-bit DAC
    rpu.forward.out_res  = 1 / 256.    # 8-bit ADC
    # Short-term (recurrent) weight + output noise
    rpu.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu.forward.w_noise  = 0.02
    rpu.forward.out_noise = 0.02
    # Long-term PCM noise model (programming noise + drift)
    rpu.noise_model = PCMLikeNoiseModel(g_max=g_max)
    # Drift compensation
    rpu.drift_compensation = GlobalDriftCompensation()
    return rpu

# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── ASR (Adversarial Success Rate) ───────────────────────────────────────────

def compute_asr(model, loader, n_samples: int, n_classes: int,
                img_shape: tuple) -> float:
    """
    ASR = fraction of correctly-classified clean samples that are
    misclassified after PGD attack (Carlini et al. metric, Lammie et al.).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    art_model = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=img_shape,
        nb_classes=n_classes,
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
        clip_values=(0.0, 1.0),
    )
    attack = ProjectedGradientDescent(
        estimator=art_model,
        norm=np.inf,
        eps=PGD_EPS,
        eps_step=PGD_ALPHA,
        max_iter=PGD_STEPS,
        verbose=False,
    )

    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x); all_y.append(y)
        if sum(t.size(0) for t in all_x) >= n_samples:
            break
    X = torch.cat(all_x)[:n_samples].numpy()
    Y = torch.cat(all_y)[:n_samples].numpy()

    # Clean predictions
    with torch.no_grad():
        clean_preds = model(torch.tensor(X).to(DEVICE)).argmax(1).cpu().numpy()

    # Keep only correctly classified
    mask        = clean_preds == Y
    X_correct   = X[mask]
    Y_correct   = Y[mask]
    if len(X_correct) == 0:
        return 0.0

    # Generate adversarial examples
    X_adv = attack.generate(X_correct)

    # Evaluate on adversarial
    with torch.no_grad():
        adv_preds = model(torch.tensor(X_adv).to(DEVICE)).argmax(1).cpu().numpy()

    asr = (adv_preds != Y_correct).mean()
    return float(asr)

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default=DATASET)
    parser.add_argument("--epochs_fp",  type=int, default=EPOCHS_FP)
    parser.add_argument("--epochs_hwa", type=int, default=EPOCHS_HWA)
    parser.add_argument("--skip_train", action="store_true",
                        help="Load saved checkpoints instead of training")
    args = parser.parse_args()

    train_loader, test_loader, C, H, n_cls = get_dataloaders(args.dataset)
    img_shape = (C, H, H)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: FP32 baseline ────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f" Phase 1: FP32 Baseline  [{args.dataset}]")
    print("="*55)

    fp_model = build_model(C, H, n_cls).to(DEVICE)
    ckpt_fp  = os.path.join(RESULTS_DIR, f"fp32_{args.dataset}.pt")

    if args.skip_train and os.path.exists(ckpt_fp):
        fp_model.load_state_dict(torch.load(ckpt_fp, map_location=DEVICE))
        print(f"  Loaded from {ckpt_fp}")
    else:
        optimizer = optim.Adam(fp_model.parameters(), lr=LR)
        for ep in range(args.epochs_fp):
            loss, acc = train_epoch(fp_model, train_loader, optimizer, criterion)
            print(f"  Epoch {ep+1:2d}/{args.epochs_fp}  loss={loss:.4f}  acc={acc:.4f}")
        torch.save(fp_model.state_dict(), ckpt_fp)

    fp_acc = evaluate(fp_model, test_loader)
    print(f"\n  FP32 clean accuracy : {fp_acc*100:.2f}%")
    fp_asr = compute_asr(fp_model, test_loader, N_ADV_SAMPLES, n_cls, img_shape)
    print(f"  FP32 PGD ASR        : {fp_asr*100:.2f}%")

    # ── Phase 2: HWA retraining ───────────────────────────────────────────────
    print("\n" + "="*55)
    print(" Phase 2: Hardware-Aware (HWA) Retraining")
    print("="*55)

    rpu_config = build_rpu_config()
    hwa_model  = convert_to_analog(
        build_model(C, H, n_cls), rpu_config
        ).to(DEVICE)

    # Initialise analog weights from the trained FP32 model
    hwa_model.load_state_dict(fp_model.state_dict(), strict=False)

    ckpt_hwa = os.path.join(RESULTS_DIR, f"hwa_{args.dataset}.pt")

    if args.skip_train and os.path.exists(ckpt_hwa):
        hwa_model.load_state_dict(torch.load(ckpt_hwa, map_location=DEVICE), strict=False)
        print(f"  Loaded from {ckpt_hwa}")
    else:
        optimizer = optim.Adam(hwa_model.parameters(), lr=LR * 0.1)
        for ep in range(args.epochs_hwa):
            loss, acc = train_epoch(hwa_model, train_loader, optimizer, criterion)
            print(f"  Epoch {ep+1:2d}/{args.epochs_hwa}  loss={loss:.4f}  acc={acc:.4f}")
        torch.save(hwa_model.state_dict(), ckpt_hwa)

    hwa_acc = evaluate(hwa_model, test_loader)
    print(f"\n  HWA (FP eval) clean accuracy : {hwa_acc*100:.2f}%")
    hwa_asr = compute_asr(hwa_model, test_loader, N_ADV_SAMPLES, n_cls, img_shape)
    print(f"  HWA (FP eval) PGD ASR        : {hwa_asr*100:.2f}%")

    # ── Phase 3: Analog inference at multiple t_inference ────────────────────
    print("\n" + "="*55)
    print(" Phase 3: Analog Inference (PCMLikeNoiseModel)")
    print("="*55)
    print(f"  {'t_inference':>12}  {'Clean Acc':>10}  {'PGD ASR':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

    analog_results = {}
    for t in T_INFERENCES:
        # Program weights fresh for each t
        analog_model = AnalogSequential(
            convert_to_analog(build_model(C, H, n_cls), rpu_config)
        ).to(DEVICE)
        analog_model.load_state_dict(hwa_model.state_dict(), strict=False)
        analog_model.eval()

        # Apply drift + noise at time t
        analog_model.drift_analog_weights(t)

        acc = evaluate(analog_model, test_loader)
        asr = compute_asr(analog_model, test_loader, N_ADV_SAMPLES, n_cls, img_shape)
        analog_results[t] = {"acc": acc, "asr": asr}

        label = (f"{t:.0f}s" if t < 60 else
                 f"{t/60:.0f}min" if t < 3600 else
                 f"{t/3600:.0f}hr" if t < 86400 else "1day")
        print(f"  {label:>12}  {acc*100:>9.2f}%  {asr*100:>9.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(" Summary")
    print("="*55)
    print(f"  Platform              Clean Acc    PGD ASR")
    print(f"  FP32 Original         {fp_acc*100:>7.2f}%    {fp_asr*100:>7.2f}%")
    print(f"  HWA Retrained (FP)    {hwa_acc*100:>7.2f}%    {hwa_asr*100:>7.2f}%")
    for t, r in analog_results.items():
        label = (f"Analog t={t:.0f}s" if t < 60 else
                 f"Analog t={t/3600:.0f}hr" if t >= 3600 else
                 f"Analog t={t/60:.0f}min")
        print(f"  {label:<22}{r['acc']*100:>7.2f}%    {r['asr']*100:>7.2f}%")
    print()


if __name__ == "__main__":
    main()