"""
cifar10_sim.py
==============
Replicates Lammie et al. (Nat. Commun. 2025) on CIFAR-10 using aihwkit.

Pipeline:
  1. Train ResNet-9 (FP32 baseline)
  2. HWA retrain with PCMLikeNoiseModel
  3. Analog inference at t = {1s, 1min, 1hr, 1day}
  4. Report clean accuracy + ASR (PGD, Square, OnePixel)

Usage
-----
  pip install aihwkit adversarial-robustness-toolbox torch torchvision
  python cifar10_sim.py
  python cifar10_sim.py --skip_train          # reuse saved checkpoints
  python cifar10_sim.py --epochs_fp 100       # longer baseline training
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# aihwkit
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn import AnalogSequential
from aihwkit.nn.conversion import convert_to_analog

# ART attacks
from art.attacks.evasion import (
    ProjectedGradientDescent,
    SquareAttack,
    PixelAttack,
)
from art.estimators.classification import PyTorchClassifier

# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE      = 128
EPOCHS_FP       = 60        # paper trains ~60–100 epochs on CIFAR-10
EPOCHS_HWA      = 10
LR              = 0.01
T_INFERENCES    = [1.0, 60.0, 3600.0, 86400.0]
N_ADV_SAMPLES   = 1000      # subset for ASR (full set = 10000)
RESULTS_DIR     = "results"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── ResNet-9 ──────────────────────────────────────────────────────────────────
# Compact architecture used in the paper for CIFAR-10 (~92% FP accuracy)

def conv_bn(in_ch, out_ch, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn(ch, ch, kernel_size=3, padding=1),
            conv_bn(ch, ch, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class ResNet9(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3,  64,  kernel_size=3, padding=1),   # 32x32
            conv_bn(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),                               # 16x16
            ResBlock(128),
            conv_bn(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),                               # 8x8
            conv_bn(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),                               # 4x4
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1),                       # 1x1
            nn.Flatten(),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.net(x)

# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders():
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.CIFAR10("data", train=True,  download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── RPU Config ────────────────────────────────────────────────────────────────

def build_rpu_config():
    rpu = InferenceRPUConfig()
    rpu.forward.inp_res       = 1 / 64.     # 6-bit DAC
    rpu.forward.out_res       = 1 / 256.    # 8-bit ADC
    rpu.forward.w_noise_type  = WeightNoiseType.ADDITIVE_CONSTANT
    rpu.forward.w_noise       = 0.02        # short-term recurrent weight noise
    rpu.forward.out_noise     = 0.02        # output noise
    rpu.noise_model           = PCMLikeNoiseModel(g_max=25.0)
    rpu.drift_compensation    = GlobalDriftCompensation()
    return rpu

# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (model(x).argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── ASR ───────────────────────────────────────────────────────────────────────

def build_art_classifier(model):
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
        clip_values=(0.0, 1.0),
        preprocessing=(MEAN, STD),    # data is already normalised
    )

def collect_samples(loader, n):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(t.size(0) for t in xs) >= n:
            break
    return torch.cat(xs)[:n].numpy(), torch.cat(ys)[:n].numpy()

def compute_asr(model, X, Y, attack) -> float:
    """Fraction of correctly-classified samples fooled by attack."""
    model.eval()
    with torch.no_grad():
        clean_preds = model(torch.tensor(X).to(DEVICE)).argmax(1).cpu().numpy()
    mask     = clean_preds == Y
    X_c, Y_c = X[mask], Y[mask]
    if len(X_c) == 0:
        return 0.0
    X_adv    = attack.generate(X_c)
    with torch.no_grad():
        adv_preds = model(torch.tensor(X_adv).to(DEVICE)).argmax(1).cpu().numpy()
    return float((adv_preds != Y_c).mean())

def run_asr_suite(model, X, Y) -> dict:
    """Run PGD, Square, OnePixel and return ASR for each."""
    art = build_art_classifier(model)
    eps = 8 / 255    # standard CIFAR-10 L-inf budget

    attacks = {
        "PGD":      ProjectedGradientDescent(art, norm=np.inf, eps=eps,
                        eps_step=2/255, max_iter=20, verbose=False),
        "Square":   SquareAttack(art, norm=np.inf, eps=eps,
                        max_iter=1000, verbose=False),
        "OnePixel": PixelAttack(art, th=1, es=1, verbose=False),
    }
    return {name: compute_asr(model, X, Y, atk) for name, atk in attacks.items()}

# ── Main ──────────────────────────────────────────────────────────────────────

def print_row(label, acc, asr_dict):
    asrs = "  ".join(f"{v*100:6.2f}%" for v in asr_dict.values())
    print(f"  {label:<26} {acc*100:6.2f}%    {asrs}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_fp",  type=int, default=EPOCHS_FP)
    parser.add_argument("--epochs_hwa", type=int, default=EPOCHS_HWA)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    train_loader, test_loader = get_loaders()
    X_test, Y_test = collect_samples(test_loader, N_ADV_SAMPLES)
    criterion = nn.CrossEntropyLoss()

    results = {}   # label → {acc, asr}

    # ── Phase 1: FP32 baseline ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Phase 1: FP32 Baseline")
    print("="*60)
    fp_model = ResNet9().to(DEVICE)
    ckpt_fp  = f"{RESULTS_DIR}/fp32_cifar10.pt"

    if args.skip_train and os.path.exists(ckpt_fp):
        fp_model.load_state_dict(torch.load(ckpt_fp, map_location=DEVICE))
        print(f"  Loaded {ckpt_fp}")
    else:
        optimizer = optim.SGD(fp_model.parameters(), lr=LR,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = OneCycleLR(optimizer, max_lr=0.1,
                               epochs=args.epochs_fp,
                               steps_per_epoch=len(train_loader))
        for ep in range(args.epochs_fp):
            loss, acc = train_epoch(fp_model, train_loader,
                                    optimizer, criterion, scheduler)
            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1:3d}  loss={loss:.4f}  train_acc={acc:.4f}")
        torch.save(fp_model.state_dict(), ckpt_fp)

    fp_acc = evaluate(fp_model, test_loader)
    fp_asr = run_asr_suite(fp_model, X_test, Y_test)
    results["FP32 Original"] = {"acc": fp_acc, "asr": fp_asr}
    print(f"\n  FP32 clean accuracy: {fp_acc*100:.2f}%")

    # ── Phase 2: HWA retraining ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Phase 2: HWA Retraining")
    print("="*60)
    rpu_config = build_rpu_config()
    hwa_model  = convert_to_analog(ResNet9(), rpu_config
        ).to(DEVICE)
    hwa_model.load_state_dict(fp_model.state_dict(), strict=False)
    ckpt_hwa = f"{RESULTS_DIR}/hwa_cifar10.pt"

    if args.skip_train and os.path.exists(ckpt_hwa):
        hwa_model.load_state_dict(torch.load(ckpt_hwa, map_location=DEVICE),
                                  strict=False)
        print(f"  Loaded {ckpt_hwa}")
    else:
        optimizer = optim.Adam(hwa_model.parameters(), lr=LR * 0.1)
        for ep in range(args.epochs_hwa):
            loss, acc = train_epoch(hwa_model, train_loader, optimizer, criterion)
            print(f"  Epoch {ep+1:2d}  loss={loss:.4f}  train_acc={acc:.4f}")
        torch.save(hwa_model.state_dict(), ckpt_hwa)

    hwa_acc = evaluate(hwa_model, test_loader)
    hwa_asr = run_asr_suite(hwa_model, X_test, Y_test)
    results["HWA Retrained (FP)"] = {"acc": hwa_acc, "asr": hwa_asr}
    print(f"\n  HWA clean accuracy: {hwa_acc*100:.2f}%")

    # ── Phase 3: Analog inference ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Phase 3: Analog Inference (PCMLikeNoiseModel)")
    print("="*60)

    for t in T_INFERENCES:
        analog_model = convert_to_analog(ResNet9(), rpu_config).to(DEVICE)
        analog_model.load_state_dict(hwa_model.state_dict(), strict=False)
        analog_model = AnalogSequential(analog_model).eval()
        analog_model.drift_analog_weights(t)

        acc = evaluate(analog_model, test_loader)
        asr = run_asr_suite(analog_model, X_test, Y_test)

        label = (f"Analog {t:.0f}s"    if t < 60   else
                 f"Analog {t/60:.0f}min" if t < 3600 else
                 f"Analog {t/3600:.0f}hr" if t < 86400 else "Analog 1day")
        results[label] = {"acc": acc, "asr": asr}
        print(f"  {label}: clean={acc*100:.2f}%  "
              + "  ".join(f"{k}={v*100:.2f}%" for k,v in asr.items()))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  {'Platform':<26} {'Clean Acc':>9}    {'PGD ASR':>8}  {'Square ASR':>10}  {'1px ASR':>8}")
    print("  " + "-"*66)
    for label, r in results.items():
        print_row(label, r["acc"], r["asr"])
    print()

if __name__ == "__main__":
    main()