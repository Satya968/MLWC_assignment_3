"""Two-stage ChannelNet training.

Stage 'sr'  : train SRCNN to map the interpolated LS image to the true channel.
Stage 'dn'  : freeze SRCNN, feed its outputs to DnCNN, train DnCNN to the truth.
Stage 'both': do the two in sequence.

A separate network is trained for each (model, SNR, pilot-count).  Each
checkpoint stores the weights, the normalisation scale, the training history
and the hyper-parameters, so testing needs nothing but the checkpoint.

Examples
    python run_train.py --model VehA --snr 12 --pilots 48 --stage both
    python run_train.py --model SUI5 --snr 22 --pilots 48 --stage sr
    quick check:  add  --limit 2000 --epochs-sr 2 --epochs-dn 1
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channelnet import settings as S
from channelnet.channels import load_splits
from channelnet.imaging import pilot_input, to_images, ImageScaler
from channelnet.nets import SRCNN, DnCNN, n_params


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def forward_all(net: nn.Module, X: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    net.eval()
    return torch.cat([net(X[i:i + batch]) for i in range(0, len(X), batch)])


@torch.no_grad()
def mean_sq_error(net: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                  batch: int = 2048) -> float:
    net.eval()
    total = 0.0
    for i in range(0, len(X), batch):
        total += torch.sum((net(X[i:i + batch]) - Y[i:i + batch]) ** 2).item()
    return total / Y.numel()


def train_loop(net, X_tr, Y_tr, X_va, Y_va, *, epochs, lr, batch,
               patience, amp, name):
    """Adam + MSE with plateau LR decay and best-validation early stopping."""
    device = X_tr.device
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(2, patience // 3))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.MSELoss()

    history = {"epoch": [], "train": [], "val": [], "lr": []}
    best_val, best_weights, stale = float("inf"), None, 0
    n = len(X_tr)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        order = torch.randperm(n, device=device)
        running, seen = 0.0, 0
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            xb, yb = X_tr[sel], Y_tr[sel]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(net(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(sel)
            seen += len(sel)

        tr = running / seen
        va = mean_sq_error(net, X_va, Y_va)
        sched.step(va)
        history["epoch"].append(ep)
        history["train"].append(tr)
        history["val"].append(va)
        history["lr"].append(opt.param_groups[0]["lr"])

        mark = ""
        if va < best_val:
            best_val, stale, mark = va, 0, " *"
            best_weights = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        print(f"[{name}] ep {ep:3d}  train {tr:.4e}  val {va:.4e}  "
              f"lr {history['lr'][-1]:.1e}  {time.time()-t0:4.1f}s{mark}",
              flush=True)
        if stale >= patience:
            print(f"[{name}] early stop after {patience} stale epochs")
            break

    net.load_state_dict(best_weights)
    return history, best_val


def plot_curves(history, title, path):
    plt.figure(figsize=(5.5, 4))
    plt.semilogy(history["epoch"], history["train"], label="train")
    plt.semilogy(history["epoch"], history["val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("normalised MSE")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VehA", choices=list(S.MAT_FILE))
    ap.add_argument("--snr", type=float, default=12.0)
    ap.add_argument("--pilots", type=int, default=S.PILOTS_MAIN)
    ap.add_argument("--stage", default="both", choices=["sr", "dn", "both"])
    ap.add_argument("--epochs-sr", type=int, default=100)
    ap.add_argument("--epochs-dn", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap train frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    S.make_dirs()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    name = S.tag(args.model, args.snr, args.pilots)
    print(f"device {device} | gpus {torch.cuda.device_count()} | amp {amp} | {name}")

    # ---------------------------------------------------------------- data
    data = load_splits(args.model)
    H_tr, H_va = data["train"], data["val"]
    if args.limit:
        H_tr = H_tr[:args.limit]
        H_va = H_va[:max(200, args.limit // 8)]

    scaler = ImageScaler.fit(H_tr)
    push = lambda arr: torch.from_numpy(arr).float().to(device)
    X_tr = push(scaler.encode(pilot_input(H_tr, args.pilots, args.snr, "train")))
    X_va = push(scaler.encode(pilot_input(H_va, args.pilots, args.snr, "val")))
    Y_tr = push(to_images(H_tr) / scaler.scale)
    Y_va = push(to_images(H_va) / scaler.scale)
    print(f"images: train {tuple(X_tr.shape)} val {tuple(X_va.shape)} "
          f"scale {scaler.scale:.4f}")

    meta = {"model": args.model, "snr_db": args.snr, "pilots": args.pilots,
            "scale": scaler.scale, "seed": args.seed, "lr": args.lr,
            "batch": args.batch, "n_train": len(H_tr), "n_val": len(H_va)}
    sr_path = S.WEIGHT_DIR / f"{name}_sr.pt"
    dn_path = S.WEIGHT_DIR / f"{name}_dn.pt"
    # per-pixel normalised MSE -> complex-channel MSE
    denorm = 2 * scaler.scale ** 2

    # -------------------------------------------------------- stage 1: SRCNN
    if args.stage in ("sr", "both"):
        sr = SRCNN().to(device)
        print(f"SRCNN params {n_params(sr)} | "
              f"input val MSE {mean_sq_error(nn.Identity(), X_va, Y_va):.4e}")
        hist, best = train_loop(sr, X_tr, Y_tr, X_va, Y_va,
                                epochs=args.epochs_sr, lr=args.lr,
                                batch=args.batch, patience=args.patience,
                                amp=amp, name=name + ":sr")
        torch.save({"state_dict": sr.state_dict(), "history": hist,
                    "best_val": best, "epochs": args.epochs_sr, **meta}, sr_path)
        plot_curves(hist, f"SRCNN  {args.model} {args.snr:g}dB {args.pilots}p",
                    S.FIG_DIR / f"loss_{name}_sr.png")
        json.dump(hist, open(S.FIG_DIR / f"loss_{name}_sr.json", "w"))
        print(f"SRCNN complex-MSE (val) {best * denorm:.4e}")

    # -------------------------------------------------- stage 2: DnCNN (frozen SR)
    if args.stage in ("dn", "both"):
        sr = SRCNN().to(device)
        sr.load_state_dict(torch.load(sr_path, map_location=device,
                                      weights_only=False)["state_dict"])
        sr.requires_grad_(False)
        Z_tr, Z_va = forward_all(sr, X_tr), forward_all(sr, X_va)
        del X_tr, X_va

        seed_everything(args.seed)
        dn = DnCNN().to(device)
        wrapped = nn.DataParallel(dn) if torch.cuda.device_count() > 1 else dn
        print(f"DnCNN params {n_params(dn)} | "
              f"SR-only val MSE {mean_sq_error(nn.Identity(), Z_va, Y_va):.4e}")
        hist, best = train_loop(wrapped, Z_tr, Y_tr, Z_va, Y_va,
                                epochs=args.epochs_dn, lr=args.lr,
                                batch=args.batch, patience=args.patience,
                                amp=amp, name=name + ":dn")
        torch.save({"state_dict": dn.state_dict(), "history": hist,
                    "best_val": best, "epochs": args.epochs_dn, **meta}, dn_path)
        plot_curves(hist, f"DnCNN  {args.model} {args.snr:g}dB {args.pilots}p",
                    S.FIG_DIR / f"loss_{name}_dn.png")
        json.dump(hist, open(S.FIG_DIR / f"loss_{name}_dn.json", "w"))
        print(f"ChannelNet complex-MSE (val) {best * denorm:.4e}")


if __name__ == "__main__":
    main()
