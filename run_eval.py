"""Task 6: evaluate everything from the saved checkpoints.

For each channel model and a fine 0-30 dB sweep (48 pilots) it measures the
complex-channel MSE of LS, estimated MMSE, ideal MMSE, the SRCNN-only output,
and the full ChannelNet trained at 12 dB and at 22 dB - all on the same test
frames and the same noise.  It also sweeps the pilot count at 12 dB, draws
example estimated images, and dumps every number to JSON and CSV.

Run:  python run_eval.py [--models VehA SUI5] [--switch-db 17]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channelnet import settings as S
from channelnet.channels import load_splits
from channelnet.imaging import to_images, from_images
from channelnet.pilots import least_squares
from channelnet.classical import (ls_interpolate, autocorr, mmse_matrix,
                                   mmse_apply, autocorr_from_noisy, complex_mse)
from channelnet.nets import SRCNN, DnCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelNetRunner:
    """Holds a trained SR (+ optional DN) and runs inference on complex grids."""

    def __init__(self, sr, dn, scale):
        self.sr, self.dn, self.scale = sr, dn, scale

    @classmethod
    def load(cls, model, snr, pilots):
        name = S.tag(model, snr, pilots)
        sr_path = S.WEIGHT_DIR / f"{name}_sr.pt"
        dn_path = S.WEIGHT_DIR / f"{name}_dn.pt"
        if not sr_path.exists():
            return None
        ck = torch.load(sr_path, map_location=DEVICE, weights_only=False)
        sr = SRCNN().to(DEVICE)
        sr.load_state_dict(ck["state_dict"])
        sr.eval()
        dn = None
        if dn_path.exists():
            dn = DnCNN().to(DEVICE)
            dn.load_state_dict(torch.load(dn_path, map_location=DEVICE,
                                          weights_only=False)["state_dict"])
            dn.eval()
        return cls(sr, dn, ck["scale"])

    @torch.no_grad()
    def __call__(self, H_input, batch=2048):
        """Return (SRCNN-only estimate, full ChannelNet estimate)."""
        X = torch.from_numpy(to_images(H_input) / self.scale).float()
        sr_out, full_out = [], []
        for i in range(0, len(X), batch):
            z = self.sr(X[i:i + batch].to(DEVICE))
            sr_out.append(z.cpu())
            full_out.append((self.dn(z) if self.dn else z).cpu())
        sr_img = torch.cat(sr_out).numpy() * self.scale
        full_img = torch.cat(full_out).numpy() * self.scale
        return from_images(sr_img), from_images(full_img)


def evaluate(model, switch_db, est_frames=1000):
    data = load_splits(model)
    H_train, H_test = data["train"], data["test"]
    R_true = autocorr(H_train)
    P = S.PILOTS_MAIN

    runners = {snr: ChannelNetRunner.load(model, snr, P) for snr in S.TRAIN_SNR}
    lo_snr, hi_snr = S.TRAIN_SNR

    res = {"snr_db": [float(s) for s in S.SNR_FINE],
           "LS": [], "MMSE_est": [], "MMSE_ideal": [],
           f"SR_{lo_snr:g}": [], f"SR_{hi_snr:g}": [],
           f"CN_{lo_snr:g}": [], f"CN_{hi_snr:g}": [], "CN_switch": []}

    for snr in S.SNR_FINE:
        h_ls = least_squares(H_test, P, snr)         # shared test noise
        H_ls = ls_interpolate(h_ls, P)
        res["LS"].append(complex_mse(H_ls, H_test))
        res["MMSE_ideal"].append(
            complex_mse(mmse_apply(h_ls, mmse_matrix(R_true, P, snr)), H_test))
        h_few = least_squares(H_train[:est_frames], P, snr,
                              seed=99 + int(round(10 * snr)))
        R_hat = autocorr_from_noisy(ls_interpolate(h_few, P), P, snr)
        res["MMSE_est"].append(
            complex_mse(mmse_apply(h_ls, mmse_matrix(R_hat, P, snr)), H_test))

        for snr_tr in S.TRAIN_SNR:
            runner = runners[snr_tr]
            if runner is None:
                res[f"SR_{snr_tr:g}"].append(None)
                res[f"CN_{snr_tr:g}"].append(None)
                continue
            H_sr, H_cn = runner(H_ls)
            res[f"SR_{snr_tr:g}"].append(complex_mse(H_sr, H_test))
            res[f"CN_{snr_tr:g}"].append(complex_mse(H_cn, H_test))

        lo, hi = res[f"CN_{lo_snr:g}"][-1], res[f"CN_{hi_snr:g}"][-1]
        if lo is None or hi is None:
            res["CN_switch"].append(None)
        else:
            res["CN_switch"].append(lo if snr < switch_db else hi)
        shown = "  ".join(f"{k} {v[-1]:.2e}" for k, v in res.items()
                          if k != "snr_db" and v[-1] is not None)
        print(f"[{model}] {snr:4.1f} dB  {shown}", flush=True)

    _example_figure(model, runners[lo_snr], R_true, P, H_test)
    res["pilot_sweep"] = _pilot_sweep(model, R_true, H_test)
    return res


def _example_figure(model, runner, R_true, P, H_test):
    if runner is None:
        return
    h_ls = least_squares(H_test[:1], P, 12, seed=7)
    H_ls = ls_interpolate(h_ls, P)
    H_sr, H_cn = runner(H_ls)
    H_mi = mmse_apply(h_ls, mmse_matrix(R_true, P, 12))
    grids = [H_test[0], H_ls[0], H_mi[0], H_sr[0], H_cn[0]]
    titles = ["Ground truth", "LS + interp.", "Ideal MMSE", "SRCNN only", "ChannelNet"]
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.5))
    for r, part in enumerate(["real", "imag"]):
        ref = getattr(H_test[0], part)
        for ax, grid, title in zip(axes[r], grids, titles):
            ax.imshow(getattr(grid, part), origin="lower", aspect="auto",
                      vmin=ref.min(), vmax=ref.max())
            ax.set_title(f"{title}\n{part}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{model}: test channel at 12 dB, 48 pilots")
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / f"estimates_{model}.png", dpi=150)
    plt.close(fig)


def _pilot_sweep(model, R_true, H_test):
    snr = S.PILOT_SWEEP_SNR
    sweep = {"snr_db": snr, "pilots": [], "LS": [], "MMSE_ideal": [], "ChannelNet": []}
    for p in S.PILOTS_SWEEP:
        h_ls = least_squares(H_test, p, snr)
        H_ls = ls_interpolate(h_ls, p)
        runner = ChannelNetRunner.load(model, snr, p)
        sweep["pilots"].append(p)
        sweep["LS"].append(complex_mse(H_ls, H_test))
        sweep["MMSE_ideal"].append(
            complex_mse(mmse_apply(h_ls, mmse_matrix(R_true, p, snr)), H_test))
        sweep["ChannelNet"].append(
            complex_mse(runner(H_ls)[1], H_test) if runner else None)
    print(f"[{model}] pilot sweep @ {snr} dB: {sweep}")
    return sweep


# ------------------------------------------------------------------- plots
_STYLE = {
    "LS": ("o-", "LS + interp."),
    "MMSE_est": ("s-", "Estimated MMSE"),
    "MMSE_ideal": ("^-", "Ideal MMSE"),
    "CN_12": ("D-", "ChannelNet (12 dB)"),
    "CN_22": ("v-", "ChannelNet (22 dB)"),
    "SR_12": ("D:", "SRCNN only (12 dB)"),
    "SR_22": ("v:", "SRCNN only (22 dB)"),
    "CN_switch": ("*-", "ChannelNet (switched)"),
}


def _curve(res, keys, title, path):
    plt.figure(figsize=(6.5, 4.8))
    for k in keys:
        series = res.get(k)
        if series and all(v is not None for v in series):
            plt.semilogy(res["snr_db"], series, _STYLE[k][0],
                         label=_STYLE[k][1], ms=5)
    for s in S.TRAIN_SNR:
        plt.axvline(s, color="grey", lw=0.6, ls="--")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["VehA", "SUI5"])
    ap.add_argument("--switch-db", type=float, default=17.0)
    args = ap.parse_args()
    S.make_dirs()

    everything = {}
    for model in args.models:
        res = evaluate(model, args.switch_db)
        everything[model] = res
        _curve(res, ["LS", "MMSE_est", "MMSE_ideal", "CN_12", "CN_22"],
               f"{model}: MSE vs SNR, 48 pilots",
               S.FIG_DIR / f"mse_snr_{model}.png")
        _curve(res, ["LS", "MMSE_est", "MMSE_ideal", "CN_22"],
               f"{model}: deep high-SNR network vs baselines",
               S.FIG_DIR / f"highsnr_{model}.png")
        _curve(res, ["LS", "MMSE_ideal", "SR_12", "CN_12", "SR_22", "CN_22",
                     "CN_switch"],
               f"{model}: SRCNN-only vs full ChannelNet",
               S.FIG_DIR / f"ablation_{model}.png")

    # pilot sweep panel
    fig, axes = plt.subplots(1, len(args.models),
                             figsize=(6 * len(args.models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], args.models):
        sweep = everything[model]["pilot_sweep"]
        for key, style, lab in [("LS", "o-", "LS + interp."),
                                ("MMSE_ideal", "^-", "Ideal MMSE"),
                                ("ChannelNet", "D-", "ChannelNet")]:
            pts = [(p, v) for p, v in zip(sweep["pilots"], sweep[key])
                   if v is not None]
            if pts:
                ax.semilogy(*zip(*pts), style, label=lab)
        ax.set_title(f"{model}: MSE vs #pilots @ {sweep['snr_db']} dB")
        ax.set_xlabel("number of pilots")
        ax.set_ylabel("MSE")
        ax.set_xticks(sweep["pilots"])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / "mse_pilots.png", dpi=150)
    plt.close(fig)

    with open(S.FIG_DIR / "results.json", "w") as fh:
        json.dump(everything, fh, indent=2)
    with open(S.FIG_DIR / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        for model, res in everything.items():
            cols = [k for k in res if k not in ("snr_db", "pilot_sweep")]
            w.writerow([model, "SNR_dB"] + cols)
            for i, snr in enumerate(res["snr_db"]):
                row = [f"{res[k][i]:.4e}" if res[k][i] is not None else ""
                       for k in cols]
                w.writerow(["", snr] + row)
    print("evaluation done ->", S.FIG_DIR)


if __name__ == "__main__":
    main()
