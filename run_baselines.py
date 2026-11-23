"""Task 1 (dataset visualisation) + Task 2 (classical estimation).

Writes to ``figures/``:
    truth_<model>.png        normalised real/imag images of one channel
    pilot_layouts.png        the 24/36/48/72 lattice patterns
    example_<model>.png      truth vs LS-interp vs ideal MMSE at 12 dB
    baseline_mse_snr.png     LS / estimated-MMSE / ideal-MMSE vs SNR
and ``figures/baseline_<model>.json`` with the raw numbers.

Run:  python run_baselines.py [--models VehA SUI5] [--est-frames 1000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channelnet import settings as S
from channelnet.channels import load_splits
from channelnet.pilots import least_squares, grid_positions
from channelnet.classical import (ls_interpolate, autocorr, mmse_matrix,
                                   mmse_apply, autocorr_from_noisy, complex_mse)


# ------------------------------------------------------------------ figures
def draw_truth(H, model):
    frame = H[0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2))
    for ax, part, name in zip(axes, [frame.real, frame.imag], ["Real", "Imag"]):
        norm = (part - part.min()) / (np.ptp(part) + 1e-12)
        im = ax.imshow(norm, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"{model}: {name}(H), normalised")
        ax.set_xlabel("OFDM symbol")
        ax.set_ylabel("Subcarrier")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / f"truth_{model}.png", dpi=150)
    plt.close(fig)


def draw_layouts():
    densities = list((24, 36, 48, 72))
    fig, axes = plt.subplots(1, len(densities), figsize=(3 * len(densities), 4.3))
    for ax, p in zip(axes, densities):
        board = np.zeros((S.N_SUB, S.N_SYM))
        rows, cols = grid_positions(p)
        board[rows, cols] = 1.0
        ax.imshow(board, origin="lower", aspect="auto", cmap="Greys")
        ax.set_title(f"{p} pilots")
        ax.set_xlabel("OFDM symbol")
        ax.set_ylabel("Subcarrier")
    fig.suptitle("Lattice pilot layouts on the 72x14 grid")
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / "pilot_layouts.png", dpi=150)
    plt.close(fig)


def draw_example(H_true, H_ls, H_mmse, model):
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.2))
    lo, hi = H_true[0].real.min(), H_true[0].real.max()
    panels = [(H_true[0], "Ground truth"),
              (H_ls[0], "LS + interpolation"),
              (H_mmse[0], "Ideal MMSE")]
    for ax, (grid, title) in zip(axes, panels):
        ax.imshow(grid.real, origin="lower", aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("OFDM symbol")
    axes[0].set_ylabel("Subcarrier")
    fig.suptitle(f"{model}: Real(H), 48 pilots, 12 dB")
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / f"example_{model}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- evaluation
def run_model(model: str, est_frames: int) -> dict:
    data = load_splits(model)
    H_train, H_test = data["train"], data["test"]
    print(f"[{model}] train {H_train.shape}  test {H_test.shape}")

    draw_truth(H_test, model)
    R_true = autocorr(H_train)
    P = S.PILOTS_MAIN

    out = {"snr_db": list(S.SNR_COARSE), "pilots": P,
           "LS": [], "MMSE_ideal": [], "MMSE_est": []}

    for snr in S.SNR_COARSE:
        h_ls = least_squares(H_test, P, snr)
        H_ls = ls_interpolate(h_ls, P)
        H_mi = mmse_apply(h_ls, mmse_matrix(R_true, P, snr))
        h_few = least_squares(H_train[:est_frames], P, snr, seed=99 + snr)
        R_hat = autocorr_from_noisy(ls_interpolate(h_few, P), P, snr)
        H_me = mmse_apply(h_ls, mmse_matrix(R_hat, P, snr))
        out["LS"].append(complex_mse(H_ls, H_test))
        out["MMSE_ideal"].append(complex_mse(H_mi, H_test))
        out["MMSE_est"].append(complex_mse(H_me, H_test))
        print(f"  {snr:2d} dB | LS {out['LS'][-1]:.3e} | "
              f"MMSE* {out['MMSE_ideal'][-1]:.3e} | "
              f"MMSE~ {out['MMSE_est'][-1]:.3e}")

    h_ls = least_squares(H_test[:1], P, 12)
    draw_example(H_test[:1], ls_interpolate(h_ls, P),
                 mmse_apply(h_ls, mmse_matrix(R_true, P, 12)), model)

    snr = S.PILOT_SWEEP_SNR
    sweep = {"snr_db": snr, "pilots": list(S.PILOTS_SWEEP), "LS": [], "MMSE_ideal": []}
    for p in S.PILOTS_SWEEP:
        h_ls = least_squares(H_test, p, snr)
        sweep["LS"].append(complex_mse(ls_interpolate(h_ls, p), H_test))
        sweep["MMSE_ideal"].append(
            complex_mse(mmse_apply(h_ls, mmse_matrix(R_true, p, snr)), H_test))
    out["pilot_sweep"] = sweep
    print(f"  pilot sweep @ {snr} dB: {sweep}")

    with open(S.FIG_DIR / f"baseline_{model}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["VehA", "SUI5"])
    ap.add_argument("--est-frames", type=int, default=1000)
    args = ap.parse_args()

    S.make_dirs()
    draw_layouts()

    fig, axes = plt.subplots(1, len(args.models),
                             figsize=(6 * len(args.models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], args.models):
        res = run_model(model, args.est_frames)
        for key, style, lab in [("LS", "o-", "LS + interp."),
                                ("MMSE_est", "s-", "Estimated MMSE"),
                                ("MMSE_ideal", "^-", "Ideal MMSE")]:
            ax.semilogy(res["snr_db"], res[key], style, label=lab)
        ax.set_title(f"{model}, {S.PILOTS_MAIN} pilots")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("MSE")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(S.FIG_DIR / "baseline_mse_snr.png", dpi=150)
    plt.close(fig)
    print("baselines done ->", S.FIG_DIR)


if __name__ == "__main__":
    main()
