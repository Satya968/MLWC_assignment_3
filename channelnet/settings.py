"""Experiment-wide constants.

Keeping every tunable number in one module means the whole experiment is
described in a single place and the scripts never hard-code a magic value.
Input/output locations are read from environment variables so the same code
runs unchanged on a laptop, a cluster node or a Kaggle kernel:

    CHAN_DATA   directory holding H_VehA.mat / H_SUI5.mat   (default ./data)
    CHAN_OUT    directory where figures/ and weights/ are written (default .)
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------- locations
DATA_ROOT = Path(os.environ.get("CHAN_DATA", "./data"))
OUT_ROOT = Path(os.environ.get("CHAN_OUT", "."))
FIG_DIR = OUT_ROOT / "results"
WEIGHT_DIR = OUT_ROOT / "checkpoints"

MAT_FILE = {"VehA": "H_VehA.mat", "SUI5": "H_SUI5.mat"}

# ----------------------------------------------------------------- OFDM grid
N_SUB = 72   # subcarriers -> image height (frequency axis)
N_SYM = 14   # OFDM symbols -> image width  (time axis)
N_RE = N_SUB * N_SYM

# --------------------------------------------------------------- data splits
SPLIT = {"train": 32000, "val": 4000, "test": 4000}
SPLIT_SEED = 0

# ---------------------------------------------------------------- experiments
# Coarse sweep used for the quick-look baseline plot.
SNR_COARSE = [0, 5, 10, 15, 20, 25, 30]
# Fine sweep (2.5 dB) used for the paper-style Fig. 3/4 curves.
SNR_FINE = [round(0.0 + 2.5 * k, 1) for k in range(13)]   # 0 .. 30
# The two SNRs at which networks are trained.
TRAIN_SNR = [12.0, 22.0]
# Pilot densities. 48 is the headline comparison; the rest form the sweep.
PILOTS_MAIN = 48
PILOTS_SWEEP = [24, 36, 48, 72]
PILOT_SWEEP_SNR = 12.0

# Base seed for all AWGN draws; offsets keep train/val/test noise disjoint.
NOISE_SEED = 1234


def make_dirs() -> None:
    """Create the output directories if they do not already exist."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)


def tag(model: str, snr: float, pilots: int) -> str:
    """Canonical experiment name, e.g. 'VehA_12dB_48p'."""
    return f"{model}_{snr:g}dB_{pilots}p"
