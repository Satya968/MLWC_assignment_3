"""Read the perfect channels produced by the MATLAB generator and split them.

The MATLAB script stores the real and imaginary parts as two single-precision
arrays ``Hr``/``Hi`` of shape ``[N_SUB x N_SYM x N]`` inside a ``-v7.3`` file
(which is really HDF5). h5py exposes HDF5 datasets with the axis order
reversed relative to MATLAB, so what we read back has shape
``(N, N_SYM, N_SUB)`` and must be transposed to ``(N, N_SUB, N_SYM)``.

A plain ``.mat`` (pre-v7.3) can also be read through SciPy as a fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import settings as S


def _read_parts(path: Path):
    """Return (Hr, Hi) as (N, N_SUB, N_SYM) float32 arrays."""
    try:
        import h5py

        with h5py.File(path, "r") as fh:
            hr = np.array(fh["Hr"], dtype=np.float32)
            hi = np.array(fh["Hi"], dtype=np.float32)
        # HDF5 gives (N, N_SYM, N_SUB) -> swap the last two axes
        hr = np.swapaxes(hr, 1, 2)
        hi = np.swapaxes(hi, 1, 2)
    except (OSError, ImportError, KeyError):
        from scipy.io import loadmat

        blob = loadmat(path)
        # SciPy keeps MATLAB order (N_SUB, N_SYM, N) -> bring N to the front
        hr = np.rollaxis(blob["Hr"].astype(np.float32), 2)
        hi = np.rollaxis(blob["Hi"].astype(np.float32), 2)
    return hr, hi


def load_perfect(model: str, data_root: Path | None = None) -> np.ndarray:
    """Load the ground-truth channels for one model as complex64.

    Returns an array ``H`` of shape ``(N, N_SUB, N_SYM)``.
    """
    root = Path(data_root) if data_root is not None else S.DATA_ROOT
    hr, hi = _read_parts(root / S.MAT_FILE[model])
    H = hr + 1j * hi
    if H.shape[1:] != (S.N_SUB, S.N_SYM):
        raise ValueError(f"{model}: got grid {H.shape[1:]}, expected "
                         f"{(S.N_SUB, S.N_SYM)}")
    return H.astype(np.complex64)


def partition(n_total: int, seed: int = S.SPLIT_SEED) -> dict[str, np.ndarray]:
    """Deterministic train/val/test index sets from a single permutation.

    Requested sizes are capped so the function also works on a tiny dataset
    (useful for smoke tests): we never ask for more frames than exist.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_tr = min(S.SPLIT["train"], int(0.8 * n_total))
    n_va = min(S.SPLIT["val"], int(0.1 * n_total))
    n_te = min(S.SPLIT["test"], n_total - n_tr - n_va)
    cut1, cut2 = n_tr, n_tr + n_va
    return {
        "train": order[:cut1],
        "val": order[cut1:cut2],
        "test": order[cut2:cut2 + n_te],
    }


def load_splits(model: str, data_root: Path | None = None) -> dict[str, np.ndarray]:
    """Load the channels and return the three splits as separate arrays."""
    H = load_perfect(model, data_root)
    idx = partition(len(H))
    return {name: H[sel] for name, sel in idx.items()}
