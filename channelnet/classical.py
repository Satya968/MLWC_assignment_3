"""Classical estimators: LS + 2-D interpolation, and MMSE (ideal / estimated).

Interpolation.  Scattered-point linear interpolation of the pilot values onto
the full grid is a *linear* operation, so the whole map can be written as one
constant matrix ``B`` with ``vec(H_interp) = B @ h_pilots``.  We build ``B``
once per pilot layout by interpolating the canonical unit vectors e_k: column
k of ``B`` is the grid obtained when only pilot k is 1 and the rest are 0.
Grid points outside the convex hull of the pilots fall back to the nearest
pilot.  The same interpolated image is what feeds the SRCNN stage.

MMSE.  With unit pilots the linear MMSE estimator is

    h_MMSE = A h_LS,   A = R_hp (R_pp + sigma^2 I)^-1,

where ``R`` is the channel autocorrelation vec(H)vec(H)^H.  Two flavours:

  * ideal      - R from the (large) set of true training channels; an
                 unrealisable lower reference.
  * estimated  - R from a handful of noisy, interpolated frames with the known
                 noise contribution sigma^2 B B^H removed; the practical case.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from . import settings as S
from .pilots import grid_positions, flat_positions, sigma2

_B_CACHE: dict[int, np.ndarray] = {}


# ----------------------------------------------------------- interpolation
def interp_operator(pilots: int) -> np.ndarray:
    """Constant matrix ``B`` of shape (N_RE, pilots): vec(H_hat) = B h_pilot."""
    if pilots in _B_CACHE:
        return _B_CACHE[pilots]

    rows, cols = grid_positions(pilots)
    src = np.column_stack([rows, cols]).astype(float)
    rr, cc = np.meshgrid(np.arange(S.N_SUB), np.arange(S.N_SYM), indexing="ij")
    dst = np.column_stack([rr.ravel(), cc.ravel()]).astype(float)

    B = np.empty((S.N_RE, pilots), dtype=np.float64)
    for k in range(pilots):
        basis = np.zeros(pilots)
        basis[k] = 1.0
        lin = griddata(src, basis, dst, method="linear")
        near = griddata(src, basis, dst, method="nearest")
        B[:, k] = np.where(np.isnan(lin), near, lin)

    _B_CACHE[pilots] = B
    return B


def ls_interpolate(h_pilot: np.ndarray, pilots: int) -> np.ndarray:
    """Apply LS + 2-D interpolation. (N, pilots) -> (N, N_SUB, N_SYM)."""
    B = interp_operator(pilots)
    full = h_pilot @ B.T
    return full.reshape(len(h_pilot), S.N_SUB, S.N_SYM).astype(np.complex64)


# ------------------------------------------------------------------- MMSE
def autocorr(H: np.ndarray) -> np.ndarray:
    """Sample autocorrelation R = E[vec(H) vec(H)^H] over the frames.

    R[d, d'] = E[ h_d * conj(h_d') ], i.e. v^T v* / N (not v* ^T v).
    """
    v = H.reshape(len(H), -1).astype(np.complex128)
    return (v.T @ v.conj()) / len(v)


def mmse_matrix(R: np.ndarray, pilots: int, snr_db: float) -> np.ndarray:
    """Estimator matrix A = R_hp (R_pp + sigma^2 I)^-1 (solved, not inverted)."""
    pos = flat_positions(pilots)
    R_hp = R[:, pos]                 # (N_RE, pilots)
    R_pp = R[np.ix_(pos, pos)]       # (pilots, pilots)
    reg = R_pp + sigma2(snr_db) * np.eye(len(pos))
    # A = R_hp reg^-1  <=>  A^T = reg^-T R_hp^T
    return np.linalg.solve(reg.conj().T, R_hp.conj().T).conj().T


def mmse_apply(h_pilot: np.ndarray, A: np.ndarray) -> np.ndarray:
    """H_hat = A h_pilot, reshaped to the grid."""
    full = h_pilot @ A.T
    return full.reshape(len(h_pilot), S.N_SUB, S.N_SYM).astype(np.complex64)


def autocorr_from_noisy(H_interp: np.ndarray, pilots: int,
                        snr_db: float) -> np.ndarray:
    """Autocorrelation recovered from noisy interpolated frames.

    The interpolated noisy channel is B(h + z), whose correlation is
    B R_pp B^H + sigma^2 B B^H.  Subtracting the known second term leaves a
    (smoothed, biased) estimate of the channel correlation - exactly what a
    receiver with no access to the true channel could build.
    """
    R = autocorr(H_interp)
    B = interp_operator(pilots)
    return R - sigma2(snr_db) * (B @ B.T)


# ----------------------------------------------------------------- metric
def complex_mse(H_hat: np.ndarray, H_ref: np.ndarray) -> float:
    """Mean |H_hat - H|^2 over every frame and resource element."""
    return float(np.mean(np.abs(H_hat - H_ref) ** 2))
