"""Lattice pilot patterns, additive noise and least-squares at the pilots.

Received resource element (unit-power pilots, E|H|^2 = 1):

    Y[i,k] = H[i,k] * X[i,k] + Z[i,k],   Z ~ CN(0, sigma^2),   SNR = 1/sigma^2

With X = 1 at every pilot, the least-squares estimate is simply the received
pilot, h_LS = Y/X = H + Z.  A diamond (lattice) layout is used: pilot-bearing
OFDM symbols carry pilots on a regular subcarrier grid, and consecutive pilot
symbols are offset by half the spacing so the holes interleave.  The 48-pilot
case matches LTE cell-specific reference signals on antenna port 0.
"""
from __future__ import annotations

import numpy as np

from . import settings as S

# density -> (pilot symbols, subcarrier spacing, stagger between symbols)
_LAYOUT = {
    24: ([0, 4, 7, 11], 12, 6),
    36: ([0, 4, 7, 11], 8, 4),
    48: ([0, 4, 7, 11], 6, 3),
    72: ([0, 2, 4, 7, 9, 11], 6, 3),
}


def grid_positions(pilots: int):
    """Row (subcarrier) and column (symbol) indices of every pilot.

    Returns two 1-D integer arrays of length ``pilots``.
    """
    symbols, spacing, stagger = _LAYOUT[pilots]
    rows, cols = [], []
    for order, sym in enumerate(symbols):
        start = stagger if (order % 2) else 0
        sub = np.arange(start, S.N_SUB, spacing)
        rows.append(sub)
        cols.append(np.full(sub.size, sym))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    if rows.size != pilots:
        raise ValueError(f"layout for {pilots} produced {rows.size} pilots")
    return rows, cols


def flat_positions(pilots: int) -> np.ndarray:
    """Pilot indices into a row-major flattened (N_SUB*N_SYM) grid."""
    rows, cols = grid_positions(pilots)
    return rows * S.N_SYM + cols


def sigma2(snr_db: float) -> float:
    """Noise variance for a target SNR in dB, given unit signal power."""
    return 10.0 ** (-snr_db / 10.0)


def _rng_for(snr_db: float, seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = S.NOISE_SEED + int(round(10 * snr_db))
    return np.random.default_rng(seed)


def least_squares(H: np.ndarray, pilots: int, snr_db: float,
                  seed: int | None = None) -> np.ndarray:
    """Noisy LS estimates at the pilot positions.

    ``H`` is (N, N_SUB, N_SYM); the result is (N, pilots) complex64, one row
    of pilot observations per frame.  Passing an explicit ``seed`` lets the
    caller pin the noise realisation (used so that every estimator sees the
    same test noise).
    """
    rng = _rng_for(snr_db, seed)
    pos = flat_positions(pilots)
    h_pilot = H.reshape(len(H), -1)[:, pos]
    s2 = sigma2(snr_db)
    noise = np.sqrt(s2 / 2.0) * (
        rng.standard_normal(h_pilot.shape) + 1j * rng.standard_normal(h_pilot.shape)
    )
    return (h_pilot + noise).astype(np.complex64)   # X = 1 so Y/X = H + Z
