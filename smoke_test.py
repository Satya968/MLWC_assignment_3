"""End-to-end sanity check on tiny synthetic channels (no MATLAB, no GPU).

Fabricates small H_VehA.mat / H_SUI5.mat in a temporary folder, then exercises
the full chain: load + split, pilots + LS, interpolation, MMSE, image codec, a
one-epoch SRCNN+DnCNN training, and the evaluation runner. It only checks that
everything runs and the numbers are finite and sane - not that the model is any
good. Run:  python smoke_test.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# point the package at a throwaway workspace BEFORE importing it
_TMP = Path(tempfile.mkdtemp(prefix="channelnet_smoke_"))
os.environ["CHAN_DATA"] = str(_TMP / "data")
os.environ["CHAN_OUT"] = str(_TMP / "out")
(_TMP / "data").mkdir(parents=True, exist_ok=True)


def _fake_channels(n, n_sub, n_sym, seed):
    """Smooth-ish complex channels normalised to mean power 1."""
    rng = np.random.default_rng(seed)
    taps = rng.standard_normal((n, 4)) + 1j * rng.standard_normal((n, 4))
    delays = np.arange(4)[None, :]
    freq = np.arange(n_sub)[:, None]
    phase = np.exp(-2j * np.pi * freq * delays / n_sub)          # (n_sub, 4)
    base = taps @ phase.T                                        # (n, n_sub)
    ramp = np.exp(1j * np.linspace(0, 0.3, n_sym))[None, None, :]
    H = base[:, :, None] * ramp
    H /= np.sqrt(np.mean(np.abs(H) ** 2))
    return H.astype(np.complex64)


def _write_mat(path, H):
    """Store like MATLAB -v7.3: datasets read back as (N, N_SYM, N_SUB)."""
    import h5py

    hr = np.ascontiguousarray(H.real.transpose(0, 2, 1))   # (N, N_SYM, N_SUB)
    hi = np.ascontiguousarray(H.imag.transpose(0, 2, 1))
    with h5py.File(path, "w") as fh:
        fh.create_dataset("Hr", data=hr.astype(np.float32))
        fh.create_dataset("Hi", data=hi.astype(np.float32))


def main():
    from channelnet import settings as S
    from channelnet.channels import load_splits, load_perfect
    from channelnet.pilots import least_squares, grid_positions
    from channelnet.classical import (ls_interpolate, autocorr, mmse_matrix,
                                       mmse_apply, autocorr_from_noisy, complex_mse)
    from channelnet.imaging import pilot_input, to_images, from_images, ImageScaler

    S.make_dirs()
    N = 400
    for model, seed in (("VehA", 1), ("SUI5", 2)):
        _write_mat(S.DATA_ROOT / S.MAT_FILE[model], _fake_channels(N, S.N_SUB, S.N_SYM, seed))

    # ---- load / split --------------------------------------------------
    H = load_perfect("VehA")
    assert H.shape == (N, S.N_SUB, S.N_SYM), H.shape
    data = load_splits("VehA")
    sizes = {k: len(v) for k, v in data.items()}
    assert sum(sizes.values()) == N, sizes
    print("load/split OK", sizes)

    # ---- pilots + image round-trip ------------------------------------
    for p in S.PILOTS_SWEEP:
        r, c = grid_positions(p)
        assert r.size == p and r.max() < S.N_SUB and c.max() < S.N_SYM
    H_test = data["test"]
    assert np.allclose(from_images(to_images(H_test)), H_test, atol=1e-5)
    print("pilots + image round-trip OK")

    # ---- classical estimators -----------------------------------------
    P = S.PILOTS_MAIN
    R = autocorr(data["train"])
    for snr in (0, 12, 30):
        h_ls = least_squares(H_test, P, snr)
        e_ls = complex_mse(ls_interpolate(h_ls, P), H_test)
        e_mi = complex_mse(mmse_apply(h_ls, mmse_matrix(R, P, snr)), H_test)
        h_few = least_squares(data["train"][:100], P, snr, seed=99 + snr)
        R_hat = autocorr_from_noisy(ls_interpolate(h_few, P), P, snr)
        e_me = complex_mse(mmse_apply(h_ls, mmse_matrix(R_hat, P, snr)), H_test)
        assert np.isfinite([e_ls, e_mi, e_me]).all()
        assert e_mi <= e_ls + 1e-6, (snr, e_mi, e_ls)   # ideal MMSE never worse
        print(f"classical @ {snr:2d} dB  LS {e_ls:.3e}  MMSE* {e_mi:.3e}  MMSE~ {e_me:.3e}")

    # ---- tiny training + eval runner ----------------------------------
    import torch
    from channelnet.nets import SRCNN, DnCNN

    H_tr = data["train"]
    scaler = ImageScaler.fit(H_tr)
    X = torch.from_numpy(scaler.encode(pilot_input(H_tr, P, 12, "train"))).float()
    Y = torch.from_numpy(to_images(H_tr) / scaler.scale).float()
    sr = SRCNN()
    opt = torch.optim.Adam(sr.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad(); loss = ((sr(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    dn = DnCNN(depth=6)
    with torch.no_grad():
        Z = sr(X)
    opt = torch.optim.Adam(dn.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad(); loss = ((dn(Z) - Y) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        est = from_images((dn(sr(X)).numpy()) * scaler.scale)
    assert np.isfinite(complex_mse(est, H_tr))
    print("training step OK  (final loss %.3e)" % loss.item())

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
