# COM837 Assignment 3 — Deep-Learning Channel Estimation (ChannelNet)

Satyaram Mangena (IMT2023576)

A from-scratch reproduction of **ChannelNet** (Soltani et al., *IEEE
Communications Letters*, 2019): an SRCNN super-resolution stage followed by a
DnCNN denoising stage that estimate an OFDM channel from pilot observations.
It is compared against LS + 2-D interpolation and against ideal/estimated MMSE,
on the VehA and SUI5 channel models, across SNR and pilot density.

## Layout

```
matlab/generate_channels.m   Vienna 5G simulator -> perfect channel grids
channelnet/settings.py       all constants (grid, splits, SNRs, pilots, paths)
channelnet/channels.py       read H_*.mat, fixed train/val/test split
channelnet/pilots.py         lattice pilots, AWGN, least-squares at pilots
channelnet/classical.py      LS+interpolation and MMSE (ideal / estimated)
channelnet/imaging.py        complex grid <-> real/imag images, input builder
channelnet/nets.py           SRCNN and DnCNN (PyTorch)
run_baselines.py             Task 1 figures + Task 2 classical baselines
run_train.py                 two-stage training (SR then DN)
run_eval.py                  Task 6 evaluation from saved checkpoints
requirements.txt             Python dependencies
checkpoints/                 trained SRCNN/DnCNN weights (*.pt)
results/                     figures, loss curves, final_table.csv, results.json
report/                      main.tex, compiled PDF, Images/
```

Outputs go to `results/` and `checkpoints/` under `$CHAN_OUT` (default `.`).

## Environment

* Channel generation: MATLAB with the Vienna 5G Link-Level Simulator.
* Everything else: Python 3.10+, PyTorch 2.x, NumPy, SciPy, h5py, Matplotlib
  (`pip install -r requirements.txt`).

Point the code at the data and output locations:

```bash
export CHAN_DATA=./data    # folder containing H_VehA.mat and H_SUI5.mat
export CHAN_OUT=.          # results/ and checkpoints/ are created here
```

## Steps

1. **Generate the channels** (MATLAB, from the project root with the simulator
   unzipped into `5GLLSimulator/`):
   ```matlab
   generate_channels          % full 40000-frame dataset per model
   ```
   writes `data/H_VehA.mat` and `data/H_SUI5.mat`.

2. **Baselines + Task 1 figures:**
   ```bash
   python run_baselines.py
   ```

3. **Train** (48 pilots at 12 dB and 22 dB, then the pilot sweep at 12 dB):
   ```bash
   for m in VehA SUI5; do for s in 12 22; do
     python run_train.py --model $m --snr $s --pilots 48 --stage both
   done; done
   for m in VehA SUI5; do for p in 24 36 72; do
     python run_train.py --model $m --snr 12 --pilots $p --stage both --epochs-dn 20
   done; done
   ```

4. **Evaluate from the checkpoints:**
   ```bash
   python run_eval.py
   ```

A quick end-to-end check without the real data (tiny synthetic channels):
`python smoke_test.py`.

## Reproducibility

The split uses seed 0. AWGN seeds derive from base 1234, offset per split and
SNR, so the test noise is identical for every estimator. Training fixes the
Python/NumPy/PyTorch seeds and makes cuDNN deterministic. Optimiser: Adam, lr
1e-3, batch 128, `ReduceLROnPlateau`, early stopping (patience 10); best
validation weights are kept and stored in each checkpoint.
