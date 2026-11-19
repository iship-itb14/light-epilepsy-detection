"""Fix the CHB-MIT patient splits and the statistics used for normalisation.

Input:  data/clean/chbmit/chbXX.npz (from script 02)
Output: data/splits/chbmit_splits.json        which patients are train / val / test
        data/splits/chbmit_channel_stats.npz  per-patient, per-channel sums for normalisation

Two kinds of split are saved:
  - "fixed": one train / validation / test split, used while developing models.
  - "lopo":  leave-one-patient-out folds, used for the final results. Each fold
             tests on one patient and trains on the other 13.

No patient is ever in more than one set of a split (decision 3.3).

Normalisation (z-score per channel) must use training patients only. This script
stores each patient's per-channel count, sum and sum of squares, so the training
code can combine exactly the training patients of any split with
`normalisation_stats(train_patients)` and never touch test-patient data.

Run from the project root:  python code/03_split_chbmit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CLEAN = Path("data/clean/chbmit")
OUT = Path("data/splits")
PATIENTS = ["chb01", "chb03", "chb05", "chb06", "chb08", "chb10", "chb13",
            "chb14", "chb15", "chb16", "chb18", "chb20", "chb23", "chb24"]

# Fixed development split. Test and validation patients have typical seizure
# counts (5-8 seizures each); the patients with the most seizure windows
# (chb08, chb15) stay in training, where they help most.
FIXED = {
    "test": ["chb03", "chb10", "chb20"],
    "val": ["chb05", "chb23"],
}
FIXED["train"] = [p for p in PATIENTS if p not in FIXED["test"] + FIXED["val"]]

CHUNK = 2000                    # windows per chunk, keeps memory use low


def patient_stats(path):
    """Per-channel count, sum and sum of squares over one patient's windows.

    Uses non-shifted windows (aug=0) without a flat channel, so the statistics
    describe ordinary recorded signal.
    """
    d = np.load(path)
    keep = (d["aug"] == 0) & ~d["flat"]
    X = d["X"]
    n_ch = X.shape[1]
    n, s, ss = np.zeros(n_ch), np.zeros(n_ch), np.zeros(n_ch)
    idx = np.flatnonzero(keep)
    for i in range(0, len(idx), CHUNK):
        x = X[idx[i:i + CHUNK]].astype(np.float64)            # windows x channels x samples
        n += x.shape[0] * x.shape[2]
        s += x.sum(axis=(0, 2))
        ss += (x ** 2).sum(axis=(0, 2))
    counts = {"windows": int(keep.sum()),
              "seizure": int(d["y"][keep].sum())}
    return n, s, ss, counts


def normalisation_stats(train_patients, stats_path=OUT / "chbmit_channel_stats.npz"):
    """Per-channel mean and std from the given (training) patients only."""
    st = np.load(stats_path)
    n = sum(st[f"{p}_n"] for p in train_patients)
    s = sum(st[f"{p}_sum"] for p in train_patients)
    ss = sum(st[f"{p}_sumsq"] for p in train_patients)
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    return mean, std


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, counts = {}, {}
    for p in PATIENTS:
        n, s, ss, c = patient_stats(CLEAN / f"{p}.npz")
        arrays[f"{p}_n"], arrays[f"{p}_sum"], arrays[f"{p}_sumsq"] = n, s, ss
        counts[p] = c
        print(f"{p}: {c['windows']} windows, {c['seizure']} seizure")
    np.savez(OUT / "chbmit_channel_stats.npz", **arrays)

    splits = {
        "fixed": FIXED,
        "lopo": [{"test": [p], "train": [q for q in PATIENTS if q != p]} for p in PATIENTS],
    }
    with open(OUT / "chbmit_splits.json", "w") as fh:
        json.dump(splits, fh, indent=2)

    for name in ("train", "val", "test"):
        pts = FIXED[name]
        w = sum(counts[p]["windows"] for p in pts)
        z = sum(counts[p]["seizure"] for p in pts)
        print(f"fixed {name:5s}: {len(pts)} patients, {w} windows, {z} seizure ({100 * z / w:.2f}%)")
    mean, std = normalisation_stats(FIXED["train"])
    print("fixed-split training mean (uV):", np.round(mean, 2))
    print("fixed-split training std  (uV):", np.round(std, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
