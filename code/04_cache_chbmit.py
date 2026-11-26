"""Unpack the cleaned CHB-MIT windows into a cache that training can read quickly.

Input:  data/clean/chbmit/chbXX.npz (from script 02)
Output: data/cache/chbmit/chbXX_X.npy     windows, float16, windows x 18 channels x 512 samples
        data/cache/chbmit/chbXX_meta.npz  labels and window details (small)

Why a cache: the cleaned files are compressed, so reading any window means
unpacking the whole patient (up to 1.1 GB in memory). A plain .npy file can be
memory-mapped instead: training reads only the windows each batch needs, so
memory use stays small on a laptop. The cache takes about 8 GB of disk. It is
ignored by git, and it is safe to delete data/cache/ and rerun this script.

Run from the project root:  python code/04_cache_chbmit.py
"""
import sys
from pathlib import Path

import numpy as np

CLEAN = Path("data/clean/chbmit")
CACHE = Path("data/cache/chbmit")
PATIENTS = ["chb01", "chb03", "chb05", "chb06", "chb08", "chb10", "chb13",
            "chb14", "chb15", "chb16", "chb18", "chb20", "chb23", "chb24"]
META_KEYS = ["y", "seizure_frac", "start_s", "edf_file", "flat", "extreme", "aug"]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for p in PATIENTS:
        x_path, meta_path = CACHE / f"{p}_X.npy", CACHE / f"{p}_meta.npz"
        d = np.load(CLEAN / f"{p}.npz")
        n = len(d["y"])
        if x_path.exists() and meta_path.exists():
            if np.load(x_path, mmap_mode="r").shape[0] == n:
                print(f"{p}: already cached ({n} windows)")
                continue
        np.save(x_path, d["X"].astype(np.float16))
        np.savez(meta_path, **{k: d[k] for k in META_KEYS})
        print(f"{p}: cached {n} windows ({x_path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
