"""Make the numbers behind the Data & prep page figures.

Input:  data/raw/chbmit/chb01/chb01_03.edf          (script 01)
        data/clean/chbmit/chb01.npz, windows.csv   (script 02)
Output: assets/data/data-page-figures.js            (read by assets/js/data-page.js)

Three figures use real data:
  - trace:    88 s of cleaned EEG around the chb01_03 seizure (2,996-3,036 s), 6 channels, shown at 32 Hz
  - filt:     power spectrum of channel FP1-F7, first 10 minutes, before and after the 0.5-40 Hz band-pass
  - patients: per patient [windows, seizure windows, extreme-flagged windows, shifted copies]

Run from the project root:  python code/site/make_data_page_figures.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

RAW = Path("data/raw/chbmit/chb01/chb01_03.edf")
CLEAN = Path("data/clean/chbmit")
OUT = Path("assets/data/data-page-figures.js")
TRACE_CHANNELS = ["FP1-F7", "T7-P7", "C3-P3", "P8-O2", "FZ-CZ", "F8-T8"]


def patient_counts():
    counts = defaultdict(lambda: [0, 0, 0, 0])
    with open(CLEAN / "windows.csv") as fh:
        for r in csv.DictReader(fh):
            c = counts[r["patient"]]
            if r["aug"] == "0":
                c[0] += 1
                c[1] += int(r["label"])
                c[2] += r["extreme"] == "True"
            else:
                c[3] += 1
    return dict(counts)


def seizure_trace():
    d = np.load(CLEAN / "chb01.npz")
    keep = (d["edf_file"] == "chb01_03.edf") & (d["aug"] == 0) & (d["start_s"] >= 2972) & (d["start_s"] < 3060)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(d["start_s"][idx])]
    sig = np.concatenate(list(d["X"][idx].astype(np.float32)), axis=1)    # channels x samples at 128 Hz
    chs = list(d["channels"])
    out = {ch: np.round(sig[chs.index(ch), ::4], 1).tolist() for ch in TRACE_CHANNELS}
    out.update(t0=float(d["start_s"][idx[0]]), fs=32, y=d["y"][idx].tolist(),
               frac=np.round(d["seizure_frac"][idx], 2).tolist())
    return out


def spectrum():
    raw = mne.io.read_raw_edf(RAW, preload=True, verbose="error")
    x = raw.get_data(picks=["FP1-F7"])[0] * 1e6
    sos = butter(4, [0.5, 40], btype="bandpass", fs=256, output="sos")    # same filter as script 02
    f = sosfiltfilt(sos, x)
    fr, pr = welch(x[:600 * 256], fs=256, nperseg=1024)
    _, pf = welch(f[:600 * 256], fs=256, nperseg=1024)
    m = fr <= 100
    a, b = 2950 * 256, 2960 * 256
    return {"raw": np.round(x[a:b:2], 1).tolist(),
            "clean": np.round(resample_poly(f[a:b], 1, 2), 1).tolist(),
            "psd_f": np.round(fr[m], 2).tolist(),
            "psd_raw": np.round(10 * np.log10(pr[m]), 2).tolist(),
            "psd_clean": np.round(10 * np.log10(pf[m] + 1e-12), 2).tolist()}


def main():
    data = {"filt": spectrum(), "trace": seizure_trace(), "patients": patient_counts()}
    OUT.write_text("/* Made by code/site/make_data_page_figures.py. Do not edit by hand. */\n"
                   "window.DATA=" + json.dumps(data) + ";\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
