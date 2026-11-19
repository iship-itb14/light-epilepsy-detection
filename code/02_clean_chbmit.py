"""Clean CHB-MIT recordings into labelled 4-second windows.

Input:  data/raw/chbmit/chbXX/*.edf and chbXX-summary.txt (from scripts 00 and 01)
Output: data/clean/chbmit/chbXX.npz, one file per patient, plus
        data/clean/chbmit/windows.csv, one row per window

Steps for each recording:
  1. Keep the 18 bipolar channels every chosen patient shares (duplicate T8-P8 dropped).
  2. Band-pass 0.5-40 Hz (4th-order Butterworth, zero phase).
  3. Downsample 256 -> 128 Hz (safe after the 40 Hz low-pass; halves storage).
  4. Cut 4 s windows with no overlap. Windows never cross file boundaries.
  5. Label each window by how much of it falls inside an annotated seizure.
  6. Flag (not drop) windows with a flat channel (before filtering) or a filtered peak over 1000 uV.
  7. Add extra seizure windows shifted by 2 s, marked aug=1, for training only.

Normalisation is NOT done here. It is fitted on the training patients only,
after the split, so no test-patient statistics leak into training.

Run from the project root:  python code/02_clean_chbmit.py
"""
import csv
import re
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

RAW = Path("data/raw/chbmit")
OUT = Path("data/clean/chbmit")
PATIENTS = ["chb01", "chb03", "chb05", "chb06", "chb08", "chb10", "chb13",
            "chb14", "chb15", "chb16", "chb18", "chb20", "chb23", "chb24"]
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3",
            "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
            "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
FS_IN, FS_OUT = 256, 128
WIN_S = 4                       # window length in seconds
WIN = WIN_S * FS_OUT            # 512 samples
SEIZURE_FRAC = 0.5              # label 1 if at least half the window is seizure
FLAT_UV = 0.5                   # channel std below this (microvolts) = flat
EXTREME_UV = 1000               # filtered peak beyond this (microvolts) = extreme

SOS = butter(4, [0.5, 40], btype="bandpass", fs=FS_IN, output="sos")


def parse_summary(path):
    """Return {file name: [(start_s, end_s), ...]} from a chbXX-summary.txt."""
    seizures, current, start = {}, None, None
    for line in path.read_text().splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            current = m.group(1)
            seizures[current] = []
            continue
        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            start = int(m.group(1))
            continue
        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m and current is not None:
            seizures[current].append((start, int(m.group(1))))
    return seizures


def pick_channels(raw):
    """Return the 18 wanted channels in a fixed order, or None if any is missing.

    MNE renames duplicate channel names to e.g. 'T8-P8-0', 'T8-P8-1'; we take the first.
    """
    lookup = {}
    for name in raw.ch_names:
        key = re.sub(r"-\d+$", "", name.upper()) if name.upper().count("-") > 1 else name.upper()
        lookup.setdefault(key, name)
    if not all(ch in lookup for ch in CHANNELS):
        return None
    return [lookup[ch] for ch in CHANNELS]


def seizure_overlap(t0, t1, intervals):
    """Fraction of [t0, t1) that lies inside any seizure interval."""
    covered = sum(max(0, min(t1, b) - max(t0, a)) for a, b in intervals)
    return covered / (t1 - t0)


def make_windows(x, unfiltered, intervals, offset_s=0):
    """Cut x (channels x samples at FS_OUT) into WIN-sample windows starting at offset_s.

    The flat check uses the unfiltered signal, because filtering smooths flat
    lines away. The extreme check uses the filtered signal, so slow drift is ignored.
    """
    out = []
    start = offset_s * FS_OUT
    while start + WIN <= x.shape[1]:
        t0 = start / FS_OUT
        seg = x[:, start:start + WIN]
        seg_raw = unfiltered[:, start:start + WIN]
        frac = seizure_overlap(t0, t0 + WIN_S, intervals)
        flat = bool((seg_raw.std(axis=1) < FLAT_UV).any())
        extreme = bool((np.abs(seg) > EXTREME_UV).any())
        out.append((seg, t0, frac, flat, extreme))
        start += WIN
    return out


def clean_file(edf_path, intervals):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    if int(raw.info["sfreq"]) != FS_IN:
        return None, f"sampling rate {raw.info['sfreq']}"
    names = pick_channels(raw)
    if names is None:
        return None, "missing standard channels"
    raw_uv = raw.get_data(picks=names) * 1e6             # volts -> microvolts
    x = sosfiltfilt(SOS, raw_uv, axis=1)
    x = resample_poly(x, FS_OUT, FS_IN, axis=1)
    unfiltered = raw_uv[:, ::FS_IN // FS_OUT][:, :x.shape[1]]   # same time grid, for artifact flags
    rows = [(*w, 0) for w in make_windows(x, unfiltered, intervals)]
    if intervals:                                        # extra shifted seizure windows
        rows += [(*w, 1) for w in make_windows(x, unfiltered, intervals, offset_s=WIN_S // 2)
                 if w[2] >= SEIZURE_FRAC]
    return rows, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_rows, skipped = [], []
    for p in PATIENTS:
        seizures = parse_summary(RAW / p / f"{p}-summary.txt")
        X, meta = [], []
        for edf in sorted((RAW / p).glob("*.edf")):
            intervals = seizures.get(edf.name, [])
            rows, why = clean_file(edf, intervals)
            if rows is None:
                skipped.append((edf.name, why))
                print(f"  skipped {edf.name}: {why}")
                continue
            for seg, t0, frac, flat, extreme, aug in rows:
                X.append(seg.astype(np.float16))
                meta.append((edf.name, t0, frac, int(frac >= SEIZURE_FRAC), flat, extreme, aug))
        if not X:
            print(f"{p}: no usable files")
            continue
        files, t0, frac, y, flat, extreme, aug = map(np.array, zip(*meta))
        np.savez_compressed(OUT / f"{p}.npz", X=np.stack(X), y=y.astype(np.int8),
                            seizure_frac=frac.astype(np.float32), start_s=t0.astype(np.float32),
                            edf_file=files, flat=flat, extreme=extreme, aug=aug.astype(np.int8),
                            channels=np.array(CHANNELS), fs=FS_OUT)
        index_rows += [(p, *m) for m in meta]
        print(f"{p}: {len(X)} windows, {int(y[aug == 0].sum())} seizure "
              f"(+{int(aug.sum())} shifted), {int(flat.sum())} flat, {int(extreme.sum())} extreme")
    with open(OUT / "windows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "file", "start_s", "seizure_frac", "label", "flat", "extreme", "aug"])
        w.writerows(index_rows)
    if skipped:
        print(f"Skipped {len(skipped)} files; see list above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
