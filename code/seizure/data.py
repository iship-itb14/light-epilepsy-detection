"""Patient splits, normalisation statistics and batches for training.

All paths are relative to the project root, like the numbered scripts.
"""
import importlib.util
import json
from pathlib import Path

import keras
import numpy as np

CACHE = Path("data/cache/chbmit")
SPLITS = Path("data/splits/chbmit_splits.json")
SCRIPT_03 = Path(__file__).resolve().parents[1] / "03_split_chbmit.py"
MIN_SCALE = 1.0                 # microvolts; floor for the spread of a flat channel
PATIENTS = ["chb01", "chb03", "chb05", "chb06", "chb08", "chb10", "chb13",
            "chb14", "chb15", "chb16", "chb18", "chb20", "chb23", "chb24"]


def load_patient(p):
    """Return (X, meta) for one patient. X is memory-mapped, so nothing is read yet."""
    X = np.load(CACHE / f"{p}_X.npy", mmap_mode="r")
    meta = dict(np.load(CACHE / f"{p}_meta.npz"))
    return X, meta


def get_split(split, fold=None, n_val=2):
    """Return (train, val, test) patient lists.

    fixed: the saved development split (n_val is ignored).
    lopo:  fold k tests on one patient. Its validation patients are the next
           n_val patients in the list after the test patient (wrapping round),
           and the rest train. This spreads validation duty across all patients
           and is fully determined by the fold number.
    """
    splits = json.loads(SPLITS.read_text())
    if split == "fixed":
        s = splits["fixed"]
        return s["train"], s["val"], s["test"]
    test = splits["lopo"][fold]["test"]
    i = PATIENTS.index(test[0])
    val = [PATIENTS[(i + k) % len(PATIENTS)] for k in range(1, n_val + 1)]
    train = [p for p in PATIENTS if p not in test + val]
    return train, val, test


def normalisation_stats(train_patients):
    """Per-channel mean and std from training patients only (reuses script 03)."""
    spec = importlib.util.spec_from_file_location("split_chbmit", SCRIPT_03)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mean, std = mod.normalisation_stats(train_patients)
    return mean.astype(np.float32), std.astype(np.float32)


def recording_stats(p):
    """Per-recording, per-channel median and robust spread for one patient.

    The spread is 1.4826 x the median absolute deviation, which matches the
    standard deviation for normal data but hardly moves when a short seizure or
    artifact is present. Labels are never used. Computed from every 4th sample
    of the ordinary windows, then cached in data/cache/chbmit/chbXX_recstats.npz.
    Returns (files, center, scale), with center and scale shaped files x channels.
    """
    path = CACHE / f"{p}_recstats.npz"
    if path.exists():
        d = np.load(path)
        return d["files"], d["center"], d["scale"]
    X, m = load_patient(p)
    files = np.unique(m["edf_file"])
    center = np.zeros((len(files), X.shape[1]), np.float32)
    scale = np.zeros_like(center)
    for i, f in enumerate(files):
        idx = np.flatnonzero((m["edf_file"] == f) & (m["aug"] == 0))
        x = np.asarray(X[idx][:, :, ::4], np.float32).transpose(1, 0, 2).reshape(X.shape[1], -1)
        med = np.median(x, axis=1)
        mad = np.median(np.abs(x - med[:, None]), axis=1)
        center[i], scale[i] = med, np.maximum(1.4826 * mad, MIN_SCALE)
    np.savez(path, files=files, center=center, scale=scale)
    print(f"  {p}: recording statistics saved for {len(files)} recordings")
    return files, center, scale


class RecordingNorm:
    """Scale each window by its own recording's per-channel median and spread.

    norm="train" leaves windows untouched (the model's own layer then applies the
    training patients' z-score). norm="recording" applies (x - median) / spread
    for the recording the window came from.
    """

    def __init__(self, patients, norm):
        self.on = norm == "recording"
        self.center, self.scale = {}, {}
        if not self.on:
            return
        for p in patients:
            files, c, s = recording_stats(p)
            row = np.searchsorted(files, load_patient(p)[1]["edf_file"])
            self.center[p], self.scale[p] = c[row], s[row]       # one row per window

    def __call__(self, p, idx, x):
        if not self.on:
            return x
        return (x - self.center[p][idx][:, :, None]) / self.scale[p][idx][:, :, None]


class BalancedBatches(keras.utils.PyDataset):
    """Training batches with every seizure window and a fresh non-seizure sample each epoch.

    Seizure windows include the shifted copies (aug=1). Non-seizure windows are
    ordinary windows (aug=0), drawn without replacement at neg_per_pos for each
    seizure window and redrawn every epoch. Flagged windows are kept, so the model
    also learns that artifacts are not seizures. Seizure windows get pos_weight in
    the loss (weighted binary cross-entropy).
    """

    def __init__(self, patients, batch_size=256, neg_per_pos=4, pos_weight=2.0, seed=0, norm="train",
                 augment=False):
        super().__init__()
        self.augment = augment
        self.patients, self.norm = patients, RecordingNorm(patients, norm)
        self.batch_size, self.neg_per_pos, self.pos_weight = batch_size, neg_per_pos, pos_weight
        self.rng = np.random.default_rng(seed)
        self.X, pos, neg = [], [], []
        for k, p in enumerate(patients):
            X, m = load_patient(p)
            self.X.append(X)
            pos_i = np.flatnonzero(m["y"] == 1)
            neg_i = np.flatnonzero((m["y"] == 0) & (m["aug"] == 0))
            pos.append(np.stack([np.full_like(pos_i, k), pos_i], axis=1))
            neg.append(np.stack([np.full_like(neg_i, k), neg_i], axis=1))
        self.pos, self.neg = np.concatenate(pos), np.concatenate(neg)
        self.n_neg = min(len(self.neg), neg_per_pos * len(self.pos))
        self.on_epoch_end()

    def on_epoch_end(self):
        neg = self.neg[self.rng.choice(len(self.neg), self.n_neg, replace=False)]
        rows = np.concatenate([self.pos, neg])
        labels = np.concatenate([np.ones(len(self.pos)), np.zeros(len(neg))])
        order = self.rng.permutation(len(rows))
        self.rows, self.labels = rows[order], labels[order].astype(np.float32)

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.batch_size))

    def __getitem__(self, b):
        sl = slice(b * self.batch_size, (b + 1) * self.batch_size)
        rows, y = self.rows[sl], self.labels[sl]
        x = np.empty((len(rows),) + self.X[0].shape[1:], np.float32)
        for k in np.unique(rows[:, 0]):
            at = np.flatnonzero(rows[:, 0] == k)
            idx = rows[at, 1]
            order = np.argsort(idx)                   # sorted reads are faster on a memory map
            x[at[order]] = self.norm(self.patients[k], idx[order], self.X[k][idx[order]])
        if self.augment:
            x = self._augment(x)
        w = np.where(y == 1, self.pos_weight, 1.0).astype(np.float32)
        return x, y, w


    def _augment(self, x):
        """Small random changes to training windows only (decision 3.23).

        - amplitude: each window scaled by a factor between 0.8 and 1.25
        - noise: Gaussian noise at 5% of each channel's own spread in that window
        - channel dropout: in 30% of windows, one random channel is set to zero,
          as if an electrode lost contact
        """
        n, c, _ = x.shape
        x = x * np.exp(self.rng.uniform(np.log(0.8), np.log(1.25), (n, 1, 1))).astype(np.float32)
        x = x + 0.05 * x.std(axis=2, keepdims=True) * self.rng.standard_normal(x.shape).astype(np.float32)
        drop = np.flatnonzero(self.rng.random(n) < 0.3)
        x[drop, self.rng.integers(0, c, len(drop)), :] = 0.0
        return x


def predict_patients(model, patients, chunk=2048, norm="train"):
    """Seizure probability for every ordinary window (aug=0) of the given patients.

    Returns a dict of arrays: patient, edf_file, start_s, seizure_frac, y, prob.
    Validation and test data are never rebalanced or shifted.
    """
    out = {k: [] for k in ("patient", "edf_file", "start_s", "seizure_frac", "y", "prob")}
    rec = RecordingNorm(patients, norm)
    for p in patients:
        X, m = load_patient(p)
        idx = np.flatnonzero(m["aug"] == 0)
        probs = [model.predict_on_batch(rec(p, idx[i:i + chunk],
                                            np.asarray(X[idx[i:i + chunk]], np.float32))).ravel()
                 for i in range(0, len(idx), chunk)]
        out["prob"].append(np.concatenate(probs))
        out["patient"].append(np.full(len(idx), p))
        for k in ("edf_file", "start_s", "seizure_frac", "y"):
            out[k].append(m[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}
