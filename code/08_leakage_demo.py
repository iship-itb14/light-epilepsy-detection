"""Split-leakage demo on the Bonn EEG sets, cut the way the UCI table was (decision 3.26).

The UCI/Kaggle "Epileptic Seizure Recognition" table is the Bonn data
(Andrzejak et al. 2001: 5 sets x 100 single-channel recordings of 23.6 s at
173.61 Hz) with every recording cut into 23 one-second chunks of 178 samples.
UCI no longer hosts the table, so we rebuild it from the Bonn files. That also
tells us for certain which recording each chunk came from.

The same small CNN is trained and tested twice on the same 11,500 chunks:
  shuffled  chunks are shuffled and cut into 5 folds, the way most tutorials do it.
            Chunks from one recording land in both training and test.
  grouped   whole recordings are cut into 5 folds, so every chunk of a recording
            is in training or in test, never both.
The gap between the two is how much a careless split inflates the score.

Two tasks: "binary" (seizure, set S, against the other four sets, as most papers
do) and "five" (all five sets). Sets N and F come from the same patients and
sets Z and O from the same volunteers, so a model that recognises a recording
rather than its class gains the most there.

Important limit: the Bonn files carry no person ids. "grouped" keeps recordings
apart, not people. Each set comes from only five people, so even the grouped
score is likely optimistic. This is why the main study uses CHB-MIT with
leave-one-patient-out.

Input:  data/raw/bonn/{Z,O,N,F,S}/*.txt (script 07)
Output: results/leakage/folds.csv    one row per task, split and fold
        results/leakage/summary.csv  mean and std over folds, and the gap
        results/leakage/config.json  settings

Run from the project root:
  python code/08_leakage_demo.py
  python code/08_leakage_demo.py --task binary --epochs 5   (quick try)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import keras
import numpy as np

DATA = Path("data/raw/bonn")
RESULTS = Path("results/leakage")
CHUNKS, CHUNK_LEN = 23, 178    # as in the UCI table: 23 x 178 = 4094 of 4097 samples
# Class numbers follow the UCI table: 1 seizure (S), 2 epileptogenic zone (F),
# 3 opposite hemisphere (N), 4 eyes closed (O), 5 eyes open (Z). Stored here as 0..4.
SETS = ["S", "F", "N", "O", "Z"]
N_CLASSES = 5


def load(root):
    """Cut every Bonn recording into 23 one-second chunks of 178 samples.

    Returns the chunks, their class (0..4, see SETS), the recording each came
    from, and the class of each recording. The last 3 samples of each recording
    are dropped, as 23 x 178 = 4094. Stops if a set does not hold 100 recordings
    of 4097 samples.
    """
    x, y, group, group_class, names = [], [], [], [], []
    for c, s in enumerate(SETS):
        files = sorted(p for p in (root / s).iterdir() if p.suffix.lower() == ".txt")
        if len(files) != 100:
            sys.exit(f"{root / s} has {len(files)} recordings, expected 100. Run code/07_get_bonn.sh first.")
        for f in files:
            sig = np.loadtxt(f, dtype=np.float32)
            if sig.shape != (4097,):
                sys.exit(f"{f} has {sig.shape[0]} samples, expected 4097")
            x.append(sig[:CHUNKS * CHUNK_LEN].reshape(CHUNKS, CHUNK_LEN))
            y += [c] * CHUNKS
            group += [len(names)] * CHUNKS
            group_class.append(c)
            names.append(f.stem)
    return np.concatenate(x), np.array(y), np.array(group), np.array(group_class)


def stratified_folds(labels, k, rng):
    """Give each item a fold 0..k-1 so that every fold has about the same class mix."""
    fold = np.empty(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def make_model(n_out, seed):
    """A small 1D CNN on one channel of 178 samples (about 15k weights)."""
    keras.utils.set_random_seed(seed)
    layers = [keras.Input((178, 1))]
    for filters, kernel in [(16, 7), (32, 5), (64, 3)]:
        layers += [keras.layers.Conv1D(filters, kernel, padding="same", use_bias=False),
                   keras.layers.BatchNormalization(), keras.layers.ReLU(),
                   keras.layers.MaxPooling1D(2)]
    layers += [keras.layers.GlobalAveragePooling1D(), keras.layers.Dropout(0.3),
               keras.layers.Dense(1 if n_out == 2 else n_out,
                                  activation="sigmoid" if n_out == 2 else "softmax")]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy" if n_out == 2 else "sparse_categorical_crossentropy")
    return model


def auroc(t, p):
    """Area under the ROC curve (Mann-Whitney, ties shared)."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = t.sum(), len(t) - t.sum()
    return float((ranks[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(task, t, p):
    if task == "binary":
        pred = (p >= 0.5).astype(int)
        tp, tn = np.sum((pred == 1) & (t == 1)), np.sum((pred == 0) & (t == 0))
        fp, fn = np.sum((pred == 1) & (t == 0)), np.sum((pred == 0) & (t == 1))
        return {"accuracy": (tp + tn) / len(t), "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp), "f1": 2 * tp / (2 * tp + fp + fn),
                "auroc": auroc(t, p)}
    pred = p.argmax(1)
    f1s = []
    for c in range(N_CLASSES):
        tp = np.sum((pred == c) & (t == c))
        f1s.append(2 * tp / (np.sum(pred == c) + np.sum(t == c)))
    return {"accuracy": float(np.mean(pred == t)), "macro_f1": float(np.mean(f1s))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--task", choices=["binary", "five", "both"], default="both")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args()

    x, y, group, group_class = load(a.data)
    print(f"{len(x)} chunks from {group.max() + 1} recordings; chunks per set "
          + ", ".join(f"{s} {n}" for s, n in zip(SETS, np.bincount(y))))

    rng = np.random.default_rng(a.seed)
    folds = {"shuffled": stratified_folds(y, a.folds, rng),
             "grouped": stratified_folds(group_class, a.folds, rng)[group]}
    tasks = ["binary", "five"] if a.task == "both" else [a.task]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        target = (y == 0).astype(int) if task == "binary" else y
        for split, fold in folds.items():
            for k in range(a.folds):
                tr, te = fold != k, fold == k
                shared = np.isin(group[te], group[tr]).mean()
                mean, std = x[tr].mean(), x[tr].std()        # training chunks only
                xtr, xte = (x[tr] - mean) / std, (x[te] - mean) / std
                model = make_model(2 if task == "binary" else N_CLASSES, a.seed + k)
                t0 = time.time()
                model.fit(xtr[..., None], target[tr], epochs=a.epochs, batch_size=a.batch,
                          shuffle=True, verbose=0)
                p = model.predict(xte[..., None], batch_size=1024, verbose=0)
                p = p[:, 0] if task == "binary" else p
                s = scores(task, target[te], p)
                row = {"task": task, "split": split, "fold": k, "test_chunks": int(te.sum()),
                       "test_chunks_with_same_recording_in_training": round(float(shared), 4),
                       **{m: round(float(v), 4) for m, v in s.items()},
                       "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"{task:6} {split:8} fold {k}: " +
                      ", ".join(f"{m} {v:.3f}" for m, v in s.items()) +
                      f"  (shared recordings {shared:.0%}, {row['seconds']} s)", flush=True)

    with open(a.out / "folds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in rows for c in r)), restval="")
        w.writeheader()
        w.writerows(rows)

    summary = []
    print("\nMean (std) over folds")
    for task in tasks:
        metrics = [m for m in rows if m["task"] == task][0]
        metrics = [m for m in metrics if m not in ("task", "split", "fold", "test_chunks", "seconds",
                                                    "test_chunks_with_same_recording_in_training")]
        for m in metrics:
            v = {s: np.array([r[m] for r in rows if r["task"] == task and r["split"] == s]) for s in folds}
            summary.append({"task": task, "metric": m,
                            "shuffled_mean": round(v["shuffled"].mean(), 4), "shuffled_std": round(v["shuffled"].std(), 4),
                            "grouped_mean": round(v["grouped"].mean(), 4), "grouped_std": round(v["grouped"].std(), 4),
                            "gap": round(v["shuffled"].mean() - v["grouped"].mean(), 4)})
            print(f"  {task:6} {m:12} shuffled {v['shuffled'].mean():.3f} ({v['shuffled'].std():.3f})   "
                  f"grouped {v['grouped'].mean():.3f} ({v['grouped'].std():.3f})   "
                  f"gap {v['shuffled'].mean() - v['grouped'].mean():+.3f}")
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(a.out / "config.json", "w") as f:
        json.dump({"data": str(a.data), "tasks": tasks, "folds": a.folds, "epochs": a.epochs,
                   "batch": a.batch, "seed": a.seed, "keras": keras.__version__,
                   "recordings": int(group.max() + 1), "chunk_samples": CHUNK_LEN,
                   "sets_in_class_order": SETS}, f, indent=2)
    print(f"\nSaved {a.out}/folds.csv, summary.csv, config.json")


if __name__ == "__main__":
    main()
