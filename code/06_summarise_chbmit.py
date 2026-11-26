"""Collect the test scores of every trained run into one table.

Input:  results/chbmit/<model>/fixed/metrics.json and results/chbmit/<model>/lopo/*/metrics.json
Output: results/chbmit/summary.csv, one row per run, and a printed table with
        the leave-one-patient-out mean and median for each model

Run from the project root:  python code/06_summarise_chbmit.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("results/chbmit")
COLS = ["sensitivity", "specificity", "f1", "event_sensitivity", "fa_per_hour"]


def main():
    rows = []
    for f in sorted(RESULTS.glob("*/*/metrics.json")) + sorted(RESULTS.glob("*/lopo/*/metrics.json")):
        m = json.loads(f.read_text())
        t = m["test"]
        rows.append({"model": m["model"], "split": m["split"], "test": "+".join(m["test_patients"]),
                     "params": m["params"], "model_kb": m["model_kb"], "threshold": round(m["threshold"], 4),
                     **{c: round(t[c], 4) for c in COLS},
                     "seizures": f"{t['seizures_detected']}/{t['seizures']}",
                     "tp": t["tp"], "fn": t["fn"], "fp": t["fp"], "tn": t["tn"]})
    if not rows:
        print("No results yet. Run code/05_train_chbmit.py first.")
        return 1
    with open(RESULTS / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':28s} {'split':5s} {'test':17s} " + " ".join(f"{c[:11]:>11s}" for c in COLS) + "  seizures")
    for r in rows:
        print(f"{r['model']:28s} {r['split']:5s} {r['test']:17s} "
              + " ".join(f"{r[c]:11.3f}" for c in COLS) + f"  {r['seizures']}")
    for model in sorted({r["model"] for r in rows}):
        lopo = [r for r in rows if r["model"] == model and r["split"] == "lopo"]
        if lopo:
            for name, fn in (("mean", np.nanmean), ("median", np.nanmedian)):
                vals = " ".join(f"{fn([r[c] for r in lopo]):11.3f}" for c in COLS)
                print(f"{model:28s} lopo  {name + f' ({len(lopo)} folds)':17s} {vals}")
    print(f"saved {RESULTS / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
