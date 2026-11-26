"""Train one model on one CHB-MIT split, pick its threshold on validation patients, and score it on the test patient(s).

Input:  data/cache/chbmit/ (script 04) and data/splits/ (script 03)
Output: results/chbmit/<run>/fixed/           for the development split
        results/chbmit/<run>/lopo/<patient>/  for one leave-one-patient-out fold
        <run> is the model name, plus "-recnorm" with --norm recording,
        "-aug" with --augment, and "-val<n>-<rule>" when LOPO uses
        --val-patients or --fa-rule other than the defaults
        Each folder holds:
          model.keras           the best model (normalisation built in)
          history.csv           training loss and validation score per epoch
          metrics.json          threshold, validation and test scores, model size
          test_predictions.npz  probability for every test window, for rescoring later
          config.json           patients and settings used

Training (decisions 3.14 and 3.18): Adam at 0.001, weighted binary cross-entropy,
batch 256, balanced batches redrawn each epoch, at most 50 epochs. After each
epoch the model is scored on all validation windows, unbalanced: we find the
threshold that gives the best sensitivity with at most --max-fa false alarms
per hour (after the 2-of-3 alarm rule), and keep the epoch with the best such
sensitivity. Epochs that cannot meet the false-alarm limit at any threshold
rank below those that can. Training stops after --patience epochs without improvement.

Normalisation (--norm): "train" z-scores every window with the training
patients' channel mean and std, inside the model. "recording" first scales each
window by its own recording's channel median and robust spread (see
seizure/data.py), so patients with different amplitudes look alike; the
model's own layer is then left as identity.

Run from the project root, for example:
  python code/05_train_chbmit.py --model cnn --split fixed
  python code/05_train_chbmit.py --model eegnet --split lopo --fold 0
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import keras
import numpy as np

from seizure.data import BalancedBatches, get_split, normalisation_stats, predict_patients
from seizure.models import MODELS
from seizure.scoring import pick_threshold, score

RESULTS = Path("results/chbmit")


class ValidationScore(keras.callbacks.Callback):
    """Score all validation windows after each epoch; keep the best weights; stop early."""

    def __init__(self, val_patients, max_fa, patience, weights_path, norm, fa_rule):
        super().__init__()
        self.norm, self.fa_rule = norm, fa_rule
        self.val_patients, self.max_fa, self.patience = val_patients, max_fa, patience
        self.weights_path = weights_path
        self.best, self.best_epoch, self.wait = None, -1, 0

    def on_epoch_end(self, epoch, logs=None):
        r = pick_threshold(predict_patients(self.model, self.val_patients, norm=self.norm),
                           self.max_fa, self.fa_rule)
        logs["val_sensitivity"], logs["val_fa_per_hour"] = r["sensitivity"], r["fa_per_hour"]
        logs["val_threshold"] = r["threshold"]
        # an epoch that keeps within the false-alarm limit always beats one that does not
        key = (r["meets_fa_limit"], r["sensitivity"], -r["fa_per_hour"])
        mark = ""
        if self.best is None or key > self.best:
            self.best, self.best_epoch, self.wait = key, epoch, 0
            self.model.save_weights(self.weights_path)
            mark = "  (best so far)"
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        print(f"  val: sensitivity {r['sensitivity']:.3f} at {r['fa_per_hour']:.2f} false alarms/h, "
              f"threshold {r['threshold']:.3f}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--split", choices=["fixed", "lopo"], default="fixed")
    ap.add_argument("--fold", type=int, help="leave-one-patient-out fold, 0-13")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--neg-per-pos", type=float, default=4)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--max-fa", type=float, default=1.0, help="false alarms per hour allowed when picking the threshold")
    ap.add_argument("--norm", choices=["train", "recording"], default="train")
    ap.add_argument("--augment", action="store_true", help="random amplitude, noise and channel dropout in training")
    ap.add_argument("--val-patients", type=int, default=2, help="LOPO only: validation patients per fold")
    ap.add_argument("--fa-rule", choices=["pooled", "median"], default="pooled",
                    help="combine validation patients' false-alarm rates by pooling or by the median")
    ap.add_argument("--skip-existing", action="store_true", help="do nothing if this run already has results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.split == "lopo" and args.fold is None:
        ap.error("--split lopo needs --fold")

    keras.utils.set_random_seed(args.seed)
    train, val, test = get_split(args.split, args.fold, args.val_patients)
    run = args.model + ("-recnorm" if args.norm == "recording" else "") + ("-aug" if args.augment else "")
    if args.split == "lopo" and (args.val_patients != 2 or args.fa_rule != "pooled"):
        run += f"-val{args.val_patients}-{args.fa_rule}"
    out = RESULTS / run / args.split
    if args.split == "lopo":
        out = out / test[0]
    if args.skip_existing and (out / "metrics.json").exists():
        print(f"{out} already has results; skipping")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run} | train {train} | val {val} | test {test}")

    if args.norm == "train":
        mean, std = normalisation_stats(train)
    else:                                                   # windows arrive already scaled
        mean, std = np.zeros(18, np.float32), np.ones(18, np.float32)
    model = MODELS[args.model](mean, std)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  weighted_metrics=[])
    n_params = model.count_params() - 2 * len(mean) - 1     # minus the fixed normalisation values
    print(f"{n_params:,} trainable and batch-norm weights")

    batches = BalancedBatches(train, args.batch_size, int(args.neg_per_pos), args.pos_weight, args.seed,
                              args.norm, args.augment)
    print(f"each epoch: {len(batches.pos)} seizure + {batches.n_neg} non-seizure windows")
    weights_path = out / "best.weights.h5"
    val_cb = ValidationScore(val, args.max_fa, args.patience, weights_path, args.norm,
                             args.fa_rule)
    t0 = time.time()
    hist = model.fit(batches, epochs=args.epochs, callbacks=[val_cb], verbose=2)
    train_minutes = (time.time() - t0) / 60

    model.load_weights(weights_path)
    model.save(out / "model.keras")
    weights_path.unlink()
    val_r = pick_threshold(predict_patients(model, val, norm=args.norm), args.max_fa, args.fa_rule)
    test_p = predict_patients(model, test, norm=args.norm)
    test_r = score(test_p, val_r["threshold"])
    np.savez_compressed(out / "test_predictions.npz", **test_p)

    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(hist.history)
        w.writerow(["epoch"] + keys)
        for i in range(len(hist.history[keys[0]])):
            w.writerow([i + 1] + [hist.history[k][i] for k in keys])
    metrics = {"model": run, "split": args.split, "test_patients": test,
               "best_epoch": val_cb.best_epoch + 1, "epochs_run": len(hist.history["loss"]),
               "train_minutes": round(train_minutes, 1), "params": int(n_params),
               "model_kb": round((out / "model.keras").stat().st_size / 1024, 1),
               "threshold": val_r["threshold"], "val": val_r, "test": test_r}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(args), "train": train, "val": val,
                                                 "test": test}, indent=2))

    print(f"\nbest epoch {metrics['best_epoch']}, threshold {val_r['threshold']:.3f}")
    for name, r in (("val", val_r), ("test", test_r)):
        print(f"{name:4s}: sensitivity {r['sensitivity']:.3f}, specificity {r['specificity']:.4f}, "
              f"F1 {r['f1']:.3f}, seizures caught {r['seizures_detected']}/{r['seizures']}, "
              f"false alarms/h {r['fa_per_hour']:.2f}")
    print(f"test confusion matrix: TP {test_r['tp']}  FN {test_r['fn']}  FP {test_r['fp']}  TN {test_r['tn']}")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
