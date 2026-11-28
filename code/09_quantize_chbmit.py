"""Shrink the saved CHB-MIT models to int8 TensorFlow Lite and score them like the float models (decision 3.28).

Input:  results/chbmit/<run>/lopo/<patient>/model.keras and config.json (script 05)
        data/cache/chbmit/ (script 04)
Output: results/chbmit/<run>-int8/lopo/<patient>/
          model_float32.tflite  the same model as TensorFlow Lite, still float32 (size baseline)
          model_int8.tflite     int8 weights, activations, input and output
          metrics.json          threshold, validation and test scores, sizes, float vs int8 agreement
          test_predictions.npz  int8 probability for every test window
          config.json           where it came from and the settings used
        results/chbmit/int8_timing.csv   CPU time per window, one row per run

No retraining. For each saved model:
1. Convert to float32 TFLite and check it gives the same probabilities as the
   Keras model on the first test windows.
2. Convert to full int8. The int8 scales are calibrated on --calib windows
   (half seizure, half not) from that fold's own training patients, prepared
   exactly as in training (same normalisation and channels). Test and
   validation patients are never used for calibration.
3. Predict every validation and test window with the int8 model, pick the
   threshold again on the validation patients with the fold's own rule
   (3.25: at most 1 false alarm per hour, median over patients), and score the
   test patient with the 2-of-3 alarm rule (seizure/scoring.py). The test
   patient is also scored at the float model's threshold, for comparison.
4. Once per run, time one window at a time on the CPU with one thread:
   the Keras model, float32 TFLite and int8 TFLite.

The int8 runs land next to the float runs, so code/06_summarise_chbmit.py
lists them too.

Run from the project root:
  python code/09_quantize_chbmit.py                      # all six reported runs, all folds
  python code/09_quantize_chbmit.py --runs eegnet-recnorm-val3-median --folds chb01
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # hide TensorFlow's info messages
import keras
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated")
try:                                    # TensorFlow's own interpreter is being moved to this package
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter

from seizure.data import CHANNEL_SETS, RecordingNorm, channel_index, load_patient, predict_patients
from seizure.scoring import pick_threshold, score

RESULTS = Path("results/chbmit")
RUNS = ["eegnet-recnorm-val3-median", "eegnet-recnorm-temporal4-val3-median",
        "eegnet-recnorm-temporal2-val3-median", "cnn-reg-recnorm-aug-val3-median",
        "cnn-reg-recnorm-aug-temporal4-val3-median", "cnn-reg-recnorm-aug-temporal2-val3-median"]


class TFLiteModel:
    """A TFLite model with the predict_on_batch() that data.predict_patients expects.

    For an int8 model, float windows are quantized with the model's input
    scale before running, and the int8 output is turned back into a probability.
    """

    def __init__(self, content, threads=None):
        self.interp = Interpreter(model_content=content, num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.batch = None

    def predict_on_batch(self, x):
        x = np.asarray(x, np.float32)
        if len(x) != self.batch:                            # resize only when the batch size changes
            self.interp.resize_tensor_input(self.inp["index"], x.shape)
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.batch = len(x)
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


def calibration_windows(train, norm, channels, n, seed):
    """n training windows, half seizure and half not, prepared as in training."""
    rng = np.random.default_rng(seed)
    rec, ch = RecordingNorm(train, norm), channel_index(channels)
    pools = {0: [], 1: []}                                  # rows of (patient number, window index)
    for k, p in enumerate(train):
        m = load_patient(p)[1]
        for label in (0, 1):
            idx = np.flatnonzero((m["aug"] == 0) & (m["y"] == label))
            pools[label].append(np.stack([np.full_like(idx, k), idx], axis=1))
    picks = []
    for label in (0, 1):
        rows = np.concatenate(pools[label])
        picks.append(rows[rng.choice(len(rows), min(n // 2, len(rows)), replace=False)])
    picks = np.concatenate(picks)
    xs = []
    for k, p in enumerate(train):
        idx = np.sort(picks[picks[:, 0] == k, 1])
        if len(idx):
            X = load_patient(p)[0]
            xs.append(rec(p, idx, np.asarray(X[idx], np.float32))[:, ch])
    x = np.concatenate(xs)
    return x[rng.permutation(len(x))]


def convert(model, calib=None):
    """TFLite flatbuffer: float32 if calib is None, otherwise full int8 calibrated on calib."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if calib is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    with contextlib.redirect_stdout(io.StringIO()):     # the converter prints every tensor it exports
        return conv.convert()


def ms_per_window(fn, x, repeats):
    """Median milliseconds for one window, after a warm-up."""
    for _ in range(20):
        fn(x)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def timing(run, params, model, f32, i8, x, repeats):
    keras_call = tf.function(lambda a: model(a, training=False))
    x = x[:1]
    lite32, lite8 = TFLiteModel(f32, threads=1), TFLiteModel(i8, threads=1)
    return {"run": run, "n_channels": x.shape[1], "params": params,
            "keras_ms": round(ms_per_window(lambda a: keras_call(a).numpy(), x, repeats), 4),
            "tflite_float32_ms": round(ms_per_window(lite32.predict_on_batch, x, repeats), 4),
            "tflite_int8_ms": round(ms_per_window(lite8.predict_on_batch, x, repeats), 4)}


def quantize_fold(run, src, args):
    cfg = json.loads((src / "config.json").read_text())
    fm = json.loads((src / "metrics.json").read_text())
    norm, channels = cfg.get("norm", "train"), cfg.get("channels", "all")
    max_fa, fa_rule = cfg.get("max_fa", 1.0), cfg.get("fa_rule", "pooled")
    train, val, test = cfg["train"], cfg["val"], cfg["test"]
    out = RESULTS / f"{run}-int8" / "lopo" / test[0]
    if args.skip_existing and (out / "metrics.json").exists():
        print(f"{out} already has results; skipping")
        return None, None
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = keras.models.load_model(src / "model.keras", compile=False)
    f32 = convert(model)
    calib = calibration_windows(train, norm, channels, args.calib, args.seed)
    i8 = convert(model, calib)
    (out / "model_float32.tflite").write_bytes(f32)
    (out / "model_int8.tflite").write_bytes(i8)

    # float32 TFLite must match Keras; otherwise the conversion itself changed the model
    X, m = load_patient(test[0])
    idx = np.flatnonzero(m["aug"] == 0)[:512]
    xs = RecordingNorm(test, norm)(test[0], idx, np.asarray(X[idx], np.float32))[:, channel_index(channels)]
    f32_gap = float(np.max(np.abs(TFLiteModel(f32).predict_on_batch(xs).ravel()
                                  - model.predict_on_batch(xs).ravel())))

    lite = TFLiteModel(i8)
    val_r = pick_threshold(predict_patients(lite, val, norm=norm, channels=channels), max_fa, fa_rule)
    test_p = predict_patients(lite, test, norm=norm, channels=channels)
    test_r = score(test_p, val_r["threshold"])
    test_at_float = score(test_p, fm["threshold"])
    np.savez_compressed(out / "test_predictions.npz", **test_p)

    float_p = dict(np.load(src / "test_predictions.npz"))
    same = (float_p["edf_file"] == test_p["edf_file"]).all() and np.allclose(float_p["start_s"], test_p["start_s"])
    diff = np.abs(test_p["prob"] - float_p["prob"]) if same else np.array([np.nan])
    metrics = {"model": f"{run}-int8", "split": "lopo", "test_patients": test,
               "channels": fm.get("channels") or CHANNEL_SETS[channels], "source_run": run, "best_epoch": fm["best_epoch"],
               "params": fm["params"], "model_kb": round(len(i8) / 1024, 1),
               "float32_tflite_kb": round(len(f32) / 1024, 1), "keras_file_kb": fm["model_kb"],
               "weights_kb_float32": round(4 * fm["params"] / 1024, 1),
               "weights_kb_int8": round(fm["params"] / 1024, 1),
               "float32_tflite_max_gap": f32_gap, "calibration_windows": int(len(calib)),
               "prob_diff_mean": float(np.mean(diff)), "prob_diff_max": float(np.max(diff)),
               "threshold": val_r["threshold"], "float_threshold": fm["threshold"],
               "val": val_r, "test": test_r, "test_at_float_threshold": test_at_float,
               "float_test": fm["test"], "minutes": round((time.time() - t0) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(args), "source": str(src), "norm": norm,
                                                 "channels": channels, "max_fa": max_fa, "fa_rule": fa_rule,
                                                 "train": train, "val": val, "test": test}, indent=2))
    ft = fm["test"]
    print(f"  {test[0]}: {len(f32) / 1024:.0f} KB -> {len(i8) / 1024:.0f} KB, "
          f"seizures {ft['seizures_detected']}/{ft['seizures']} -> {test_r['seizures_detected']}/{test_r['seizures']}, "
          f"sens {ft['sensitivity']:.3f} -> {test_r['sensitivity']:.3f}, "
          f"FA/h {ft['fa_per_hour']:.2f} -> {test_r['fa_per_hour']:.2f}, "
          f"mean |p diff| {metrics['prob_diff_mean']:.4f}, float TFLite gap {f32_gap:.1e}  "
          f"({metrics['minutes']} min)")
    return metrics, (fm["params"], model, f32, i8, calib)


def summary_line(run, rows):
    def med(key, part):
        return float(np.nanmedian([r[part][key] for r in rows]))
    caught = lambda part: sum(r[part]["seizures_detected"] for r in rows)
    total = sum(r["test"]["seizures"] for r in rows)
    return (f"{run:44s} {len(rows):2d} folds | seizures {caught('float_test')} -> {caught('test')} of {total} "
            f"(at float threshold {caught('test_at_float_threshold')}) | median sens "
            f"{med('sensitivity', 'float_test'):.2f} -> {med('sensitivity', 'test'):.2f} | median FA/h "
            f"{med('fa_per_hour', 'float_test'):.2f} -> {med('fa_per_hour', 'test'):.2f} | "
            f"{rows[0]['float32_tflite_kb']:.0f} KB -> {rows[0]['model_kb']:.0f} KB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=RUNS, help="float runs to quantize (folders in results/chbmit)")
    ap.add_argument("--folds", nargs="+", help="test patients to do, e.g. chb01 chb03 (default: all saved)")
    ap.add_argument("--calib", type=int, default=1000, help="calibration windows from the training patients")
    ap.add_argument("--timing-repeats", type=int, default=200)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="skip folds that already have int8 results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.no_timing:                                  # must be set before TensorFlow starts
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.utils.set_random_seed(args.seed)

    timings, lines = [], []
    for run in args.runs:
        folds = sorted((RESULTS / run / "lopo").glob("chb*/model.keras"))
        if args.folds:
            folds = [f for f in folds if f.parent.name in args.folds]
        if not folds:
            print(f"{run}: no saved LOPO models found; skipping")
            continue
        print(f"{run}: {len(folds)} folds")
        timed = False
        for f in folds:
            _, kept = quantize_fold(run, f.parent, args)
            if kept and not args.no_timing and not timed:
                t = timing(run, *kept, args.timing_repeats)
                timings.append(t)
                timed = True
                print(f"  CPU time per window, 1 thread: Keras {t['keras_ms']:.3f} ms, "
                      f"float32 TFLite {t['tflite_float32_ms']:.3f} ms, int8 TFLite {t['tflite_int8_ms']:.3f} ms")
        done = [json.loads(p.read_text()) for p in sorted((RESULTS / f"{run}-int8" / "lopo").glob("chb*/metrics.json"))]
        if done:
            lines.append(summary_line(run, done))

    if timings:
        path = RESULTS / "int8_timing.csv"
        old = []
        if path.exists():                                   # keep timings of runs not redone now
            with open(path) as fh:
                old = [r for r in csv.DictReader(fh) if r["run"] not in {t["run"] for t in timings}]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(timings[0]))
            w.writeheader()
            w.writerows(old + timings)
        print(f"saved {path}")
    print("\nfloat -> int8 (int8 threshold re-picked on validation)")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
