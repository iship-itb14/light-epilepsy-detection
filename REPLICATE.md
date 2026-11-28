# How to replicate our results

This guide takes you from nothing to our results table: get the code, download the data, clean it, train both models and score them. It is written for students; no EEG background is needed.

> **Status (27 November 2026).** The whole CHB-MIT pipeline (scripts 00–06) has been run end to end on real data, including leave-one-patient-out for both models (final numbers 26 November 2026). The split-leakage demo (scripts 07–08, section 6b) was run on 27 November 2026. The expected outputs below are the ones we got.

For the reasoning behind every step, read `RESEARCH_NOTES.md`. For the official decision log, read `decision.md`.

---

## 0. What you need

| Item | What we used | Notes |
|---|---|---|
| Computer | MacBook Air (Apple silicon), CPU only | No GPU needed; a GPU will just be faster |
| Free disk | **about 30 GB** | ~20 GB raw + 7.4 GB cleaned + ~8 GB cache. You can delete the raw files after cleaning if space is tight |
| Python | 3.13 (Anaconda) | |
| Packages | `tensorflow` 2.20, `keras` 3.11, `numpy` 2.4, `scipy`, `mne` | See the Keras note in section 7 |
| Tools | `bash`, `curl`, `shasum` | Standard on macOS. See section 7 for Linux |
| Time | Download 1–3 h (depends on your connection), cleaning about 10 min, caching a few minutes, training about 30 minutes for the fixed split and 4–5 hours for leave-one-patient-out (both models, CPU) | |

Install the Python packages:

```bash
python3 -m pip install tensorflow mne scipy numpy
```

---

## 1. Get the code

Fork or clone the repository:

```bash
git clone https://github.com/iship-itb14/light-epilepsy-detection.git
cd light-epilepsy-detection
```

**Run every command from this project root.** The scripts use paths like `data/raw/...` relative to it.

What you get: the numbered scripts in `code/`, the shared package `code/seizure/`, and the saved patient splits in `data/splits/`. You do **not** get any EEG data. The data has its own licence and is about 20 GB, so you download it yourself from PhysioNet.

---

## 2. Download CHB-MIT (scripts 00 and 01)

The data is the CHB-MIT Scalp EEG Database on PhysioNet: https://physionet.org/content/chbmit/1.0.0/ (Open Data Commons Attribution License v1.0, DOI 10.13026/C2K01R). No account is needed.

**Step 2a: summaries (a few KB).**

```bash
./code/00_get_chbmit_summaries.sh
```

It prints nothing when it works. If a file comes back empty, it stops and names the file.

**Step 2b: the recordings (~20 GB).**

```bash
./code/01_get_chbmit_edf.sh
```

- It downloads the 14 patients we use: chb01, 03, 05, 06, 08, 10, 13, 14, 15, 16, 18, 20, 23, 24.
- It refuses to start with less than 22 GB free.
- It is safe to stop and rerun: verified files are skipped, partial files resume.
- **Note for Linux users:** this script was written and tested on macOS. Its free-space check uses `df -g`, which is a macOS-only option, so on Linux the script stops at that line. Check your free space with `df -h .` and see section 7 for the workaround. We have not changed the script yet.
- It ends with **`All files verified.`** (every file is checked against PhysioNet's SHA-256 checksums). Do not go on until you see this.

---

## 3. Clean the data (script 02)

```bash
python3 code/02_clean_chbmit.py
```

About 10 minutes. It writes `data/clean/chbmit/chbXX.npz` (one per patient) and `windows.csv`.

**Check your output against ours.** You should see exactly these lines:

```
chb01: 36610 windows, 112 seizure (+114 shifted), 11 flat, 304 extreme
chb03: 34304 windows, 103 seizure (+103 shifted), 3 flat, 469 extreme
chb05: 35243 windows, 140 seizure (+141 shifted), 0 flat, 570 extreme
chb06: 60100 windows, 40 seizure (+40 shifted), 6 flat, 1293 extreme
chb08: 18236 windows, 231 seizure (+231 shifted), 2 flat, 87 extreme
chb10: 45132 windows, 114 seizure (+112 shifted), 1 flat, 829 extreme
chb13: 29837 windows, 138 seizure (+137 shifted), 3 flat, 705 extreme
chb14: 23444 windows, 43 seizure (+44 shifted), 0 flat, 52 extreme
chb15: 36514 windows, 501 seizure (+506 shifted), 0 flat, 272 extreme
chb16: 17124 windows, 25 seizure (+24 shifted), 0 flat, 532 extreme
chb18: 32150 windows, 80 seizure (+80 shifted), 0 flat, 25 extreme
chb20: 24914 windows, 76 seizure (+75 shifted), 0 flat, 98 extreme
chb23: 24007 windows, 108 seizure (+107 shifted), 0 flat, 13 extreme
chb24: 19303 windows, 127 seizure (+137 shifted), 2 flat, 1256 extreme
```

("windows" includes the shifted seizure copies.) Small differences in the *extreme* column could come from a different filter implementation in another SciPy version; differences in the *seizure* column mean something is wrong.

---

## 4. Make the splits (script 03)

The splits are already in the repository (`data/splits/chbmit_splits.json`). Running script 03 recomputes them and the per-patient channel sums used for normalisation:

```bash
python3 code/03_split_chbmit.py
```

The last lines should read:

```
fixed train: 9 patients, 271981 windows, 1297 seizure (0.48%)
fixed val  : 2 patients, 59002 windows, 248 seizure (0.42%)
fixed test : 3 patients, 104058 windows, 292 seizure (0.28%)
```

followed by a training mean of 0 µV on every channel and a standard deviation of about 51–73 µV per channel.

**Do not change the splits** if you want to compare with our numbers. If you want to try your own, add a new key to the JSON rather than editing ours.

---

## 5. Build the cache (script 04)

```bash
python3 code/04_cache_chbmit.py
```

Writes about 8 GB to `data/cache/chbmit/`. It prints one line per patient, for example `chb01: cached 36610 windows (0.67 GB)`. You can delete `data/cache/` at any time and rerun this.

---

## 6. Train and score (scripts 05 and 06)

**Quick check (2 epochs):**

```bash
python3 code/05_train_chbmit.py --model eegnet --split fixed --epochs 2
```

**Fixed development split, both models:**

```bash
python3 code/05_train_chbmit.py --model eegnet  --split fixed --norm recording
python3 code/05_train_chbmit.py --model cnn-reg --split fixed --norm recording --augment
```

These are our final settings (`decision.md` 3.24). To reproduce the earlier steps in section 7.6 of `RESEARCH_NOTES.md`, drop `--norm recording` and/or `--augment`, or use `--model cnn`. The first `--norm recording` run spends a few extra minutes computing each recording's statistics, then caches them.

**Final results, leave-one-patient-out (14 folds per model):**

```bash
caffeinate -i bash -c 'for fold in $(seq 0 13); do
  python3 code/05_train_chbmit.py --model eegnet  --split lopo --fold $fold --norm recording --val-patients 3 --fa-rule median --skip-existing
  python3 code/05_train_chbmit.py --model cnn-reg --split lopo --fold $fold --norm recording --augment --val-patients 3 --fa-rule median --skip-existing
done' 2>&1 | tee results/chbmit/lopo_val3_log.txt
```

**Collect everything into one table:**

```bash
python3 code/06_summarise_chbmit.py
```

This writes `results/chbmit/summary.csv` and prints the LOPO mean and median per model.

Each run saves, in `results/chbmit/<model>/<split>/` (or `.../lopo/<patient>/`): `model.keras`, `history.csv`, `metrics.json`, `config.json`, and `test_predictions.npz` (the score for every test window, so you can rescore without retraining).

**Default settings** (all from `decision.md` 3.14, 3.18 and 3.21; change them with command-line flags):

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 50 | maximum epochs |
| `--patience` | 10 | stop after this many epochs without improvement |
| `--batch-size` | 256 | |
| `--neg-per-pos` | 4 | non-seizure windows per seizure window in each epoch |
| `--pos-weight` | 2.0 | loss weight of seizure windows |
| `--max-fa` | 1.0 | false alarms per hour allowed when picking the threshold |
| `--norm` | `train` | `train`: scale by the training patients' averages; `recording`: scale each recording by itself (our final runs) |
| `--augment` | off | random amplitude, noise and channel dropout on training windows (our final CNN) |
| `--val-patients` | 2 | LOPO only: validation patients per fold (our final runs use 3) |
| `--fa-rule` | `pooled` | `pooled` or `median` false-alarm rate across validation patients (our final runs use `median`) |
| `--channels` | `all` | `all` (18), `temporal4` or `temporal2`: which channels the model sees (decision 3.27) |
| `--skip-existing` | off | skip runs that already have results, so an interrupted loop can restart |
| `--seed` | 0 | random seed |

On macOS, `caffeinate -i` stops the computer sleeping during the long LOPO loop; leave it out on Linux.

**Our numbers to compare against** (leave-one-patient-out; full table in `results/chbmit/summary.csv` and section 13 of `RESEARCH_NOTES.md`):

| | EEGNet | CNN (`cnn-reg`) |
|---|---|---|
| Seizures caught | 80 / 128 | 83 / 128 |
| Window sensitivity, median per patient | 0.58 | 0.54 |
| False alarms / h, median per patient | 1.19 | 1.54 |

Without `--val-patients 3 --fa-rule median` (decision 3.24) you should get about 67 and 71 seizures caught instead; section 13.2 of `RESEARCH_NOTES.md` explains the difference.

 Expect some run-to-run variation, even with the same seed, because TensorFlow on different hardware is not bit-for-bit identical.

---

## 6b. Optional: the split-leakage demo (scripts 07 and 08)

This side study does not need CHB-MIT. It shows how much a shuffled split inflates scores, using the small Bonn EEG set (about 3 MB) [Andrzejak et al. 2001], cut into the same one-second chunks as the UCI/Kaggle "Epileptic Seizure Recognition" table. UCI no longer hosts that table, so we rebuild it.

```bash
bash code/07_get_bonn.sh
mkdir -p results/leakage
python3 code/08_leakage_demo.py 2>&1 | tee results/leakage/run_log.txt
```

- Script 07 downloads the five sets from the University of Bonn, checks each zip against the SHA-256 we recorded, and ends with **`All files verified: 5 sets x 100 recordings in data/raw/bonn`**. If the checksum differs, the site has changed the files; tell us.
- Script 08 needs only `keras` and `numpy`. It takes about 10 minutes on a MacBook Air (about 25 s per fold, 20 folds). It writes `results/leakage/folds.csv`, `summary.csv` and `config.json`.
- TensorFlow may print a warning about `tf.function retracing`. It is harmless.

**Our numbers to compare against** (mean over 5 folds):

| | Shuffled | Grouped |
|---|---|---|
| Seizure vs rest: accuracy | 0.988 | 0.974 |
| Seizure vs rest: sensitivity | 0.958 | 0.887 |
| Five classes: accuracy | 0.773 | 0.754 |

Section 13.5 of `RESEARCH_NOTES.md` explains what they mean.

---

## 6c. Optional: fewer channels (wearable case)

The same leave-one-patient-out runs with 4 or 2 temporal channels (`decision.md` 3.27). The 18-channel runs from section 6 are the comparison, so they are not repeated. This is 56 runs and took us about 13 hours of training on a MacBook Air; EEGNet goes first, and `--skip-existing` lets you stop and restart.

```bash
caffeinate -i bash -c 'for ch in temporal4 temporal2; do for fold in $(seq 0 13); do
  python3 code/05_train_chbmit.py --model eegnet --split lopo --fold $fold --norm recording --val-patients 3 --fa-rule median --channels $ch --skip-existing
done; done
for ch in temporal4 temporal2; do for fold in $(seq 0 13); do
  python3 code/05_train_chbmit.py --model cnn-reg --split lopo --fold $fold --norm recording --augment --val-patients 3 --fa-rule median --channels $ch --skip-existing
done; done' 2>&1 | tee -a results/chbmit/lopo_channels_log.txt
python3 code/06_summarise_chbmit.py
```

**Our numbers to compare against** (seizures caught of 128; median false alarms per hour):

| | 18 channels | 4 channels | 2 channels |
|---|---|---|---|
| EEGNet | 80; 1.19 | 83; 1.42 | 80; 1.43 |
| CNN (`cnn-reg`) | 83; 1.54 | 88; 1.29 | 85; 1.54 |

Section 13.6 of `RESEARCH_NOTES.md` explains what they mean.

---

## 7. Known problems and fixes

| Problem | Cause | What to do |
|---|---|---|
| Files from script 00 are 0 bytes | PhysioNet redirects to its Amazon S3 copy | Already fixed in the script (`curl -L`). If you edited it, keep `-L` |
| `Unable to broadcast tensor of shape [18] to tensor of shape [1,18,1]` when building a model | Older code passed the mean and variance to the normalisation layer in a shape Keras 3.11 rejects | Fixed in `code/seizure/models.py` (tested on Keras 3.11.3 and 3.15). Make sure you have the latest code |
| Script 01 fails at `df -g` on Linux | `df -g` is a macOS option | Check free space yourself with `df -h .`, then remove the free-space check or replace it with `df -BG` |
| `shasum: command not found` on Linux | Not installed by default on some distributions | Install the `perl` package (which provides `shasum`), or adapt the script to `sha256sum` |
| Out of memory during training | Reading the compressed files directly | Make sure script 04 ran; training reads from `data/cache/` |
| Out of disk | Raw + cleaned + cache is about 30 GB | After script 02 succeeds you can delete `data/raw/chbmit/*/*.edf`; script 01 can always download them again |

---

## 8. Add your own model

1. Open `code/seizure/models.py`.
2. Write a function `my_model(mean, std)` that returns a Keras model taking input shape `(18, 512)` in microvolts and giving one sigmoid output. Start it with the shared `_input(mean, std, 18, 512)` helper so it normalises the same way as ours.
3. Add it to the `MODELS` dictionary, e.g. `"my_model": my_model`.
4. Run `python3 code/05_train_chbmit.py --model my_model --split fixed --norm recording`, then the LOPO loop.

Because the splits, thresholds and scoring are shared, your numbers are directly comparable with ours.

---

## 9. Please cite the data

If you use CHB-MIT, PhysioNet asks you to cite:

- Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R
- Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*. PhD thesis, MIT.
- The PhysioNet platform citation shown on the dataset page.

If you use the Bonn data (leakage demo), the University of Bonn asks you to cite:

- Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C. E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E*, 64(6), 061907. https://doi.org/10.1103/PhysRevE.64.061907

The full reference list for the project is at the end of `RESEARCH_NOTES.md`.
