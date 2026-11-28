# Listening for Seizures: research notes

**What this file is.** A study companion to our project. It explains *what* we built, *how* it works, and *why* we made each choice, including the problems and bugs we ran into along the way. It is written for students who know some programming and some machine learning, not for EEG experts.

**What this file is not.** It is not a paper and it does not claim a new state of the art. The CHB-MIT results for both models are in section 13 (26 November 2026), the split-leakage demo is in section 13.5 (27 November 2026), and channel reduction is in section 13.6 (28 November 2026). Compression and the Siena test have not been run yet, and are marked *pending*. We would rather leave a gap than guess.

**Related files.**
- `decision.md` is the short, official log of every decision (numbered 3.1–3.27). This file explains those decisions at more length; when the two disagree, `decision.md` wins.
- `REPLICATE.md` is the step-by-step guide for rerunning everything.
- `code/` holds the scripts, numbered in the order they run.

Course: Neural Networks and Deep Learning (CSCI 5122), University of Colorado Boulder, Fall 2026.
Team: Sejal Hukare, Ishita Pawar, Swejal Gabhane.
Last updated: 27 November 2026.

---

## Contents

1. Background: epilepsy, EEG and why detection matters
2. Our research question
3. Related work and literature review
4. The data: what we used and how we cleaned it
5. Input and output, explained
6. The models: why these two, and how they work
7. How we train
8. How we score (evaluation)
9. Code walkthrough
10. Problems and bugs we caught and fixed
11. Known weak spots and limitations
12. Our decisions, in our own words
13. Results
14. What comes next
15. Glossary
16. References

---

## 1. Background: epilepsy, EEG and why detection matters

### 1.1 Epilepsy

The World Health Organization estimates that "around 50 million people worldwide have epilepsy, making it one of the most common neurological diseases globally" [WHO 2024]. A seizure is a short burst of abnormal electrical activity in the brain. The International League Against Epilepsy (ILAE) gives the clinical definition of epilepsy used by doctors [Fisher et al. 2014]; we do not need its details here, only the idea that seizures are sudden, often unpredictable events.

The WHO also notes that "up to 70% of people living with epilepsy could live seizure-free if properly diagnosed and treated" [WHO 2024]. Diagnosis and treatment depend on knowing *when* and *how often* seizures happen, and people often cannot report their own seizures reliably. That is one reason automatic detection is studied.

### 1.2 EEG

An **electroencephalogram (EEG)** records small voltages on the scalp, measured in microvolts (µV). Electrodes are placed in standard positions (the "10-20 system"). In a **bipolar montage**, each channel is the difference between two neighbouring electrodes, for example `FP1-F7`. Our data uses 18 such channels.

A seizure on EEG often looks like rhythmic, high-amplitude activity that starts in some channels and may spread. But EEG is also full of things that are *not* seizures: eye blinks, chewing, movement, loose electrodes and mains electrical hum. Telling these apart is the core difficulty.

### 1.3 Why automatic detection, and why small models

Reading days of EEG by hand is slow. A detector that flags likely seizures helps clinicians and, if it is small enough, could run on a **wearable device** at home. A wearable has a tiny processor and battery, so the model must be small and fast. That is why our project cares about model size (number of weights, size in KB) as much as accuracy.

A detector is only useful if people can live with it. Clinical standards for seizure detection devices ask for outcome measures such as sensitivity and false alarms to be reported in a standard way [Beniczky & Ryvlin 2018]. This shaped our choice of metrics (section 8).

---

## 2. Our research question

> **How well can a very small neural network detect seizures in EEG from a patient it has never seen, and how much detection do we lose as the model gets smaller?**

Broken into parts:

1. **Main question (CHB-MIT).** On children's EEG, with patients kept strictly separate between training and testing, how do a larger baseline CNN (328k weights) and the tiny EEGNet (1.7k weights) compare?
2. **Compression.** If we shrink EEGNet further (int8 quantization, then pruning), how much detection is kept?
3. **Honest evaluation (Bonn/UCI demo, done, section 13.5).** How much does a careless "shuffled" split inflate the scores compared with a split that keeps recordings apart?
4. **Wearable fit (done, section 13.6).** How much does detection drop with fewer channels (18 → 4 → 2)?
5. **Outside test (Siena, planned).** Does a model trained on children still work on adults from another hospital?

We expect (but have not shown) that the tiny model loses some sensitivity, and that results vary a lot between patients. These are expectations, not findings.

---

## 3. Related work and literature review

This is a short, honest review of the sources we actually read and checked. It is not a complete survey of the field. Every source here is listed in section 16 with a DOI or URL.

### 3.1 Public EEG seizure datasets

| Dataset | Who | Size (from the source) | Our use |
|---|---|---|---|
| **CHB-MIT Scalp EEG** [Guttag 2010; Shoeb 2009] | Children and young people at Boston Children's Hospital | 23 cases from 22 subjects, 256 Hz, 198 seizures in the collection | Main dataset |
| **Siena Scalp EEG** [Detti 2020; Detti et al. 2020] | Adults, ages 20–71 | 14 patients, 512 Hz, 47 seizures in about 128 hours | Outside test only (planned) |
| **TUH EEG Seizure Corpus** [Shah et al. 2018] | Hospital patients | Very large; SzCORE reports v2.0.0 as 675 subjects and 1,476 hours [Dan et al. 2024] | Not used (registration, mixed setups) |
| **Bonn** [Andrzejak et al. 2001] | Five sets, A–E | "Five sets (denoted A–E) each containing 100 single-channel EEG segments of 23.6-sec duration" | Leakage demo, cut into UCI-style chunks (section 13.5) |

**CHB-MIT** is the most used public scalp EEG set for seizure detection. The PhysioNet page notes that "case chb21 was obtained 1.5 years after case chb01, from the same female subject" [Guttag 2010], which matters for splitting (section 4.2). It is free under the Open Data Commons Attribution License, so we can link to it but we do not redistribute it.

**The UCI/Kaggle "Epileptic Seizure Recognition" dataset** is popular in student projects. Sources describe it as 500 recordings of about 23.6 s, cut into one-second chunks of 178 samples [Al-Marzouki 2025]. The UCI page confirms it is the Bonn data [Andrzejak et al. 2001]: it points to the Bonn epileptology site for "the original data" and says "We divided and shuffled every 4097 data points into 23 chunks". By 27 November 2026 UCI no longer offered the file (its download link returned 404 and its data API reported the dataset as not found), so we rebuilt the same table from the Bonn files, which are still free to download (section 13.5).

### 3.2 Deep learning on EEG

- **Reviews.** Roy et al. [2019] systematically reviewed deep learning studies on EEG. It is a good starting point for students who want the broader picture.
- **CNNs on raw EEG.** Schirrmeister et al. [2017] studied convolutional networks that learn directly from raw EEG instead of hand-made features. Our baseline CNN follows this general idea: give the network the signal and let it learn the filters.
- **EEGNet.** Lawhern et al. [2018] designed a compact CNN for EEG-based brain–computer interfaces. Its layers mirror classic EEG processing: temporal filters (frequency bands), then spatial filters (which channels to combine). The paper reports EEGNet-8,2 with roughly 1.7k–2.3k trainable parameters depending on the task, against about 150k–175k for a larger "DeepConvNet". This tiny size is exactly why we chose it for the wearable case. Note that EEGNet was designed for brain–computer interfaces, not seizure detection; using it for seizures is our choice to test, not something the paper claims.

### 3.3 How the split changes the result

This is the theme that most shaped our project.

- **SzCORE** [Dan et al. 2024] is a proposed framework for evaluating seizure detection. It says methodological choices "can lead to radically different results" and that subject-independent models are valid only if "independence of subjects between training and test sets is maintained". It also notes that a single fixed split on a small dataset "can lead to more variability in estimated performance".
- **Leakage in ML research.** Kapoor & Narayanan [2023] describe *data leakage* across many fields. Their top-level type is a "lack of clean separation of training and test set": if the two are not kept apart "during all pre-processing, modeling and evaluation steps, the model has access to information in the test set before its performance is evaluated."

Two recent papers illustrate the gap (we read them only in part):

| Paper | Data | Split | Headline result |
|---|---|---|---|
| Ayala-Britez et al. [2026] | CHB-MIT, 10 s windows, classical ML | Patient-level ("no patient appeared in both the training and test sets") | 86.59% accuracy |
| Al-Marzouki [2025] | UCI/Kaggle, BiLSTM | 80/10/10 split of a balanced subset; patient separation not stated | 98.70% validation accuracy |

These two differ in data and model too, so we **cannot** blame the whole gap on the split. That is exactly why we ran our own controlled demo (section 13.5): same model, same data, two splits.

### 3.4 How detectors should be scored

- Beniczky & Ryvlin [2018] propose standards for testing and validating seizure detection devices, including which outcome measures should be reported.
- SzCORE [Dan et al. 2024] recommends event-based scoring with specific rules. It advises "splitting events longer than 5 minutes", "merging events separated by less than 90 seconds", a "30 seconds pre-ictal tolerance" and a "60 seconds post-ictal tolerance". We adopted the 5-minute split (after a bug, see section 10.5) but **not** the merge or tolerance rules yet. So our false-alarm numbers will not be directly comparable with SzCORE-scored results. We say this up front.

### 3.5 Making models smaller

- **Quantization.** Jacob et al. [2018] showed how to run networks with integer-only arithmetic. Google's post-training quantization guide lists full integer quantization as about "4x smaller, 3x+ speedup" and says it needs "a representative dataset to calibrate" [Google AI Edge]. This is our first compression step (planned).
- **Pruning.** Han, Mao & Dally [2016] combined pruning, quantization and coding to shrink networks a lot. We plan a simpler version: prune 50–80% of weights, then fine-tune.

### 3.6 Where our project fits

We are not proposing a new architecture. Our contribution is modest and practical:

1. a fully scripted, rerunnable pipeline from raw download to results, with saved patient splits;
2. a careful, patient-wise comparison of a baseline CNN and a tiny EEGNet on the same data, scored by window and by event, including false alarms per hour;
3. a controlled demo of how much a shuffled split inflates scores (section 13.5), a channel-reduction study for the wearable case (section 13.6), and a planned outside test on another dataset.

---

## 4. The data: what we used and how we cleaned it

### 4.1 Which patients

We used **14 CHB-MIT patients**: chb01, 03, 05, 06, 08, 10, 13, 14, 15, 16, 18, 20, 23, 24 (decision 3.11).

- **Rule.** Every patient with 5 or more seizures whose files all contain the 18 standard bipolar channels.
- **chb12 excluded.** It has the most seizures (40), but 3 of its files use a different electrode layout.
- **chb21 not used.** It is the same child as chb01. If it is ever added, it must be treated as the same subject.
- **Size.** About 480 hours and 128 seizures; about 20 GB raw, 7.4 GB cleaned.

### 4.2 Cleaning pipeline (script `02_clean_chbmit.py`)

| Step | What | Why |
|---|---|---|
| 1. Channels | Keep 18 bipolar channels in a fixed order; drop the duplicate `T8-P8` | Every patient must give the model the same inputs in the same order |
| 2. Filter | 0.5–40 Hz band-pass, 4th-order Butterworth, zero phase (`sosfiltfilt`) | Removes slow drift (below 0.5 Hz) and 60 Hz mains hum. Zero phase means the filter does not shift events in time |
| 3. Downsample | 256 → 128 Hz | After a 40 Hz low-pass, 128 Hz still captures everything (the Nyquist limit is 64 Hz). It halves storage |
| 4. Windows | 4-second windows, no overlap, never crossing a file boundary | A window is the unit the model classifies. Crossing files would glue unrelated recordings together |
| 5. Labels | Seizure (1) if at least 50% of the window lies inside an annotated seizure; the exact fraction is stored | A clear rule; storing the fraction lets us change the rule later without cleaning again |
| 6. Flags | *Flat*: any channel with std below 0.5 µV in the unfiltered signal. *Extreme*: filtered peak above 1000 µV. Windows are flagged, not dropped | Lets us study artifacts without losing data. Seizure windows are never dropped |
| 7. Extra seizure windows | Seizure windows shifted by 2 s, marked `aug=1`, used for training only | Doubles the rare class without inventing data. Test data never contains these copies |

Normalisation is **not** done here. It must come from training patients only, after the split.

**Output.** 435,067 ordinary windows (about 483 hours), of which **1,838 are seizure windows (0.42%, about 1 in 236)**. Plus 1,851 shifted seizure copies for training. 6,505 windows (1.5%) are flagged extreme.

### 4.3 Splits (script `03_split_chbmit.py`)

No patient is ever in more than one set (decision 3.3).

- **Fixed development split.** Test: chb03, chb10, chb20. Validation: chb05, chb23. Training: the other 9.

  | Set | Patients | Windows | Seizure windows |
  |---|---|---|---|
  | Train | 9 | 271,981 | 1,297 (0.48%) |
  | Validation | 2 | 59,002 | 248 (0.42%) |
  | Test | 3 | 104,058 | 292 (0.28%) |

  (Counts exclude flat windows.) Test and validation patients have typical seizure counts; the seizure-heavy patients (chb08, chb15) stay in training.
- **Leave-one-patient-out (LOPO), for final results.** 14 folds. Fold *k* tests on one patient, validates on the next two patients in the list (wrapping round), and trains on the other 11.

The script also stores each patient's per-channel count, sum and sum of squares, so the training code can compute the mean and standard deviation of **exactly the training patients** of any split.

---

## 5. Input and output, explained

### 5.1 Input

One input is a single 4-second window:

```
shape:  (18, 512)      18 channels × 512 samples (4 s at 128 Hz)
units:  microvolts (µV), filtered 0.5–40 Hz, NOT normalised
type:   float32 (stored as float16 on disk)
```

The model normalises internally (section 6.1), so it takes raw microvolts. A batch is `(batch_size, 18, 512)`.

In our final runs (`--norm recording`, section 7.6) each window is first divided by its own recording's typical spread on each channel, so what reaches the model is close to unit scale, not microvolts.

### 5.2 Output

```
shape:  (1,)   one number between 0 and 1 (sigmoid)
meaning: the model's seizure score for this window
```

The score is **not** a calibrated probability. Training sees about 1 seizure window per 4 non-seizure windows, while real recordings have about 1 in 236, so a score of 0.8 does not mean "80% chance".

### 5.3 From score to alarm

1. **Threshold.** A window is *flagged* if its score ≥ threshold. The threshold is picked on validation patients (section 7.5), never on test patients.
2. **2-of-3 rule.** Window *i* raises an **alarm** if at least 2 of windows *i*-2, *i*-1, *i* are flagged. It only looks backwards (a wearable cannot see the future) and never counts windows across different files or across time gaps.
3. **Events.** A run of alarm windows is an alarm event. Runs longer than 5 minutes are split into 5-minute events.

The alarm output, not the raw flag, is what we score.

---

## 6. The models: why these two, and how they work

Both models live in `code/seizure/models.py`.

### 6.1 Shared first layer: normalisation inside the model

```python
norm = layers.Normalization(axis=1)
x = norm(inp)                                   # build the layer first
norm.set_weights([mean, std ** 2, count])       # then load the training patients' statistics
norm.finalize_state()
```

Each channel is z-scored: `(x - mean) / std`, with the mean and std of the **training patients only**, fixed at build time.

**Why inside the model?** A saved model (and later the int8 version on a wearable) then carries its own scaling. Nobody can feed it wrongly scaled data by forgetting a preprocessing step.

**What we changed after the first results (decision 3.22).** Scaling everyone by the *training patients'* averages turned out to be the biggest weakness: scores did not carry over between patients (section 7.6). The final runs therefore scale each window by **its own recording's** per-channel median and robust spread (1.4826 × median absolute deviation) before the model, and the layer above is left as "do nothing" (mean 0, std 1). The original behaviour is still the default (`--norm train`) so the first results can be reproduced.

### 6.2 Baseline: 1D CNN (about 328k weights)

**Why.** A plain convolutional network is simple, strong on raw signals and easy to explain. It gives a fair reference point: "this is what a reasonably sized network gets".

**How.**

```
input (18, 512) → normalise → transpose to (512 time steps, 18 channels)
Block 1: [Conv1D 32, kernel 5 → BatchNorm → ReLU] × 2 → MaxPool 4   (512 → 128)
Block 2: [Conv1D 64, kernel 5 → BatchNorm → ReLU] × 2 → MaxPool 4   (128 → 32)
Block 3: [Conv1D 128, kernel 5 → BatchNorm → ReLU] × 2 → MaxPool 4  (32 → 8)
Block 4: [Conv1D 128, kernel 5 → BatchNorm → ReLU] × 2
GlobalAveragePooling → Dropout 0.5 → Dense 1 (sigmoid)
```

- *Conv1D* slides small filters along time; each filter learns a pattern across all 18 channels at once.
- *Max pooling* shortens the sequence, so deeper layers see longer stretches of time.
- *BatchNorm* [Ioffe & Szegedy 2015] keeps layer outputs in a stable range, which makes training easier.
- *Global average pooling* averages each feature over time, so the network asks "did this pattern appear anywhere in the window?"
- *Dropout* randomly switches off half the features during training to reduce overfitting.

**The version in the final results, `cnn-reg` (decision 3.23).** The same network plus weight decay (L2 = 1e-4 on every convolution and the output layer) and spatial dropout (0.2, switching off whole feature maps after each block). The plain CNN memorised its training patients (section 7.6); these two additions limit that. The weight count does not change.

**Where the weights are.** Almost all are in the convolutions: 325,440 convolution weights + 2,816 batch-norm values + 129 in the final layer = 328,385 (our arithmetic, matching the code's count).

### 6.3 Lightweight: EEGNet-8,2 (about 1.7k weights)

**Why.** EEGNet [Lawhern et al. 2018] is built for EEG, widely cited, and tiny. Its paper reports about 1.7k–2.3k trainable parameters for EEGNet-8,2. Ours has 1.7k, a little under the 2–5k we first estimated (decision 3.21).

**How.** EEGNet treats the window as a small image: 18 rows (channels) × 512 columns (time).

```
input (18, 512) → normalise → reshape to (18, 512, 1)
1. Temporal conv: 8 filters of size (1, 64)          learn 8 frequency filters (64 samples = 0.5 s)
   → BatchNorm
2. Depthwise conv: size (18, 1), depth multiplier 2  for each frequency filter, learn 2 ways of combining the 18 channels
   → BatchNorm → ELU → AvgPool (1, 4) → Dropout 0.25
3. Separable conv: 16 filters of size (1, 16)        summarise each feature over time, then mix features
   → BatchNorm → ELU → AvgPool (1, 8) → Dropout 0.25
4. Flatten (16 features × 16 time steps = 256) → Dense 1 (sigmoid)
```

The idea, in plain words: first find useful **rhythms** (step 1), then find **where on the head** they appear (step 2), then look at **how they change over time** (step 3). Depthwise and separable convolutions are the trick that keeps the weight count tiny.

**Dropout 0.25.** The EEGNet authors "set the dropout probability to 0.25 in cross-subject classification" (0.5 for within-subject). Our task is cross-subject, so we followed that.

### 6.4 Side by side

| | Baseline CNN | EEGNet-8,2 |
|---|---|---|
| Weights | 328k | 1.7k (about 190× fewer) |
| Input | (18, 512) µV | same |
| Output | one score 0–1 | same |
| Main idea | stacked 1D convolutions over time | frequency filters → spatial filters → temporal summary |
| Role in the study | reference point | wearable candidate, to be compressed |

---

## 7. How we train (script `05_train_chbmit.py`)

### 7.1 The imbalance problem

Only 1 window in 236 is a seizure. If we trained on windows in their natural ratio, the model could predict "no seizure" every time and be 99.6% accurate while catching nothing. Accuracy is useless here.

### 7.2 Balanced batches

Each epoch we use **every** seizure window (with the shifted copies) plus **4 non-seizure windows for each seizure window**, drawn at random without replacement and **redrawn every epoch**. Over many epochs the model sees a wide variety of normal EEG, but each batch still has enough seizures to learn from. Validation and test data are **never** rebalanced.

Windows flagged as artifacts stay in training, so the model learns that a big blink is not a seizure.

### 7.3 Loss and optimiser

- **Weighted binary cross-entropy.** Seizure windows weigh 2 in the loss ("mild" extra weight on top of the balancing).
- **Adam** [Kingma & Ba 2015], learning rate 0.001, batch size 256, at most 50 epochs, fixed random seed.

These are standard, stable defaults. We did not tune them; tuning on test patients would be leakage, and tuning on validation patients is future work.

### 7.4 Memory: the cache (script `04_cache_chbmit.py`)

The cleaned files are compressed, so reading one window means unpacking a whole patient (up to 1.1 GB). Script 04 unpacks them once into plain `.npy` files (about 8 GB). Training then **memory-maps** them: the operating system loads only the windows each batch needs. This keeps a laptop from running out of memory.

### 7.5 Picking the threshold and the best epoch

After every epoch we score **all** validation windows (unbalanced, as recorded):

1. For many candidate thresholds, apply the 2-of-3 alarm rule and count false alarms per hour.
2. Among thresholds with **at most 1 false alarm per hour**, keep the one with the best window sensitivity.
3. The epoch with the best such sensitivity is saved. Epochs that cannot meet the limit at any threshold always rank below those that can.
4. Stop after 10 epochs without improvement.

**Why this and not validation loss?** A wearer cares about "does it catch my seizures without waking me up every hour", not about cross-entropy. 1 false alarm per hour is a common target, though for daily life it is still a lot; we report the real number either way.

The threshold candidates include quantiles of the predicted scores, so a very confident model (scores bunched near 0 or 1) still gets usable thresholds.

### 7.6 What the first results taught us, and what we changed

We developed on the fixed split (test chb03, chb10, chb20; validation chb05, chb23). Every change below was **chosen by the validation patients**, never by the test patients.

| Step | Run | Validation sensitivity | Test seizures caught | Test false alarms / h |
|---|---|---|---|---|
| 1. First run | CNN | 0.43 | 4 / 22 | 0.10 |
| | EEGNet | 0.40 | 5 / 22 | 0.05 |
| 2. Per-recording normalisation (3.22) | CNN | 0.63 | 13 / 22 | 0.28 |
| | EEGNet | **0.69** | **17 / 22** | 0.49 |
| 3. Weight decay, spatial dropout, augmentation (3.23) | CNN (`cnn-reg`) | **0.66** | 13 / 22 | **0.03** |
| | EEGNet + augmentation | 0.63 | 13 / 22 | 2.33 |

**Step 1 → 2.** The first scores looked hopeless, but a check of the saved test scores showed the models *ranked* windows reasonably (window AUROC 0.90 for the CNN, 0.76 for EEGNet). The problem was the threshold: one chosen on the validation patients was far too strict for the test patients, because each patient's scores sat at a different level. Patients differ a lot in amplitude, so we scaled each recording by itself. This roughly tripled the seizures caught.

**Step 2 → 3.** The CNN's training loss fell to 0.03 while its validation score jumped between 0.19 and 0.63 from epoch to epoch: it was memorising the 9 training patients. Weight decay, spatial dropout and **augmentation** (training windows only: random amplitude 0.8–1.25×, 5% noise, and in 30% of windows one channel set to zero, like a loose electrode) made it steadier, with almost no false alarms. Augmentation **hurt** EEGNet: with 1.7k weights it has little room to memorise, so the extra noise mostly got in the way.

**So the final settings (3.24) are:** both models use per-recording normalisation; the CNN is `cnn-reg` with augmentation; EEGNet has no augmentation.

These were single runs with one seed on three test patients, so small differences may be chance. That is why the numbers we report come from leave-one-patient-out (section 13).

---

## 8. How we score (evaluation, `code/seizure/scoring.py`)

| Metric | Plain meaning | Why we report it |
|---|---|---|
| **Sensitivity** (recall) | Of the seizure windows, how many raised an alarm? | Missing seizures is the worst failure |
| **Specificity** | Of the normal windows, how many stayed quiet? | Shows the other side, but is close to 1 for almost any model because normal windows are so common |
| **F1** | Balance of precision and sensitivity | One number that punishes both misses and false alarms |
| **Event sensitivity** | Of the seizures (events), how many had at least one alarm window? | Closer to what a clinician cares about |
| **False alarms per hour** | Alarm events touching no seizure, per hour recorded | What a wearer would live with |
| **Confusion matrix** | TP, FN, FP, TN on alarm windows | Required by the course; shows everything at once |
| **Size and speed** | Weights, KB, CPU time per prediction | The wearable argument |

We also report accuracy because the course asks for it, but we never use it to choose anything.

**Leave-one-patient-out** gives 14 test results per model. We report the mean **and** the median, because one hard patient can drag the mean a long way.

Each run saves the score for every test window (`test_predictions.npz`), so anyone can rescore with different rules without retraining.

---

## 9. Code walkthrough

Run everything from the project root, in order.

| Script | Input | Output | One-line job |
|---|---|---|---|
| `00_get_chbmit_summaries.sh` | internet | `data/raw/chbmit/` text files | Fetch the per-patient seizure summaries and checksum list |
| `01_get_chbmit_edf.sh` | summaries | `data/raw/chbmit/chbXX/*.edf` (~20 GB) | Download the 14 patients' recordings and verify SHA-256 checksums; resumable |
| `02_clean_chbmit.py` | `.edf` files | `data/clean/chbmit/chbXX.npz`, `windows.csv` | Filter, downsample, window, label, flag (section 4.2) |
| `03_split_chbmit.py` | cleaned files | `data/splits/chbmit_splits.json`, `chbmit_channel_stats.npz` | Fix the patient splits and save per-patient sums for normalisation |
| `04_cache_chbmit.py` | cleaned files | `data/cache/chbmit/` (~8 GB) | Unpack into memory-mappable arrays |
| `05_train_chbmit.py` | cache + splits | `results/chbmit/<model>/<split>/` | Train one model on one split, pick the threshold, score the test patients |
| `06_summarise_chbmit.py` | all `metrics.json` | `results/chbmit/summary.csv` | One table of every run, with LOPO mean and median |

Shared code in `code/seizure/`:

- `data.py`: `get_split` (including the LOPO validation rule), `normalisation_stats` (reuses script 03 so there is one source of truth), `recording_stats` and `RecordingNorm` (per-recording scaling, cached in `data/cache/chbmit/chbXX_recstats.npz`), `BalancedBatches` (with optional augmentation), `predict_patients`.
- `models.py`: `baseline_cnn`, `cnn-reg` (the same with weight decay and spatial dropout) and `eegnet`, registered in `MODELS`. **To add your own model**, write a function `my_model(mean, std)` that returns a Keras model taking `(18, 512)` and giving one sigmoid output, add it to `MODELS`, and run `05_train_chbmit.py --model my_model`.
- `scoring.py`: the alarm rule, all metrics, and `pick_threshold`.

### 9.1 Two pieces of logic worth reading slowly

**Normalisation without leakage.** Script 03 stores, for each patient and channel, *n* (sample count), Σx and Σx². For any list of training patients the mean and std come from simple sums:

```
mean = Σx / n          std = sqrt(Σx² / n − mean²)
```

So the code never needs to load test patients to compute training statistics.

**The 2-of-3 rule with segment boundaries.** Windows are sorted by file and start time. A *segment id* changes whenever the file changes or there is a time gap. The rule only counts neighbours in the same segment:

```python
count = flag.copy()
count[1:] += flag[:-1] * (seg[1:] == seg[:-1])   # previous window, same segment
count[2:] += flag[:-2] * (seg[2:] == seg[:-2])   # two back, same segment
alarm = count >= 2
```

---

## 10. Problems and bugs we caught and fixed

We think a list of mistakes is one of the most useful things a student project can publish. Each entry says what we saw, why it happened, how we fixed it, and what we learned.

| # | Where | Symptom | Cause | Fix | Status |
|---|---|---|---|---|---|
| 1 | `00_get_chbmit_summaries.sh` | 26 of 27 downloaded files were 0 bytes, with no error | PhysioNet answered with an HTTP 302 redirect to its Amazon S3 copy; `curl` without `-L` saved the empty redirect body | Added `-L` (follow redirects) and a check that stops the script if any file is empty | Fixed |
| 2 | `02_clean_chbmit.py` (draft) | Test run failed when saving | A name clash when writing the output file | Corrected in the draft before it was added to the project (details not recorded) | Fixed before first real run |
| 3 | `02_clean_chbmit.py` (draft) | On a fake recording, the planted flat channel was not flagged | The band-pass filter smooths a flat line into a small wiggle, so "flat" never triggered | Flat check moved to the **unfiltered** signal | Fixed before first real run |
| 4 | `02_clean_chbmit.py` | 23% of all windows, and **47% of seizure windows**, flagged "extreme" | A 500 µV limit on the unfiltered signal is too low for children's EEG, and seizures themselves are large (in chb08 the old rule flagged 76% of seizure windows) | Limit changed to a 1000 µV peak on the **filtered** signal. Flags fell to 1.5% of windows. Seizure windows are never dropped | Fixed, data recleaned |
| 5 | `scoring.py` (draft) | A model that alarms **all the time** scored **0 false alarms per hour** | One endless alarm that touches a seizure somewhere counts as one "true" event and zero false ones | Alarm runs longer than 5 minutes are split into 5-minute events (as SzCORE advises); epochs that cannot meet the false-alarm limit rank below those that can | Fixed (found in a synthetic test) |
| 6 | `models.py` | On Keras 3.11: `Unable to broadcast tensor of shape [18] to tensor of shape [1,18,1]` when building the model | Our sandbox used a newer Keras; passing the mean and variance to the `Normalization` constructor needs differently shaped arrays in different Keras versions | The layer is now built first, then given its mean and variance with `set_weights` and `finalize_state`. Tested on Keras 3.11.3 and 3.15 | Fixed, confirmed on the first real run |

### 10.1 What these taught us

- **Silent failures are the dangerous ones (bugs 1 and 5).** An empty file and a perfect false-alarm score both *look* fine. We now add explicit checks ("stop if empty") and test with deliberately bad inputs (a model that always says "seizure").
- **Test on fake data with planted answers (bugs 2, 3, 5).** We built a 10-minute fake recording with a seizure at 100–140 s, a flat channel at 300 s and a spike at 400 s, then checked that the labels and flags landed exactly there. Three bugs were caught this way before touching real data.
- **Look at the numbers, not just "it ran" (bug 4).** The cleaning script ran without errors. Only reading the per-patient counts showed that the artifact flag was nonsense.
- **Pin your library versions (bug 6).** Code that works on one Keras version can break on another. `REPLICATE.md` lists the versions we used.

### 10.2 Plan changes that were not bugs

Not every change was a fix. Some plans changed as we learned more:

- **8 patients → 14.** We first suggested 8 patients because the laptop had 30 GB free. The team freed space, and we used all 14 eligible patients, because leave-one-patient-out is more trustworthy with more patients.
- **Drop artifacts → keep and flag them.** The first plan dropped artifact windows from training. After bug 4 showed how many seizure windows look "extreme", we kept everything and stored flags instead.
- **256 Hz → 128 Hz.** Halved storage with no loss after the 40 Hz filter.
- **Normalisation moved out of cleaning.** Doing it in script 02 would have used test patients' statistics. It now happens after the split, and inside the model.
- **Training-patient scaling → per-recording scaling.** The first real results showed that one scale for everyone made thresholds fail on new patients (section 7.6).
- **A plain CNN → a regularised one, and augmentation for the CNN only.** The plain CNN memorised its training patients; augmentation helped the CNN but hurt EEGNet on validation.
- **UCI table → the original Bonn files.** UCI stopped hosting the leakage-demo table, so we cut the Bonn recordings into the same one-second chunks ourselves. A side benefit: we know for certain which recording each chunk came from, instead of reading it from an undocumented id column.

---

## 11. Known weak spots and limitations

We listed these before we had results, so we could not be accused of explaining them away afterwards. Notes added after the results are marked *(after results)*.

1. **Children only.** CHB-MIT is a paediatric dataset from one hospital. Results may not hold for adults (the planned Siena test checks this).
2. **Few seizures per patient.** Validation patients chb05 and chb23 have about 12 seizures between them, so validation sensitivity moves in big steps and the "best epoch" partly depends on luck. LOPO averages this out but does not remove it.
3. **Saturated scores.** On synthetic data the CNN gave many windows a score of exactly 1.0, and then no threshold can separate them. If the chosen threshold shows as 1.000 on real data, we will search thresholds on the model's raw output before the sigmoid. *(After results: this happened in the first CNN run, threshold 0.9994. With per-recording scaling and regularisation the thresholds came down to about 0.35–0.6, so we did not need the change.)*
4. **Batch-norm statistics.** The CNN learns its batch-norm averages from 1:4 balanced batches, but real data is 1:236. The validation threshold mostly corrects for this, but the scores are not true probabilities.
5. **One scale for everyone.** Normalisation uses training patients' averages; a patient with unusual amplitudes may score worse. The 1.5% extreme windows also widen the spread a little. *(After results: this was the biggest problem, and we switched to per-recording scaling, section 7.6.)*
5a. **Per-recording scaling looks ahead.** *(After results.)* Each recording's median and spread are computed over the whole recording (about an hour), including the future. A wearable would need a running estimate from the minutes so far; we have not tested that.
5b. **Some patients fail completely.** *(After results.)* Both models catch no seizures in chb06, chb14 and chb16. chb16's seizures are very short (median 8 s), so 4-second windows with the 2-of-3 rule can barely catch them; chb06's seizures score no higher than normal EEG. Overlapping windows (for example a 1–2 s step) would give short seizures more windows; we have not tried this because it means re-cleaning the data.
5c. **The threshold rule is a trade-off.** *(After results.)* With few validation patients per fold, one unusual patient can pull the threshold too high (chb16, first rule) or too low (chb14, final rule). Section 13.2 shows both. More validation patients, or a threshold tuned per wearer after a short calibration period, would help.
6. **Detection delay.** The 2-of-3 rule needs two flagged windows, so an alarm comes at least 4–8 seconds after the first seizure window.
7. **Border windows.** A window that is 30% seizure is labelled non-seizure and can be used as a negative in training. That is a strict choice that may teach the model to ignore seizure onsets. (The false-alarm count does treat any window touching a seizure as "not false".)
8. **No tolerance windows.** Unlike SzCORE we do not yet allow 30 s before or 60 s after a seizure, and we do not merge nearby alarms. Our false-alarm rate is therefore not directly comparable with SzCORE-scored work.
9. **One seed, no tuning.** Each run uses one random seed and default hyperparameters. Differences between models smaller than seed-to-seed noise should not be over-read.
10. **Download script is macOS-only.** `01_get_chbmit_edf.sh` checks free space with `df -g`, which only exists on macOS, so it fails on Linux. `REPLICATE.md` gives the workaround. The script itself has not been changed yet.
11. **Not a medical device.** Nothing here has been clinically validated.
12. **The leakage demo keeps recordings apart, not people.** *(After results.)* The Bonn files have no person ids, and each set comes from only five people, so even our "grouped" split shares people between training and test. Its scores are therefore still optimistic. The demo is also one seed on very clean, hand-picked clips, so its exact gaps are rough.
13. **Only one choice of reduced channels.** *(After results.)* We tested one 4-channel and one 2-channel set, chosen in advance, not the best possible sets, and not the electrodes a real ear-worn device would use (those are not standard 10-20 positions). The CNN's augmentation also zeroes one of the two channels in 30% of training windows at 2 channels, a much bigger loss than with 18. A wearable would also record more noise from movement than a hospital cap.

---

## 12. Our decisions, in our own words

This part is written the way we would explain the project to a classmate.

**We decided to frame the task as window classification.** A 4-second window goes in, a seizure score comes out. We did this because it is the simplest task a tiny model on a wearable could run continuously, and it maps directly onto the confusion matrix the course asks for.

**We chose CHB-MIT as our main dataset** because it is free, widely used (so our numbers can be compared with others'), recorded continuously, and has enough patients for a real patient-wise split. We looked at TUH, which is far bigger, but decided against it because it needs registration and its recordings use mixed sampling rates and channels. We also looked at the Bonn/UCI clips and decided not to train our main model on them, because they carry no patient information and so cannot support our most important rule.

**We decided that no patient may appear in both training and testing.** This is the single biggest reliability choice in the project. EEG from one person is very similar from hour to hour, so mixing a person across sets lets the model "recognise the person" rather than the seizure. We went one step further: because chb01 and chb21 are the same child, we left chb21 out so it could never leak.

**We picked 14 patients** because they are all the patients with at least 5 seizures and a consistent set of 18 channels. We left out chb12 even though it has the most seizures, because part of it uses a different electrode layout and we did not want to invent channels.

**We filtered to 0.5–40 Hz and downsampled to 128 Hz** because this removes drift and mains hum while keeping the seizure rhythms, and it halves the storage. We used a zero-phase filter so that events are not shifted in time.

**We decided to flag artifacts instead of deleting them.** Our first plan dropped extreme windows. When we ran it, almost half of the seizure windows were flagged. We realised that seizures *are* extreme signals, and that deleting them would throw away exactly what we need. We raised the limit to 1000 µV on the filtered signal, and we kept every window.

**We kept two split schemes.** We use a fixed split to develop quickly, and leave-one-patient-out for the final numbers, because SzCORE warns that a single split on a small dataset varies a lot. We saved both splits as files so that anyone can test on exactly the same patients.

**We chose two models on purpose.** We wanted a fair "normal-sized" reference (a 1D CNN, 328k weights) and a genuinely tiny candidate (EEGNet, 1.7k weights). We picked EEGNet because it was designed for EEG and is small enough for a microcontroller. The question we care about is not "which is best" but "how much do we lose as the model shrinks".

**We put normalisation inside the model** because we expect to convert the model to int8 for a wearable. If scaling were a separate step, it would be easy to forget it and feed the model the wrong numbers.

**We changed to scaling each recording by itself after our first results.** The first models caught only 4–5 of 22 test seizures. Instead of tuning blindly, we looked at the saved scores and saw that the models ranked windows well but each patient's scores sat at a different level, so one threshold could not fit everyone. Scaling each recording by its own typical amplitude fixed most of that. We know it looks ahead in time, and we say so.

**We changed the validation rule after finding that one patient was setting the threshold for others.** chb16's noisy EEG forced the threshold so high in two folds that chb15's seizures were missed, although the models ranked them well. We saw this on the validation patients themselves, so we fixed the rule (three validation patients, median false alarms), reran everything, and report the new numbers. They are better for seizures caught and worse for false alarms, and we show both rules side by side so nobody has to take our word for it.

**We chose every improvement on the validation patients, not the test patients.** It would have been easy to keep whatever looked best on test. That would make our test numbers a lie. When augmentation helped the CNN but hurt EEGNet on validation, we kept it for one and dropped it for the other.

**We dealt with imbalance by balancing batches, not by changing the test data.** We show the model every seizure window and a fresh sample of normal windows each epoch. We never rebalance validation or test data, because a wearer will not get rebalanced EEG.

**We stopped training on sensitivity at a fixed false-alarm rate, not on loss.** We did this because it matches what a wearer cares about: catching seizures without constant false alarms.

**We added a 2-of-3 alarm rule** because a single noisy window should not wake anyone up. We made it look backwards only, because a device cannot see the future.

**We split long alarms into 5-minute events after our own test fooled us.** A model that alarmed all the time scored zero false alarms. We would not have noticed if we had only tested "normal" models.

**We chose TensorFlow/Keras** because converting to an int8 TensorFlow Lite model is built in and is a standard route onto microcontrollers.

**We decided to ship code, not data.** The raw data is about 20 GB and has its own licence. Instead we ship numbered scripts, the saved splits, and the scoring code, so others can rebuild everything themselves and check our work.

**We decided to write down our mistakes.** A project that only shows its final numbers hides how fragile those numbers can be. Our bug list (section 10) is part of the result.

---

## 13. Results

Final CHB-MIT results, **leave-one-patient-out**: each of the 14 patients is tested by a model that never saw them. Each fold trains on 10 patients and uses the next 3 in the list as validation, with the median false-alarm rule (decisions 3.24 and 3.25). Run on 26 November 2026 on a MacBook Air (CPU only), one seed. Every number comes from `results/chbmit/summary.csv` (runs named `...-val3-median`).

### 13.1 Summary

| | EEGNet-8,2 | Baseline CNN (`cnn-reg`) |
|---|---|---|
| Weights | **1,729** | 328,385 (about 190× more) |
| Saved model size | **92 KB** | 3,983 KB |
| Seizures caught, all 14 patients | 80 / 128 (63%) | 83 / 128 (65%) |
| Event sensitivity, mean / median per patient | **0.68 / 0.94** | 0.66 / 0.89 |
| Window sensitivity, mean / median | **0.48 / 0.58** | 0.44 / 0.54 |
| Specificity, mean / median | 0.979 / 0.995 | 0.982 / 0.996 |
| F1, mean / median | 0.30 / 0.26 | 0.30 / 0.24 |
| False alarms per hour, mean / median | 3.80 / **1.19** | 3.80 / 1.54 |

"Seizures caught" counts every seizure across all patients. The other rows average over patients, so each patient counts once however many seizures they had. The false-alarm means are pulled up by chb08 (about 20 per hour for both models) and chb16 (15–17), so the median is the fairer "typical patient" figure.

**What this says.** The tiny EEGNet does about as well as a CNN 190 times its size: three fewer seizures in total, but more of each typical patient's seizure windows and fewer false alarms for the typical patient. That is the project's main claim, and it holds on our data. The typical patient gets a little more than 1 false alarm per hour, slightly above our target. CPU time per prediction is *pending* (measured with compression).

### 13.2 How much the validation rule matters

The same models trained the same way, scored with the first rule (2 validation patients, pooled false alarms; decision 3.24) and with the final rule (3 patients, median; decision 3.25):

| | EEGNet, first rule | EEGNet, final | CNN, first rule | CNN, final |
|---|---|---|---|---|
| Seizures caught (of 128) | 67 | **80** | 71 | **83** |
| Window sensitivity, median per patient | 0.51 | **0.58** | 0.32 | **0.54** |
| False alarms / h, median per patient | **0.62** | 1.19 | **0.83** | 1.54 |
| False alarms / h, mean | 2.51 | 3.80 | 1.91 | 3.80 |

Under the first rule, chb16 (a noisy patient) was a validation patient for the chb14 and chb15 folds and pushed their thresholds to 0.99–1.00, so chb15's 20 seizures were almost all missed. The final rule fixes that (the CNN now catches 19 of 20). The price is the opposite case: when one validation patient is very quiet, the median rule lets the threshold drop. In the chb08 fold, chb14 (no false alarms at all) was a validation patient, and chb08 went to about 20 false alarms per hour. **The threshold rule moves the balance between catching seizures and false alarms**; neither rule is free. We fixed the final rule before seeing its results, and report it as the headline.

### 13.3 Per patient (final rule)

Seizures caught (window sensitivity, false alarms per hour):

| Test patient | Seizures | EEGNet | CNN (`cnn-reg`) |
|---|---|---|---|
| chb01 | 7 | 7 (0.90, 0.52) | 7 (0.77, 0.27) |
| chb03 | 7 | 7 (0.83, 0.45) | 7 (0.76, 0.13) |
| chb05 | 5 | 5 (0.77, 0.51) | 5 (0.89, 4.79) |
| chb06 | 10 | 0 (0.00, 2.43) | 0 (0.00, 0.40) |
| chb08 | 5 | 5 (0.75, 20.39) | 5 (0.70, 20.59) |
| chb10 | 7 | 7 (0.75, 3.32) | 7 (0.61, 1.24) |
| chb13 | 12 | 2 (0.17, 0.76) | 4 (0.20, 2.09) |
| chb14 | 8 | 0 (0.00, 0.00) | 0 (0.00, 0.46) |
| chb15 | 20 | 12 (0.21, 2.30) | 19 (0.47, 2.20) |
| chb16 | 10 | 0 (0.00, 16.79) | 0 (0.00, 14.63) |
| chb18 | 6 | 6 (0.78, 1.63) | 5 (0.74, 0.56) |
| chb20 | 8 | 7 (0.38, 0.40) | 3 (0.08, 0.04) |
| chb23 | 7 | 6 (0.47, 0.68) | 5 (0.31, 3.99) |
| chb24 | 16 | 16 (0.69, 2.96) | 16 (0.63, 1.83) |
| **Total** | **128** | **80** | **83** |

**The honest reading.** Results depend heavily on the patient. For chb01, 03, 05, 10, 18, 24 both models caught every seizure, or all but one. For chb06, 14 and 16 both missed every seizure under either rule: chb16's seizures last a median of 8 seconds (9 of 10 at most 12 s), too short for 4-second windows with a 2-of-3 alarm rule, and chb06's seizures (median 15 s) score no higher than normal EEG (window AUROC 0.46 for EEGNet, 0.64 for the CNN). chb08 and chb16 also give far too many false alarms. This spread is normal for models tested on people they have never seen, and it is why we show every patient instead of only an average.

### 13.4 Still pending

| Model | Split | Result |
|---|---|---|
| EEGNet int8 | LOPO | *pending* |
| EEGNet pruned | LOPO | *pending* |
| CPU time per window, both models | | *pending* |

### 13.5 Leakage demo: shuffled chunks against whole recordings

The same small 1D CNN (about 15k weights), trained and tested on the same 11,500 one-second chunks with 5-fold cross-validation; only the split changes (decision 3.26). The chunks are the Bonn recordings [Andrzejak et al. 2001] cut exactly as in the UCI/Kaggle table. **Shuffled** mixes chunks into folds, as most tutorials do, so every test chunk has chunks from its own recording in training (100% in every fold). **Grouped** puts whole recordings into folds (0% shared). Run on 27 November 2026 on a MacBook Air, one seed, 30 epochs. Numbers from `results/leakage/summary.csv`, mean (std) over the 5 folds.

| | Shuffled | Grouped | Gap |
|---|---|---|---|
| Seizure vs rest: accuracy | 0.988 (0.001) | 0.974 (0.009) | +0.014 |
| Seizure vs rest: sensitivity | 0.958 (0.019) | 0.887 (0.057) | **+0.071** |
| Seizure vs rest: specificity | 0.995 (0.004) | 0.995 (0.005) | 0.000 |
| Seizure vs rest: F1 | 0.970 (0.002) | 0.930 (0.025) | +0.039 |
| Seizure vs rest: AUROC | 0.999 (0.000) | 0.987 (0.021) | +0.012 |
| Five classes: accuracy | 0.773 (0.009) | 0.754 (0.022) | +0.019 |
| Five classes: macro-F1 | 0.766 (0.009) | 0.745 (0.027) | +0.021 |

**What this says.**
- **The shuffled split inflates every score except specificity.** The model has seen other seconds of the same recording, so part of what it "detects" is the recording itself.
- **Accuracy hides most of the leak.** 80% of the chunks are non-seizure and specificity does not change, so accuracy moves only 1.4 points. Sensitivity, the share of seizure chunks caught, moves 7 points. This is one more reason we never report accuracy alone (decision 3.4).
- **Shuffling also hides how unstable a result is.** The spread across folds is three times larger with recordings kept apart (sensitivity std 0.057 against 0.019; the worst grouped fold catches 0.81). A shuffled split makes a model look both better and steadier than it is.
- **The split is not the whole story.** Even the grouped split scores 97% accuracy, far above anything on CHB-MIT. Bonn clips are short, clean and hand-picked, and people are still shared across our grouped folds (limitation 12). So the near-perfect numbers in UCI tutorials come mostly from the data, with the shuffled split adding on top. Comparing them with a patient-wise CHB-MIT result is not a fair comparison either way.
- **The gap is likely a lower bound.** We could only keep recordings apart. Keeping people apart, as we do on CHB-MIT, would remove more of the shared signal.

This is one seed, so treat the exact gaps as rough; the direction is the same for every metric except specificity.

### 13.6 Channel reduction: 18, 4 and 2 channels

A wearable cannot carry an 18-channel cap, so we retrained both models on fewer channels picked in advance (decision 3.27): **4 temporal** channels (F7-T7, T7-P7, F8-T8, T8-P8) and **2**, one on each side nearest the ears (T7-P7, T8-P8). Everything else is the same as the final runs above (leave-one-patient-out, 3 validation patients, median rule), so only the channels change. The 18-channel column is the result from section 13.1. Run on 27–28 November 2026 on a MacBook Air, one seed. Numbers from `results/chbmit/summary.csv` (runs named `...-temporal4-val3-median` and `...-temporal2-val3-median`).

| | EEGNet 18 | EEGNet 4 | EEGNet 2 | CNN 18 | CNN 4 | CNN 2 |
|---|---|---|---|---|---|---|
| Weights | 1,729 | 1,505 | 1,473 | 328,385 | 326,145 | 325,825 |
| Saved model size | 92 KB | 89 KB | 89 KB | 3,983 KB | 3,957 KB | 3,953 KB |
| Seizures caught (of 128) | 80 | 83 | 80 | 83 | **88** | 85 |
| Event sensitivity, mean / median | 0.68 / 0.94 | 0.67 / 0.90 | 0.66 / 0.97 | 0.66 / 0.89 | 0.71 / 1.00 | 0.69 / 1.00 |
| Window sensitivity, mean / median | 0.48 / 0.58 | 0.45 / 0.50 | 0.46 / 0.52 | 0.44 / 0.54 | 0.50 / 0.62 | 0.52 / 0.67 |
| Specificity, mean / median | 0.979 / 0.995 | 0.983 / 0.995 | 0.987 / 0.993 | 0.982 / 0.996 | 0.990 / 0.995 | 0.988 / 0.993 |
| F1, mean / median | 0.30 / 0.26 | 0.28 / 0.25 | 0.26 / 0.19 | 0.30 / 0.24 | 0.39 / 0.42 | 0.34 / 0.32 |
| False alarms / h, mean / median | 3.80 / 1.19 | 3.39 / 1.42 | 2.69 / 1.43 | 3.80 / 1.54 | 2.29 / 1.29 | 2.37 / 1.54 |

Per patient, seizures caught (window sensitivity, false alarms per hour):

| Test patient | Seizures | EEGNet 18 | EEGNet 4 | EEGNet 2 | CNN 18 | CNN 4 | CNN 2 |
|---|---|---|---|---|---|---|---|
| chb01 | 7 | 7 (0.90, 0.52) | 7 (0.88, 2.49) | 7 (0.73, 1.95) | 7 (0.77, 0.27) | 7 (0.87, 0.20) | 7 (0.80, 0.42) |
| chb03 | 7 | 7 (0.83, 0.45) | 7 (0.93, 8.84) | 7 (0.93, 6.53) | 7 (0.76, 0.13) | 7 (0.85, 1.26) | 7 (0.84, 1.58) |
| chb05 | 5 | 5 (0.77, 0.51) | 5 (0.82, 1.49) | 5 (0.76, 0.77) | 5 (0.89, 4.79) | 5 (0.75, 1.31) | 5 (0.84, 2.38) |
| chb06 | 10 | 0 (0.00, 2.43) | 0 (0.00, 0.84) | 0 (0.00, 1.29) | 0 (0.00, 0.40) | 0 (0.00, 1.20) | 0 (0.00, 1.50) |
| chb08 | 5 | 5 (0.75, 20.39) | 5 (0.58, 11.25) | 5 (0.46, 9.50) | 5 (0.70, 20.59) | 5 (0.59, 6.05) | 5 (0.48, 7.70) |
| chb10 | 7 | 7 (0.75, 3.32) | 6 (0.43, 0.02) | 7 (0.89, 0.68) | 7 (0.61, 1.24) | 7 (0.66, 0.06) | 7 (0.92, 0.46) |
| chb13 | 12 | 2 (0.17, 0.76) | 2 (0.12, 1.58) | 0 (0.00, 0.00) | 4 (0.20, 2.09) | 2 (0.14, 1.88) | 1 (0.12, 0.55) |
| chb14 | 8 | 0 (0.00, 0.00) | 0 (0.00, 0.62) | 0 (0.00, 0.08) | 0 (0.00, 0.46) | 0 (0.00, 1.23) | 0 (0.00, 0.77) |
| chb15 | 20 | 12 (0.21, 2.30) | 19 (0.37, 1.35) | 17 (0.24, 1.50) | 19 (0.47, 2.20) | 20 (0.75, 1.62) | 20 (0.85, 1.97) |
| chb16 | 10 | 0 (0.00, 16.79) | 0 (0.00, 13.89) | 0 (0.00, 7.21) | 0 (0.00, 14.63) | 0 (0.00, 12.79) | 0 (0.00, 9.53) |
| chb18 | 6 | 6 (0.78, 1.63) | 6 (0.84, 3.34) | 6 (0.81, 4.57) | 5 (0.74, 0.56) | 6 (0.79, 1.57) | 6 (0.78, 0.98) |
| chb20 | 8 | 7 (0.38, 0.40) | 4 (0.18, 0.36) | 4 (0.16, 0.25) | 3 (0.08, 0.04) | 7 (0.39, 0.62) | 5 (0.29, 0.62) |
| chb23 | 7 | 6 (0.47, 0.68) | 6 (0.56, 0.79) | 7 (0.81, 1.96) | 5 (0.31, 3.99) | 6 (0.53, 0.83) | 7 (0.75, 3.13) |
| chb24 | 16 | 16 (0.69, 2.96) | 16 (0.62, 0.66) | 15 (0.58, 1.36) | 16 (0.63, 1.83) | 16 (0.64, 1.41) | 15 (0.58, 1.60) |
| **Total** | **128** | **80** | **83** | **80** | **83** | **88** | **85** |

**What this says.**
- **Two channels keep almost all of the detection.** Seizures caught stay at 80–88 of 128 for every setting. With one seed and 14 patients, differences of a few seizures are within run-to-run noise, so we read this as "about the same", not "better".
- **The CNN does not get worse with fewer channels, and may get a little better.** Its median window sensitivity goes from 0.54 to 0.67 and its mean false alarms fall from 3.8 to 2.4 per hour. One possible reason is that with fewer inputs it has less to memorise about the training patients; we have not tested this.
- **EEGNet loses a little window sensitivity** (median 0.58 → 0.52) and its typical patient gets slightly more false alarms (median 1.19 → 1.43 per hour), but it catches the same 80 seizures.
- **The mean false-alarm rate falls for both models**, mostly because chb08 drops from about 20 to 6–11 false alarms per hour and chb16 from about 15 to 7–10.
- **Some patients do worse.** chb13 falls to 0 (EEGNet) and 1 (CNN) of 12 seizures with 2 channels, chb20 falls from 7 to 4 for EEGNet, and EEGNet's false alarms on chb03 rise from 0.45 to 6.5–8.8 per hour. Their seizures may be clearer away from the temporal channels; we have not checked where each patient's seizures start.
- **Some patients do better.** chb15 goes from 12 to 17–19 seizures for EEGNet and 19 to 20 for the CNN.
- **chb06, chb14 and chb16 are still missed at every channel count**, so their problem is not the number of channels (section 13.3).
- **The model barely shrinks.** Only the first spatial layer depends on the channel count, so EEGNet goes from 1,729 to 1,473 weights. The saving is in electrodes, data and power, not in model size; compression (section 14) is what makes the model itself smaller.

---

## 14. What comes next

In order (decision 3.10):

1. ~~**Finish CHB-MIT training.**~~ Done on 26 November 2026 (section 13). Open question: overlapping windows for short seizures (chb16, chb06).
2. ~~**Leakage demo on UCI/Kaggle.**~~ Done on 27 November 2026 with the Bonn data cut as in the UCI table (section 13.5).
3. ~~**Channel reduction.**~~ Done on 28 November 2026 (section 13.6). Two temporal channels keep almost all of the detection.
4. **Siena outside test.** Run the CHB-MIT model, without retraining, on adult recordings (needs 512 → 128 Hz resampling and matching Siena's electrodes to CHB-MIT's bipolar pairs).
5. **Compression.** int8 post-training quantization, then 50–80% pruning with fine-tuning.
6. **Research page** on the website, describing the scripts, splits and scoring for others.

---

## 15. Glossary

- **Artifact.** Signal not produced by the brain (blinks, movement, electrode noise).
- **Bipolar montage.** Each channel is the voltage difference between two neighbouring electrodes.
- **Data leakage.** When information from the test set reaches the model during training, making results look better than they are.
- **Epoch.** One pass over the (balanced) training data.
- **Event.** A continuous run of alarm windows (split at 5 minutes).
- **False alarm.** An alarm event that touches no annotated seizure.
- **Ictal / pre-ictal / post-ictal.** During / just before / just after a seizure.
- **LOPO.** Leave-one-patient-out: test on each patient in turn, train on the others.
- **Memory-map.** Reading parts of a file from disk on demand, as if it were in memory.
- **Quantization (int8).** Storing weights as 8-bit integers instead of 32-bit floats; about 4× smaller.
- **Pruning.** Setting the least important weights to zero.
- **Sensitivity.** True positives ÷ (true positives + false negatives).
- **Specificity.** True negatives ÷ (true negatives + false positives).
- **Window.** A 4-second slice of EEG; the unit the model classifies.
- **Zero-phase filter.** A filter applied forwards and backwards so it does not shift the signal in time.

---

## 16. References

Every reference below was checked online on 25 November 2026. Where we could only verify a preprint, we cite the preprint.

**Datasets**

1. Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R
2. Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*. PhD thesis, Massachusetts Institute of Technology. (Citation requested by the CHB-MIT PhysioNet page.)
3. Shoeb, A., & Guttag, J. (2010). Application of machine learning to epileptic seizure detection. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, Haifa, Israel. https://archive.physionet.org/pn6/chbmit/shoeb-icml-2010.pdf
4. Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. Ch., Mark, R. G., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101, e215–e220. https://amarallab.github.io/media/publications/2000/Goldberger-2000-Circulation-101-E215.pdf
5. Detti, P. (2020). *Siena Scalp EEG Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/5d4a-j060
6. Detti, P., Vatti, G., & Zabalo Manrique de Lara, G. (2020). EEG synchronization analysis for seizure prediction: A study on data of noninvasive recordings. *Processes*, 8(7), 846. https://doi.org/10.3390/pr8070846
7. Shah, V., von Weltin, E., Lopez, S., McHugh, J. R., Veloso, L., Golmohammadi, M., Obeid, I., & Picone, J. (2018). The Temple University Hospital Seizure Detection Corpus. *Frontiers in Neuroinformatics*, 12. https://doi.org/10.3389/fninf.2018.00083
8. Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C. E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E*, 64(6), 061907. https://doi.org/10.1103/PhysRevE.64.061907

**Models and deep learning on EEG**

9. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15, 056013. https://doi.org/10.1088/1741-2552/aace8c
10. Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*. https://doi.org/10.1002/hbm.23730
11. Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: A systematic review. arXiv:1901.05498. https://arxiv.org/abs/1901.05498
12. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37, 448–456. https://proceedings.mlr.press/v37/ioffe15.html
13. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

**Evaluation, leakage and clinical standards**

14. Dan, J., Pale, U., Amirshahi, A., Cappelletti, W., Ingolfsson, T. M., Wang, X., et al. (2024). SzCORE: A Seizure Community Open-source Research Evaluation framework for the validation of EEG-based automated seizure detection algorithms. arXiv:2402.13005. https://arxiv.org/abs/2402.13005 (framework page: https://epilepsybenchmarks.com/framework/)
15. Beniczky, S., & Ryvlin, P. (2018). Standards for testing and clinical validation of seizure detection devices. *Epilepsia*, 59(S1), 9–13. https://doi.org/10.1111/epi.14049
16. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. Preprint: arXiv:2207.07048, https://arxiv.org/abs/2207.07048; project page: https://reproducible.cs.princeton.edu/

**Model compression**

17. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2704–2713. https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html
18. Han, S., Mao, H., & Dally, W. J. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1510.00149
19. Google AI Edge. *Post-training quantization*. https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization

**Background**

20. World Health Organization (7 February 2024). *Epilepsy* (fact sheet). https://www.who.int/news-room/fact-sheets/detail/epilepsy
21. Fisher, R. S., Acevedo, C., Arzimanoglou, A., Bogacz, A., Cross, J. H., Elger, C. E., et al. (2014). ILAE official report: A practical clinical definition of epilepsy. *Epilepsia*, 55(4), 475–482. https://doi.org/10.1111/epi.12550

**Comparison papers (read in part)**

22. Ayala-Britez, E. H., Frutos, L., Pinto-Roa, D. P., Abente, S., Molinas, A., Kim, N.-Y., et al. (2026). Real-time epileptic seizure detection from raw EEG using classical machine learning and time-domain feature. *Machine Learning and Knowledge Extraction*, 8(8), 231. https://doi.org/10.3390/make8080231
23. Al-Marzouki, S. (2025). Advancing epileptic seizure recognition through bidirectional LSTM networks. *Frontiers in Computational Neuroscience*, 19. https://doi.org/10.3389/fncom.2025.1668358
