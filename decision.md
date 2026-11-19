# Decision log: Listening for Seizures

This file records the project's logic and the decisions behind it, each with its reason. It covers the detection work (data, models, evaluation). Website design rules are in `CONTRIBUTING.md`.

Each decision is marked **Decided** or **Open**. When an open question is settled, move it to Decided, write the reason, and add a line to the change log at the bottom.

Last updated: 19 November 2026.

---

## 1. What the project is

| Item | Value |
|---|---|
| Course | Neural Networks and Deep Learning (CSCI 5122), University of Colorado Boulder, Fall 2026 |
| Team | Sejal Hukare, Ishita Pawar, Swejal Gabhane |
| Goal | Classify short windows of EEG as seizure or non-seizure, with a model small enough to run on a wearable |
| Priorities | High accuracy and reliability, reported alongside model size and speed |
| Deliverable | A website (plain HTML on GitHub Pages) that links to all raw data, cleaned data and code |

## 2. Current state (18 November 2026)

- The website is built: six pages, with Data & prep, Models and Results laid out as empty sections that match the rubric.
- **No modelling code exists yet.** `code/`, `data/clean/` and `results/` hold only README placeholders.
- No dataset has been downloaded or confirmed.

## 3. Decided

### 3.1 The task is window-level binary classification
**Decision.** Input is a short window of multi-channel EEG. Output is seizure or non-seizure.
**Why.** It is the simplest framing that a small on-device model can run continuously, and it maps directly to the rubric's confusion matrix requirement.

### 3.2 Two models: a baseline and a lightweight network
**Decision.** Compare a larger baseline network against a lightweight network, then compress the lightweight one after training.
**Why.** The project's claim is about how much detection is kept as the model shrinks. A baseline gives a fair reference point for that trade-off. (Models page sections: baseline, lightweight, making it smaller.)

### 3.3 Patient-wise split
**Decision.** No patient appears in both training and testing.
**Why.** EEG from the same person is highly correlated. Mixing a patient across splits inflates accuracy and does not reflect how the model would work on a new person. This is the single biggest reliability safeguard.

### 3.4 Metrics beyond accuracy
**Decision.** Report sensitivity, specificity, F1 and false alarms per hour for each model, at least one confusion matrix, plus parameters, size in KB and CPU time per prediction.
**Why.** Seizure windows are rare, so a model that always says "no seizure" can score high accuracy while being useless. Sensitivity says how many seizures are caught; false alarms per hour says whether a wearer could live with it.

### 3.5 Raw data stays out of the repository
**Decision.** Raw `.edf` recordings are ignored by git. `data/raw/README.md` explains where to download them.
**Why.** GitHub rejects files over 100 MB, and the full candidate dataset is about 40 GB.

### 3.6 Code is numbered in run order
**Decision.** Notebooks and scripts in `code/` are numbered in the order they run.
**Why.** Anyone, including the grader, can reproduce the pipeline from raw data to results without guessing.

### 3.7 About us page: click to open each person
**Decision.** Three equal 800 × 800 square portraits. Clicking one enlarges it, slides out two square personal photos, and opens a short panel. The other two shrink into a row above and stay clickable.
**Why.** The page stays tidy by default but still has room for fuller, more personal bios. Equal crops keep the grid even, and the motion is turned off for visitors who choose reduced motion.

### 3.8 The project is a reproducible study and toolkit
**Decision.** Ship numbered scripts from raw download to results, the saved patient splits, and the scoring code, so others can rerun the work and add their own models. Cleaned data is not redistributed.
**Why.** Others can check and extend the results, which is the reliability claim made concrete. Shipping code instead of data avoids licence and file-size problems.

### 3.9 A Research page follows Conclusion on the website
**Decision.** Add a Research page after Conclusion. It covers dataset links and licence notes, the scripts in run order, the splits, how results are scored, how to add your own model, and open questions. It is built near the end, once the scripts exist.
**Why.** It gives other researchers one place to start, and it describes files that already exist rather than plans.

### 3.10 Three datasets, each for one argument
**Decision.** CHB-MIT is the main dataset (patient-wise train and test). Siena is an outside test on adults only, with no training on it. UCI/Kaggle is used only to show how a shuffled split inflates accuracy. A channel-reduction study on CHB-MIT covers the wearable case. TUH is left out. The work order is CHB-MIT core, then the leakage demo, then channel reduction, then Siena.
**Why.** Each dataset supports a separate claim (performance on a new patient, generalisation across age and hospital, honest evaluation, wearable fit). The order keeps the core result safe if time runs short. TUH needs registration and has mixed recordings.

### 3.11 CHB-MIT patient set
**Decision.** 14 patients: chb01, 03, 05, 06, 08, 10, 13, 14, 15, 16, 18, 20, 23, 24 (about 480 hours, 128 seizures). chb12 is excluded. chb21 is not used, and if it is ever added, it counts as the same subject as chb01.
**Why.** These are all the patients with 5 or more seizures and the 18 standard channels in every file. chb12 has files with a different electrode layout. chb01 and chb21 are the same child, so splitting them across training and testing would leak. 14 patients fit on the laptop (about 20 GB raw, 8 GB cleaned).

### 3.12 Cleaning and windowing
**Decision.** 18 bipolar channels; 0.5–40 Hz zero-phase band-pass; downsample to 128 Hz; 4-second windows with no overlap that never cross files. A window is seizure if at least 50% of it is inside an annotation (the exact fraction is stored). Flat or extreme windows are flagged, not dropped. Extreme = filtered peak above 1000 µV. Seizure windows are never dropped for artifacts, even when flagged. Extra seizure windows shifted by 2 s are stored for training only. Normalisation is fitted after the split, on training patients only.
**Why.** The filter removes drift and mains hum. 128 Hz loses nothing after a 40 Hz filter and halves storage. Storing the fraction and flags lets later choices change without cleaning again. Fitting normalisation on training patients only prevents leakage.

### 3.13 Patient splits
**Decision.** Fixed development split: test chb03, chb10, chb20; validation chb05, chb23; training the other 9. Final results use leave-one-patient-out over all 14 patients. Both are saved in `data/splits/chbmit_splits.json`. Normalisation statistics come from each split's training patients only.
**Why.** Test and validation patients have typical seizure counts, and the seizure-heavy patients stay in training. SzCORE notes that a single fixed split varies more on small datasets, so the final numbers come from leave-one-patient-out. Saving the splits lets anyone reproduce the same test sets.

### 3.14 Class imbalance
**Decision.** Balanced training batches: all seizure windows (including shifted copies) plus non-seizure windows sampled about 1:4, redrawn each epoch, with a mild class weight in the loss. The threshold is chosen on validation patients. An alarm needs 2 of 3 consecutive windows flagged. Validation and test data are never rebalanced. Focal loss is the fallback.
**Why.** Only about 1 window in 236 is seizure. Balanced batches let the model see enough seizures. The 2-of-3 rule and validation-set threshold control false alarms per hour.

### 3.15 Baseline model
**Decision.** A 1D CNN: 4 convolution blocks, global average pooling, about 200–400k parameters.
**Why.** Simple, strong on raw EEG and easy to explain. It gives a fair reference for the lightweight model.

### 3.16 Lightweight model
**Decision.** EEGNet (about 2–5k parameters).
**Why.** Designed for EEG, widely cited, and small enough for a wearable chip.

### 3.17 Compression
**Decision.** int8 post-training quantization first, then pruning (50–80% of weights, with fine-tuning) as a second experiment. Both are compared with the uncompressed model.
**Why.** Quantization is a single step that makes the model about 4× smaller. Together they show how much detection is kept as the model shrinks.

### 3.18 Training setup
**Decision.** Adam (learning rate 0.001), weighted binary cross-entropy, batch size 256, at most 50 epochs, early stopping on validation sensitivity at a fixed false-alarm rate, fixed random seeds. Final results are 14 leave-one-patient-out runs per model.
**Why.** These are standard, stable defaults. Stopping on sensitivity at a set false-alarm rate matches what a wearer cares about better than accuracy does.

### 3.19 Framework
**Decision.** TensorFlow/Keras.
**Why.** Converting to an int8 TensorFlow Lite model is built in and is the standard route onto microcontrollers, which suits the wearable claim.

### 3.20 Data & prep page: game plan and two reading levels
**Decision.** The Data & prep page opens with an eight-step route map of the whole project (four steps done, "we are here" on training), with brackets showing which site page covers each step. Every section has a plain-language story and a closed "Show the working" panel beside it with the exact settings, a numbered excerpt of our script and a GitHub link; one switch opens all panels. Figures use real CHB-MIT data, made by `code/site/make_data_page_figures.py`. The page links raw data to PhysioNet plus our scripts and splits; cleaned data is not hosted (3.8). The plan for the models is explained here; the Models page will hold the built architectures. Page styles live in `assets/css/data-page.css`, so the shared `site.css` is unchanged.
**Why.** The course asks for a page that works for both general readers and data scientists. Closed panels keep the page calm for the first group while giving the second a fixed place to find the code. The route map answers "where are we?" in a project with many parts. It goes here and not on the home page, because the Introduction must stay non-technical.

## 4. Open

None right now.

## 5. Working rules

- Every code edit and debugging step is shown to Sejal and approved before it is applied.
- Never debug a teammate's code without their permission.
- No AI tool is credited as a contributor or co-author anywhere, including commit messages.

## 6. Change log

| Date | Change |
|---|---|
| 18 Nov 2026 | File created from the website skeleton and README files. No modelling code existed yet. |
| 18 Nov 2026 | Added 3.7: interactive About us page. |
| 18 Nov 2026 | Added 3.8 (reproducible study and toolkit), 3.9 (Research page) and 3.10 (three datasets). Closed open question 4.1. |
| 18 Nov 2026 | Added 3.11 (patient set) and 3.12 (cleaning and windowing). Closed 4.2, 4.3 and 4.4. |
| 18 Nov 2026 | 3.12: extreme-window threshold set to a 1000 µV filtered peak (500 µV flagged 47% of seizure windows). |
| 18 Nov 2026 | Added 3.13 (patient splits: fixed development split and leave-one-patient-out). |
| 18 Nov 2026 | Added 3.14–3.19 (imbalance, baseline, lightweight model, compression, training setup, framework). Closed 4.5–4.10. |
| 19 Nov 2026 | Added 3.20 (Data & prep page layout: route map, two reading levels, real-data figures). |
