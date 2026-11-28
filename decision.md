# Decision log: Listening for Seizures

This file records the project's logic and the decisions behind it, each with its reason. It covers the detection work (data, models, evaluation). Website design rules are in `CONTRIBUTING.md`.

Each decision is marked **Decided** or **Open**. When an open question is settled, move it to Decided, write the reason, and add a line to the change log at the bottom.

Last updated: 27 November 2026.

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

### 3.21 Training and scoring details
**Decision.**
- *Cache.* `code/04_cache_chbmit.py` unpacks the cleaned files into uncompressed `.npy` files in `data/cache/` (about 8 GB, ignored by git). Training memory-maps them and reads only the windows each batch needs.
- *Normalisation inside the model.* Each model's first layer is a fixed per-channel z-score holding its training patients' mean and std. Models take raw microvolts.
- *Validation in leave-one-patient-out.* For the fold that tests patient *k*, the next two patients in the patient list (wrapping round) are validation and the other 11 train. Normalisation uses those 11 only.
- *Batches.* Every seizure window (with shifted copies) plus 4 non-seizure windows per seizure window, redrawn each epoch. Flagged windows stay in. Seizure windows weigh 2 in the loss.
- *Alarm and false alarms.* The 2-of-3 rule looks back only (windows *i*-2 to *i*) and never joins windows across files or gaps. Window metrics use the alarm output. An alarm event is a run of alarm windows, split every 5 minutes. A false alarm is an event that touches no seizure. A seizure is caught if any of its windows alarms.
- *Threshold and early stopping.* After each epoch, the threshold on validation patients is the one with the best window sensitivity at no more than 1 false alarm per hour. The epoch with the best such sensitivity is kept. Epochs that cannot meet the limit at any threshold rank lower. Training stops after 10 epochs without improvement.
- *Sizes.* The baseline CNN (two convolutions per block, 32/64/128/128 filters, kernel 5) has 328k weights. EEGNet-8,2 has 1.7k, a little under the 2–5k estimated in 3.16.
- *Outputs.* Each run saves its model, history, metrics, settings and test-window probabilities in `results/chbmit/<model>/<split>/`, so test scores can be recomputed without retraining. `code/06_summarise_chbmit.py` collects them into `results/chbmit/summary.csv`.

**Why.** The cache keeps memory use small on a laptop (a full training set is 5–7.5 GB). Normalisation inside the model means a saved model, and later the int8 version, cannot be fed wrongly scaled data. Rotating validation spreads it across patients and is fixed by the fold number, so it is reproducible. Keeping flagged windows teaches the model that artifacts are not seizures. A looking-back alarm rule is what a wearable could run. Splitting long alarms (as SzCORE does) stops a model that alarms all the time from scoring zero false alarms; this showed up in testing. 1 false alarm per hour is a common target for seizure detectors.

### 3.22 Experiment: normalise each recording on its own
**Decision.** Add `--norm recording` to `code/05_train_chbmit.py`. Each window is scaled by its own recording's per-channel median and robust spread (1.4826 × median absolute deviation, at least 1 µV). The model's own normalisation layer is then left as identity. Statistics are computed once per patient without using labels, and cached in `data/cache/chbmit/chbXX_recstats.npz`. Runs are saved as `<model>-recnorm`. The default stays `--norm train` (3.21), so earlier results can still be reproduced. This is tested on its own before any other change (more validation patients, regularisation, thresholds on raw outputs).
**Why.** On the fixed split (25 Nov 2026), test seizures caught were 5/22 (EEGNet) and 4/22 (CNN) at the validation threshold. Yet the window AUROC was 0.76 and 0.90, and a threshold suited to the test patients would have caught 12–14/22 at 1 false alarm per hour (a diagnostic only, not a result). So the models rank windows reasonably well, but their scores shift from patient to patient. Patients differ a lot in amplitude, and scaling each recording by itself removes most of that. The median and MAD hardly move during a short seizure. Using the whole recording looks ahead in time; on a wearable it would be a running estimate from the minutes so far. This limitation is noted for the report.

### 3.23 Experiment: a steadier CNN and training-window augmentation
**Decision.** Add a model `cnn-reg`: the same CNN as 3.15 and 3.21, plus weight decay (L2 1e-4 on every convolution and the output layer) and spatial dropout (0.2, dropping whole feature maps after each block). Add `--augment` to `code/05_train_chbmit.py`, which changes training windows only: each window is scaled by a random factor between 0.8 and 1.25, Gaussian noise is added at 5% of each channel's spread in that window, and in 30% of windows one random channel is set to zero. Validation and test windows are never changed. Both are off by default, so earlier runs can still be reproduced. Runs are saved as `cnn-reg-recnorm-aug` and `eegnet-recnorm-aug`. EEGNet gets the same augmentation, so both models are trained the same way. If this helps, leave-one-patient-out uses `cnn-reg` and `eegnet` with `--norm recording --augment`.
**Why.** With per-recording normalisation (3.22), the CNN's training loss fell to 0.03 while its validation sensitivity swung between 0.19 and 0.63 from epoch to epoch. It was memorising the 9 training patients. Weight decay and spatial dropout limit how much it can memorise. Augmentation shows it the kinds of variation new patients bring: different amplitudes, noise and a lost electrode. A steadier baseline makes the comparison with EEGNet fair.

### 3.24 Settings for the leave-one-patient-out runs
**Decision.** Final runs use per-recording normalisation for both models. The CNN is `cnn-reg` with `--augment`. EEGNet is `eegnet` without augmentation. Everything else follows 3.18 and 3.21. All 14 folds are run for each model.
**Why.** Each model's settings were chosen by its validation score (chb05, chb23), not by test results. CNN: validation sensitivity at 1 false alarm per hour rose from 0.63 (`cnn-recnorm`) to 0.66 with weight decay, spatial dropout and augmentation, and training was steadier. On test it caught the same 13/22 seizures with false alarms down from 0.28 to 0.03 per hour. EEGNet: validation fell from 0.69 to 0.63 with augmentation, and its best epoch moved to epoch 1. With only 1.7k weights it has little room to memorise, so the extra noise mostly gets in its way. The test results agreed (17/22 → 13/22 seizures, 0.49 → 2.3 false alarms per hour). These come from one seed and three test patients, so small differences may be chance. Leave-one-patient-out gives the numbers that are reported.

### 3.25 Sturdier validation in leave-one-patient-out
**Decision.** Each LOPO fold uses the next **three** patients as validation (10 train), and the threshold rule uses the **median** of the validation patients' own false-alarm rates instead of pooling them: the best window sensitivity whose median false alarms per hour is at most 1. Sensitivity is still pooled over all validation seizure windows. New options in `code/05_train_chbmit.py`: `--val-patients 3 --fa-rule median`; runs are saved as `<run>-val3-median`. Defaults stay 2 and pooled, so the runs under 3.24 can still be reproduced. The fixed split is unchanged. All 14 folds are rerun for both models with the 3.24 settings plus this rule, and these become the reported results. `--skip-existing` lets an interrupted loop restart where it stopped.
**Why.** In the 3.24 runs, chb16 was a validation patient in the chb14 and chb15 folds. Its normal EEG gives about 12 false alarms per hour, so keeping the pooled rate under 1 pushed the threshold to 0.99–1.00. Both models then caught 0–6 of chb15's 20 seizures, although they ranked its windows well (AUROC 0.94 and 0.91). In the chb14 fold, EEGNet found no usable threshold and kept its epoch-1 weights (validation sensitivity 0). The problem shows on the validation patients themselves, so the fix does not use test results. With three patients and the median, one unusual patient cannot set the threshold for a whole fold. The cost is one fewer training patient per fold. Two other findings are recorded, not fixed: chb16's seizures are very short (median 8 s, 9 of 10 at most 12 s), so 4-second windows with the 2-of-3 rule can barely catch them; and chb06's seizures (median 15 s) score no higher than normal EEG (AUROC 0.46 and 0.64).

### 3.26 Leakage demo: shuffled chunks against whole recordings, same model
**Decision.** `code/07_get_bonn.sh` downloads the five Bonn EEG sets (Andrzejak et al. 2001; 500 single-channel recordings of 23.6 s at 173.61 Hz) into `data/raw/bonn/`, checking each zip against the SHA-256 we recorded on 27 Nov 2026 (not in git, like all raw data in 3.5). `code/08_leakage_demo.py` cuts every recording into 23 one-second chunks of 178 samples, which rebuilds the UCI/Kaggle "Epileptic Seizure Recognition" table (11,500 chunks, same five classes). It trains one small 1D CNN (three convolution blocks, about 15k weights) twice on these chunks with 5-fold cross-validation: once with the chunks shuffled into folds ("shuffled", the common tutorial way), once with whole recordings assigned to folds ("grouped"). Both splits keep the class mix equal in every fold and use the same seeds, 30 epochs with no early stopping, Adam at 0.001 and batch 64, and z-scoring from training chunks only. Two tasks are run: seizure (set S) against the rest, and all five sets. Reported: accuracy, sensitivity, specificity, F1 and AUROC for the seizure task, and accuracy and macro-F1 for five classes, each as mean and std over folds, plus the gap. The script also reports the share of test chunks whose recording has chunks in training. Results go to `results/leakage/`.
**Why.** This is the third argument in 3.10: the same model on the same data, with only the split changed, so any gap comes from the split alone (the paper comparison in the research notes mixes data and model too). UCI no longer hosts the table (its download link returns 404 and its data API reports the dataset as not found, 27 Nov 2026), so we rebuild it from the Bonn files it was made from. That also tells us for certain which recording each chunk came from. The Bonn files carry no person ids, and each set comes from five people, so the grouped split keeps recordings apart, not people. It is still optimistic, which is why the main study uses CHB-MIT with leave-one-patient-out. Fixed epochs avoid a validation choice that could differ between the two splits. The five-class task is included because sets N and F, and Z and O, come from the same people, so recognising a recording helps most there. The seizure task alone may show only a small gap, and we report whatever comes out.

### 3.27 Channel reduction: 18, 4 and 2 channels
**Decision.** Add `--channels all|temporal4|temporal2` to `code/05_train_chbmit.py`. The sets are fixed in advance in `code/seizure/data.py`: `temporal4` is F7-T7, T7-P7, F8-T8 and T8-P8 (two pairs over each temporal lobe), and `temporal2` is T7-P7 and T8-P8 (one pair on each side, nearest the ears). The model is built for that many channels and never sees the others. Everything else is exactly as in the reported runs (3.24 and 3.25): leave-one-patient-out over all 14 patients, three validation patients and the median false-alarm rule, per-recording normalisation, `cnn-reg` with augmentation and `eegnet` without. The 18-channel numbers are the existing `-val3-median` runs, so they are not rerun. Runs are saved as `<model>-recnorm[-aug]-<set>-val3-median`, each run records its channel names, and `code/06_summarise_chbmit.py` adds a channel-count column. The default stays `all`, so earlier runs can still be reproduced.
**Why.** A wearable cannot carry a full 18-channel cap; a few electrodes around the ears is realistic. Choosing the channels before training, rather than picking the best channels from the results, means no test patient helps choose them. Temporal channels are where many seizures are seen and where an ear-worn device would sit. They are a reasonable guess for a wearable, not the best possible set, and this study does not test other sets. Keeping every other setting fixed means any change in detection comes from the channels alone. The model barely shrinks (EEGNet 1,729 → 1,505 → 1,473 weights; the CNN about 328k → 326k), because only its first spatial layer depends on the channel count; the saving is in electrodes, data and power, not in model size. With two channels the CNN's augmentation zeroes one of the two channels in 30% of training windows, which is a much larger loss than with 18; we keep it so that only the channels change, and note it when reading the CNN results. Patients whose seizures start away from the temporal lobes are expected to lose the most, and the per-patient table will show this.

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
| 25 Nov 2026 | Added 3.21 (training and scoring details). |
| 25 Nov 2026 | Added 3.22 (per-recording normalisation experiment) after the first fixed-split results. |
| 25 Nov 2026 | Added 3.23 (regularised CNN and augmentation experiment) after per-recording normalisation raised test seizures caught to 17/22 (EEGNet) and 13/22 (CNN). |
| 25 Nov 2026 | Added 3.24 (settings for leave-one-patient-out, chosen on validation patients). |
| 26 Nov 2026 | Added 3.25 (three validation patients and a median false-alarm rule for leave-one-patient-out), after chb16 was found to set the threshold in the chb14 and chb15 folds. |
| 27 Nov 2026 | Added 3.26 (leakage demo on the Bonn data cut as in the UCI table: shuffled chunks against whole recordings, same model). |
| 27 Nov 2026 | Added 3.27 (channel reduction: 18, 4 and 2 temporal channels, same LOPO settings as 3.25). |
