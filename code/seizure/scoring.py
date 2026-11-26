"""How a model's output is scored (decisions 3.4 and 3.14).

1. A window is flagged when its seizure probability is at or above the threshold.
2. The alarm rule: window i raises an alarm when at least 2 of windows i-2, i-1, i
   are flagged. It only looks backwards, as a wearable would. Windows from
   different recordings, or with a gap between them, never count together.
3. Window metrics (confusion matrix, sensitivity, specificity, F1) use the alarm
   output, since that is what the wearer gets.
4. An alarm event is a run of consecutive alarm windows. Runs longer than
   5 minutes are split into 5-minute events (as in SzCORE), so a model that
   alarms all the time cannot pass as having few false alarms. A false alarm is
   an event that touches no seizure. False alarms per hour = false alarms / hours
   recorded.
5. A seizure counts as detected if any of its windows raises an alarm.

Scoring uses ordinary windows only (no shifted copies).
"""
import numpy as np

WIN_S = 4.0
MAX_EVENT = int(300 / WIN_S)        # 5 minutes of windows


def _segments(edf_file, start_s):
    """Sort order, and a segment id that changes at a new file or a time gap."""
    order = np.lexsort((start_s, edf_file))
    f, s = edf_file[order], start_s[order]
    new = np.ones(len(order), bool)
    new[1:] = (f[1:] != f[:-1]) | (np.abs(s[1:] - s[:-1] - WIN_S) > 1e-3)
    return order, np.cumsum(new)


def _runs(mask, seg):
    """Start and end (exclusive) of runs of True in mask that stay within one segment."""
    m = mask.astype(np.int8)
    starts = np.flatnonzero(m & np.r_[1, (1 - m[:-1]) | (seg[1:] != seg[:-1])])
    ends = np.flatnonzero(m & np.r_[(1 - m[1:]) | (seg[1:] != seg[:-1]), 1]) + 1
    return starts, ends


def _prepare(p):
    """Put windows in time order once, so many thresholds can be scored quickly."""
    if "_seg" in p:
        return p
    order, seg = _segments(p["edf_file"], p["start_s"])
    return {"_seg": seg, "prob": p["prob"][order], "y": p["y"][order].astype(bool),
            "touches": p["seizure_frac"][order] > 0}


def score(p, threshold):
    """Score predictions p (dict from data.predict_patients) at one threshold."""
    q = _prepare(p)
    seg, y, touches = q["_seg"], q["y"], q["touches"]
    flag = (q["prob"] >= threshold).astype(np.int8)

    count = flag.copy()
    count[1:] += flag[:-1] * (seg[1:] == seg[:-1])
    count[2:] += flag[:-2] * (seg[2:] == seg[:-2])
    alarm = count >= 2

    tp = int((alarm & y).sum()); fn = int((~alarm & y).sum())
    fp = int((alarm & ~y).sum()); tn = int((~alarm & ~y).sum())

    a_start, a_end = _runs(alarm, seg)
    false_alarms = 0
    for a, b in zip(a_start, a_end):                      # split long alarms into 5-minute events
        for c in range(a, b, MAX_EVENT):
            false_alarms += not touches[c:min(b, c + MAX_EVENT)].any()
    s_start, s_end = _runs(y, seg)
    detected = sum(alarm[a:b].any() for a, b in zip(s_start, s_end))
    hours = len(y) * WIN_S / 3600

    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
        "accuracy": (tp + tn) / len(y),
        "seizures": len(s_start), "seizures_detected": int(detected),
        "event_sensitivity": detected / len(s_start) if len(s_start) else float("nan"),
        "false_alarms": int(false_alarms), "hours": hours,
        "fa_per_hour": false_alarms / hours,
    }


def pick_threshold(p, max_fa_per_hour=1.0, fa_rule="pooled"):
    """Threshold with the best window sensitivity whose false alarms per hour stay within the limit.

    fa_rule says how the false-alarm rate of several validation patients is
    combined before it is compared with the limit:
      pooled: all false alarms / all hours (one noisy patient can dominate)
      median: the median of each patient's own rate, so one noisy patient out
              of three cannot set the threshold for everyone (decision 3.25)
    Sensitivity is always pooled over all validation seizure windows.

    Candidates are a fixed grid plus quantiles of the predicted probabilities,
    so very confident models still get usable thresholds. If no candidate meets
    the limit, the one with the fewest false alarms is returned, marked
    meets_fa_limit = False.
    """
    per_patient = None
    if fa_rule == "median":
        per_patient = [_prepare({k: v[p["patient"] == pt] for k, v in p.items()})
                       for pt in np.unique(p["patient"])]
    p = _prepare(p)
    q = np.quantile(p["prob"], 1 - np.logspace(-5, -0.3, 150))
    cands = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), q]))
    results = [score(p, t) for t in cands]
    if per_patient is not None:
        for r in results:
            r["fa_per_hour_pooled"] = r["fa_per_hour"]
            r["fa_per_hour"] = float(np.median([score(pp, r["threshold"])["fa_per_hour"]
                                                for pp in per_patient]))
    ok = [r for r in results if r["fa_per_hour"] <= max_fa_per_hour]
    if ok:
        best = max(ok, key=lambda r: (r["sensitivity"], r["threshold"]))
    else:
        best = min(results, key=lambda r: (r["fa_per_hour"], -r["sensitivity"]))
    return {**best, "meets_fa_limit": bool(ok)}
