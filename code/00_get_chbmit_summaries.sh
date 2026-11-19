#!/usr/bin/env bash
# Fetch CHB-MIT summary files (no EEG yet) into data/raw/chbmit/
set -euo pipefail
BASE=https://physionet.org/files/chbmit/1.0.0
OUT=data/raw/chbmit
mkdir -p "$OUT"
for f in RECORDS RECORDS-WITH-SEIZURES SHA256SUMS.txt; do
  curl -fsSL -o "$OUT/$f" "$BASE/$f"
  [ -s "$OUT/$f" ] || { echo "Empty download: $f" >&2; exit 1; }
done
for i in $(seq -w 1 24); do
  mkdir -p "$OUT/chb$i"
  curl -fsSL -o "$OUT/chb$i/chb$i-summary.txt" "$BASE/chb$i/chb$i-summary.txt"
  [ -s "$OUT/chb$i/chb$i-summary.txt" ] || { echo "Empty download: chb$i summary" >&2; exit 1; }
done
