#!/usr/bin/env bash
# Download the chosen CHB-MIT recordings (.edf) and verify their checksums.
# Safe to rerun: verified files are skipped, partial ones resume.
set -euo pipefail
BASE=https://physionet.org/files/chbmit/1.0.0
OUT=data/raw/chbmit
PATIENTS="chb01 chb03 chb05 chb06 chb08 chb10 chb13 chb14 chb15 chb16 chb18 chb20 chb23 chb24"

# Checksum list in the two-space format shasum expects
sed 's/ /  /' "$OUT/SHA256SUMS.txt" > "$OUT/.sha256-local.txt"

free_gb=$(df -g . | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 22 ]; then
  echo "Only ${free_gb} GB free; about 22 GB is needed. Free up space and rerun." >&2
  exit 1
fi

for p in $PATIENTS; do
  grep "^$p/" "$OUT/RECORDS" | while read -r rec; do
    if [ -f "$OUT/$rec" ] && grep "  $rec\$" "$OUT/.sha256-local.txt" \
         | (cd "$OUT" && shasum -a 256 -c --status -); then
      continue  # already downloaded and verified
    fi
    echo "Downloading $rec"
    curl -fL -C - --retry 3 -o "$OUT/$rec" "$BASE/$rec"
  done
done

echo "Verifying checksums..."
grep -E "  ($(echo $PATIENTS | tr ' ' '|'))/.*\.edf$" "$OUT/.sha256-local.txt" \
  | (cd "$OUT" && shasum -a 256 -c --quiet -)
echo "All files verified."
