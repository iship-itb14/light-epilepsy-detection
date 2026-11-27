#!/usr/bin/env bash
# Fetch the Bonn EEG sets (Andrzejak et al. 2001; about 3 MB) into data/raw/bonn/
# Source page: https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/
# These are the recordings behind the UCI/Kaggle "Epileptic Seizure Recognition" table,
# which UCI no longer hosts. Used only for the split-leakage demo (decision 3.26). Not stored in git.
set -euo pipefail
BASE=https://www.ukbonn.de/site/assets/files
OUT=data/raw/bonn
mkdir -p "$OUT"
cd "$OUT"
# set letter, file id on the site, SHA-256 of the zip we downloaded on 27 Nov 2026
while read -r SET ID SUM; do
  curl -fsSL -o "$SET.zip" "$BASE/$ID/$SET.zip"
  echo "$SUM  $SET.zip" | shasum -a 256 -c --quiet || { echo "Checksum differs for $SET.zip (the site may have changed the file)" >&2; exit 1; }
  unzip -o -q "$SET.zip" -x '__MACOSX/*'
  rm "$SET.zip"
done <<'LIST'
z 21874 dd7dd53b84fdbdfde16915c608b8333e926112944861aab933f1b76cedb8ad05
o 21872 82b806f39aa04bf010640154dc1bbd548fc8defc2f267653070aaf09129e72da
n 21871 98ab85976b5cc8e0b77cc4d4f76a98c6457ea58184751f6f4141a8c30ff5a2c1
f 21870 f6cc5a5ce9bcb1f16a00cbfde7bb501cd7fe3874b549118182501aa3d509d578
s 21875 befda5a0f352e3bd222b2a8dafa715eb4dc6337ac2bf7a050de62a467615fc3a
LIST
rm -rf __MACOSX
for d in Z O N F S; do
  n=$(find "$d" -maxdepth 1 -iname '*.txt' | wc -l | tr -d ' ')
  [ "$n" = 100 ] || { echo "Folder $d has $n files, expected 100" >&2; exit 1; }
done
echo "All files verified: 5 sets x 100 recordings in $OUT"
