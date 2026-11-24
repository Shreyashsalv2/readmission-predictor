#!/usr/bin/env bash
# Download the Diabetes 130-US Hospitals dataset from the UCI repository (CC BY 4.0)
# and check it against a pinned SHA-256 before unpacking.
set -euo pipefail
URL="https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
SHA256="f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
OUT="${1:-data}"
mkdir -p "$OUT"
if [ ! -f "$OUT/diabetic_data.csv" ]; then
  curl -fsSL --retry 3 -o "$OUT/dataset.zip" "$URL"
  echo "$SHA256  $OUT/dataset.zip" | sha256sum -c -
  unzip -o -q "$OUT/dataset.zip" -d "$OUT"
  # some versions nest the files in a folder
  find "$OUT" -mindepth 2 -name '*.csv' -exec mv {} "$OUT"/ \;
  rm -f "$OUT/dataset.zip"
fi
ls -la "$OUT"
wc -l "$OUT/diabetic_data.csv"
