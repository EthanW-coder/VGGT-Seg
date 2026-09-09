#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT IMAGE_DIR OUTPUT_NPZ" >&2
  exit 2
fi

python infer.py --variant full --config "$1" --checkpoint "$2" --image-dir "$3" --output "$4"
