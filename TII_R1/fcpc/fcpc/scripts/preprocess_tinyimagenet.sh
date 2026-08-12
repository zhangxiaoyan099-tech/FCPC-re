#!/usr/bin/env bash
set -euo pipefail

python3 scripts/preprocess_tinyimagenet.py --root "${1:-data/tiny-imagenet-200}" --mode copy

