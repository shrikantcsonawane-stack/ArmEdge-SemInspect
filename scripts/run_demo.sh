#!/usr/bin/env bash
# One-shot demo setup: synthetic data -> short training -> evaluation -> dashboard.
# For the real dataset: python scripts/download_dataset.py && python scripts/prepare_dataset.py
#                       && python scripts/train.py && python scripts/evaluate.py
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -f data/demo/manifest.csv ] || $PY scripts/prepare_dataset.py --demo
[ -f models/mobilenetv3_small_best.pt ] || $PY scripts/train.py --demo
[ -f reports/eval_demo.json ] || $PY scripts/evaluate.py --demo
exec $PY -m streamlit run app.py --server.headless true
