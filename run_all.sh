#!/usr/bin/env bash
# Reproduce everything.  Needs the CoinMetrics community CSVs:
#   git clone --depth 1 https://github.com/coinmetrics/data /path/to/coinmetrics-data
# The compact panels committed under data/ were built from commit f1a36af (2026-05-24); rebuild only if you
# want fresher data:   python scripts/build_dataset.py /path/to/coinmetrics-data/csv
set -euo pipefail
cd "$(dirname "$0")"
python run_backtest.py          # headline metrics + charts   -> results/
python validate.py              # robustness suite (~5 min)   -> results/validation.json
