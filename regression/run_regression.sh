#!/usr/bin/env bash
# Reproduces the regression study of the paper (Section VII-D): ~25 min on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
python hadstack_thlstm.py --mode seeds --out results/seeds && cp results/seeds/results.json results/results_seeds.json
python hadstack_thlstm.py --mode folds --out results/folds && cp results/folds/results.json results/results_folds.json
python fig_forecast.py      # Fig. 8 (fixed split, seed 1)
python tau_sens.py          # data for Fig. 7
python make_paper_figures.py
python paper_stats.py > results/paper_stats.txt
