#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HCS_ORDER=adagb TF_CPP_MIN_LOG_LEVEL=3
python hadstack_thlstm.py --mode seeds --out results/seeds > results/seeds.log 2>&1 && cp results/seeds/results.json results/results_seeds.json
python hadstack_thlstm.py --mode folds --out results/folds > results/folds.log 2>&1 && cp results/folds/results.json results/results_folds.json
python fig_forecast.py > results/fig_forecast.log 2>&1
python tau_sens.py > results/tau_sens.log 2>&1
python make_paper_figures.py
python paper_stats.py > results/paper_stats.txt
echo DONE
