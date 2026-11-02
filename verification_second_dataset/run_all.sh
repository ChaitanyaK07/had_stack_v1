#!/usr/bin/env bash
# Full verification on one GPU.  Usage:  bash run_all.sh /path/to/creditcard.csv
set -euo pipefail
cd "$(dirname "$0")"
python tests/test_equivalence.py --cuda          # must print ALL EQUIVALENCE CHECKS PASSED
python prep.py --csv "$1"
# Track A: FraudTransformer trained from scratch (paper protocol).
# seed1 trains the shared fixed-split baseline, so it runs first; seed2/seed3 reuse it.
python run_gpu.py --track scratch --unit seed1
python run_gpu.py --track scratch --unit seed2
python run_gpu.py --track scratch --unit seed3
for k in 0 1 2 3 4; do python run_gpu.py --track scratch --unit fold$k; done
# Track B: pre-trained MLP (fixed split, three seeds).
for s in 1 2 3; do python run_gpu.py --track pretrained --unit seed$s; done
python summarize.py
zip -r results_to_send.zip results -x "*.pt"
echo "Done. Send results_to_send.zip back."
