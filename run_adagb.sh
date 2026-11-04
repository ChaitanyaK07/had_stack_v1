#!/usr/bin/env bash
# Re-runs Study A (UPI fraud data and card data) with Delta = Re<a^dag b> (Eqs. 7-8).
# usage: bash run_adagb.sh /path/to/creditcard.csv      (needs fraud_dataset.csv in the repository root)
#        SKIP_UPI=1 or SKIP_CARD=1 to skip a part. Every stage can be restarted with the same command.
set -euo pipefail
cd "$(dirname "$0")"
export HCS_ORDER=adagb PYTHONUTF8=1
mkdir -p run_logs models logs
step() { local name=$1 dir=$2; shift 2; echo "=== $name ==="; (cd "$dir" && python "$@") 2>&1 | tee "run_logs/$name.txt"; }
python -c "import had_stack_v2 as hs, hamiltonian_v2 as hv; print('order', hv.HCS_ORDER, 'eps range', hs.EPS_MIN, hs.EPS_MAX); assert hv.HCS_ORDER == 'adagb' and hs.EPS_MAX == 0.6428"
if [ -z "${SKIP_UPI:-}" ]; then
  step 01_data_processing . data_processing.py
  step 02_baseline . baseline_model.py
  step 03_confirmation_all . run_confirmation.py --part all
  step 04_confirmation_lambda . run_confirmation.py --part lambda
  step 05_final_experiments . run_final_experiments.py
  step 06_analyze_final . analyze_final.py
  step 07_ablations_train . experiments_and_ablations.py --part train
  step 08_ablations_eval . experiments_and_ablations.py --part eval
  step 09_ablations_summary . experiments_and_ablations.py --part summary
  step 10_single_v3 . had_stack_v2.py --method v3
  step 11_single_pgdat . had_stack_v2.py --method pgdat
  step 12_diag_masking . diag_masking.py v3=models/v2_v3_best.pt
  step 13_trajectory_figure . make_trajectory_figure.py
fi
if [ -z "${SKIP_CARD:-}" ]; then
  CSV=$(realpath "${1:?pass the path to creditcard.csv, or set SKIP_CARD=1}")
  step 20_card_equivalence verification_second_dataset tests/test_equivalence.py --cuda
  step 21_card_prep verification_second_dataset prep.py --csv "$CSV"
  for u in seed1 seed2 seed3 fold0 fold1 fold2 fold3 fold4; do step 22_card_$u verification_second_dataset run_gpu.py --track scratch --unit $u; done
  step 23_card_summary verification_second_dataset summarize.py
  step 24_card_figure verification_second_dataset make_fig_ulb.py
fi
zip -qr results_adagb_studyA.zip run_logs logs results_final results_advisor fig_trajectory.png verification_second_dataset/results 2>/dev/null || true
echo "All stages finished. Send results_adagb_studyA.zip."
