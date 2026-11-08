#!/usr/bin/env bash
# Study A on the card data on a CUDA GPU, Delta = Re<a^dag b> (HCS_ORDER=adagb). Linux / Colab / lab machine.
# usage: bash run_card_gpu.sh /path/to/creditcard.csv [--smoke]
# Outputs: verification_second_dataset/results_gpu/ (smoke: results_gpu_smoke/), then results_card_gpu.zip.
set -euo pipefail
cd "$(dirname "$0")"
CSV=$(realpath "${1:?pass the path to creditcard.csv}"); SMOKE=${2:-}
export HCS_ORDER=adagb PYTHONUTF8=1
RES=results_gpu; EXTRA=(); [ "$SMOKE" = "--smoke" ] && { RES=results_gpu_smoke; EXTRA=(--smoke --rounds 2); }
mkdir -p run_logs_gpu
step() { local name=$1; shift; echo "=== $name ==="; (cd verification_second_dataset && python "$@") 2>&1 | tee "run_logs_gpu/$name.txt"; }
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'CUDA',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
python -c "import had_stack_v2 as hs, hamiltonian_v2 as hv; print('order', hv.HCS_ORDER, 'eps range', hs.EPS_MIN, hs.EPS_MAX); assert hv.HCS_ORDER == 'adagb' and hs.EPS_MAX == 0.6428"
step g20_equivalence tests/test_equivalence.py --cuda
step g21_prep prep.py --csv "$CSV"
for u in seed1 seed2 seed3 fold0 fold1 fold2 fold3 fold4; do
  step g22_$u run_gpu.py --track scratch --unit $u --device cuda --results $RES ${EXTRA[@]+"${EXTRA[@]}"}
  [ -n "$SMOKE" ] && break
done
if [ -n "$SMOKE" ]; then step g23_summary summarize.py $RES --include-smoke
else step g23_summary summarize.py $RES; step g24_figure make_fig_ulb.py $RES $RES/f_ulb_dynamics.pdf; fi
find verification_second_dataset/$RES -name "*.pt" -delete
zip -qr results_card_gpu${SMOKE:+_smoke}.zip run_logs_gpu verification_second_dataset/$RES
echo "Finished. Send results_card_gpu${SMOKE:+_smoke}.zip"
