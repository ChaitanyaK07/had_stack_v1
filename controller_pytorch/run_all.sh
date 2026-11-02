#!/usr/bin/env bash
# Same pipeline as run_all.ps1 for a Linux GPU box:
#   bash run_all.sh [CSV] [TARGET] [DEVICE] [OUT]      e.g.  nohup bash run_all.sh &
set -euo pipefail
CSV=${1:-../data/india_bond_yields_2010_2025.csv}; TARGET=${2:-10Y}; DEVICE=${3:-auto}; OUT=${4:-results}
mkdir -p "$OUT"; LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
{
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python had_stack_bond.py --check-hamiltonian
python had_stack_bond.py --csv "$CSV" --target "$TARGET" --device "$DEVICE" --sweep
python had_stack_bond.py --csv "$CSV" --target "$TARGET" --device "$DEVICE" --mode all --out "$OUT/univariate"
python had_stack_bond.py --csv "$CSV" --target "$TARGET" --device "$DEVICE" --mode seeds --seeds 1,2,3,4,5 --out "$OUT/univariate_5seeds" --no-plots
python had_stack_bond.py --csv "$CSV" --target "$TARGET" --device "$DEVICE" --mode all --lr 1e-4 --out "$OUT/univariate_lr1e-4" --no-plots
echo "Done. Summaries: $OUT/*/summary.md"
} 2>&1 | tee "$LOG"
