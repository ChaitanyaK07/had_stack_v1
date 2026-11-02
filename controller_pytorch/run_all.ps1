# Runs everything end to end and writes logs + results. Usage (from this folder):
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Csv data\your_yields.csv -Device cuda
param(
  [string]$Csv = "..\data\india_bond_yields_2010_2025.csv",
  [string]$Target = "10Y",
  [string]$Device = "auto",
  [string]$Out = "results"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $Out "run_$stamp.log"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | Tee-Object -FilePath $log
# 1) Hamiltonian layer check (reproduces Table VI; aborts on mismatch)
python had_stack_bond.py --check-hamiltonian 2>&1 | Tee-Object -FilePath $log -Append
# 2) Reference-budget sweep on the undefended model
python had_stack_bond.py --csv $Csv --target $Target --device $Device --sweep 2>&1 | Tee-Object -FilePath $log -Append
# 3) Main run: 3 seeds on fixed split + 5 walk-forward folds (univariate, as in the bond paper)
python had_stack_bond.py --csv $Csv --target $Target --device $Device --mode all --out "$Out\univariate" 2>&1 | Tee-Object -FilePath $log -Append
# 4) Robustness of the conclusion: 5 seeds on the fixed split
python had_stack_bond.py --csv $Csv --target $Target --device $Device --mode seeds --seeds 1,2,3,4,5 --out "$Out\univariate_5seeds" --no-plots 2>&1 | Tee-Object -FilePath $log -Append
# 5) Paper's original loop lr (1e-4) for comparison
python had_stack_bond.py --csv $Csv --target $Target --device $Device --mode all --lr 1e-4 --out "$Out\univariate_lr1e-4" --no-plots 2>&1 | Tee-Object -FilePath $log -Append
Write-Host "Done. Summaries: $Out\*\summary.md   Log: $log"
