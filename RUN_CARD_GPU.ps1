# Study A on the card data (Dal Pozzolo et al. 2015) on a CUDA GPU, with Delta = Re<a^dag b> (HCS_ORDER=adagb).
# Run from the repository root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\RUN_CARD_GPU.ps1 -CardCsv C:\path\to\creditcard.csv
# Add -Smoke for a 5-minute test on a small subset first (results_gpu_smoke\, numbers mean nothing).
# Outputs go to verification_second_dataset\results_gpu\ (the CPU results in results\ are not touched).
# Every unit is skipped if its results.json exists, so the same command resumes an interrupted run.
param([Parameter(Mandatory = $true)][string]$CardCsv, [switch]$Smoke)
$ErrorActionPreference = "Stop"
$env:HCS_ORDER = "adagb"; $env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path $CardCsv)) { throw "creditcard.csv not found at $CardCsv" }
$card = (Resolve-Path $CardCsv).Path
$res = if ($Smoke) { "results_gpu_smoke" } else { "results_gpu" }
New-Item -ItemType Directory -Force run_logs_gpu | Out-Null

function Step([string]$name, [string[]]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $log = Join-Path (Resolve-Path run_logs_gpu) "$name.txt"
    Push-Location verification_second_dataset
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & python @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old; Pop-Location }
    if ($code -ne 0) { throw "$name failed (exit $code); see $log" }
}

& python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot see a CUDA GPU. Install the CUDA build of PyTorch (pytorch.org) and try again." }
& python -c "import had_stack_v2 as hs, hamiltonian_v2 as hv; print('order', hv.HCS_ORDER, 'eps range', hs.EPS_MIN, hs.EPS_MAX); assert hv.HCS_ORDER == 'adagb' and hs.EPS_MAX == 0.6428"
if ($LASTEXITCODE -ne 0) { throw "controller check failed" }

Step "g20_equivalence" @("tests/test_equivalence.py", "--cuda")
Step "g21_prep"        @("prep.py", "--csv", $card)
$extra = if ($Smoke) { @("--smoke", "--rounds", "2") } else { @() }
foreach ($u in @("seed1", "seed2", "seed3", "fold0", "fold1", "fold2", "fold3", "fold4")) {
    Step "g22_$u" (@("run_gpu.py", "--track", "scratch", "--unit", $u, "--device", "cuda", "--results", $res) + $extra)
    if ($Smoke) { break }
}
$sumargs = if ($Smoke) { @("summarize.py", $res, "--include-smoke") } else { @("summarize.py", $res) }
Step "g23_summary" $sumargs
if (-not $Smoke) { Step "g24_figure" @("make_fig_ulb.py", $res, "$res/f_ulb_dynamics.pdf") }

Get-ChildItem "verification_second_dataset\$res" -Recurse -Filter *.pt | Remove-Item -Force -ErrorAction SilentlyContinue
$zip = if ($Smoke) { "results_card_gpu_smoke.zip" } else { "results_card_gpu.zip" }
Compress-Archive -Force -Path run_logs_gpu, "verification_second_dataset\$res" -DestinationPath $zip
Write-Host "`nFinished. Send $zip." -ForegroundColor Green
