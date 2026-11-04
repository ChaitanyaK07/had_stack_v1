# Re-runs Study A (UPI fraud data and card data) with the order parameter Delta = Re<a^dag b> (Eqs. 7-8).
# Run from the repository root in PowerShell:   powershell -ExecutionPolicy Bypass -File .\RUN_ADAGB.ps1 -CardCsv C:\path\to\creditcard.csv
# Needs fraud_dataset.csv in the repository root. Every stage can be restarted with the same command.
# Skip parts with -SkipUPI or -SkipCard. Output of every stage goes to run_logs\.
param([string]$CardCsv = "", [switch]$SkipUPI, [switch]$SkipCard)
$ErrorActionPreference = "Stop"
$env:HCS_ORDER = "adagb"                 # "ab" reproduces the original release
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force run_logs, models, logs | Out-Null

function Step([string]$name, [string[]]$cmd, [string]$dir = ".") {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $log = Join-Path (Resolve-Path run_logs) "$name.txt"
    Push-Location $dir
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"   # Python warnings on stderr must not abort the run
    try {
        & python @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old; Pop-Location }
    if ($code -ne 0) { throw "$name failed (exit $code); see $log" }
}

& python -c "import had_stack_v2 as hs, hamiltonian_v2 as hv; print('order', hv.HCS_ORDER, 'eps range', hs.EPS_MIN, hs.EPS_MAX); assert hv.HCS_ORDER == 'adagb' and hs.EPS_MAX == 0.6428"
if ($LASTEXITCODE -ne 0) { throw "controller check failed" }

if (-not $SkipUPI) {
    if (-not (Test-Path fraud_dataset.csv)) { throw "fraud_dataset.csv not found in the repository root" }
    Step "01_data_processing"     @("data_processing.py")
    Step "02_baseline"            @("baseline_model.py")
    Step "03_confirmation_all"    @("run_confirmation.py", "--part", "all")
    Step "04_confirmation_lambda" @("run_confirmation.py", "--part", "lambda")
    Step "05_final_experiments"   @("run_final_experiments.py")
    Step "06_analyze_final"       @("analyze_final.py")
    Step "07_ablations_train"     @("experiments_and_ablations.py", "--part", "train")
    Step "08_ablations_eval"      @("experiments_and_ablations.py", "--part", "eval")
    Step "09_ablations_summary"   @("experiments_and_ablations.py", "--part", "summary")
    Step "10_single_v3"           @("had_stack_v2.py", "--method", "v3")
    Step "11_single_pgdat"        @("had_stack_v2.py", "--method", "pgdat")
    Step "12_diag_masking"        @("diag_masking.py", "v3=models/v2_v3_best.pt")
    Step "13_trajectory_figure"   @("make_trajectory_figure.py")
}

if (-not $SkipCard) {
    if ($CardCsv -eq "" -or -not (Test-Path $CardCsv)) { throw "pass -CardCsv C:\path\to\creditcard.csv (or use -SkipCard)" }
    $card = (Resolve-Path $CardCsv).Path
    Step "20_card_equivalence" @("tests/test_equivalence.py", "--cuda") "verification_second_dataset"
    Step "21_card_prep"        @("prep.py", "--csv", $card) "verification_second_dataset"
    foreach ($u in @("seed1", "seed2", "seed3", "fold0", "fold1", "fold2", "fold3", "fold4")) {
        Step "22_card_$u" @("run_gpu.py", "--track", "scratch", "--unit", $u) "verification_second_dataset"
    }
    Step "23_card_summary" @("summarize.py") "verification_second_dataset"
    Step "24_card_figure"  @("make_fig_ulb.py") "verification_second_dataset"
}

# Collect everything needed for the paper into one zip to send back
$items = @("run_logs", "logs", "results_final", "results_advisor", "fig_trajectory.png", "verification_second_dataset/results") | Where-Object { Test-Path $_ }
Compress-Archive -Force -Path $items -DestinationPath results_adagb_studyA.zip
Write-Host "`nAll stages finished. Send results_adagb_studyA.zip." -ForegroundColor Green
