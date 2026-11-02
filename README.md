# HAD-Stack: Hamiltonian-Driven Stackelberg Adversarial Training for Fraud Detection, Bond Forecasting.

Code and results for *A Hamiltonian-Stackelberg Game Approach for Fraud Detection in Digital Banking Transactions*.

HAD-Stack trains a Transformer fraud detector as the leader of a Stackelberg game against an $L_\infty$ PGD attacker.
Each round, the attacker's budget ε and the clean/adversarial loss weight λ are derived from the **complex eigenspectra of
non-Hermitian Hamiltonians** (H1, H2 and the full dissipative Eq. 29). A validation robustness-gap signal rescales their coupling constants.

## Main results (five-fold cross-validation, worst-case attack ensemble)

Robust recall = share of all fraud cases still detected under the strongest of: white-box PGD (100 steps × 5 restarts,
BCE and margin losses) and transfer attacks from the other models.

| Model | Robust recall, ε = 0.5 | Robust recall, ε = 1.0 | Clean cost (100·FN + FP, per fold) |
|---|---|---|---|
| Baseline Transformer | 51.6% | 18.1% | 9,358 |
| HAD-Stack | 70.3% | 53.0% | 13,981 |
| FB-AT (λ = 0.5) | 70.6% | 50.8% | 12,903 |
| FB-AT (λ = 0.66) | 70.1% | 51.3% | 14,221 |

- **Robustness vs. the undefended detector:** a large, consistent gain in every seed and fold
  (paired bootstrap at ε = 1.0: +34.9 points, 95% CI [+33.4, +36.2]).
- **Is it the Hamiltonians?** No. Fixed-budget training with the same attack weights and a non-Hamiltonian controller driven by the same
  signal are equally robust (paired differences at ε = 1.0: +0.2 and -0.3 points, not significant),
  and standard PGD-AT is as robust at lower clean cost. HAD-Stack does beat a linear ε ramp (all 5 folds, +7.5 points).
  Pre-registered tests therefore do not credit the Hamiltonian controller.
- **Cost:** every defence trades clean detection for robustness.
- **Evaluation note:** a 10-step attack overstated HAD-Stack's robustness up to ~10× (gradient masking); all reported
  numbers use the worst-case ensemble.

The data are **synthetic** and near-separable; perturbations are in standardised feature units. See the paper for all
caveats (confounds, support leakage, limitations).

## Repository layout

| File | Purpose |
|---|---|
| `data_processing.py` | Leakage-audited preprocessing (18 features), 70/15/15 split, support-leakage check |
| `baseline_model.py` | FraudTransformer baseline (training, evaluation) |
| `final_hamiltonian_solver.py` | Hamiltonian operators H1, H2, Eq. 29 (reference definitions) |
| `hamiltonian_v2.py` | Complex-spectrum solvers built on the operators above (`numpy.linalg.eig`) |
| `adversarial_game.py` | Stackelberg trainer, PGD attacker, robustness-gap signal |
| `had_stack_v2.py` | HAD-Stack controller (`--method v3`) and fixed-budget defences |
| `run_confirmation.py` | 3 seeds + 5-fold CV (`--part all`) and the λ-matched control (`--part lambda`) |
| `run_final_experiments.py` | Per-transaction worst-case evaluation (ε = 0.25–1.0) and XGBoost ablation |
| `analyze_final.py` | Reproduction check, bootstrap CIs, XGBoost ablation tables, cost/lift/recall curves |
| `experiments_and_ablations.py` | Controls, baselines (PGD-AT, TRADES), component ablation, APGD/Square and constrained attacks |
| `time_costs.py` | Computational-cost measurements |
| `h29_latex.py` | Generates and verifies the LaTeX of the full H29 operator |
| `strong_eval.py`, `diag_masking.py` | Worst-case evaluation and gradient-masking diagnostics (single split) |
| `eval_extras.py` | Shared helpers (data loading, metrics, XGBoost configuration) |
| `make_arch_figure.py`, `make_trajectory_figure.py` | Figures 1 and the controller trajectory |
| `results/` | The result files the paper's numbers come from |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Place `fraud_dataset.csv` in the repository root (see *Data*).

## Reproducing the paper

Run from the repository root, one command at a time, in one terminal (scripts share checkpoint files).
All runs are on CPU; interrupted runs can be restarted with the same command.

```bash
python data_processing.py                       # processed_data/
python baseline_model.py                        # baseline checkpoint (seed 42)
python run_confirmation.py --part all           # HAD-Stack + FB-AT (λ=0.5): 3 seeds, 5 folds
python run_confirmation.py --part lambda        # λ-matched FB-AT control
python run_final_experiments.py                 # per-transaction worst-case outputs + XGBoost ablation
python analyze_final.py                         # checks reproduction, writes results_final/analysis/
python experiments_and_ablations.py --part train     # controls, baselines, ablations (5 folds)
python experiments_and_ablations.py --part eval      # extended + constrained attacks
python experiments_and_ablations.py --part summary   # pre-registered verdicts -> results_advisor/advisor_summary.json
```
Single-split diagnostics (seed 42; gradient-masking table and trajectory figure):
```bash
python had_stack_v2.py --method v3
python had_stack_v2.py --method pgdat           # FB-AT, eps = 0.5 (transfer source for the diagnostics)
python diag_masking.py v3=models/v2_v3_best.pt
python make_trajectory_figure.py
python make_arch_figure.py
```
Expect several hours per stage on a CPU. `analyze_final.py` stops with an error if the re-evaluation does not
reproduce `logs/lambda_control_results.jsonl` exactly.

<<<<<<< HEAD
## Combined paper: Hamiltonian–Stackelberg game loops

Scripts added for *Hamiltonian–Stackelberg Game Loops for Adversarial Learning* (Study A = HAD-Stack). Run them from the
repository root unless stated otherwise. They need only `requirements.txt`, plus `matplotlib` for the figure scripts.

| File | What it produces | Paper |
|---|---|---|
| `check_hamiltonians.py` | Spectra of H1, H2, H_V under both spectral treatments (complex `eig`, Hermitian `eigh`) and the φ column of the controller-response table | Table II, Table VI |
| `hv_convergence.py` | Self-consistent H_V field iteration: 24 runs (2 treatments × 3 coupling scales × 4 starting values), iterations, fixed points, residuals, contraction factors → `results/hamiltonian_layer/hv_convergence.json` | Table III, Section IV-E |
| `make_fig_hv_convergence.py` | Figure from the file above → `results/hamiltonian_layer/hv_convergence.pdf` | Fig. 2 |
| `torch_hamiltonian_solver.py` | Batched PyTorch solver for H1, H2, H_V (same code on CPU or CUDA), checked against the NumPy reference solver; CPU timings → `results/hamiltonian_layer/torch_solver_check.json` | Algorithm 1, Table IV, Section IV-G |
| `check_hv_pauli_sums.py` | Compares the Pauli sums of H_V in the derivation with those built by the solver | Section IV-D, Appendix A |
| `rerun_hamiltonian_variants.py` | Re-solves the Hamiltonian layer under the derivation's order parameter and H_V terms, and propagates the change to every controller quantity → `results/hamiltonian_layer/hamiltonian_variants.json` | Section IV-B, IV-D |
| `check_source_values.py` | Independent re-computation of quantities reported in the two source papers | consistency checks |
| `benchmarks/bench.py`, `benchmarks/make_tables.py` | Runtime and scaling benchmark of FraudTransformer on inputs of the data's shape (`python bench.py methods` / `python bench.py scaling`, run inside `benchmarks/`); stored outputs `bench_*.json` | Section VII-D, Table IX |
| `verification_second_dataset/` | Study A on the public credit-card fraud dataset (GPU or CPU), with its own README, tests, stored results and figure script | Section VII-C, Table VIII, Fig. 5 |
| `regression/` | Study A on the bond-yield forecaster (Two-Head LSTM of the forecasting baseline), with stored results; see *Regression study* below | Sections V-F, VII-D, Tables X–XIII, Figs. 6–8 |
| `data/` | Bond data released with the forecasting baseline, and its notebook | Section V-F |
| `replication/` | Re-runs the forecasting baseline's notebook and six-model protocol | — |
| `theory/verify_theorems.py` | Numerical check of Theorems 1–3 and Corollary 1 | Sections IV-B, IV-E, V-D, VI-E |
| `controller_pytorch/` | Earlier stand-alone PyTorch prototype of the regression game (exploratory, not the source of the paper's numbers) | — |
| `paper/` | LaTeX source (IEEEtran, two-column), `refs.bib` and the three regression figures | — |

```bash
python check_hamiltonians.py
python hv_convergence.py && python make_fig_hv_convergence.py
python torch_hamiltonian_solver.py               # CPU check and timings; for GPU use TorchSolver("cuda") in your own code
python check_hv_pauli_sums.py
python rerun_hamiltonian_variants.py
```

## Regression study: bond-yield forecasting

Scripts in `data/`, `regression/`, `replication/`, `theory/`, `controller_pytorch/` and `paper/` resolve paths relative to
their own folder, so they run from anywhere. Everything runs on CPU (Python 3.10+). They need TensorFlow, which the
fraud code does not, so they have their own requirements file:

```bash
pip install -r requirements-regression.txt
bash regression/run_regression.sh            # whole regression study, ~25 min on 2 CPU cores
python theory/verify_theorems.py
```

### regression/ (paper, Section VII-D)
| file | what it does |
|---|---|
| `hadstack_thlstm.py` | the forecaster exactly as in the baseline's notebook (cell 13) and the unchanged HAD-Stack loop: Hamiltonian controller (Alg. 2, checks Table VI on start-up), weighted PGD on market features (Alg. 6), game loop (Alg. 4), worst-case ensemble (Alg. 7), fixed-budget comparison, block bootstrap |
| `fig_forecast.py` | Fig. 8: fixed split, seed 1, clean vs attacked forecasts |
| `tau_sens.py` | per-day worst-case errors for the threshold sensitivity (Fig. 7) |
| `make_paper_figures.py` | Figs. 6 and 7 from saved results |
| `paper_stats.py` | every regression number in Section VII-D and Tables X-XIII |
| `analyze.py` | compact summary tables from one or more `results.json` |
| `run_regression.sh` | all of the above, in order (~25 min on 2 CPU cores) |
| `results/` | the outputs used in the paper (`results_seeds.json`, `results_folds.json`, `tau_sens.json`, `fig_forecast.json`, logs) |

The loop is deterministic given its inputs: walk-forward fold 4 shares data and seed with fixed-split seed 1
and reproduces it round for round.

### replication/
| file | what it does |
|---|---|
| `run_nb.py` | runs `data/India_Bonds_DL_MLP.ipynb` cell by cell as given and scores each model (MAE, RMSE, directional accuracy, no-change forecast); `--save-plots` keeps the plots, `--grid` also runs the 243-model grid search of cell 7; the notebook sets no random seed, so scores vary from run to run |
| `run_cell7.py` | the grid-search cell on its own |
| `paper_protocol.py` | the six models of the baseline's Section 3 under its stated protocol (20-day window, last 250 days as test), 3 seeds |
| `compare_figures.py` | each figure of the baseline paper next to the same notebook cell re-run (needs the paper PDF and `pdfimages`) |

### theory/
`verify_theorems.py` checks the closed-form spectrum against the eigensolver (to 4e-16), reproduces Table VI,
the derivative and Lipschitz constants of Corollary 1, the 43-iteration count of Theorem 2, the Hermitian-surrogate
spectra used in Theorem 3, and the energy-fidelity inequality on 20,000 random instances.

### controller_pytorch/
`had_stack_bond.py` is an earlier PyTorch re-implementation (univariate input, own forecaster) used to prototype
the regression game before running it on the baseline model as released; it is not the source of the paper's
numbers. `run_all.sh` / `run_all.ps1` run its full pipeline.

### paper/
`journal_paper.tex` (IEEEtran, two-column) and `refs.bib`. `figs/` holds the three regression figures; Fig. 2 is
`results/hamiltonian_layer/hv_convergence.pdf` and Fig. 5 is `verification_second_dataset/results/f_ulb_dynamics.pdf`.

=======
>>>>>>> 75391000f36841e26fc17e00e5c42f991f5b65fe
## Data

`fraud_dataset.csv`: 26,393 synthetic UPI/banking transactions (17.2% fraud). [Add source and licence here.
If redistribution is not permitted, keep the file out of the repository and describe how to obtain it.]

<<<<<<< HEAD
`data/india_bond_yields_2010_2025.csv`, `data/bond_yield_technical_indicators.csv`: Indian government bond yields and
technical indicators released with the forecasting baseline, used by the regression study.

=======
>>>>>>> 75391000f36841e26fc17e00e5c42f991f5b65fe
## Citation

[Add the citation once the paper is published.]

## Licence

[Choose a licence, e.g. MIT, and add a LICENSE file.]
