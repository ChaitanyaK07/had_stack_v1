# HAD-Stack: Hamiltonian-Driven Stackelberg Adversarial Training for UPI Fraud Detection

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

## Data

`fraud_dataset.csv`: 26,393 synthetic UPI/banking transactions (17.2% fraud). [Add source and licence here.
If redistribution is not permitted, keep the file out of the repository and describe how to obtain it.]

## Citation

[Add the citation once the paper is published.]

## Licence

[Choose a licence, e.g. MIT, and add a LICENSE file.]
