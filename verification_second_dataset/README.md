# Verification of Study A on a second dataset (paper Section VII-C)

Re-runs the paper's Study A algorithms on the public credit-card fraud dataset of
Dal Pozzolo et al. (IEEE SSCI 2015): `creditcard.csv`, 284,807 transactions, 492 fraud.
It checks the two things the paper claims for Study A:

1. **The game loop works**: the loop diagnostics of Definition 8 (in-loop attack success,
   fresh vs. stale responses, PGD-50×2, exploitability gap, budget trajectory).
2. **What the loop produces**: HAD-Stack against the undefended FraudTransformer under the
   worst-case ensemble of Algorithm 5 (white-box PGD-100×5 with BCE and margin losses, plus transfer).

The training code is the released code in the repository root (`baseline_model.py`,
`adversarial_game.py`, `had_stack_v2.py`, `hamiltonian_v2.py`, `final_hamiltonian_solver.py`), unchanged and
imported from `..`. `run_gpu.py` only adds device handling, the per-round diagnostics, and an attack
that runs on fraud rows only. That attack gives exactly the same perturbation as the released one, which
`tests/test_equivalence.py` checks.

| Track | Starting detector θ₀ | Protocol | Units | In the paper |
|---|---|---|---|---|
| **A, scratch** | FraudTransformer trained from scratch (30 features: Time, V1–V28, log(1+Amount)) | 70/15/15 split, seed 42; five-fold CV | seed1–3, fold0–4 | **Yes: Table VIII, Fig. 5** |
| B, pretrained | pre-trained MLP from [PHNPR/Credit-Card-Fraud-Detection](https://github.com/PHNPR/Credit-Card-Fraud-Detection) (MIT; weights and licence in `pretrained/`) | the model's own split (80/20, random_state 4) | seed1–3 | No (internal check only) |

## Reproduce

The dataset is not redistributed here. Download `creditcard.csv` yourself and pass its path.

```bash
cd verification_second_dataset
pip install -r requirements.txt                 # install the CUDA build of PyTorch for your driver first
python tests/test_equivalence.py --cuda          # -> ALL EQUIVALENCE CHECKS PASSED
python prep.py --csv /path/to/creditcard.csv     # writes data/scratch_fixed, data/scratch_fold0..4, data/pretrained_fixed

python run_gpu.py --track scratch --unit seed1   # run first: trains the shared fixed-split baseline
python run_gpu.py --track scratch --unit seed2
python run_gpu.py --track scratch --unit seed3
for k in 0 1 2 3 4; do python run_gpu.py --track scratch --unit fold$k; done
python summarize.py                              # results/summary.md and results/summary.json
python make_fig_ulb.py                           # results/f_ulb_dynamics.pdf (Fig. 5)
```

You can also run `bash run_all.sh /path/to/creditcard.csv`. It runs everything, including track B.
`run_gpu.py` skips any unit whose `results.json` already exists, so delete a unit's folder to re-run it.
Use `--device cpu` without a GPU (track A takes hours per unit on CPU), and `--smoke` for a two-round
test on a small subset whose numbers mean nothing.

## Results in this folder

`results/` holds the runs reported in the paper (track A on a CUDA GPU) and the track B runs (CPU).
Model weights (`*.pt`) are not included.

* `results/<track>_<unit>/results.json`: clean metrics, robust recall and broken rate at ε = 0.5 and 1.0
  for the baseline, HAD-Stack (`v3`) and the fixed-budget transfer source (`fbat05`), plus per-row outcomes
  for the paired bootstrap
* `results/<track>_<unit>/traj_v3.csv`, `traj_fbat05.csv`: per-round ε, λ, robust validation AUC, feedback
  signal and loop diagnostics
* `results/summary.md`: the tables behind Table VIII and Section VII-C

## Notes

* The fixed-budget detector (`fbat05`, ε = 0.6586, λ = 0.5) is trained only as an extra transfer source
  for the evaluation ensemble, as in the paper.
* The data contain 1,081 exact duplicate transactions, which are kept. Removing the test fraud cases that
  have a copy in the training rows changes the five-fold robust-recall gains by at most 0.3 points (paper, Section VII-C).
* Results on GPU differ slightly from CPU because of floating-point non-determinism.
