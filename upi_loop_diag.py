"""Per-round loop diagnostics of HAD-Stack on the UPI data (paper: Section 7.1, Table "game dynamics", Fig. dynamics).

Re-trains HAD-Stack (method v3) for seeds 1-3 from models/baseline_best.pt exactly as run_confirmation.py does
(had_stack_v2.train: AdamW 1e-4, cosine over 30 rounds, robust-AUC selection at eps 0.5, patience 6) and logs the
diagnostics of Definition 8 every round: in-loop attack success on training fraud (train_flip), PGD-10 against the
current detector (fresh), last round's PGD-10 examples replayed (stale) and PGD-50x2 (strong) on validation fraud.
The training loop is verification_second_dataset/run_gpu.train_adv with the released attack, which reproduces
had_stack_v2.train round for round (its diagnostics save and restore the random state); this script checks that by comparing eps, lambda, robust
validation AUC and the feedback signal with logs/v2_traj_v3_seed{s}.csv written by run_confirmation.py.

Run from the repository root after run_confirmation.py:   python upi_loop_diag.py            (CPU, ~1 h per seed)
Outputs: logs/diag_seed{s}/traj_v3.csv and results/upi_loop_diag.json.  Use --seeds 1 to run one seed.
"""
import os, sys, csv, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "verification_second_dataset"))
import numpy as np, torch
import adversarial_game as ag
_released_attack = ag.HamiltonianAttacker.evasion_attack
import run_gpu as rg                       # patches the released trainer with the diagnostics (same updates)
ag.HamiltonianAttacker.evasion_attack = _released_attack   # keep the released attack (run_gpu's fraud-only variant draws
                                                           # fewer random numbers, so trajectories would drift)
import had_stack_v2 as hs, hamiltonian_v2 as hv

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--base", default="models/baseline_best.pt"); ap.add_argument("--data", default="processed_data")
a = ap.parse_args()
rg.set_device("cpu")
tr, va, te, S = rg.loaders(a.data, smoke=False); nf = S["train"][0].shape[1]
out = {"order": hv.HCS_ORDER, "eps_max": hs.EPS_MAX, "seeds": {}}
for sd in [int(s) for s in a.seeds.split(",")]:
    d = f"logs/diag_seed{sd}"; os.makedirs(d, exist_ok=True)
    rg.train_adv("v3", sd, "scratch", nf, tr, va, S, a.base, d, a.rounds)
    T = list(csv.DictReader(open(f"{d}/traj_v3.csv")))
    ref_f = f"logs/v2_traj_v3_seed{sd}.csv"; dev = None
    if os.path.exists(ref_f):
        R = list(csv.DictReader(open(ref_f)))
        dev = max(abs(float(t[k]) - float(r[k])) for t, r in zip(T, R) for k in ("eps", "lam", "robust_val_auc", "next_signal"))
        dev = dict(max_abs_diff=dev, rounds_diag=len(T), rounds_reference=len(R))
        print(f"seed {sd}: reproduction of run_confirmation's trajectory: {dev}", flush=True)
    out["seeds"][sd] = dict(reproduction=dev, traj=T)
    os.makedirs("results", exist_ok=True); json.dump(out, open("results/upi_loop_diag.json", "w"), indent=1)
print("wrote results/upi_loop_diag.json")
