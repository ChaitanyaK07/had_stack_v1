#!/usr/bin/env python3
"""Non-Hamiltonian schedule baselines on the bond data (reviewer item: does the Hamiltonian controller beat a tuned
schedule?). For one split this trains, from the same pretrained forecaster and with the same game loop, follower,
optimiser, checkpointing and early stopping as HAD-Stack (hadstack_thlstm.game):
  HAD-Stack       : the Hamiltonian controller (retrained; the loop is deterministic, so it reproduces the reported run)
  Linear ramp     : eps_r = EPS_MIN + (EPS_MAX - EPS_MIN)(r-1)/(R-1), lam 0.5, the controller's base step weights
  Cosine ramp     : eps_r = EPS_MIN + (EPS_MAX - EPS_MIN)(1 - cos(pi (r-1)/(R-1)))/2, lam 0.5, base step weights
  Tuned constant  : fixed-budget training (uniform weights, lam 0.5) at eps in {EPS_MIN, 0.5, EPS_MAX}, the eps chosen by
                    the checkpoint criterion (robust validation RMSE at eps 0.5), i.e. without test data
and evaluates {undefended, HAD-Stack, linear ramp, cosine ramp, tuned constant} jointly with the worst-case ensemble
(white-box PGD-100x5, two losses, plus transfer from every other model), with paired block-bootstrap intervals.
Usage: python run_schedules.py --split fixed|fold1..fold5 [--seed 1] [--out results/schedules]"""
import argparse, json, math, os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import hadstack_thlstm as H

REAL = H.Controller
EPS_MIN = REAL(18).step(+1.0)[0]


class Schedule:
    """Stand-in for hadstack_thlstm.Controller: same interface, budget from a fixed schedule over rounds."""
    kind, rounds = "linear", 30

    def __init__(self, d):
        self._c = REAL(d); self.base_w = self._c.base_w; self.k = 0

    def step(self, s):
        r = max(self.k, 1); self.k += 1; x = (r - 1) / max(self.rounds - 1, 1)
        f = x if self.kind == "linear" else (1 - math.cos(math.pi * x)) / 2
        eps = EPS_MIN + (H.EPS_MAX - EPS_MIN) * f
        return eps, 0.5, self.base_w, dict(schedule=self.kind)


def data(split):
    if split == "fixed": return H.cell13_data()
    k = int(split[4:]) - 1; N = H.cell13_data()["n_rows"]
    D = H.cell13_data(split_at=int((0.5 + 0.1 * k) * N)); nte = int(0.1 * N) + D["window"]
    D["te"] = {kk: v[:nte - D["window"]] for kk, v in D["te"].items()}
    return D


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", required=True); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rounds", type=int, default=30); p.add_argument("--out", default="results/schedules")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    D = data(a.split); base = H.cell13_train(D, seed=0); tag = f"{a.split}/s{a.seed}"
    had, hlog, hbr = H.game(base, D, a.seed, rounds=a.rounds, controller=True, tag=tag + " HAD", verbose=False)
    models, logs = {"Two-Head LSTM (undefended)": base, "HAD-Stack": had}, {"HAD-Stack": (hlog, hbr)}
    real = REAL; print(f"{tag} HAD done, best round {hbr}", flush=True)
    for kind in ("linear", "cosine"):
        Schedule.kind, Schedule.rounds = kind, a.rounds; H.Controller = Schedule
        try:
            m, lg, br = H.game(base, D, a.seed, rounds=a.rounds, controller=True, tag=f"{tag} {kind}", verbose=False)
        finally:
            H.Controller = real
        print(f"{tag} {kind} ramp done, best round {br}", flush=True); models[f"{kind.capitalize()} ramp"] = m; logs[f"{kind.capitalize()} ramp"] = (lg, br)
    cands = {}
    for e in (EPS_MIN, 0.5, H.EPS_MAX):
        m, lg, br = H.game(base, D, a.seed, rounds=a.rounds, controller=False, fixed=(e, 0.5), tag=f"{tag} const {e:.4f}", verbose=False)
        cands[e] = (m, lg, br, lg[br]["val_rrmse"] if br > 0 else lg[1]["val_rrmse"]); print(f"{tag} const {e:.4f} done, val rRMSE {cands[e][3]:.4f}", flush=True)
    e_best = min(cands, key=lambda e: cands[e][3])
    models["Tuned constant"] = cands[e_best][0]; logs["Tuned constant"] = (cands[e_best][1], cands[e_best][2])
    res, rows = H.evaluate(models, D)
    ci = {k: H.bootstrap(rows[k], rows["HAD-Stack"], (0.5, 1.0)) for k in models if k not in ("HAD-Stack", "Two-Head LSTM (undefended)")}
    ci["HAD-Stack vs undefended"] = H.bootstrap(rows["HAD-Stack"], rows["Two-Head LSTM (undefended)"], (0.5, 1.0))
    out = dict(split=a.split, seed=a.seed, eps_min=EPS_MIN, eps_max=H.EPS_MAX, tuned_eps=e_best,
               tuned_val={f"{e:.4f}": c[3] for e, c in cands.items()}, metrics=res, ci_vs_had=ci,
               rounds={k: (len(v[0]) - 1, v[1]) for k, v in logs.items()},
               eps_traj={k: [x["eps"] for x in v[0]] for k, v in logs.items()}, minutes=(time.time() - t0) / 60)
    json.dump(out, open(f"{a.out}/{a.split}_s{a.seed}.json", "w"), indent=1, default=float)
    for k, v in res.items():
        print(f"{tag} {k:28s} RMSE {v['RMSE']:.3f} | wcRMSE {v['wcRMSE@0.5']:.3f}/{v['wcRMSE@1.0']:.3f} | broken {v['broken@0.5']:.1f}/{v['broken@1.0']:.1f}", flush=True)
    print(f"{tag} tuned eps {e_best:.4f}; minutes {out['minutes']:.1f}", flush=True)


if __name__ == "__main__":
    main()
