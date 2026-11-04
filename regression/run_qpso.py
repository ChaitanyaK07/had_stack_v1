#!/usr/bin/env python3
"""
Study A on the bond data with a derivative-free follower (paper, Section VII-D, "QPSO follower").

For every seed / fold of hadstack_thlstm.py this script
  1. re-runs the original pipeline in the original order (undefended Two-Head LSTM, HAD-Stack with the PGD-10
     follower, fixed-budget AT, worst-case PGD ensemble) and checks it reproduces results_seeds/folds.json;
  2. runs the same Hamiltonian-controlled game with the follower replaced by QPSO (Study B's Alg. 10, 20 particles
     x 15 iterations) and by the better of PGD-10 and QPSO per window ("dual");
  3. evaluates the two new forecasters with the same PGD ensemble as HAD-Stack (white-box PGD-100x5 with the
     squared-error and margin losses, transfer margin-PGD from the undefended and fixed-budget forecasters);
  4. attacks all five forecasters at test time with a larger QPSO (50 particles x 100 iterations, both losses),
     which uses no gradients, and reports it alone and joined to the PGD ensemble.
Controller, feedback law, step weights, optimiser, schedule, patience and budgets are unchanged.
The order parameter of the controller is set by HCS_ORDER (see hadstack_thlstm.py).
"""
import argparse, json, os, sys, time
import numpy as np
from hadstack_thlstm import (cell13_data, cell13_train, game, evaluate, bootstrap, make_pgd, make_qpso, predict,
                             degraded, check_table_vi, FB_LABEL, HCS_ORDER, EPS_MAX)

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
U, FB, H, HQ, HD = "Two-Head LSTM (undefended)", FB_LABEL, "HAD-Stack", "HAD-Stack (QPSO)", "HAD-Stack (PGD+QPSO)"
BUD = (0.5, 1.0)


def metrics_from(e0, finals, true, prev, tau):
    """Broken rate and worst-case errors of one forecaster for a list of attacked forecasts (Alg. 7, lines 8-10)."""
    worst = np.max([np.abs(f - true) for f in finals], 0)
    brk = np.zeros(len(e0), bool)
    for f in finals: brk |= degraded(e0, f - true, tau)
    return brk, worst


def pgd_ensemble(m, D, sources, steps, restarts):
    """Alg. 7 for one forecaster: white-box SE and margin PGD + transfer margin-PGD from `sources`."""
    te, tau, mask = D["te"], D["sd_delta"], D["market"]
    pg = make_pgd(m); f0 = predict(m, te["X"], te["prev"]); e0 = f0 - te["true"]
    out = dict(MAE=float(np.abs(e0).mean()), RMSE=float(np.sqrt((e0 ** 2).mean())))
    nz = np.sign(te["true"] - te["prev"]) != 0
    out["DA"] = float(100 * ((np.sign(f0 - te["prev"]) == np.sign(te["true"] - te["prev"])) & nz).sum() / nz.sum())
    rows = dict(e0=e0)
    for eps in BUD:
        finals = [predict(m, te["X"] + pg(te["X"], te["true"], te["prev"], eps, None, mask, steps, restarts, k), te["prev"]) for k in ("se", "margin")]
        for s in sources:
            finals.append(predict(m, te["X"] + make_pgd(s)(te["X"], te["true"], te["prev"], eps, None, mask, steps, 1, "margin"), te["prev"]))
        brk, worst = metrics_from(e0, finals, te["true"], te["prev"], tau)
        out[f"broken@{eps}"] = float(100 * brk.mean()); out[f"robust@{eps}"] = float(100 * (~brk).mean())
        out[f"wcMAE@{eps}"] = float(worst.mean()); out[f"wcRMSE@{eps}"] = float(np.sqrt((worst ** 2).mean()))
        rows[f"brk@{eps}"] = brk; rows[f"worst@{eps}"] = worst
    return out, rows


def qpso_attack(m, D, rows, n_p, n_it):
    """Test-time QPSO (both losses) alone, and joined to the PGD ensemble whose per-day results are in `rows`."""
    te, tau, mask = D["te"], D["sd_delta"], D["market"]
    qp = make_qpso(m); e0 = rows["e0"]; out, qrows = {}, dict(e0=e0)
    for eps in BUD:
        finals = [predict(m, te["X"] + qp(te["X"], te["true"], te["prev"], eps, None, mask, n_p, n_it, k), te["prev"]) for k in ("se", "margin")]
        brk, worst = metrics_from(e0, finals, te["true"], te["prev"], tau)
        out[f"q_broken@{eps}"] = float(100 * brk.mean()); out[f"q_wcRMSE@{eps}"] = float(np.sqrt((worst ** 2).mean()))
        out[f"q_wcMAE@{eps}"] = float(worst.mean())
        ba, wa = brk | rows[f"brk@{eps}"], np.maximum(worst, rows[f"worst@{eps}"])
        out[f"all_broken@{eps}"] = float(100 * ba.mean()); out[f"all_wcRMSE@{eps}"] = float(np.sqrt((wa ** 2).mean()))
        out[f"all_wcMAE@{eps}"] = float(wa.mean())
        qrows[f"brk@{eps}"] = ba; qrows[f"worst@{eps}"] = wa
    return out, qrows


def run(D, tag, seeds, args, ref):
    print(f"\n=== {tag}: fit {len(D['fit']['X'])} val {len(D['val']['X'])} test {len(D['te']['X'])}, tau = {D['sd_delta']:.4f}", flush=True)
    base = cell13_train(D, seed=0); out = []
    for sd in seeds:
        t0 = time.time()
        # 1. original pipeline, original order (reproduction check)
        had, hlog, hbr = game(base, D, sd, rounds=args.rounds, controller=True, verbose=False)
        fb, flog, fbr = game(base, D, sd, rounds=args.rounds, controller=False, verbose=False)
        res, rows = evaluate({U: base, FB: fb, H: had}, D, steps=args.eval_steps, restarts=args.eval_restarts)
        r0 = next((r for r in ref if r["split"] == tag and r["seed"] == sd), None)
        dev = None
        if r0 is not None:
            dev = max(abs(res[k][q] - r0["metrics"][k][q]) for k in res for q in res[k])
            print(f"{tag}/s{sd} reproduction of the reported run: max |difference| over all metrics = {dev:.2e}", flush=True)
        # 2. the same game with a derivative-free follower
        hq, qlog, qbr = game(base, D, sd, rounds=args.rounds, controller=True, follower="qpso", tag=f"{tag}/s{sd} HAD-QPSO", verbose=args.verbose)
        hd, dlog, dbr = game(base, D, sd, rounds=args.rounds, controller=True, follower="dual", tag=f"{tag}/s{sd} HAD-dual", verbose=args.verbose)
        # 3. same PGD ensemble as HAD-Stack for the new forecasters
        for name, m in ((HQ, hq), (HD, hd)):
            res[name], rows[name] = pgd_ensemble(m, D, [base, fb], args.eval_steps, args.eval_restarts)
        # 4. test-time QPSO on all five
        models = {U: base, FB: fb, H: had, HQ: hq, HD: hd}; qrows = {}
        for name, m in models.items():
            q, qrows[name] = qpso_attack(m, D, rows[name], args.q_particles, args.q_iters); res[name].update(q)
        ci = {name: bootstrap(rows[name], rows[U], BUD) for name in (H, HQ, HD)}
        ci_all = {name: bootstrap(qrows[name], qrows[U], BUD) for name in (H, HQ, HD)}
        out.append(dict(split=tag, seed=sd, order=HCS_ORDER, eps_max=EPS_MAX, best_round={H: hbr, FB: fbr, HQ: qbr, HD: dbr},
                        metrics=res, ci=ci, ci_all=ci_all, had_log=hlog, qpso_log=qlog, dual_log=dlog, reproduced_max_diff=dev))
        for k, v in res.items():
            print(f"{tag}/s{sd} {k:28s} MAE {v['MAE']:.3f} RMSE {v['RMSE']:.3f} | PGD-ens broken {v['broken@0.5']:.1f}/{v['broken@1.0']:.1f} "
                  f"wcRMSE {v['wcRMSE@0.5']:.3f}/{v['wcRMSE@1.0']:.3f} | QPSO broken {v['q_broken@0.5']:.1f}/{v['q_broken@1.0']:.1f} "
                  f"wcRMSE {v['q_wcRMSE@0.5']:.3f}/{v['q_wcRMSE@1.0']:.3f} | all wcRMSE {v['all_wcRMSE@0.5']:.3f}/{v['all_wcRMSE@1.0']:.3f}", flush=True)
        print(f"{tag}/s{sd} done in {time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(args.partial, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seeds", "folds"], required=True)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-restarts", type=int, default=5)
    p.add_argument("--q-particles", type=int, default=50)
    p.add_argument("--q-iters", type=int, default=100)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args()
    check_table_vi()
    ref = json.load(open(os.path.join(RES, f"results_{args.mode}.json")))
    outf = os.path.join(RES, f"qpso_{args.mode}" + ("" if args.mode == "seeds" else "_" + args.folds.replace(",", "")) + ".json")
    args.partial = outf + ".partial"
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.mode == "seeds":
        allout = run(cell13_data(), "fixed", seeds, args, ref)
    else:
        allout, N = [], cell13_data()["n_rows"]
        for k in [int(f) - 1 for f in args.folds.split(",")]:
            D = cell13_data(split_at=int((0.5 + 0.1 * k) * N)); nte = int(0.1 * N) + D["window"]
            D["te"] = {kk: v[:nte - D["window"]] for kk, v in D["te"].items()}
            D["test_span"] = (str(D["te"]["dates"][0])[:10], str(D["te"]["dates"][-1])[:10])
            args.partial = outf + f".fold{k+1}.partial"
            allout += run(D, f"fold{k+1}", seeds[:1], args, ref)
    json.dump(allout, open(outf, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
