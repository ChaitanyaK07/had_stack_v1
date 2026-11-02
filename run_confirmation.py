"""
CONFIRMATION RUN: HAD-Stack v3 vs matched PGD-AT (eps = 0.6586)
===============================================================
  python run_confirmation.py --part seeds     # 3 seeds on the fixed split  (~6 runs)
  python run_confirmation.py --part cv        # 5-fold stratified CV        (~15 runs)
  python run_confirmation.py --part all       # both, then the summary
  python run_confirmation.py --part summary   # re-print the verdict from saved results
  add --quick for a smoke test (1 round, weak attack) — NOT for reporting.

Protocol (identical to the reported single-split results):
  * training: had_stack_v2.train (v3 / pgdat066), 30 rounds, robust-AUC selection
  * evaluation: worst case over white-box PGD-100 x 5 restarts (BCE and margin loss)
    and margin-PGD-100 transfer from the other models in the same seed/fold
  * robust recall = share of ALL fraud rows still detected under that worst case
Seeds vary the adversarial stage (batch order, attack starts, dropout); the baseline
checkpoint is shared. CV varies the data: each fold refits the scaler, retrains the
baseline, then trains both defences from that fold's baseline.

Decision rule (fixed before running):
  v3 is "better than standard adversarial training" only if its robust recall is
  higher than PGD-AT's at BOTH eps = 0.5 and eps = 1.0 in all 3 seeds AND in at least
  4 of 5 folds. Otherwise the two are reported as comparable.
Results are appended to logs/confirmation_results.jsonl after EVERY model, and a run
that is interrupted resumes where it stopped.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # force CPU: deterministic, and the pipeline
                                                 # keeps the model on CPU (avoids a device mismatch)
import argparse, json, shutil
import numpy as np, torch
import baseline_model as bm, eval_extras as ex, had_stack_v2 as hs

RES = "logs/confirmation_results.jsonl"
SEEDS, FOLDS, EPS = (1, 2, 3), 5, (0.5, 1.0)
SMOKE_ONE_FOLD = False
os.makedirs("logs", exist_ok=True); os.makedirs("models", exist_ok=True)


def done():
    if not os.path.exists(RES): return set()
    return {(r["exp"], r["unit"], r["model"]) for r in map(json.loads, open(RES))}


def pgd(m, x, eps, loss, K, R, seed=0):
    m.eval(); a = 2.5 * eps / K; best = x.clone(); bz = m(x).detach()
    for r in range(R):
        g = torch.Generator().manual_seed(seed + r)
        eta = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
        for _ in range(K):
            eta.requires_grad_(True); z = m(x + eta)
            L = (torch.nn.functional.softplus(-z) if loss == "bce" else -z).sum()
            gr, = torch.autograd.grad(L, eta)
            with torch.no_grad(): eta = torch.clamp(eta + a * gr.sign(), -eps, eps)
        with torch.no_grad():
            z = m(x + eta); better = z < bz; best[better] = (x + eta)[better]; bz = torch.minimum(bz, z)
    assert (best - x).abs().max() <= eps + 1e-5
    return best


def worst_case(ckpts, quick):
    """ckpts: {name: path}. Same protocol as strong_eval.py. Returns list of rows."""
    K, R = (3, 1) if quick else (100, 5)
    X, y = ex.load_split("test"); Xf = torch.tensor(X[y == 1]); n_fraud = int((y == 1).sum())
    M = {k: ex.load_transformer(v) for k, v in ckpts.items()}
    crafted = {(k, e): pgd(m, Xf, e, "margin", K, 1) for k, m in M.items() for e in EPS}
    rows = []
    for k, m in M.items():
        pc = ex.probs(m, X); c = ex.metrics(y, pc); det = torch.tensor(pc[y == 1] >= 0.5)
        row = dict(model=k, clean_auc=c["auc"], f1=c["f1"], fp=c["fp"], fn=c["fn"], cost=c["cost"])
        for e in EPS:
            adv = {l: pgd(m, Xf, e, l, K, R) for l in ("bce", "margin")}
            with torch.no_grad():
                broken = torch.zeros_like(det)
                for l in adv: broken |= (m(adv[l]) < 0) & det
                for s in M:
                    if s != k: broken |= (m(crafted[(s, e)]) < 0) & det
            row[f"worst@{e}"] = 100 * broken.sum().item() / max(det.sum().item(), 1)
            row[f"robust_recall@{e}"] = 100 * (det & ~broken).sum().item() / n_fraud
        rows.append(row)
    return rows


def record(exp, unit, rows):
    with open(RES, "a") as f:
        for r in rows:
            f.write(json.dumps(dict(exp=exp, unit=unit, **r), default=float) + "\n")
    for r in rows:
        print(f"[{exp} {unit}] {r['model']:9s} clean cost {r['cost']:6d} | robust recall "
              f"eps0.5 {r['robust_recall@0.5']:5.1f}%  eps1.0 {r['robust_recall@1.0']:5.1f}%", flush=True)


def run_seeds(epochs, quick):
    fin = done()
    for sd in SEEDS:
        if all(("seeds", sd, m) in fin for m in ("v3", "pgdat066")): continue
        ck = {}
        for meth in ("v3", "pgdat066"):
            hs.train(meth, epochs=epochs, seed=sd, base_ckpt="models/baseline_best.pt",
                     out_tag=f"{meth}_seed{sd}")
            ck[meth] = f"models/v2_{meth}_seed{sd}_best.pt"
        ck["baseline"] = "models/baseline_best.pt"
        record("seeds", sd, worst_case(ck, quick))


def make_fold(k, tr, te, X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = f"cv_folds/fold{k}"; os.makedirs(d, exist_ok=True)
    Xtr, Xva, ytr, yva = train_test_split(X[tr], y[tr], test_size=0.15, random_state=42, stratify=y[tr])
    sc = StandardScaler().fit(Xtr)
    for n, a in [("X_train", sc.transform(Xtr)), ("X_val", sc.transform(Xva)), ("X_test", sc.transform(X[te])),
                 ("y_train", ytr), ("y_val", yva), ("y_test", y[te])]:
        np.save(f"{d}/{n}.npy", a.astype(np.float32 if n[0] == "X" else np.int64))
    return d


def run_cv(epochs, quick):
    import data_processing as dp
    from sklearn.model_selection import StratifiedKFold
    df = dp.engineer_features(dp.select_and_clean(dp.load_data(dp.DATA_PATH)))
    feats = [c for c in df.columns if c != dp.TARGET_COL]
    X = df[feats].values.astype(np.float32); y = df[dp.TARGET_COL].values.astype(np.int64)
    fin = done()
    orig = (bm.OUTPUT_DIR, bm.MODEL_DIR, ex.DATA)
    try:
        for k, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(X, y)):
            if all(("cv", k, m) in fin for m in ("baseline", "v3", "pgdat066")): continue
            d = make_fold(k, tr, te, X, y)
            bm.OUTPUT_DIR, ex.DATA = d, d
            bm.MODEL_DIR = f"models_cv/fold{k}"; os.makedirs(bm.MODEL_DIR, exist_ok=True)
            base = f"{bm.MODEL_DIR}/baseline_best.pt"
            if not os.path.exists(base):
                if quick: bm.HP["epochs"] = 1
                bm.train_baseline()
            ck = {"baseline": base}
            for meth in ("v3", "pgdat066"):
                hs.train(meth, epochs=epochs, seed=42, base_ckpt=base, out_tag=f"{meth}_fold{k}")
                ck[meth] = f"models/v2_{meth}_fold{k}_best.pt"
            record("cv", k, worst_case(ck, quick))
            if SMOKE_ONE_FOLD: break
    finally:
        bm.OUTPUT_DIR, bm.MODEL_DIR, ex.DATA = orig


LAM_RES = "logs/lambda_control_results.jsonl"


def lam_done():
    if not os.path.exists(LAM_RES): return set()
    return {(r["exp"], r["unit"]) for r in map(json.loads, open(LAM_RES))}


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — the lambda control reuses the checkpoints "
                                f"from the confirmation run; run --part all first.")
    return path


def run_lambda_control(epochs, quick):
    """PGD-AT with eps = lambda = 0.6586 (v3's final values). All four models of each
    seed/fold are re-evaluated together so every model faces the same transfer sources."""
    global LAM_RES
    if quick: LAM_RES = "logs/SMOKE_TEST_lambda.jsonl"
    fin = lam_done()
    for sd in SEEDS:
        if ("seeds", sd) in fin: continue
        tag = f"pgdat066lam_seed{sd}"
        if not os.path.exists(f"models/v2_{tag}_best.pt"):
            hs.train("pgdat066lam", epochs=epochs, seed=sd, base_ckpt="models/baseline_best.pt", out_tag=tag)
        ck = {"baseline": _need("models/baseline_best.pt"),
              "v3": _need(f"models/v2_v3_seed{sd}_best.pt"),
              "pgdat066": _need(f"models/v2_pgdat066_seed{sd}_best.pt"),
              "pgdat066lam": f"models/v2_{tag}_best.pt"}
        _record_lam("seeds", sd, worst_case(ck, quick))
    import data_processing as dp
    from sklearn.model_selection import StratifiedKFold
    df = dp.engineer_features(dp.select_and_clean(dp.load_data(dp.DATA_PATH)))
    feats = [c for c in df.columns if c != dp.TARGET_COL]
    X = df[feats].values.astype(np.float32); y = df[dp.TARGET_COL].values.astype(np.int64)
    orig = (bm.OUTPUT_DIR, bm.MODEL_DIR, ex.DATA)
    try:
        for k, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(X, y)):
            if ("cv", k) in fin: continue
            d = make_fold(k, tr, te, X, y)                     # deterministic: same split/scaler
            bm.OUTPUT_DIR, ex.DATA = d, d
            base = _need(f"models_cv/fold{k}/baseline_best.pt")
            tag = f"pgdat066lam_fold{k}"
            if not os.path.exists(f"models/v2_{tag}_best.pt"):
                hs.train("pgdat066lam", epochs=epochs, seed=42, base_ckpt=base, out_tag=tag)
            ck = {"baseline": base, "v3": _need(f"models/v2_v3_fold{k}_best.pt"),
                  "pgdat066": _need(f"models/v2_pgdat066_fold{k}_best.pt"),
                  "pgdat066lam": f"models/v2_{tag}_best.pt"}
            _record_lam("cv", k, worst_case(ck, quick))
            if SMOKE_ONE_FOLD: break
    finally:
        bm.OUTPUT_DIR, bm.MODEL_DIR, ex.DATA = orig
    lambda_summary()


def _record_lam(exp, unit, rows):
    with open(LAM_RES, "a") as f:
        for r in rows: f.write(json.dumps(dict(exp=exp, unit=unit, **r), default=float) + "\n")
    for r in rows:
        print(f"[lambda-control {exp} {unit}] {r['model']:12s} cost {r['cost']:6d} | robust recall "
              f"eps0.5 {r['robust_recall@0.5']:5.1f}%  eps1.0 {r['robust_recall@1.0']:5.1f}%", flush=True)


def lambda_summary():
    if not os.path.exists(LAM_RES): print("no lambda-control results yet"); return
    R = {(r["exp"], r["unit"], r["model"]): r for r in map(json.loads, open(LAM_RES))}
    ok_all = True
    for exp, units, need in [("seeds", SEEDS, len(SEEDS)), ("cv", range(FOLDS), 4)]:
        have = [u for u in units if (exp, u, "v3") in R and (exp, u, "pgdat066lam") in R]
        print(f"\n== lambda control: {exp} ({len(have)} complete) ==")
        wins = 0
        for u in have:
            a, b, c = R[(exp, u, "v3")], R[(exp, u, "pgdat066lam")], R[(exp, u, "pgdat066")]
            d1 = a["robust_recall@1.0"] - b["robust_recall@1.0"]; d5 = a["robust_recall@0.5"] - b["robust_recall@0.5"]
            wins += d1 > 0
            print(f"  {exp} {u}: v3 - PGD-AT(lam=0.66) robust recall eps1.0 {d1:+5.1f} pts, eps0.5 {d5:+5.1f} pts"
                  f" | clean cost v3 {a['cost']} / lam-matched {b['cost']} / lam=0.5 {c['cost']}")
        for m in ("baseline", "v3", "pgdat066", "pgdat066lam"):
            v = [R[(exp, u, m)] for u in have]
            if v:
                f = lambda k: f"{np.mean([x[k] for x in v]):6.1f} ± {np.std([x[k] for x in v]):4.1f}"
                print(f"  mean {m:12s} robust recall eps0.5 {f('robust_recall@0.5')} eps1.0 {f('robust_recall@1.0')}"
                      f" | clean cost {f('cost')}")
        ok = len(have) == len(list(units)) and wins >= need
        print(f"  rule: v3 must beat the lambda-matched control at eps=1.0 in >= {need} of "
              f"{len(list(units))} -> {'PASSED' if ok else 'NOT passed'} ({wins} wins)")
        ok_all &= ok
    print("\nLAMBDA-CONTROL VERDICT:", "the eps=1.0 edge is credited to the Hamiltonian curriculum" if ok_all
          else "the eps=1.0 edge is NOT credited to the Hamiltonians (explained by lambda / noise)")


def summary():
    if not os.path.exists(RES): print("no results yet"); return
    R = {}
    for r in map(json.loads, open(RES)): R[(r["exp"], r["unit"], r["model"])] = r
    verdict_ok = True
    for exp, units, need in [("seeds", SEEDS, len(SEEDS)), ("cv", range(FOLDS), 4)]:
        have = [u for u in units if ("%s" % exp, u, "v3") in R and (exp, u, "pgdat066") in R]
        if not have: print(f"{exp}: no complete units yet"); verdict_ok = False; continue
        print(f"\n== {exp} ({len(have)} complete) ==")
        wins = 0
        for u in have:
            a, b = R[(exp, u, "v3")], R[(exp, u, "pgdat066")]
            d5 = a["robust_recall@0.5"] - b["robust_recall@0.5"]; d1 = a["robust_recall@1.0"] - b["robust_recall@1.0"]
            w = d5 > 0 and d1 > 0; wins += w
            print(f"  {exp} {u}: v3 - PGD-AT robust recall  eps0.5 {d5:+5.1f} pts  eps1.0 {d1:+5.1f} pts  "
                  f"| clean cost v3 {a['cost']} vs {b['cost']}  -> {'v3 wins' if w else 'no win'}")
        for m in ("baseline", "v3", "pgdat066"):
            v = [R[(exp, u, m)] for u in have if (exp, u, m) in R]
            if v:
                f = lambda key: f"{np.mean([x[key] for x in v]):5.1f} ± {np.std([x[key] for x in v]):4.1f}"
                print(f"  mean {m:9s} robust recall eps0.5 {f('robust_recall@0.5')}  eps1.0 "
                      f"{f('robust_recall@1.0')} | clean cost {f('cost')}")
        ok = len(have) == len(list(units)) and wins >= need
        print(f"  rule: v3 must win at both eps in >= {need} of {len(list(units))} -> "
              f"{'PASSED' if ok else 'NOT passed'} ({wins} wins)")
        verdict_ok &= ok
    print("\nVERDICT:", "v3 is better than standard adversarial training (claimable)" if verdict_ok
          else "v3 and standard adversarial training are comparable (do not claim 'better')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["seeds", "cv", "all", "summary", "lambda", "lambda_summary"], default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ep = 1 if a.quick else 30
    if a.quick:   # smoke test: separate results file, 1 seed, 1 fold
        RES = "logs/SMOKE_TEST_results.jsonl"; SEEDS = (1,); SMOKE_ONE_FOLD = True
    if a.part == "lambda": run_lambda_control(ep, a.quick); raise SystemExit
    if a.part == "lambda_summary": lambda_summary(); raise SystemExit
    if a.part in ("seeds", "all"): run_seeds(ep, a.quick)
    if a.part in ("cv", "all"): run_cv(ep, a.quick)
    summary()
