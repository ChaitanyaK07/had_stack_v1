"""
FINAL EXPERIMENTS — per-transaction outputs for bootstrap CIs, curves and the XGBoost ablation
============================================================================================
  python run_final_experiments.py            # all 3 seeds + 5 folds (resumable)
  python run_final_experiments.py --quick    # smoke test (1 seed, 1 fold, weak attack) -> results_quick/
Reuses the checkpoints of run_confirmation.py (--part all and --part lambda). Nothing is retrained
except the two XGBoost variants of each seed/fold.

For every seed/fold and every model it saves (results_final/<exp>_<unit>.npz):
  * clean fraud probabilities for every test row
  * for each eps in EPS: which DETECTED fraud rows are broken by (a) white-box PGD-100x5 with BCE and
    margin losses, (b) margin-PGD-100 transfer from each other Transformer of the same unit
Worst-case protocol identical to strong_eval.py / run_confirmation.py (same seeds -> at eps 0.5 and 1.0
the Transformer results must reproduce lambda_control_results.jsonl; analyze_final.py checks this).

XGBoost ablation (gradient attacks do not apply to trees, so robustness is transfer-only):
  * xgb_plain  : XGBoost on the unit's training split (paper configuration)
  * xgb_advaug : same, with each training fraud row duplicated as an L-inf PGD-10 perturbation
                 (eps = 0.6586, margin loss) crafted on the unit's baseline Transformer
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json
import numpy as np, torch
import baseline_model as bm, eval_extras as ex, run_confirmation as rc

EPS = (0.25, 0.5, 0.75, 1.0)
TF = ("baseline", "v3", "pgdat066", "pgdat066lam")


def ckpts(exp, unit):
    if exp == "seeds":
        return {"baseline": "models/baseline_best.pt", "v3": f"models/v2_v3_seed{unit}_best.pt",
                "pgdat066": f"models/v2_pgdat066_seed{unit}_best.pt",
                "pgdat066lam": f"models/v2_pgdat066lam_seed{unit}_best.pt"}
    return {"baseline": f"models_cv/fold{unit}/baseline_best.pt", "v3": f"models/v2_v3_fold{unit}_best.pt",
            "pgdat066": f"models/v2_pgdat066_fold{unit}_best.pt",
            "pgdat066lam": f"models/v2_pgdat066lam_fold{unit}_best.pt"}


def xgb_models(Xtr, ytr, base, K):
    plain = ex.xgb_model(Xtr, ytr)
    Xf = torch.tensor(Xtr[ytr == 1])
    adv = rc.pgd(base, Xf, 0.6586, "margin", K, 1).numpy()
    aug = ex.xgb_model(np.concatenate([Xtr, adv]), np.concatenate([ytr, np.ones(len(adv), np.int64)]))
    return {"xgb_plain": plain, "xgb_advaug": aug}


def run_unit(exp, unit, out, quick):
    K, R = (3, 1) if quick else (100, 5)
    X, y = ex.load_split("test"); Xtr, ytr = ex.load_split("train")
    Xf = torch.tensor(X[y == 1])
    paths = ckpts(exp, unit)
    for p in paths.values():
        if not os.path.exists(p): raise FileNotFoundError(p)
    M = {k: ex.load_transformer(v) for k, v in paths.items()}
    save = {"y": y}
    crafted = {}
    for e in EPS:
        for k, m in M.items():
            crafted[(k, e)] = rc.pgd(m, Xf, e, "margin", K, 1)
    for k, m in M.items():
        save[f"{k}__clean"] = ex.probs(m, X)
        det = torch.tensor(save[f"{k}__clean"][y == 1] >= 0.5)
        for e in EPS:
            wb = torch.zeros_like(det)
            for loss in ("bce", "margin"):
                a = rc.pgd(m, Xf, e, loss, K, R)
                with torch.no_grad(): wb |= (m(a) < 0) & det
            tr = torch.zeros_like(det)
            with torch.no_grad():
                for s in M:
                    if s != k: tr |= (m(crafted[(s, e)]) < 0) & det
            save[f"{k}__wb__{e}"] = wb.numpy(); save[f"{k}__tr__{e}"] = tr.numpy()
    for k, xg in xgb_models(Xtr, ytr, M["baseline"], 3 if quick else 10).items():
        p = xg.predict_proba(X)[:, 1]; save[f"{k}__clean"] = p
        det = p[y == 1] >= 0.5
        for e in EPS:
            tr = np.zeros_like(det)
            for s in M:
                tr |= (xg.predict_proba(crafted[(s, e)].numpy())[:, 1] < 0.5) & det
            save[f"{k}__tr__{e}"] = tr
    np.savez_compressed(out, **save)
    print(f"[done] {exp} {unit} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    D = "results_quick" if a.quick else "results_final"; os.makedirs(D, exist_ok=True)
    seeds, folds = ((1,), (0,)) if a.quick else ((1, 2, 3), range(5))
    for s in seeds:
        out = f"{D}/seeds_{s}.npz"
        if not os.path.exists(out): run_unit("seeds", s, out, a.quick)
    import data_processing as dp
    from sklearn.model_selection import StratifiedKFold
    df = dp.engineer_features(dp.select_and_clean(dp.load_data(dp.DATA_PATH)))
    feats = [c for c in df.columns if c != dp.TARGET_COL]
    Xa = df[feats].values.astype(np.float32); ya = df[dp.TARGET_COL].values.astype(np.int64)
    orig = ex.DATA
    try:
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xa, ya)):
            if k not in folds: continue
            out = f"{D}/cv_{k}.npz"
            if os.path.exists(out): continue
            ex.DATA = rc.make_fold(k, tr, te, Xa, ya)
            run_unit("cv", k, out, a.quick)
    finally:
        ex.DATA = orig
    print("ALL DONE — now run: python analyze_final.py" + (" --quick" if a.quick else ""))
