"""
=============================================================================
FILE 4: SUPPLEMENTARY EVALUATION — every table/figure not produced by
adversarial_game.py --mode suite
=============================================================================
Run AFTER:  python data_processing.py
            python baseline_model.py
            python adversarial_game.py --mode suite

  python eval_extras.py --part all        (or: xgb | bootstrap | noise |
                                               transfer | curves | sensitivity
                                               | cv)

Outputs (logs/):
  extras_reference.csv   XGBoost + out-of-legit-range rule on the test set
  bootstrap_ci.csv       95% bootstrap CIs (1,000 resamples, seed 42)
  noise_corruption.csv   F1 under random Gaussian feature noise on a
                         fraction of TEST rows (NOT training-set poisoning)
  transfer.csv           PGD crafted on the baseline Transformer, evaluated
                         on XGBoost
  sensitivity.csv        closed-form eps / lambda under +-20% gamma_1/gamma_2
  cv_results.csv         5-fold stratified CV (preprocessing refit per fold)
  fig2_curves.png        PR / cost / lift curves under a PGD eps sweep
=============================================================================
"""
import argparse, csv, os, pickle
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import baseline_model as bm
import adversarial_game as ag

LOG, MOD, DATA = "logs", "models", "processed_data"
EPS_SWEEP = (0.0, 0.01, 0.03, 0.05, 0.10, 0.15)
os.makedirs(LOG, exist_ok=True)


# ───────────────────────── helpers ─────────────────────────
def load_split(name):
    return (np.load(f"{DATA}/X_{name}.npy").astype(np.float32),
            np.load(f"{DATA}/y_{name}.npy").astype(np.int64))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(auc=roc_auc_score(y, p), f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                specificity=tn / max(tn + fp, 1),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                cost=int(100 * fn + fp))


def load_transformer(path, n_features=18):
    m = bm.TabularTransformer(n_features, d_model=bm.HP['d_model'],
                              n_heads=bm.HP['n_heads'], n_layers=bm.HP['n_layers'],
                              d_ffn=bm.HP['d_ffn'], dropout=bm.HP['dropout'])
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def probs(model, X):
    return torch.sigmoid(model(torch.tensor(X))).numpy()


def pgd(model, X, y, eps):
    """Same attacker as the suite: base-parameter Hamiltonian step weights."""
    if eps == 0:
        return X.copy()
    att = ag.HamiltonianAttacker(ag.HamiltonianPerturbationBudget())
    out = []
    for i in range(0, len(X), 512):
        xb, yb = torch.tensor(X[i:i+512]), torch.tensor(y[i:i+512])
        out.append(att.evasion_attack(model, xb, yb, eps=eps).numpy())
    return np.concatenate(out)


def xgb_model(Xtr, ytr):
    from xgboost import XGBClassifier
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      gamma=0.1, reg_lambda=1.0, reg_alpha=0.1,
                      scale_pos_weight=spw, random_state=42, n_jobs=1,
                      tree_method="hist", eval_metric="auc")
    return m.fit(Xtr, ytr)


def model_paths():
    paths = {"baseline": f"{MOD}/baseline_best.pt"}
    for t in ["full", "pgdat_eps0.01", "pgdat_eps0.0397", "clean_finetune",
              "static_hamiltonian"]:
        p = f"{MOD}/adversarial_best_{t}.pt"
        if os.path.exists(p):
            paths[t] = p
    return paths


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[OK] {path}")


# ───────────────────────── parts ─────────────────────────
def part_reference():
    Xtr, ytr = load_split("train"); Xte, yte = load_split("test")
    rows = []
    xgb = xgb_model(Xtr, ytr)
    rows.append(dict(model="xgboost", **metrics(yte, xgb.predict_proba(Xte)[:, 1])))
    lo, hi = Xtr[ytr == 0].min(0), Xtr[ytr == 0].max(0)
    rule = ((Xte < lo) | (Xte > hi)).any(1).astype(float)
    r = metrics(yte, rule); r["auc"] = float("nan")   # a hard rule has no ranking
    rows.append(dict(model="out_of_legit_range_rule", **r))
    for name, p in model_paths().items():
        rows.append(dict(model=name, **metrics(yte, probs(load_transformer(p), Xte))))
    write_csv(f"{LOG}/extras_reference.csv", rows)
    pickle.dump(xgb, open(f"{MOD}/xgb.pkl", "wb"))


def part_bootstrap(n=1000):
    Xtr, ytr = load_split("train"); Xte, yte = load_split("test")
    preds = {k: probs(load_transformer(p), Xte) for k, p in model_paths().items()}
    preds["xgboost"] = pickle.load(open(f"{MOD}/xgb.pkl", "rb")).predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(yte), len(yte)) for _ in range(n)]
    rows = []
    for name, p in preds.items():
        pt = metrics(yte, p)
        boot = [metrics(yte[i], p[i]) for i in idx]
        for k in ["auc", "f1", "precision", "recall", "specificity", "fp", "fn"]:
            v = np.array([b[k] for b in boot])
            rows.append(dict(model=name, metric=k, point=pt[k],
                             lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    # paired bootstrap of the key difference: full - baseline
    if "full" in preds:
        for k in ["precision", "fp", "fn", "f1"]:
            d = np.array([metrics(yte[i], preds["full"][i])[k] -
                          metrics(yte[i], preds["baseline"][i])[k] for i in idx])
            rows.append(dict(model="full_minus_baseline(paired)", metric=k,
                             point=metrics(yte, preds["full"])[k] - metrics(yte, preds["baseline"])[k],
                             lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    write_csv(f"{LOG}/bootstrap_ci.csv", rows)


def part_noise(sigmas=(0.25, 0.5), rates=(0.0, 0.05, 0.10, 0.20)):
    Xte, yte = load_split("test")
    models = {k: load_transformer(p) for k, p in model_paths().items()}
    xgb = pickle.load(open(f"{MOD}/xgb.pkl", "rb"))
    rows = []
    for s in sigmas:
        for r in rates:
            rng = np.random.default_rng(42)
            Xn = Xte.copy()
            idx = rng.choice(len(Xte), int(r * len(Xte)), replace=False)
            Xn[idx] += rng.normal(0, s, size=(len(idx), Xte.shape[1])).astype(np.float32)
            row = dict(sigma=s, rate=r)
            for k, m in models.items():
                row[f"f1_{k}"] = metrics(yte, probs(m, Xn))["f1"]
            row["f1_xgboost"] = metrics(yte, xgb.predict_proba(Xn)[:, 1])["f1"]
            rows.append(row)
    write_csv(f"{LOG}/noise_corruption.csv", rows)


def part_transfer():
    Xte, yte = load_split("test")
    base = load_transformer(f"{MOD}/baseline_best.pt")
    xgb = pickle.load(open(f"{MOD}/xgb.pkl", "rb"))
    rows = []
    for e in EPS_SWEEP:
        Xa = pgd(base, Xte, yte, e)
        mt, mx = metrics(yte, probs(base, Xa)), metrics(yte, xgb.predict_proba(Xa)[:, 1])
        rows.append(dict(eps=e, surrogate_auc=mt["auc"], surrogate_recall=mt["recall"],
                         xgb_auc=mx["auc"], xgb_recall=mx["recall"], xgb_f1=mx["f1"]))
    write_csv(f"{LOG}/transfer.csv", rows)


def part_sensitivity():
    rows = []
    for which in ["gamma_1", "gamma_2"]:
        for f in (0.8, 1.0, 1.2):
            p1, p2 = dict(ag.PARAMS_H1), dict(ag.PARAMS_H2)
            if which == "gamma_1": p1["gamma_1"] *= f
            else:                  p2["gamma_2"] *= f
            e1 = ag.QuantumHamiltonianSolver(p1).solve()
            e2 = ag.H2Solver(p2).solve()
            e29 = ag.Eq29Solver(dict(ag.PARAMS_EQ29)).solve()
            eps = np.clip(((e1[1]-e1[0])*.5 * (e2[1]-e2[0])*.5 * (e29[1]-e29[0])*.1) ** (1/3), .02, .3)
            lam = 0.1 + 0.8 * np.clip(abs(e1[0]) / 0.5, 0, 1)
            rows.append(dict(param=which, factor=f, E0_H1=e1[0], gap_H2=e2.max()-e2.min(),
                             eps=eps, lam=lam))
    base = [r for r in rows if r["factor"] == 1.0][0]
    for r in rows:
        r["d_eps_pct"] = 100 * (r["eps"] / base["eps"] - 1)
        r["d_lam_pct"] = 100 * (r["lam"] / base["lam"] - 1)
    write_csv(f"{LOG}/sensitivity.csv", rows)


def part_curves():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Xte, yte = load_split("test")
    models = {k: load_transformer(p) for k, p in model_paths().items()
              if k in ("baseline", "full")}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = []
    for name, m in models.items():
        for e, ls in zip((0.0, 0.05, 0.15), ("-", "--", ":")):
            p = probs(m, pgd(m, Xte, yte, e))   # white-box: attack crafted on each model
            pr, rc, _ = precision_recall_curve(yte, p)
            ax[0].plot(rc, pr, ls, label=f"{name} ε={e}")
            thr = np.linspace(0.01, 0.99, 99)
            ax[1].plot(thr, [metrics(yte, p, t)["cost"] for t in thr], ls, label=f"{name} ε={e}")
            order = np.argsort(-p); k = np.arange(1, len(p)+1)
            ax[2].plot(k / len(p), np.cumsum(yte[order]) / k / yte.mean(), ls, label=f"{name} ε={e}")
            top = order[: len(p) // 10]
            rows.append(dict(model=name, eps=e, ap=average_precision_score(yte, p),
                             top_decile_lift=yte[top].mean() / yte.mean(),
                             min_cost=min(metrics(yte, p, t)["cost"] for t in thr)))
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    ax[1].set(xlabel="Threshold", ylabel="Cost (100·FN + FP)", title="Cost", yscale="log")
    ax[2].set(xlabel="Fraction flagged", ylabel="Lift", title="Lift", xlim=(0, 0.5))
    for a in ax: a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{LOG}/fig2_curves.png", dpi=200)
    write_csv(f"{LOG}/curves_summary.csv", rows)


def part_cv(folds=5, with_hadstack=False, adv_epochs=30):
    """5-fold stratified CV on the full processed dataset. The scaler is refit
    on each training fold; 15% of each training fold is held out for early
    stopping. with_hadstack=True also runs the full adversarial game per fold."""
    import pandas as pd, data_processing as dp
    df = dp.engineer_features(dp.select_and_clean(dp.load_data(dp.DATA_PATH)))
    feats = [c for c in df.columns if c != dp.TARGET_COL]
    X = df[feats].values.astype(np.float32); y = df[dp.TARGET_COL].values.astype(np.int64)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    rows = []
    orig_dir, orig_bm_mod, orig_ag_mod = bm.OUTPUT_DIR, bm.MODEL_DIR, ag.MODEL_DIR
    os.makedirs("cv_tmp", exist_ok=True); os.makedirs("models_cv", exist_ok=True)
    # Fold models go to models_cv/, so the main checkpoints in models/ are never touched.
    bm.MODEL_DIR = ag.MODEL_DIR = "models_cv"
    for k, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xva, ytr, yva = train_test_split(X[tr], y[tr], test_size=0.15,
                                              random_state=42, stratify=y[tr])
        sc = StandardScaler().fit(Xtr)
        for n, a in [("X_train", sc.transform(Xtr)), ("X_val", sc.transform(Xva)),
                     ("X_test", sc.transform(X[te])), ("y_train", ytr), ("y_val", yva),
                     ("y_test", y[te])]:
            np.save(f"cv_tmp/{n}.npy", a.astype(np.float32 if n[0] == "X" else np.int64))
        bm.OUTPUT_DIR = "cv_tmp"
        _, tm, _ = bm.train_baseline()
        rows.append(dict(fold=k, model="transformer", **metrics(tm["labels"], tm["probs"])))
        xgb = xgb_model(sc.transform(Xtr), ytr)
        rows.append(dict(fold=k, model="xgboost",
                         **metrics(y[te], xgb.predict_proba(sc.transform(X[te]))[:, 1])))
        if with_hadstack:
            _, r = ag.run_adversarial_training(tag=f"cv{k}", epochs=adv_epochs)
            m = load_transformer(f"models_cv/adversarial_best_cv{k}.pt")
            rows.append(dict(fold=k, model="hadstack",
                             **metrics(y[te], probs(m, sc.transform(X[te])))))
        bm.OUTPUT_DIR = orig_dir
        write_csv(f"{LOG}/cv_results.csv", rows)   # checkpoint after every fold
    bm.MODEL_DIR, ag.MODEL_DIR = orig_bm_mod, orig_ag_mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "xgb", "bootstrap", "noise", "transfer",
                             "curves", "sensitivity", "cv"])
    ap.add_argument("--cv_hadstack", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    steps = dict(xgb=part_reference, bootstrap=part_bootstrap, noise=part_noise,
                 transfer=part_transfer, curves=part_curves, sensitivity=part_sensitivity)
    if a.part == "all":
        for f in steps.values(): f()
    elif a.part == "cv":
        part_cv(with_hadstack=a.cv_hadstack)
    else:
        steps[a.part]()
