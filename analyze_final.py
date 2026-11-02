"""
ANALYSIS of run_final_experiments.py outputs.
  python analyze_final.py            (reads results_final/, writes results_final/analysis/)
  python analyze_final.py --quick    (smoke test on results_quick/)

1. Reproduction check: Transformer robust recall at eps 0.5 / 1.0 must equal
   logs/lambda_control_results.jsonl (same models, attacks and seeds). Stops if not.
2. Bootstrap CIs (B = 1000, percentile 95%), resampling TEST ROWS:
     * five-fold: the five out-of-fold test sets cover all 26,393 transactions once; rows are
       resampled within each fold and metrics computed on the pooled resample;
     * seeds: each seed separately.
   Absolute CIs for every model, and PAIRED CIs for HAD-Stack minus each comparison
   (same resampled rows for both models). "Significant" = 95% CI excludes 0.
3. XGBoost ablation: plain vs adversarially augmented, clean and under transfer attack,
   compared with the Transformers under the same transfer-only attack.
4. Curves on the pooled five-fold predictions: cost vs threshold, lift, robust recall vs eps.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); A = ap.parse_args()
D = "results_quick" if A.quick else "results_final"; OUT = f"{D}/analysis"; os.makedirs(OUT, exist_ok=True)
EPS = (0.25, 0.5, 0.75, 1.0)
TF = ("baseline", "v3", "pgdat066", "pgdat066lam"); XG = ("xgb_plain", "xgb_advaug")
NAME = {"baseline": "Baseline Transformer", "v3": "HAD-Stack", "pgdat066": r"FB-AT ($\lambda{=}0.5$)",
        "pgdat066lam": r"FB-AT ($\lambda{=}0.66$)", "xgb_plain": "XGBoost", "xgb_advaug": "XGBoost + adv.\\ aug."}
SEEDS, FOLDS = ((1,), (0,)) if A.quick else ((1, 2, 3), (0, 1, 2, 3, 4))
B = 50 if A.quick else 1000
U = {("seeds", s): dict(np.load(f"{D}/seeds_{s}.npz")) for s in SEEDS}
U.update({("cv", k): dict(np.load(f"{D}/cv_{k}.npz")) for k in FOLDS})


def unit_arrays(d, m):
    """Per-row arrays for one model in one unit: y, clean pred, and for fraud rows per eps:
    'kept' = detected and not broken (worst case), 'kept_tr' = detected and not broken by transfer."""
    y = d["y"]; p = d[f"{m}__clean"]; det = p[y == 1] >= 0.5
    kept, kept_tr = {}, {}
    for e in EPS:
        tr = d[f"{m}__tr__{e}"]; wb = d.get(f"{m}__wb__{e}", np.zeros_like(tr))
        kept[e] = det & ~(tr | wb); kept_tr[e] = det & ~tr
    return y, p, kept, kept_tr


def metrics(y, p, kept, kept_tr, idx_all, idx_fr):
    yy, pp = y[idx_all], p[idx_all]; pred = pp >= 0.5
    tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum()); fn = int((~pred & (yy == 1)).sum())
    nf = len(idx_fr); out = dict(fp=fp, fn=fn, cost=100 * fn + fp, recall=100 * tp / max(tp + fn, 1),
                                 precision=100 * tp / max(tp + fp, 1))
    for e in EPS:
        out[f"rr@{e}"] = 100 * kept[e][idx_fr].sum() / nf
        out[f"rrtr@{e}"] = 100 * kept_tr[e][idx_fr].sum() / nf
    return out


def pooled(units, m, rng=None):
    """Metrics over a set of units, optionally with rows resampled within each unit."""
    agg = None
    for d in units:
        y, p, kept, kept_tr = unit_arrays(d, m)
        n = len(y); nfr = int((y == 1).sum())
        if rng is None: ia, ifr = np.arange(n), np.arange(nfr)
        else:
            ia = rng.integers(0, n, n)
            # fraud rows of the resample, mapped to their index among fraud rows
            fr_pos = np.cumsum(y == 1) - 1
            sel = ia[y[ia] == 1]; ifr = fr_pos[sel]
        mm = metrics(y, p, kept, kept_tr, ia, ifr)
        yy, pp = y[ia], p[ia]; pred = pp >= 0.5
        raw = dict(tp=int((pred & (yy == 1)).sum()), fp=mm["fp"], fn=mm["fn"], nf=len(ifr),
                   **{f"k{e}": int(kept[e][ifr].sum()) for e in EPS}, **{f"kt{e}": int(kept_tr[e][ifr].sum()) for e in EPS})
        agg = raw if agg is None else {k: agg[k] + raw[k] for k in agg}
    r = dict(fp=agg["fp"], fn=agg["fn"], cost=100 * agg["fn"] + agg["fp"],
             recall=100 * agg["tp"] / max(agg["tp"] + agg["fn"], 1), precision=100 * agg["tp"] / max(agg["tp"] + agg["fp"], 1))
    for e in EPS:
        r[f"rr@{e}"] = 100 * agg[f"k{e}"] / agg["nf"]; r[f"rrtr@{e}"] = 100 * agg[f"kt{e}"] / agg["nf"]
    return r


def boot(units, models, B, seed=0):
    """Same resampled rows for every model (paired)."""
    rng_states = np.random.default_rng(seed).integers(0, 2**31, B)
    res = {m: [] for m in models}
    for b in range(B):
        for m in models:
            res[m].append(pooled(units, m, np.random.default_rng(rng_states[b])))
    return res


def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------- 1. reproduction check ----------------
rep = {"checked": 0, "max_abs_diff": 0.0}
if os.path.exists("logs/lambda_control_results.jsonl") and not A.quick:
    L = {(r["exp"], r["unit"], r["model"]): r for r in map(json.loads, open("logs/lambda_control_results.jsonl"))}
    for (exp, u), d in U.items():
        for m in TF:
            for e in (0.5, 1.0):
                mine = pooled([d], m)[f"rr@{e}"]; ref = L[(exp, u, m)][f"robust_recall@{e}"]
                rep["checked"] += 1; rep["max_abs_diff"] = max(rep["max_abs_diff"], abs(mine - ref))
    assert rep["max_abs_diff"] < 0.05, f"re-evaluation does not reproduce earlier results: {rep}"
if not A.quick and rep["checked"] == 0:
    raise SystemExit("logs/lambda_control_results.jsonl not found: run this in the folder of the confirmation run, "
                     "so the reproduction check can be performed.")
print("reproduction check:", rep)

# ---------------- 2. bootstrap ----------------
cv_units = [U[("cv", k)] for k in FOLDS]
point_cv = {m: pooled(cv_units, m) for m in TF + XG}
bs_cv = boot(cv_units, TF + XG, B)
KEYS = ["recall", "precision", "cost"] + [f"rr@{e}" for e in EPS] + [f"rrtr@{e}" for e in EPS]
res = {"B": B, "reproduction": rep, "cv_point": point_cv, "cv_ci": {}, "cv_paired": {}, "seed_paired": {}}
for m in TF + XG:
    res["cv_ci"][m] = {k: ci([r[k] for r in bs_cv[m]]) for k in KEYS if not (m in XG and k.startswith("rr@"))}
for c in ("baseline", "pgdat066", "pgdat066lam"):
    res["cv_paired"][c] = {}
    for k in ["recall", "precision", "cost", "rr@0.5", "rr@1.0"]:
        diff = [a[k] - b[k] for a, b in zip(bs_cv["v3"], bs_cv[c])]
        res["cv_paired"][c][k] = dict(point=point_cv["v3"][k] - point_cv[c][k], ci=ci(diff))
for s in SEEDS:
    d = [U[("seeds", s)]]; bs = boot(d, ("v3", "pgdat066", "pgdat066lam", "baseline"), B, seed=s)
    pt = {m: pooled(d, m) for m in ("v3", "pgdat066", "pgdat066lam", "baseline")}
    res["seed_paired"][s] = {c: {k: dict(point=pt["v3"][k] - pt[c][k], ci=ci([a[k] - b[k] for a, b in zip(bs["v3"], bs[c])]))
                                 for k in ["cost", "rr@0.5", "rr@1.0"]} for c in ("baseline", "pgdat066", "pgdat066lam")}
json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=1, default=float)

# ---------------- 3. LaTeX fragments ----------------
def sig(c): return "" if c[0] <= 0 <= c[1] else "$^\\ast$"
def fi(x): return f"{x:,.0f}".replace(",", "{,}")
def f1(x): return f"{x:.1f}"
rows = []
for m in TF:
    p, c = point_cv[m], res["cv_ci"][m]
    rows.append(f"{NAME[m]} & " + " & ".join(f"{f1(p[k])} [{f1(c[k][0])}, {f1(c[k][1])}]" for k in ("recall", "rr@0.5", "rr@1.0")) +
                f" & {fi(p['cost'])} [{fi(c['cost'][0])}, {fi(c['cost'][1])}]\\\\")
open(f"{OUT}/tab_boot_abs.tex", "w").write("\n".join(rows))
rows = []
for cmp_ in ("baseline", "pgdat066", "pgdat066lam"):
    cells = []
    for k in ("rr@0.5", "rr@1.0", "recall", "cost"):
        q = res["cv_paired"][cmp_][k]; fmt = (lambda x: f"{x:+,.0f}".replace(",", "{,}")) if k == "cost" else (lambda x: f"{x:+.1f}")
        cells.append(f"${fmt(q['point'])}$ [{fmt(q['ci'][0])}, {fmt(q['ci'][1])}]{sig(q['ci'])}")
    rows.append(f"{NAME[cmp_]} & " + " & ".join(cells) + "\\\\")
open(f"{OUT}/tab_boot_paired.tex", "w").write("\n".join(rows))
rows = []
for s in SEEDS:
    cells = []
    for cmp_ in ("pgdat066", "pgdat066lam"):
        for k in ("rr@0.5", "rr@1.0"):
            q = res["seed_paired"][s][cmp_][k]; cells.append(f"${q['point']:+.1f}$ [{q['ci'][0]:+.1f}, {q['ci'][1]:+.1f}]{sig(q['ci'])}")
    rows.append(f"Seed {s} & " + " & ".join(cells) + "\\\\")
open(f"{OUT}/tab_boot_seeds.tex", "w").write("\n".join(rows))
rows = []
for m in XG + TF:
    p, c = point_cv[m], res["cv_ci"][m]
    rows.append(f"{NAME[m]} & {f1(p['recall'])} & {f1(p['precision'])} & {fi(p['cost'])}" +
                " & " + " & ".join(f"{f1(p[f'rrtr@{e}'])} [{f1(c[f'rrtr@{e}'][0])}, {f1(c[f'rrtr@{e}'][1])}]" for e in (0.5, 1.0)) + "\\\\")
    if m == XG[-1]: rows.append("\\midrule")
open(f"{OUT}/tab_xgb.tex", "w").write("\n".join(rows))

# ---------------- 4. curves (pooled five-fold) ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
STY = {"baseline": ("tab:gray", "-"), "v3": ("tab:red", "-"), "pgdat066": ("tab:blue", "--"),
       "pgdat066lam": ("tab:green", ":"), "xgb_plain": ("tab:purple", "-."), "xgb_advaug": ("tab:olive", "-.")}
def lbl(m): return NAME[m].replace("$", "").replace("{=}", "=").replace("\\lambda", "λ").replace("\\ ", " ")
y = np.concatenate([d["y"] for d in cv_units])
thr = np.linspace(0.01, 0.99, 99)
for fig_name, fn in [("cost", "fig_cost.png"), ("lift", "fig_lift.png"), ("recall", "fig_recall.png")]:
    f, a = plt.subplots(figsize=(4.2, 3.3))
    for m in (TF + XG if fig_name != "recall" else TF + XG):
        col, ls = STY[m]; p = np.concatenate([d[f"{m}__clean"] for d in cv_units])
        if fig_name == "cost":
            a.plot(thr, [100 * ((p < t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum() for t in thr], ls, color=col, label=lbl(m))
        elif fig_name == "lift":
            o = np.argsort(-p); k = np.arange(1, len(p) + 1)
            a.plot(k / len(p), np.cumsum(y[o]) / k / y.mean(), ls, color=col, label=lbl(m))
        else:
            xs = [0.0] + list(EPS); key = "rrtr" if m in XG else "rr"
            ys = [point_cv[m]["recall"]] + [point_cv[m][f"{key}@{e}"] for e in EPS]
            a.plot(xs, ys, ls, color=col, marker="o", ms=3, label=lbl(m) + (" (transfer)" if m in XG else ""))
    if fig_name == "cost": a.set(xlabel="Decision threshold", ylabel="Cost (100·FN + FP)", yscale="log")
    if fig_name == "lift": a.set(xlabel="Fraction of transactions flagged", ylabel="Lift", xlim=(0, 0.5))
    if fig_name == "recall": a.set(xlabel="Attack budget ε (standardised units)", ylabel="Recall under worst-case attack (%)", ylim=(0, 100))
    a.grid(alpha=.3); a.legend(fontsize=6); f.tight_layout(); f.savefig(f"{OUT}/{fn}", dpi=300); plt.close(f)
print("analysis written to", OUT)
