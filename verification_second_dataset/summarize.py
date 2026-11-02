"""Summarise results/ into results/summary.md and results/summary.json.

  python summarize.py                 (add --include-smoke to summarise smoke runs, for testing only)

Per track and setting (seeds on the fixed split; five-fold CV):
  * clean metrics and worst-case robust recall / broken rate, mean +- std, baseline vs HAD-Stack
  * units where HAD-Stack's robust recall is above the baseline's at both budgets
  * paired bootstrap of the robust-recall difference (1,000 resamples of test fraud rows within each unit)
  * loop diagnostics of HAD-Stack (Definition 8): budget trajectory, follower payoff, fresh vs stale,
    strong attack and exploitability gap
"""
import os, sys, glob, json, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
smoke = "--include-smoke" in sys.argv
ms = lambda v, d=1: f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}"

def load():
    U = []
    for p in sorted(glob.glob(f"{RES}/*/results.json")):
        r = json.load(open(p))
        if r["smoke"] and not smoke: continue
        d = os.path.dirname(p); r["traj"] = {}
        for m in ("v3", "fbat05"):
            f = f"{d}/traj_{m}.csv"
            if os.path.exists(f): r["traj"][m] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(open(f))]
        U.append(r)
    return U

def bootstrap(units, eps, B=1000, seed=0):
    rng = np.random.RandomState(seed); diffs = []
    pairs = [(np.array(u["models"]["v3"][f"robust_detected_rows@{eps}"]), np.array(u["models"]["baseline"][f"robust_detected_rows@{eps}"])) for u in units]
    n = sum(len(a) for a, _ in pairs); point = 100 * sum((a - b).sum() for a, b in pairs) / n
    for _ in range(B):
        tot = 0
        for a, b in pairs:
            i = rng.randint(0, len(a), len(a)); tot += (a[i] - b[i]).sum()
        diffs.append(100 * tot / n)
    lo, hi = np.percentile(diffs, [2.5, 97.5]); return point, lo, hi

def diagnostics(traj):
    if not traj: return {}
    eps = [t["eps"] for t in traj]; half = traj[len(traj) // 2:]
    fs = [t["fresh"] - t["stale"] for t in half if not np.isnan(t["stale"])]
    return dict(rounds=len(traj), eps_first=eps[0], eps_max=max(eps), round_eps_max=1 + int(np.argmax(eps)),
                train_flip_first=traj[0]["train_flip"], train_flip_last=traj[-1]["train_flip"],
                strong_first=traj[0]["strong"], strong_last=traj[-1]["strong"],
                fresh_minus_stale_second_half=float(np.mean(fs)) if fs else float("nan"),
                exploitability_gap_last=traj[-1]["strong"] - traj[-1]["fresh"],
                seconds_per_round=float(np.mean([t["seconds"] for t in traj])))

def main():
    U = load(); out = {}; md = ["# HAD-Stack verification on the ULB credit-card data\n"]
    for track in ("scratch", "pretrained"):
        for setting, pref in (("seeds (fixed split)", "seed"), ("five-fold CV", "fold")):
            units = [u for u in U if u["track"] == track and u["unit"].startswith(pref)]
            if not units: continue
            key = f"{track} / {setting}"; S = out[key] = {"units": [u["unit"] for u in units]}
            md.append(f"\n## Track {'A (FraudTransformer from scratch)' if track == 'scratch' else 'B (pre-trained MLP)'} — {setting}: {', '.join(S['units'])}\n")
            md.append(f"Test set per unit: {units[0]['n_test']} rows, {units[0]['n_test_fraud']} fraud. Device: {units[0]['device']}.\n")
            md.append("| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for m, name in (("baseline", "Baseline" if track == "scratch" else "Pre-trained (no game)"), ("v3", "HAD-Stack")):
                g = lambda k: [u["models"][m][k] for u in units]
                S[m] = {k: (float(np.mean(g(k))), float(np.std(g(k)))) for k in ("auc", "recall", "fp", "fn", "cost", "robust_recall@0.5",
                                                                               "robust_recall@1.0", "broken@0.5", "broken@1.0")}
                md.append(f"| {name} | {ms(g('auc'), 4)} | {ms([100*x for x in g('recall')])} | {ms(g('fp'))} | {ms(g('fn'))} | {ms(g('cost'))} | "
                          f"{ms(g('robust_recall@0.5'))} | {ms(g('robust_recall@1.0'))} | {ms(g('broken@0.5'))} | {ms(g('broken@1.0'))} |")
            wins = sum(all(u["models"]["v3"][f"robust_recall@{e}"] > u["models"]["baseline"][f"robust_recall@{e}"] for e in (0.5, 1.0)) for u in units)
            S["units_above_baseline_both_budgets"] = f"{wins} of {len(units)}"
            md.append(f"\nHAD-Stack above the baseline at both budgets in **{wins} of {len(units)}** units.")
            for e in (0.5, 1.0):
                p, lo, hi = bootstrap(units, e); S[f"paired_bootstrap@{e}"] = (p, lo, hi)
                md.append(f"Paired bootstrap, robust recall HAD-Stack − baseline at ε={e}: **{p:+.1f}** points (95% CI [{lo:+.1f}, {hi:+.1f}]).")
            D = [diagnostics(u["traj"].get("v3", [])) for u in units]; S["diagnostics"] = D
            md.append("\n**Loop diagnostics of HAD-Stack (Definition 8), per unit:**\n")
            md.append("| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |")
            md.append("|---|---|---|---|---|---|---|---|")
            for u, d in zip(units, D):
                if d: md.append(f"| {u['unit']} | {d['rounds']} | {d['eps_first']:.4f}→{d['eps_max']:.4f} ({d['round_eps_max']}) | "
                                f"{d['train_flip_first']:.1f}→{d['train_flip_last']:.1f} | {d['strong_first']:.1f}→{d['strong_last']:.1f} | "
                                f"{d['fresh_minus_stale_second_half']:+.1f} | {d['exploitability_gap_last']:.1f} | {d['seconds_per_round']:.0f} |")
    json.dump(out, open(f"{RES}/summary.json", "w"), indent=1); open(f"{RES}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
