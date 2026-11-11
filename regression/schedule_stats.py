"""Summary of the bond schedule baselines (results/schedules/*.json from run_schedules.py)."""
import glob, json, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob("results/schedules/*.json"))]
names = ["Two-Head LSTM (undefended)", "HAD-Stack", "Linear ramp", "Cosine ramp", "Tuned constant"]
L = ["| Split | Model | RMSE | wcRMSE@0.5 | wcRMSE@1.0 | broken@0.5 | broken@1.0 | vs HAD wcRMSE@0.5 [CI] | vs HAD wcRMSE@1.0 [CI] |", "|---|---|---|---|---|---|---|---|---|"]
for r in R:
    for n in names:
        m = r["metrics"][n]; c = r["ci_vs_had"].get(n, {})
        f = lambda k: f"{c[k]['diff']:+.2f} [{c[k]['lo']:+.2f}, {c[k]['hi']:+.2f}]" if k in c else ""
        L.append(f"| {r['split']} | {n} | {m['RMSE']:.3f} | {m['wcRMSE@0.5']:.2f} | {m['wcRMSE@1.0']:.2f} | {m['broken@0.5']:.1f} | {m['broken@1.0']:.1f} | {f('wcRMSE@0.5')} | {f('wcRMSE@1.0')} |")
    L.append(f"| {r['split']} | tuned eps {r['tuned_eps']:.4f} (val rRMSE {r['tuned_val']}) | | | | | | | |")
L.append("\nCounts over splits (schedule minus HAD-Stack; negative = schedule more robust):")
for n in names[2:]:
    for b in ("0.5", "1.0"):
        d = [r["ci_vs_had"][n][f"wcRMSE@{b}"] for r in R]
        L.append(f"  {n:15s} eps {b}: lower wcRMSE than HAD in {sum(x['diff'] < 0 for x in d)}/{len(d)}, CI below 0 in {sum(x['hi'] < 0 for x in d)}, "
                 f"CI above 0 in {sum(x['lo'] > 0 for x in d)}; walk-forward mean {np.mean([r['metrics'][n]['wcRMSE@'+b] for r in R if r['split'] != 'fixed']):.2f} "
                 f"(HAD {np.mean([r['metrics']['HAD-Stack']['wcRMSE@'+b] for r in R if r['split'] != 'fixed']):.2f})")
open("results/schedules/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
