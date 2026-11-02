import json, numpy as np, sys
runs = []
for f in sys.argv[1:]:
    try: runs += json.load(open(f))
    except Exception as e: print("skip", f, e)
M = ["Two-Head LSTM (undefended)", "FB-AT (eps 0.6586)", "HAD-Stack"]
K = ["MAE", "RMSE", "DA", "robust@0.5", "robust@1.0", "broken@0.5", "broken@1.0", "wcRMSE@0.5", "wcRMSE@1.0", "robDA@0.5", "robDA@1.0"]
for grp, sel in (("Fixed split, seeds", lambda r: r["split"] == "fixed"), ("Walk-forward folds", lambda r: r["split"] != "fixed")):
    R = [r for r in runs if sel(r)]
    if not R: continue
    print(f"\n## {grp} (n={len(R)})\n| model | " + " | ".join(K) + " |")
    for m in M:
        v = np.array([[r["metrics"][m][k] for k in K] for r in R])
        print(f"| {m} | " + " | ".join(f"{a:.2f}±{b:.2f}" for a, b in zip(v.mean(0), v.std(0))) + " |")
    for r in R:
        print(f"  {r['split']}/s{r['seed']} HAD best round {r['had_best_round']}, FB-AT {r['fb_best_round']}; CI: " +
              "; ".join(f"{k} {c['diff']:+.2f}[{c['lo']:+.2f},{c['hi']:+.2f}]" for k, c in r["ci"].items()))
    # loop diagnostics
    for r in R:
        L = r["had_log"]
        eps = [l["eps"] for l in L]; tr = [l["train"] for l in L]; fr = [l["fresh"] for l in L]; sg = [l["strong"] for l in L]
        st = [(l["fresh"], l["stale"]) for l in L[1:] if l["stale"] == l["stale"]]
        half = st[len(st)//2:]
        print(f"  {r['split']}/s{r['seed']}: rounds {len(L)-1}, eps {min(eps):.4f}..{max(eps):.4f}, s {min(l['s'] for l in L):+.2f}..{max(l['s'] for l in L):+.2f} | "
              f"train {tr[0]:.1f}->{tr[1]:.1f}->{tr[-1]:.1f} | fresh {fr[0]:.1f}->{fr[-1]:.1f} | strong {sg[0]:.1f}->{sg[-1]:.1f} | "
              f"fresh-stale all {np.mean([a-b for a,b in st]):+.2f}, 2nd half {np.mean([a-b for a,b in half]):+.2f} | gap mean {np.mean([l['strong']-l['fresh'] for l in L[1:]]):+.2f}")
