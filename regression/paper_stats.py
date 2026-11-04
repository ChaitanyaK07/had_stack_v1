"""Every regression number quoted in Section VII-D and Tables X-XIII of the paper, recomputed from saved results."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
S = json.load(open(os.path.join(RES, "results_seeds.json"))); F = json.load(open(os.path.join(RES, "results_folds.json"))); R = S + F
U, H, B = "Two-Head LSTM (undefended)", "HAD-Stack", "FB-AT (eps max)"
added = lambda m, e: m[f"wcMAE@{e}"] - m["MAE"]
print("== Table XII (outcome): mean +- std ==")
for grp, RR in (("folds", F), ("seeds", S)):
    for k in (U, H, B):
        q = lambda f: (np.mean([f(r["metrics"][k]) for r in RR]), np.std([f(r["metrics"][k]) for r in RR]))
        cells = [q(lambda m, x=x: m[x]) for x in ("MAE", "RMSE", "DA", "broken@0.5", "broken@1.0", "wcRMSE@0.5", "wcRMSE@1.0")]
        cells += [q(lambda m, e=e: added(m, e)) for e in ("0.5", "1.0")]
        print(f"{grp:5s} {k[:16]:16s} " + " | ".join(f"{a:.3f}+-{b:.3f}" for a, b in cells))
print("== Table XIII (HAD-Stack minus undefended, block bootstrap 95% CI); broken = -robust ==")
for r in R:
    c = r["ci"]
    print(f"{r['split']}/s{r['seed']}: " + "; ".join(f"{k} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for k, v in c.items()))
print("== Table XI (loop diagnostics per run) ==")
for r in R:
    L = r["had_log"]; st = [(l["fresh"], l["stale"]) for l in L[1:] if l["stale"] == l["stale"]]; h = st[len(st) // 2:]
    print(f"{r['split']}/s{r['seed']}: rounds {len(L)-1} (sel {r['had_best_round']}) eps [{min(l['eps'] for l in L):.4f},{max(l['eps'] for l in L):.4f}] "
          f"s [{min(l['s'] for l in L):+.2f},{max(l['s'] for l in L):+.2f}] train {L[0]['train']:.1f}->{L[-1]['train']:.1f} strong {L[0]['strong']:.1f}->{L[-1]['strong']:.1f} "
          f"F-S {np.mean([a-b for a,b in h]):+.2f} gap {np.mean([l['strong']-l['fresh'] for l in L[1:]]):.2f} max|F-S| {max(abs(a-b) for a,b in st):.1f} sec/round {np.mean([l['sec'] for l in L[1:]]):.1f}")
print("== error added by the attack: HAD-Stack below undefended ==")
for e in ("0.5", "1.0"):
    print(e, sum(added(r["metrics"][H], e) < added(r["metrics"][U], e) for r in R), "of", len(R))
print("== Table X (seed 1 per round) ==")
for l in S[0]["had_log"]:
    print(l["round"], f"{l['s']:+.2f} {l['eps']:.4f} train {l['train']:.1f} fresh {l['fresh']:.1f} stale {l['stale']:.1f} strong {l['strong']:.1f} rRMSE {l.get('val_rrmse', float('nan')):.3f}")
