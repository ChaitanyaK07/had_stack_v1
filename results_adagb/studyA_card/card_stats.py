import csv, sys, json, numpy as np
R = sys.argv[1]; units = ["seed1", "seed2", "seed3", "fold0", "fold1", "fold2", "fold3", "fold4"]
T = {u: [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{R}/scratch_{u}/traj_v3.csv"))] for u in units}
rounds = {u: len(t) for u, t in T.items()}
print("total rounds", sum(rounds.values()), "per run", rounds)
s = [r["next_signal"] for t in T.values() for r in t[1:-1]] + [r["next_signal"] for t in T.values() for r in t[:1]]
sig_used = [r["next_signal"] for t in T.values() for r in t[:-1]]           # signals that set the next round's budget
print("signal range (all logged)", round(min(r["next_signal"] for t in T.values() for r in t), 2), round(max(r["next_signal"] for t in T.values() for r in t), 2))
print("signal range (used)", round(min(sig_used), 2), round(max(sig_used), 2))
eps = [r["eps"] for t in T.values() for r in t]; print("eps range", round(min(eps), 4), round(max(eps), 4))
print("runs with eps < 0.5:", [u for u, t in T.items() if min(r["eps"] for r in t) < 0.5 - 1e-9])
print("eps max per run", {u: round(max(r['eps'] for r in t), 4) for u, t in T.items()})
tf = {u: (t[0]["train_flip"], t[-1]["train_flip"]) for u, t in T.items()}
print("train flip first->last", {u: (round(a, 1), round(b, 1)) for u, (a, b) in tf.items()}, "falls in", sum(b < a for a, b in tf.values()))
print("  first range", round(min(a for a, b in tf.values()), 1), round(max(a for a, b in tf.values()), 1), "last range", round(min(b for a, b in tf.values()), 1), round(max(b for a, b in tf.values()), 1))
st = {u: (t[0]["strong"], t[-1]["strong"]) for u, t in T.items()}
print("strong first->last", {u: (round(a, 1), round(b, 1)) for u, (a, b) in st.items()}, "falls in", sum(b < a for a, b in st.values()))
fl = [u for u, (a, b) in st.items() if b < a]
print("  falling runs: first range", round(min(st[u][0] for u in fl), 1), round(max(st[u][0] for u in fl), 1), "last range", round(min(st[u][1] for u in fl), 1), round(max(st[u][1] for u in fl), 1))
fs = [r["fresh"] - r["stale"] for t in T.values() for r in t if r["stale"] == r["stale"]]
print("max |fresh-stale|", round(max(abs(x) for x in fs), 1))
h = []
for t in T.values():
    v = [r["fresh"] - r["stale"] for r in t if r["stale"] == r["stale"]]; h += v[len(v) // 2:]
print("fresh-stale mean over second halves", round(np.mean(h), 2), "rounds", len(h))
gap = [r["strong"] - r["fresh"] for t in T.values() for r in t]
print("strong - fresh mean over all rounds", round(np.mean(gap), 2), "last round per run", {u: round(t[-1]["strong"] - t[-1]["fresh"], 1) for u, t in T.items()})
best = {}
for u, t in T.items():
    b, br = -1, None
    for r in t:
        if r["robust_val_auc"] > b: b, br = r["robust_val_auc"], int(r["round"])
    best[u] = br
print("checkpoint rounds", best, "range", min(best.values()), max(best.values()), "| stop rounds", min(rounds.values()), max(rounds.values()))
print("seconds per round", {u: round(np.mean([r['seconds'] for r in t])) for u, t in T.items()})
S = json.load(open(f"{R}/summary.json"))
for k, v in S.items():
    print(k, "paired", v["paired_bootstrap@0.5"], v["paired_bootstrap@1.0"], v["units_above_baseline_both_budgets"])
