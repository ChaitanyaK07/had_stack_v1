"""Duplicate-row sensitivity check for Section VII-C.

creditcard.csv contains exact duplicate transactions. This script finds the test fraud cases that have
an identical copy among the training rows (same splits as prep.py), then recomputes robust recall and the
paired bootstrap of HAD-Stack minus baseline without them, from the per-row outcomes in results/.

Run from verification_second_dataset/:  python check_duplicates.py --csv /path/to/creditcard.csv
"""
import argparse, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--results", default="results")
a = ap.parse_args()
df = pd.read_csv(a.csv)
feats = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
key = pd.util.hash_pandas_object(df[feats], index=False).values; y = df["Class"].values; X = np.arange(len(df))
print(f"exact duplicate transactions (extra copies): {int(pd.Series(key).duplicated().sum())}")

def keep_mask(train_idx, test_idx):
    """True for each test fraud row (in test order) that has no identical copy among the training rows."""
    seen = set(key[train_idx]); f = test_idx[y[test_idx] == 1]
    return np.array([k not in seen for k in key[f]])

# same splits as prep.py (track A)
masks = {}
itv, ite = train_test_split(X, test_size=0.15, random_state=42, stratify=y)
itr, _ = train_test_split(itv, test_size=0.15 / 0.85, random_state=42, stratify=y[itv])
for s in (1, 2, 3): masks[f"seed{s}"] = keep_mask(itr, ite)
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    fit, _ = train_test_split(tr, test_size=0.15, random_state=42, stratify=y[tr]); masks[f"fold{k}"] = keep_mask(fit, te)

rng = np.random.RandomState(0)
for name, units in (("fixed split, three seeds", ["seed1", "seed2", "seed3"]), ("five-fold CV", [f"fold{k}" for k in range(5)])):
    print(f"\n{name}: test fraud with a copy in training = "
          f"{[int((~masks[u]).sum()) for u in units]} of {[len(masks[u]) for u in units]}")
    for e in ("0.5", "1.0"):
        rr = {"baseline": [], "v3": []}; pairs = []
        for u in units:
            M = json.load(open(os.path.join(a.results, f"scratch_{u}", "results.json")))["models"]; m = masks[u]
            for k in rr: rr[k].append(100 * np.array(M[k][f"robust_detected_rows@{e}"])[m].mean())
            pairs.append((np.array(M["v3"][f"robust_detected_rows@{e}"])[m], np.array(M["baseline"][f"robust_detected_rows@{e}"])[m]))
        n = sum(len(p) for p, _ in pairs); pt = 100 * sum((p - q).sum() for p, q in pairs) / n
        bs = []
        for _ in range(1000):
            tot = 0
            for p, q in pairs:
                i = rng.randint(0, len(p), len(p)); tot += (p[i] - q[i]).sum()
            bs.append(100 * tot / n)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  eps {e}: RR baseline {np.mean(rr['baseline']):.1f}%  HAD-Stack {np.mean(rr['v3']):.1f}%  "
              f"paired gain {pt:+.1f} [{lo:+.1f}, {hi:+.1f}]  ({n} fraud rows)")
