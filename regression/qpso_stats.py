"""Every number of the QPSO-follower part of Section VII-D (paper), from results/qpso_*.json; also writes Fig. f_bond_qpso.pdf.
Usage: python qpso_stats.py   (after run_qpso.py --mode seeds and --mode folds)"""
import glob, json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "..", "paper", "figs")
U, FB, H, HQ, HD = "Two-Head LSTM (undefended)", "FB-AT (eps max)", "HAD-Stack", "HAD-Stack (QPSO)", "HAD-Stack (PGD+QPSO)"
S = json.load(open(os.path.join(RES, "qpso_seeds.json")))
F = sorted(sum((json.load(open(f)) for f in glob.glob(os.path.join(RES, "qpso_folds_*.json"))), []), key=lambda r: r["split"])
R = S + F; N = len(R)
name = lambda r: ("Seed %d" % r["seed"]) if r["split"] == "fixed" else ("Fold " + r["split"][4:])
print("order:", R[0].get("order"), " eps max:", R[0].get("eps_max"))
print("== reproduction of the reported PGD runs (max |diff| over all metrics) ==")
for r in R: print(name(r), r["reproduced_max_diff"])

print("\n== loop diagnostics per follower ==")
for key, lab in (("had_log", "PGD"), ("qpso_log", "QPSO"), ("dual_log", "PGD+QPSO")):
    for r in R:
        L = r[key]; st = [(l["fresh"], l["stale"]) for l in L[1:] if l["stale"] == l["stale"]]; h = st[len(st) // 2:]
        br = r["best_round"][{"had_log": H, "qpso_log": HQ, "dual_log": HD}[key]]
        print(f"{lab:8s} {name(r)}: rounds {len(L)-1} ({br}) eps {min(l['eps'] for l in L):.4f}-{max(l['eps'] for l in L):.4f} "
              f"s [{min(l['s'] for l in L):+.2f},{max(l['s'] for l in L):+.2f}] train {L[0]['train']:.1f}->{L[-1]['train']:.1f} "
              f"fresh {L[0]['fresh']:.1f}->{L[-1]['fresh']:.1f} strong {L[0]['strong']:.1f}->{L[-1]['strong']:.1f} "
              f"F-S {np.mean([a-b for a,b in h]):+.2f} max|F-S| {max(abs(a-b) for a,b in st):.1f} gap {np.mean([l['strong']-l['fresh'] for l in L[1:]]):.2f} "
              f"sec {np.mean([l['sec'] for l in L[1:]]):.1f}")

print("\n== outcome per run ==")
for r in R:
    for k in (U, H, HQ, HD, FB):
        m = r["metrics"][k]
        print(f"{name(r):7s} {k[:22]:22s} RMSE {m['RMSE']:.3f} | pgd {m['wcRMSE@0.5']:.2f} {m['wcRMSE@1.0']:.2f} | q {m['q_wcRMSE@0.5']:.2f} {m['q_wcRMSE@1.0']:.2f} "
              f"| all {m['all_wcRMSE@0.5']:.2f} {m['all_wcRMSE@1.0']:.2f} | brk all {m['all_broken@0.5']:.1f} / {m['all_broken@1.0']:.1f} "
              f"| brk pgd {m['broken@0.5']:.1f} {m['broken@1.0']:.1f} q {m['q_broken@0.5']:.1f} {m['q_broken@1.0']:.1f} "
              f"| added all {m['all_wcMAE@0.5']-m['MAE']:.2f} / {m['all_wcMAE@1.0']-m['MAE']:.2f} MAE {m['MAE']:.3f}")

added = lambda m, e, p="": m[f"{p}wcMAE@{e}"] - m["MAE"]
print(f"\n== counts over the {N} runs ==")
for k in (H, HQ, HD):
    for p, lab in (("", "PGD ensemble"), ("q_", "QPSO attack"), ("all_", "PGD ensemble + QPSO")):
        for e in ("0.5", "1.0"):
            n_rm = sum(r["metrics"][k][f"{p}wcRMSE@{e}"] < r["metrics"][U][f"{p}wcRMSE@{e}"] for r in R)
            n_br = sum(r["metrics"][k][f"{p}broken@{e}"] < r["metrics"][U][f"{p}broken@{e}"] for r in R)
            n_ad = sum(added(r["metrics"][k], e, p) < added(r["metrics"][U], e, p) for r in R)
            n_fb = sum(r["metrics"][k][f"{p}wcRMSE@{e}"] < r["metrics"][FB][f"{p}wcRMSE@{e}"] for r in R)
            print(f"{k:21s} {lab:20s} eps {e}: wcRMSE<undef {n_rm}/{N}, broken<undef {n_br}/{N}, added<undef {n_ad}/{N}, wcRMSE<FB-AT {n_fb}/{N}")
for e in ("0.5", "1.0"):
    print(f"union raises wcRMSE over PGD ensemble: max {max(r['metrics'][k][f'all_wcRMSE@{e}'] - r['metrics'][k][f'wcRMSE@{e}'] for r in R for k in (U, H, HQ, HD, FB)):.4f} at eps {e}")

print("\n== CI (vs undefended), PGD ensemble + QPSO; broken = -robust ==")
for k in (H, HQ, HD):
    for r in R:
        c = r["ci_all"][k]
        print(f"{k:21s} {name(r)}: " + "; ".join(f"{q} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for q, v in c.items()))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.0)); cols = {1: "C0", 2: "C1", 3: "C2"}
for r in R:
    L = r["qpso_log"]; x = [l["round"] for l in L]
    sty = dict(lw=1.3, color=cols[r["seed"]], label=f"seed {r['seed']}") if r["split"] == "fixed" else dict(lw=.8, ls="--", color="grey", alpha=.75, label=None)
    ax[0].plot(x, [l["eps"] for l in L], **sty); ax[1].plot(x, [l["train"] for l in L], **sty); ax[2].plot(x, [l["strong"] for l in L], **sty)
EM = R[0].get("eps_max", 0.6586)
ax[0].axhline(EM, ls=":", c="k", lw=.8); ax[0].set_ylim(0.33, 0.70); ax[0].text(9, EM + 0.0094, f"max. {EM:.4f}", fontsize=6.5)
ax[0].set_ylabel(r"budget $\varepsilon$ ($=\lambda$)"); ax[1].set_ylabel("in-loop QPSO, training (%)"); ax[2].set_ylabel(r"PGD-50$\times$2, validation (%)")
for a, t in zip(ax, ["(a)", "(b)", "(c)"]): a.set_xlabel("round"); a.set_title(t, loc="left")
ax[1].plot([], [], ls="--", c="grey", lw=.8, label="folds 1–2")
h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
ax[1].legend(h[:3] + h2[-1:], l[:3] + l2[-1:], fontsize=6.5, loc="upper right", frameon=False)
fig.tight_layout(pad=0.3); os.makedirs(FIG, exist_ok=True); fig.savefig(os.path.join(FIG, "f_bond_qpso.pdf")); plt.close(fig)
print("wrote", os.path.join(FIG, "f_bond_qpso.pdf"))
