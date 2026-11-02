"""Figures 6 and 7 of the paper (game dynamics on the bond data; threshold sensitivity) from saved results.
Run after hadstack_thlstm.py (seeds and folds) and tau_sens.py."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(FIG, exist_ok=True); plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
R = json.load(open(os.path.join(RES, "results_seeds.json"))) + json.load(open(os.path.join(RES, "results_folds.json")))

# Fig. 6: budget, in-loop success on training, PGD-50x2 on validation
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.0)); cols = {1: "C0", 2: "C1", 3: "C2"}
for r in R:
    L = r["had_log"]; x = [l["round"] for l in L]
    sty = dict(lw=1.3, color=cols[r["seed"]], label=f"seed {r['seed']}") if r["split"] == "fixed" else dict(lw=.8, ls="--", color="grey", alpha=.75, label=None)
    ax[0].plot(x, [l["eps"] for l in L], **sty); ax[1].plot(x, [l["train"] for l in L], **sty); ax[2].plot(x, [l["strong"] for l in L], **sty)
ax[0].axhline(0.6586, ls=":", c="k", lw=.8); ax[0].set_ylim(0.33, 0.70); ax[0].text(9, 0.668, "max. 0.6586", fontsize=6.5)
ax[0].set_ylabel(r"budget $\varepsilon$ ($=\lambda$)"); ax[1].set_ylabel("in-loop attack, training (%)"); ax[2].set_ylabel(r"PGD-50$\times$2, validation (%)")
for a, t in zip(ax, ["(a)", "(b)", "(c)"]): a.set_xlabel("round"); a.set_title(t, loc="left")
ax[1].plot([], [], ls="--", c="grey", lw=.8, label="folds 1–5")
h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
ax[1].legend(h[:3] + h2[-1:], l[:3] + l2[-1:], fontsize=6.5, loc="upper right", frameon=False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "f_bond_dynamics.pdf")); plt.close(fig)

# Fig. 7: broken rate vs threshold (fixed split, seed 1)
d = json.load(open(os.path.join(RES, "tau_sens.json"))); tau = d["tau"]; mult = np.linspace(0.25, 6, 47)
names = [("Two-Head LSTM (undefended)", "Undefended", "C3"), ("HAD-Stack", "HAD-Stack", "C0"), ("FB-AT (eps 0.6586)", "Fixed-budget AT", "C7")]
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True)
for a, eps in zip(ax, ("0.5", "1.0")):
    for k, lab, c in names:
        r = d["rows"][k]; e0 = np.abs(np.array(r["e0"])); w = np.array(r[f"worst@{eps}"])
        a.plot(mult, [100 * np.mean(w - e0 > m * tau) for m in mult], color=c, lw=1.2, label=lab)
    a.axvline(1, ls=":", c="k", lw=.7); a.set_title(rf"$\varepsilon={eps}$", fontsize=8); a.set_xticks([1, 2, 3, 4, 5, 6])
ax[0].set_ylabel("broken (%)"); ax[0].legend(fontsize=6, frameon=False, loc="lower left")
fig.supxlabel(r"threshold (multiples of $\tau$)", fontsize=8)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "f_bond_tau.pdf")); plt.close(fig)
print("wrote", FIG)
