"""Fig. 5 (fig:ver): HAD-Stack loop on the card data (track A), from results/scratch_*/traj_v3.csv.
Run from verification_second_dataset/:  python make_fig_ulb.py   (writes results/f_ulb_dynamics.pdf)
"""
import csv, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = sys.argv[1] if len(sys.argv) > 1 else "results"; out = sys.argv[2] if len(sys.argv) > 2 else "results/f_ulb_dynamics.pdf"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 3, "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.fontsize": 5.5})
load = lambda u: list(csv.DictReader(open(f"{R}/scratch_{u}/traj_v3.csv")))
fig, ax = plt.subplots(1, 2, figsize=(3.32, 1.30))
C = {"seed1": "tab:blue", "seed2": "tab:red", "seed3": "tab:green"}
for u in ["fold0", "fold1", "fold2", "fold3", "fold4", "seed1", "seed2", "seed3"]:
    T = load(u); r = [int(t["round"]) for t in T]
    kw = dict(color=C[u], lw=1.0, label=u.replace("seed", "seed ")) if u in C else dict(color="0.55", lw=0.7, ls="--", label="folds 0–4" if u == "fold0" else None)
    ax[0].plot(r, [float(t["eps"]) for t in T], **kw)
    ax[1].plot(r, [float(t["train_flip"]) for t in T], **kw)
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import had_stack_v2 as _hs; EM = _hs.EPS_MAX   # 0.6586 under HCS_ORDER=ab
ax[0].axhline(EM, color="0.3", lw=0.5, ls=":"); ax[0].text(20.6, EM - 0.0226, f"max. {EM:.4f}", fontsize=5.5, color="0.3", va="top", ha="right")
ax[0].set_ylabel(r"Budget $\varepsilon\ (=\lambda)$"); ax[1].set_ylabel("In-loop flip (%)")
for a in ax: a.set_xlabel("Round"); a.set_xlim(1, 21); a.grid(alpha=0.25, lw=0.4)
ax[0].set_ylim(0.36, 0.68); ax[1].set_ylim(0, 52)
ax[1].legend(loc="upper right", frameon=False, handlelength=1.6, labelspacing=0.2)
fig.tight_layout(pad=0.25, w_pad=0.6); fig.savefig(out); print("wrote", out)
