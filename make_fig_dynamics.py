"""Fig. dynamics (fig:dyn): Study A game dynamics on the UPI data over rounds, seeds 1-3, from results/upi_loop_diag.json
(written by upi_loop_diag.py). Left: budget eps = lambda set by the controller. Right: success of the in-loop attack on
training fraud (solid) and of PGD-50x2 on validation fraud (dashed).
Run from the repository root:  python make_fig_dynamics.py   (writes results/f_dynamics.pdf)"""
import json, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "results/upi_loop_diag.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/f_dynamics.pdf"
D = json.load(open(src)); em = D.get("eps_max")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3)); col = {"1": "#2f6fd0", "2": "#e8643c", "3": "#1fa37a"}
for s, v in sorted(D["seeds"].items()):
    T = v["traj"]; x = [int(r["round"]) for r in T]
    ax[0].plot(x, [float(r["eps"]) for r in T], color=col[s], lw=1.4, label=f"seed {s}")
    ax[1].plot(x, [float(r["train_flip"]) for r in T], color=col[s], lw=1.4, label=f"seed {s}, in-loop (train)")
    ax[1].plot(x, [float(r["strong"]) for r in T], color=col[s], lw=1.2, ls="--", label=f"seed {s}, PGD-50$\\times$2 (val)")
if em: ax[0].axhline(em, color="0.4", lw=0.7, ls=":"); ax[0].text(x[-1], em + 0.004, f"max. {em:.4f}", fontsize=7, ha="right", va="bottom", color="0.3")
ax[0].set_ylabel(r"budget $\varepsilon=\lambda$"); ax[1].set_ylabel("detected fraud flipped (%)")
for a, t in zip(ax, ("(a) Budget", "(b) Attack success")): a.set_xlabel("round"); a.set_title(t, loc="left"); a.grid(alpha=0.3, lw=0.5)
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
from matplotlib.lines import Line2D
ax[1].legend([Line2D([], [], color="k", lw=1.4), Line2D([], [], color="k", lw=1.2, ls="--")],
             ["in-loop attack, training", r"PGD-50$\times$2, validation"], frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(pad=0.4); fig.savefig(out); print("saved", out)
