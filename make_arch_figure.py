import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.4, 5.3)); ax.set_xlim(-6, 100); ax.set_ylim(0, 73); ax.axis("off")
C = dict(data="#E8EEF7", ham="#FFF1D6", att="#FBE3E3", dfd="#E3F4E6", val="#EFE7F7", edge="#333333")

def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=C["edge"], lw=1.0))
    ax.text(x + w/2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, weight="bold")
    ax.text(x + w/2, y + h - 5.6, body, ha="center", va="top", fontsize=6.9, linespacing=1.35)

def arrow(p, q, text=None, tx=0, ty=0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, lw=1.0, color="#222",
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text: ax.text((p[0]+q[0])/2 + tx, (p[1]+q[1])/2 + ty, text, fontsize=6.6, ha="center",
                     va="center", style="italic", bbox=dict(fc="white", ec="none", pad=0.6))

# Hamiltonian controller (top)
ax.add_patch(FancyBboxPatch((1, 48), 98, 24, boxstyle="round,pad=0.4,rounding_size=1.5",
                            fc="#FFFBF2", ec="#B8860B", lw=1.0, ls="--"))
ax.text(50, 71.2, "Hamiltonian controller (complex eigenvalues, re-solved every round)", ha="center", va="top", fontsize=8.4, weight="bold", color="#8A6100")
box(3,  50, 29, 16.5, r"$H_2$  (4$\times$4)", "complex spectrum (np.linalg.eig)\nground-state decay rate $\\Gamma_2$", C["ham"])
box(35.5, 50, 29, 16.5, r"$H_1$  (4$\times$4)", "complex spectrum (np.linalg.eig)\nground-state decay rate $\\Gamma_1$", C["ham"])
box(68, 50, 29, 16.5, r"$H_{29}$  (16$\times$16)", "full Eq. 29 (incl. dissipation)\nfield $\\phi$: monitored only", C["ham"])

# derived parameters
box(8, 33, 38, 10.5, "Attack budget", "$\\varepsilon=0.5\\,\\Gamma_2/\\Gamma_2^{base}$", "white")
box(58, 33, 36, 10.5, "Loss weight", "$\\lambda=\\mathrm{clip}(0.5\\,\\Gamma_1/\\Gamma_1^{base},0.1,0.9)$", "white")
arrow((17.5, 50), (22, 44))
arrow((50, 50), (72, 44))

# game (middle)
box(3, 15, 27, 14, "Train batch $(x, y)$", "18 audited features\n(standardised)", C["data"])
box(36, 15, 28, 14, "Attacker (follower)", "$L_\\infty$ PGD, 10 steps\nfraud rows only, $\\|\\delta\\|_\\infty\\leq\\varepsilon$\nstep weights $\\geq 0.5$", C["att"])
box(70, 15, 28, 14, "Defender (leader)", "FraudTransformer\n$(1-\\lambda)\\mathcal{L}(x)+\\lambda\\mathcal{L}(x_{adv})$\nAdamW update", C["dfd"])
arrow((30, 22), (36, 22))
arrow((64, 22), (70, 22), r"$x_{adv}$", ty=2.6)
arrow((27, 33), (48, 29.5))
arrow((76, 33), (84, 29.5))
arrow((78, 14.2), (54, 14.2), r"commits $\theta$", ty=-1.9)

# validation feedback (bottom)
box(18, 0.3, 64, 10.2, "Validation (end of round $r$)",
    "gap $g_r=(\\mathcal{L}_{adv}-\\mathcal{L}_{clean})/\\mathcal{L}_{clean}$,   $s=\\mathrm{clip}(g_r/g_1-1,-1,1)$\ncheckpoint = best robust val. AUC (PGD, $\\varepsilon$=0.5); patience 6", C["val"])
arrow((90, 15), (80, 10.9))
ax.add_patch(FancyArrowPatch((18, 5.4), (-2, 58), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#8A6100",
             connectionstyle="angle,angleA=180,angleB=-90,rad=6", linestyle="--"))
ax.text(-4.2, 31, "$g_1,g_2,\\eta_r\\leftarrow$ base$\\cdot(1-0.5s)$", rotation=90, fontsize=6.8, color="#8A6100",
        ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
plt.savefig("arch_game.png", dpi=300, bbox_inches="tight"); print("ok")
