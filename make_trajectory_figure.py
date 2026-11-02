"""Fig.: HAD-Stack controller trajectory (seed 42). Reads logs/v2_traj_v3.csv, written by
`python had_stack_v2.py --method v3`, and saves fig_trajectory.png."""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
tr = pd.read_csv("logs/v2_traj_v3.csv")
f, a = plt.subplots(figsize=(4.2, 3.2))
a.plot(tr["round"], tr["eps"], "-o", ms=3, label=r"$\varepsilon=\lambda$ (Hamiltonian)")
a.set_xlabel("Round"); a.set_ylabel(r"$\varepsilon$, $\lambda$")
a2 = a.twinx(); a2.plot(tr["round"], tr["robust_val_auc"], "s--", ms=3, color="tab:red", label="Robust val. AUC")
a2.set_ylabel("Robust validation AUC")
a2.axvline(tr.loc[tr.robust_val_auc.idxmax(), "round"], color="gray", ls=":", lw=1)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right"); a.grid(alpha=.3)
f.tight_layout(); f.savefig("fig_trajectory.png", dpi=300); print("saved fig_trajectory.png")
