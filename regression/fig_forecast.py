import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
from hadstack_thlstm import *
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, '..', 'paper', 'figs')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
D = cell13_data(); base = cell13_train(D, 0)
had, log, br = game(base, D, 1, verbose=False)
te = D["te"]; out = {"best_round": br}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for a, (name, m) in zip(ax, [("Undefended", base), ("HAD-Stack", had)]):
    d = make_pgd(m)(te["X"], te["true"], te["prev"], 0.5, None, D["market"], 100, 5, "se")
    f0, fa = predict(m, te["X"], te["prev"]), predict(m, te["X"] + d, te["prev"])
    out[name] = dict(clean_rmse=float(np.sqrt(np.mean((f0-te["true"])**2))), adv_rmse=float(np.sqrt(np.mean((fa-te["true"])**2))))
    t = np.arange(len(f0))
    a.plot(t, te["true"], "k-", lw=1, label="actual"); a.plot(t, f0, lw=.9, label="forecast, clean")
    a.plot(t, fa, lw=.9, alpha=.85, label=r"forecast, PGD-100$\times$5, $\varepsilon{=}0.5$")
    a.set_title(name, fontsize=9); a.set_xlabel("test day", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("10Y series", fontsize=8); ax[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "f_bond_forecast.pdf")); json.dump(out, open(os.path.join(RES, "fig_forecast.json"), "w"), indent=1); print(out)
