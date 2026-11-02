"""Fig. 2 (fig:conv_hv): convergence of the H_V field iteration, complex treatment, phi_0 = 0.1,
at the controller's three coupling scales s in {+1, 0, -1}.

Reads the traces written by hv_convergence.py (results/hamiltonian_layer/hv_convergence.json) and draws
(a) the iterates phi_n and (b) |phi_n - phi*| on a log scale with the reference rate 0.1 * 2^-n.
Run from the repository root after hv_convergence.py:  python make_fig_hv_convergence.py
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/hamiltonian_layer/hv_convergence.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/hamiltonian_layer/hv_convergence.pdf"
runs = json.load(open(src))
sel = {r["s"]: r for r in runs if r["kind"] == "complex" and abs(r["phi0"] - 0.1) < 1e-12}
assert set(sel) == {1.0, 0.0, -1.0}, "expected complex runs with phi0=0.1 at s=+1,0,-1"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
style = {1.0: dict(color="#2f6fd0", ls="-", marker="o"), 0.0: dict(color="#e8643c", ls="--", marker="s"),
         -1.0: dict(color="#1fa37a", ls=":", marker="^")}
lab = {1.0: "+1", 0.0: "0", -1.0: "-1"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35))
for s in (1.0, 0.0, -1.0):
    r = sel[s]; tr = np.array(r["trace"]); n = np.arange(len(tr)); ps = r["phi_star"]
    kw = dict(lw=1.6, ms=4, markevery=3, **style[s])
    ax[0].plot(n, tr, label=rf"$s={lab[s]}$, $\phi^\star={ps:.5f}$", **kw)
    d = np.abs(tr - ps); m = d > 0
    ax[1].plot(n[m], d[m], label=rf"$s={lab[s]}$", **kw)
nmax = max(len(r["trace"]) for r in sel.values())
ax[1].plot(np.arange(nmax + 1), 0.1 * 2.0 ** -np.arange(nmax + 1), color="0.5", ls="-.", lw=1, label=r"$0.1\cdot2^{-n}$")
ax[0].set_yscale("log"); ax[1].set_yscale("log")
ax[0].set_title(r"(a) Field iterates, $\phi_0=0.1$"); ax[1].set_title("(b) Distance to the fixed point")
ax[0].set_ylabel(r"field $\phi_n$"); ax[1].set_ylabel(r"$|\phi_n-\phi^\star|$")
for a in ax: a.set_xlabel(r"iteration $n$"); a.grid(alpha=0.3, lw=0.5); a.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out); print("saved", out)
