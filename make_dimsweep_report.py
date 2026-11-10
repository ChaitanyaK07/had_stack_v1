"""Table and figure of the Study B dimension sweep from results/studyb_dimsweep/d{d}_seed{s}.json.
Writes results/studyb_dimsweep/summary.json, table_dimsweep.tex and q_dimsweep.pdf.
Usage: python make_dimsweep_report.py [results/studyb_dimsweep]"""
import glob, json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results/studyb_dimsweep"
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/d*_seed*.json"))]
dims = sorted({r["game"]["d"] for r in runs})
S, rows = {}, []


def ms(v, k=3):
    v = np.asarray(v, float); return f"{v.mean():.{k}f} $\\pm$ {v.std():.{k}f}"


for d in dims:
    rr = [r for r in runs if r["game"]["d"] == d]
    g = lambda m, e, k: [r[m]["eval"][e][k] for r in rr]
    st = [r["game"]["stationarity"] for r in rr]
    first = [x[1] for x in st]
    sec = [np.mean([t["sec"] for t in r["game"]["traj"]]) for r in rr]
    qp = [sum(t["qpso"] for t in r["game"]["traj"]) / sum(t["qpso"] + t["als"] for t in r["game"]["traj"]) for r in rr]
    gain = np.array(g("game", "0.1", "fid_adv_target")) - np.array(g("noattack", "0.1", "fid_adv_target"))
    dEr = np.array(g("game", "0.1", "dE")) - np.array(g("noattack", "0.1", "dE"))
    S[d] = dict(n=len(rr), q=rr[0]["game"]["q"], P=rr[0]["game"]["n_quantum"], C=rr[0]["game"]["n_classical"],
                **{f"{m}_{k}_{e}": (float(np.mean(g(m, e, k))), float(np.std(g(m, e, k))))
                   for m in ("game", "noattack", "standard") for k in ("dE", "fid_adv", "fid_clean", "fid_adv_target", "rob")
                   for e in ("0.05", "0.1", "0.2")},
                gain_fid=(float(gain.mean()), float(gain.std()), int((gain > 0).sum())),
                gain_dE=(float(dEr.mean()), float(dEr.std()), int((dEr < 0).sum())),
                rollstd=(float(np.mean([x[0] for x in st])), float(np.std([x[0] for x in st]))),
                stationary=sum(f is not None for f in first),
                first=[f for f in first],
                qpso_share=float(np.mean(qp)), sec_epoch=(float(np.min(sec)), float(np.max(sec))),
                dE_traj=np.mean([[t["dE"] for t in r["game"]["traj"]] for r in rr], 0).tolist(),
                rs_traj=np.mean([x[2] for x in st], 0).tolist())
    s = S[d]
    stat = f"{s['stationary']}/{s['n']}"
    fe = [f for f in first if f is not None]
    pair = lambda k, nd=3: f"{np.mean(g('noattack','0.1',k)):.{nd}f} / {np.mean(g('game','0.1',k)):.{nd}f}"
    rows.append(f"{d} & {s['q']} & {s['P']} & {pair('fid_clean')} & {pair('dE')} & {pair('fid_adv_target')} & "
                f"{(dEr < 0).sum()}/{len(dEr)} & {(gain > 0).sum()}/{len(gain)} & {np.mean(g('game','0.1','rob')):.2f} & "
                f"{100 * s['qpso_share']:.0f} & {np.mean([x[0] for x in st]):.4f} & {stat}" + (f" ({min(fe)}--{max(fe)})" if fe else "") +
                f" & {s['sec_epoch'][0]:.1f}--{s['sec_epoch'][1]:.1f}\\\\")
json.dump(S, open(f"{R}/summary.json", "w"), indent=1)
open(f"{R}/table_dimsweep.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
for d in dims:
    s = S[d]
    print(d, "standard fid_clean", s["standard_fid_clean_0.1"], "noattack fid_clean", s["noattack_fid_clean_0.1"],
          "dE std", s["standard_dE_0.1"], "| dE game 0.05/0.2", s["game_dE_0.05"], s["game_dE_0.2"], "noatk", s["noattack_dE_0.05"], s["noattack_dE_0.2"],
          "| gain_dE", s["gain_dE"], "| qpso", round(s["qpso_share"], 2), "| s/ep", s["sec_epoch"], "| first", s["first"])

# figure
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
x = np.arange(len(dims)); col = {"game": "#2f6fd0", "noattack": "#e8643c", "standard": "#8a8a8a"}
lab = {"game": "Stackelberg game", "noattack": "no attack ($\\delta z{=}0$)", "standard": "standard Q-VAE"}
for m in ("game", "noattack"):
    v = np.array([S[d][f"{m}_fid_adv_target_0.1"] for d in dims])
    ax[0].errorbar(x, v[:, 0], v[:, 1], marker="o", ms=3.5, lw=1.3, capsize=2, color=col[m], label=lab[m])
    v = np.array([S[d][f"{m}_dE_0.1"] for d in dims])
    ax[1].errorbar(x, v[:, 0], v[:, 1], marker="o", ms=3.5, lw=1.3, capsize=2, color=col[m], label=lab[m])
for a, t, yl in zip(ax[:2], ("(a) Fidelity under attack", "(b) Leader's energy gain"),
                    (r"$|\langle x|\psi_{\mathrm{adv}}\rangle|^2$", r"$\Delta E$ (spectral width 2)")):
    a.set_xticks(x, [str(d) for d in dims]); a.set_xlabel("Hilbert-space dimension $d$"); a.set_title(t, loc="left"); a.set_ylabel(yl); a.grid(alpha=0.3, lw=0.5)
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
pal = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
for i, d in enumerate(dims):
    ep = np.arange(1, len(S[d]["rs_traj"]) + 1)
    ax[2].plot(ep[4:], S[d]["rs_traj"][4:], color=pal[i % 4], lw=1.3, label=f"$d={d}$")   # window of 5 epochs full from epoch 5
ax[2].axhline(0.01, color="0.4", ls=":", lw=0.8); ax[2].set_yscale("log")
ax[2].set_xlabel("epoch"); ax[2].set_xlim(4, None); ax[2].set_ylabel(r"rolling std of $\Delta E$ (5 epochs)"); ax[2].set_title("(c) Stationarity", loc="left")
ax[2].legend(frameon=False, fontsize=7); ax[2].grid(alpha=0.3, lw=0.5)
fig.tight_layout(pad=0.4); fig.savefig(f"{R}/q_dimsweep.pdf"); print("saved", f"{R}/q_dimsweep.pdf")
