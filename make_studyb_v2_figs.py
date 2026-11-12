"""Figures for Study B (v2 runs): q_conv_v2.pdf (in-loop energy gain and its rolling std, H1 and H_V, five seeds) and
q_advex_v2.pdf (target, clean and attacked reconstructions of one state, seed 1, retrained deterministically).
Usage: python make_studyb_v2_figs.py"""
import glob, json, numpy as np, torch, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import studyb_v2 as S
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
R = "results/studyb_v2"
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
for a, (fam, d, name) in zip(ax, (("H1", 4, "$H_1=H_2$ ($d{=}4$)"), ("HV", 16, r"$H_V$ ($d{=}16$)"))):
    T = []
    for s in range(1, 6):
        r = json.load(open(f"{R}/runs/{fam}_d{d}_s{s}_game_fcvqc_both_energy_exact.json")); x = np.array([t["dE"] for t in r["traj"]]) / np.mean(r["w_re"])
        T.append(x); a.plot(range(1, 31), x, color="#2f6fd0", lw=0.6, alpha=0.4)
        rs = np.array([x[max(0, i - 4):i + 1].std() for i in range(len(x))]); a.plot(range(5, 31), rs[4:], color="#e8643c", lw=0.6, alpha=0.4, ls="--")
    a.plot(range(1, 31), np.mean(T, 0), color="#2f6fd0", lw=1.6, label=r"$\Delta E/w_{\mathrm{re}}$ (mean of 5 seeds)")
    a.axhline(0.005, color="0.4", ls=":", lw=0.8)
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(name, loc="left"); a.grid(alpha=0.3, lw=0.5)
ax[0].set_ylabel(r"in-loop $\Delta E$ / spectral width"); ax[0].plot([], [], color="#e8643c", ls="--", label="rolling std (5 epochs)"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(pad=0.4); fig.savefig(f"{R}/q_conv_v2.pdf")

fig, ax = plt.subplots(2, 2, figsize=(6.6, 3.6), gridspec_kw=dict(height_ratios=[2, 1]))
for j, (fam, d, name) in enumerate((("H1", 4, "$H_1=H_2$"), ("HV", 16, r"$H_V$"))):
    model, game, V, X, Hr, codec = S.run(fam, d, 1, "game", ret_model=True)
    k = 8; v, x, H = V[k], X[k], Hr[k]
    with torch.no_grad():
        mu = model.encode(v[None])[0][0]; dz, j_, _ = game.leader(mu, v, H, 0.1, torch.Generator().manual_seed(78), "both", "energy")
        pc, pa = codec.state(model.decode(mu[None])[0]), codec.state(model.decode((mu + dz)[None])[0])
    ph = lambda p: p * np.exp(-1j * np.angle(np.vdot(x.numpy(), p.numpy())))
    pc, pa = ph(pc), ph(pa); idx = np.arange(d); w = 0.27
    ax[0, j].bar(idx - w, np.abs(x.numpy()), w, color="0.6", label="target $|x_k|$")
    ax[0, j].bar(idx, np.abs(pc), w, color="#2f6fd0", label=r"clean $|\psi_k|$"); ax[0, j].bar(idx + w, np.abs(pa), w, color="#e8643c", label=r"attacked $|\psi_k|$")
    F = abs(np.vdot(pc, pa)) ** 2; Ft = abs(np.vdot(x.numpy(), pa)) ** 2
    ax[0, j].set_title(f"{name}: $\\Delta E={j_:.4f}$, $F_{{ca}}={F:.4f}$, $F_{{\\mathrm{{att}}}}={Ft:.4f}$", loc="left", fontsize=7.5)
    ax[1, j].bar(idx, np.abs(pa - pc), color="#7570b3"); ax[1, j].set_ylabel(r"$|\psi^{\mathrm{att}}_k-\psi^{\mathrm{cl}}_k|$"); ax[1, j].set_xlabel("basis state $k$")
    for a in ax[:, j]:
        a.grid(alpha=0.3, lw=0.5, axis="y"); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax[0, 0].set_ylabel("amplitude modulus"); ax[0, 0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.4); fig.savefig(f"{R}/q_advex_v2.pdf"); print("saved")
