"""Tables, figures and checks for the Study B v2 runs (results/studyb_v2/runs/*.json, from jobs_v2.txt).
Writes results/studyb_v2/report.md, summary.json and figures q_fid_v2.pdf, q_nh_sweep.pdf.
Definitions: fidelity under attack = |<x|psi_adv>|^2 to the target; clean-adversarial fidelity = |<psi_clean|psi_adv>|^2
(the paper's earlier 'fidelity'); Rob. = mean L_c / mean L_a; attack success = share of states whose energy gain under
the evaluation attack exceeds 1% of the spectral width w_re of H_re; stationarity = rolling std (5 epochs) of the
in-loop mean energy gain below 0.5% of w_re."""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "results/studyb_v2"; runs = {}
for f in glob.glob(f"{R}/runs/*.json"):
    r = json.load(open(f))
    if r.get("skip"): continue
    runs[os.path.basename(f)[:-5]] = r
L = []  # report lines
P = lambda *a: L.append(" ".join(str(x) for x in a))


def key(fam, d, s, variant="game", model="fcvqc", leader="both", payoff="energy", follower="exact", real=False):
    return f"{fam.replace(':', '')}_d{d}_s{s}_{variant}_{model}_{leader}_{payoff}_{follower}{'_real' if real else ''}"


def metrics(r, e="0.1"):
    rows = r["eval"][e]; w = np.array(r["w_re"]); dE = np.array([x["dE"] for x in rows])
    M = {k: float(np.mean([x[k] for x in rows])) for k in ("dE", "fid_cc", "fid_clean", "fid_adv", "Lc", "La")}
    M["rob"] = M["Lc"] / M["La"]; M["atk"] = float(100 * np.mean(dE > 0.01 * w))
    M["dE_rel"] = float(np.mean(dE / w))
    tr = r["traj"]; M["qpso"] = sum(t["qpso"] for t in tr); M["als"] = sum(t["als"] for t in tr)
    if r["variant"] == "game":
        x = np.array([t["dE"] for t in tr]); rs = np.array([x[max(0, i - 4):i + 1].std() for i in range(len(x))])
        thr = 0.005 * np.mean(w); ok = [i for i in range(4, len(x)) if (rs[i:] < thr).all()]
        M["rollstd_rel"] = float(rs[-1] / np.mean(w)); M["stationary_from"] = (ok[0] + 1) if ok else None
    if "jacobian_F" in r["eval"]: M["jac"] = float(np.mean(r["eval"]["jacobian_F"]))
    return M


def agg(keys, e="0.1"):
    ms = [metrics(runs[k], e) for k in keys if k in runs]
    if not ms: return None
    out = {"n": len(ms)}
    for k in ms[0]:
        v = [m[k] for m in ms if m.get(k) is not None]
        if v and isinstance(v[0], (int, float)): out[k] = (float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)))
    out["stationary"] = sum(m.get("stationary_from") is not None for m in ms) if "stationary_from" in ms[0] else None
    out["stat_epochs"] = [m.get("stationary_from") for m in ms]
    return out


def fmt(a, k, nd=3):
    return "--" if a is None or k not in a else f"{a[k][0]:.{nd}f} ± {a[k][1]:.{nd}f}"


S = {}
# ---------------------------------------------------------------- 1. benchmark game (Table 23)
P("## 1. Benchmark systems, exact gradient, Route A targets (5 seeds; H2 = H1 under the complex treatment)\n")
P("| System | Model | Variant | n | clean fid | fid under attack | clean-adv fid | dE | dE/w_re | Rob | Atk % | QPSO/ALS | stationary |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for fam, d in (("H1", 4), ("HV", 16)):
    for model in ("fcvqc", "mlp"):
        for v in ("game", "noattack", "standard"):
            a = agg([key(fam, d, s, v, model) for s in range(1, 6)])
            if a is None: continue
            S[f"bench_{fam}_{model}_{v}"] = a
            st = f"{a['stationary']}/{a['n']} {a['stat_epochs']}" if a.get("stationary") is not None else ""
            P(f"| {fam} | {model} | {v} | {a['n']} | {fmt(a,'fid_clean')} | {fmt(a,'fid_adv')} | {fmt(a,'fid_cc',4)} | {fmt(a,'dE',4)} | {fmt(a,'dE_rel',4)} | {fmt(a,'rob',2)} | {fmt(a,'atk',1)} | {a['qpso'][0]:.0f}/{a['als'][0]:.0f} | {st} |")
# paired game vs noattack
for fam, d in (("H1", 4), ("HV", 16)):
    pairs = [(metrics(runs[key(fam, d, s)]), metrics(runs[key(fam, d, s, "noattack")])) for s in range(1, 6)
             if key(fam, d, s) in runs and key(fam, d, s, "noattack") in runs]
    if pairs:
        P(f"\n{fam}: game vs no attack, dE lower in {sum(g['dE'] < n['dE'] for g, n in pairs)}/{len(pairs)}, fidelity under attack higher in "
          f"{sum(g['fid_adv'] > n['fid_adv'] for g, n in pairs)}/{len(pairs)}, clean-adv fidelity higher in {sum(g['fid_cc'] > n['fid_cc'] for g, n in pairs)}/{len(pairs)}")
    pairs = [(metrics(runs[key(fam, d, s)]), metrics(runs[key(fam, d, s, "standard")])) for s in range(1, 6)
             if key(fam, d, s) in runs and key(fam, d, s, "standard") in runs]
    if pairs:
        P(f"{fam}: game vs standard, dE lower in {sum(g['dE'] < n['dE'] for g, n in pairs)}/{len(pairs)}, fidelity under attack higher in "
          f"{sum(g['fid_adv'] > n['fid_adv'] for g, n in pairs)}/{len(pairs)}")

# ---------------------------------------------------------------- 2. Fig. 15 (H1, fidelity vs budget)
eps = ["0.02", "0.05", "0.1", "0.15", "0.2", "0.3"]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
plt.rcParams.update({"font.family": "serif", "font.size": 8})
col = {"game": "#2f6fd0", "standard": "#8a8a8a", "noattack": "#e8643c"}
lab = {"game": "Stackelberg-trained", "standard": "standard Q-VAE", "noattack": "no attack ($\\delta z{=}0$)"}
P("\n## 2. H1: fidelity under the ALS+QPSO attack vs budget (seeds 1-5; mean [min, max])\n")
for v in ("game", "standard", "noattack"):
    ks = [key("H1", 4, s, v) for s in range(1, 6) if key("H1", 4, s, v) in runs]
    if not ks: continue
    for j, m in enumerate(("fid_adv", "fid_cc")):
        Y = np.array([[metrics(runs[k], e)[m] for e in eps] for k in ks]); x = [float(e) for e in eps]
        ax[j].plot(x, Y.mean(0), marker="o", ms=3, color=col[v], label=lab[v]); ax[j].fill_between(x, Y.min(0), Y.max(0), color=col[v], alpha=0.15)
        S[f"fid_curve_{v}_{m}"] = Y.tolist()
        P(f"{v:9s} {m:8s}: " + "  ".join(f"{e}: {Y.mean(0)[i]:.3f} [{Y.min(0)[i]:.3f}, {Y.max(0)[i]:.3f}]" for i, e in enumerate(eps)))
    jac = [metrics(runs[k]).get("jac") for k in ks]
    if all(j is not None for j in jac): P(f"{v:9s} decoder Jacobian norm {np.mean(jac):.2f} [{min(jac):.2f}, {max(jac):.2f}]")
for a_, t in zip(ax, ("(a) Fidelity to the target", "(b) Clean--adversarial fidelity")):
    a_.set_xlabel(r"budget $\varepsilon$"); a_.set_title(t, loc="left"); a_.grid(alpha=0.3, lw=0.5)
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
ax[0].set_ylabel("fidelity"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(pad=0.4); fig.savefig(f"{R}/q_fid_v2.pdf")

# ---------------------------------------------------------------- 3. ablations on H1 (Table 25)
P("\n## 3. Ablations on H1 (seeds 1-3, 30 epochs, eps 0.1)\n")
P("| # | Variant | n | Lc | La | Rob | dE | clean-adv fid | fid under attack | clean fid | Atk % |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
rows = [("No attack (dz=0)", dict(variant="noattack")), ("Standard Q-VAE (clean-trained)", dict(variant="standard")),
        ("ALS only", dict(leader="als")), ("QPSO only", dict(leader="qpso")), ("ALS+QPSO", {}),
        ("Finite-difference follower", dict(follower="fd")), ("Earlier loss-level rule", dict(follower="old")),
        ("Plain VQC", dict(model="plain")), ("Min-max (simultaneous)", dict(follower="simultaneous")),
        ("Classical MLP VAE (matched)", dict(model="mlp")), ("Payoff: reconstruction error", dict(payoff="recon")),
        ("Payoff: ||dz||_2", dict(payoff="norm")), ("Payoff: random linear", dict(payoff="linear"))]
for i, (name, kw) in enumerate(rows, 1):
    a = agg([key("H1", 4, s, **kw) for s in (1, 2, 3)]); S[f"abl_H1_{name}"] = a
    if a is None: P(f"| {i} | {name} | 0 | | | | | | | | |"); continue
    P(f"| {i} | {name} | {a['n']} | {a['Lc'][0]:.3f} | {a['La'][0]:.3f} | {a['rob'][0]:.2f} | {a['dE'][0]:.4f} | {a['fid_cc'][0]:.4f} | {a['fid_adv'][0]:.3f} | {a['fid_clean'][0]:.3f} | {a['atk'][0]:.1f} |")
P("\nH_V payoff variants (seed 1):")
for p_ in ("energy", "recon", "norm", "linear"):
    a = agg([key("HV", 16, 1, payoff=p_)]); S[f"abl_HV_payoff_{p_}"] = a
    if a: P(f"  {p_:7s}: dE {a['dE'][0]:.4f}  fid under attack {a['fid_adv'][0]:.4f}  clean-adv {a['fid_cc'][0]:.4f}  clean {a['fid_clean'][0]:.4f}")

# ---------------------------------------------------------------- 4. quantum vs classical across the first sweep
P("\n## 4. FC-VQC (first sweep, real encoding) vs matched classical MLP VAE (5 seeds)\n")
P("| d | FC-VQC dE na/game | MLP dE na/game | FC-VQC fid att na/game | MLP fid att na/game | FC-VQC clean fid | MLP clean fid | MLP hidden |"); P("|---|---|---|---|---|---|---|---|")
DS = "results/studyb_dimsweep"
for d in (4, 9, 16, 25):
    fv = [json.load(open(f)) for f in sorted(glob.glob(f"{DS}/d{d}_seed*.json"))]
    g = lambda mode, k: np.mean([r[mode]["eval"]["0.1"][k] for r in fv]) if fv else float("nan")
    a = agg([key("goe", d, s, "game", "mlp", real=True) for s in range(1, 6)]); b = agg([key("goe", d, s, "noattack", "mlp", real=True) for s in range(1, 6)])
    S[f"mlp_goe_{d}"] = dict(game=a, noattack=b)
    if a and b:
        P(f"| {d} | {g('noattack','dE'):.3f} / {g('game','dE'):.3f} | {b['dE'][0]:.3f} / {a['dE'][0]:.3f} | {g('noattack','fid_adv_target'):.3f} / {g('game','fid_adv_target'):.3f} | "
          f"{b['fid_adv'][0]:.3f} / {a['fid_adv'][0]:.3f} | {g('game','fid_clean'):.3f} | {a['fid_clean'][0]:.3f} | |")
P("\nPaired per seed (game-trained models, eps 0.1): FC-VQC vs MLP")
for d in (4, 9, 16, 25):
    c1 = c2 = c3 = n = 0
    for s in range(1, 6):
        f = f"{DS}/d{d}_seed{s}.json"; k = key("goe", d, s, "game", "mlp", real=True)
        if not (os.path.exists(f) and k in runs): continue
        q = json.load(open(f))["game"]["eval"]["0.1"]; m = metrics(runs[k]); n += 1
        c1 += q["fid_adv_target"] > m["fid_adv"]; c2 += q["dE"] < m["dE"]; c3 += q["fid_clean"] > m["fid_clean"]
    hid = runs.get(key("goe", d, 1, "game", "mlp", real=True), {}).get("n_circuit")
    P(f"  d={d}: FC-VQC higher fidelity under attack in {c1}/{n}, lower dE in {c2}/{n}, higher clean fidelity in {c3}/{n}; MLP parameters per VAE (both networks) {hid}")
for fam, d in (("H1", 4), ("HV", 16)):
    c1 = c2 = n = 0
    for s in (1, 2, 3):
        a_, b_ = key(fam, d, s), key(fam, d, s, model="mlp")
        if a_ in runs and b_ in runs:
            q, m = metrics(runs[a_]), metrics(runs[b_]); n += 1; c1 += q["fid_adv"] > m["fid_adv"]; c2 += q["dE"] < m["dE"]
    if n: P(f"  {fam}: FC-VQC higher fidelity under attack in {c1}/{n}, lower dE in {c2}/{n}; circuit params {runs[key(fam,d,1)]['n_circuit']}, MLP params {runs[key(fam,d,1,model='mlp')]['n_circuit']}")
rk = key("goe", 4, 1, "game", real=True)
if rk in runs and os.path.exists(f"{DS}/d4_seed1.json"):
    o = json.load(open(f"{DS}/d4_seed1.json"))["game"]["traj"]; n = runs[rk]["traj"]
    P(f"\nReproduction of the first sweep (d=4, seed 1) by studyb_v2.py: max |dE_epoch difference| = {max(abs(a_['dE'] - b_['dE']) for a_, b_ in zip(o, n)):.2e}")

# ---------------------------------------------------------------- 5. non-Hermitian sweep
P("\n## 5. Non-Hermitian sweep (complex encoding; r = |H_anti|_F/|H_herm|_F; 0 = Hermitian arm; 0.105 = H_V; 2.531 = H1)\n")
P("| d | r | n | clean fid na/game | dE na/game | fid under attack na/game | dE lower | fid higher | Rob game | stationary |"); P("|---|---|---|---|---|---|---|---|---|---|")
NH = {}
for d in (4, 9, 16):                      # d = 25 (D = 64) does not fit in 30 epochs with the complex encoding; see report
    for r_ in ("nh:0", "nh:0.105", "nh:2.531"):
        ks = [(key(r_, d, s), key(r_, d, s, "noattack")) for s in range(1, 6)]
        ks = [(a_, b_) for a_, b_ in ks if a_ in runs and b_ in runs]
        if not ks: continue
        G = [metrics(runs[a_]) for a_, _ in ks]; N = [metrics(runs[b_]) for _, b_ in ks]
        m_ = lambda X, k: np.mean([x[k] for x in X])
        NH[(d, r_)] = dict(G=G, N=N)
        st = sum(x.get("stationary_from") is not None for x in G)
        P(f"| {d} | {r_[3:]} | {len(ks)} | {m_(N,'fid_clean'):.3f} / {m_(G,'fid_clean'):.3f} | {m_(N,'dE'):.4f} / {m_(G,'dE'):.4f} | {m_(N,'fid_adv'):.3f} / {m_(G,'fid_adv'):.3f} | "
          f"{sum(g['dE'] < n['dE'] for g, n in zip(G, N))}/{len(ks)} | {sum(g['fid_adv'] > n['fid_adv'] for g, n in zip(G, N))}/{len(ks)} | {m_(G,'rob'):.2f} | {st}/{len(ks)} |")
S["nh"] = {f"{d}_{r}": {k: [x[k] for x in v[k2]] for k2 in ("G", "N") for k in ("dE", "fid_adv", "fid_clean")} for (d, r), v in NH.items()}
if NH:
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4)); cols = {"nh:0": "#2f6fd0", "nh:0.105": "#1fa37a", "nh:2.531": "#e8643c"}
    for r_ in ("nh:0", "nh:0.105", "nh:2.531"):
        ds = [d for d in (4, 9, 16) if (d, r_) in NH]
        for j, k in enumerate(("fid_adv", "dE")):
            G = [np.mean([x[k] for x in NH[(d, r_)]["G"]]) for d in ds]; N = [np.mean([x[k] for x in NH[(d, r_)]["N"]]) for d in ds]
            ax[j].plot(range(len(ds)), G, "-o", ms=3, color=cols[r_], label=f"game, $r={r_[3:]}$"); ax[j].plot(range(len(ds)), N, "--o", ms=3, mfc="white", color=cols[r_], label=f"no attack, $r={r_[3:]}$")
            ax[j].set_xticks(range(len(ds)), [str(d) for d in ds])
    for a_, t, yl in zip(ax, ("(a) Fidelity under attack", "(b) Leader's energy gain"), (r"$|\langle x|\psi_{\mathrm{adv}}\rangle|^2$", r"$\Delta E$")):
        a_.set_xlabel("dimension $d$"); a_.set_title(t, loc="left"); a_.set_ylabel(yl); a_.grid(alpha=0.3, lw=0.5)
    ax[0].legend(frameon=False, fontsize=6, ncol=1); fig.tight_layout(pad=0.4); fig.savefig(f"{R}/q_nh_sweep.pdf")

d25 = [metrics(runs[k])["fid_clean"] for k in runs if "_d25_" in k and k.startswith("nh")]
if d25: P(f"\nd = 25 with the complex encoding (D = 64, {len(d25)} runs): clean fidelity {min(d25):.3f}-{max(d25):.3f}; not fitted in 30 epochs, excluded.")
import studyb_v2 as SV
for fam, d in (("H1", 4), ("HV", 16), ("goe", 4), ("goe", 25)):
    Hs, xs, info = SV.family(fam, d, 1); F = np.abs(xs.conj() @ xs.T) ** 2
    P(f"Family variation {fam} d={d}: min pairwise fidelity between its 16 targets {F.min():.4f}; r = {info['r']:.3f}")
# ---------------------------------------------------------------- 6. energy-fidelity bound
worst, npair = 0.0, 0
for r in runs.values():
    for e, rows_ in r["eval"].items():
        if e == "jacobian_F": continue
        for x, w in zip(rows_, r["w_re"]):
            b = w * np.sqrt(max(0.0, 1 - x["fid_cc"])); npair += 1
            if b > 0: worst = max(worst, abs(x["dE"]) / b)
P(f"\n## 6. Energy-fidelity bound |dE| <= w_re sqrt(1 - F_cc): {npair} (dE, F) pairs, max ratio {worst:.3f}")
S["bound"] = dict(pairs=npair, max_ratio=worst)
if os.path.exists(f"{R}/ps_checks.json"): S["ps_checks"] = json.load(open(f"{R}/ps_checks.json"))
if os.path.exists(f"{R}/pauli_rebuild.json"): S["pauli_rebuild"] = json.load(open(f"{R}/pauli_rebuild.json"))
open(f"{R}/report.md", "w").write("\n".join(L) + "\n"); json.dump(S, open(f"{R}/summary.json", "w"), indent=1, default=str)
print("\n".join(L))
