"""Writes results_adagb/new_tables.tex: the bodies of every table of the paper whose numbers change under
HCS_ORDER=adagb and that can be recomputed here (Hamiltonian layer, bond study with PGD and QPSO followers),
in the same column layout as the paper. Run after the reruns:  HCS_ORDER=adagb python make_adagb_tables.py"""
import json, os, numpy as np
import hamiltonian_v2 as hv
HERE = os.path.dirname(os.path.abspath(__file__)); RG = os.path.join(HERE, "regression", "results")
U, FB, H, HQ, HD = "Two-Head LSTM (undefended)", "FB-AT (eps max)", "HAD-Stack", "HAD-Stack (QPSO)", "HAD-Stack (PGD+QPSO)"
out = []; P = out.append
cz = lambda z: f"${z.real:+.5f}{z.imag:+.5f}i$".replace("$+", "$")

# --- Hamiltonian layer ---------------------------------------------------------------------------------
V = json.load(open(os.path.join(HERE, "results", "hamiltonian_layer", "hamiltonian_variants.json")))["order=adagb, h29=solver"]
T = json.load(open(os.path.join(HERE, "results", "hamiltonian_layer", "verify_theorems_adagb.json")))
a = hv.H1v2(dict(hv.PARAMS_H1)); e1 = a.solve(); b = hv.H2v2(dict(hv.PARAMS_H2)); e2 = b.solve()
P("% ===== tab:hamcheck (rows A) =====")
P(f"Complex $\\nu_0$ (A) & {cz(e1[0])} & {cz(e2[0])} & $-1.99943-0.14996i$\\\\")
P(f"Order/field (A) & $\\Delta_1={a.order:.4f}$ & $\\Delta_2={b.order:.4f}$ & $\\phi=0.00601$\\\\")
P(f"% spectrum of H1 = H2 at the fixed point: {np.round(np.sort_complex(e1), 5).tolist()}; iterations from Delta_0 = 0.1 at tol 1e-8: {a.iters}")
P("% ===== tab:ctrl (phi column unchanged) =====")
phi = {1.0: "0.00306", 0.5: "0.00455", 0.0: "0.00601", -0.5: "0.00744", -1.0: "0.00883"}
for r in V["controller"]:
    s = r["s"]; P(f"${s:+.1f}$ & {1 - 0.5 * s:.2f} & ${r['E0']:+.5f}$ & {r['G1']:.5f} & {r['eps']:.4f} & {r['lam']:.4f} & {phi[s]}\\\\".replace("$+0.0$", "$+0.0$"))
P(f"% fixed points Delta* per s: " + ", ".join(f"{x['s']:+.1f}: {x['Delta_star']:.4f}" for x in T["thm2"]))
P(f"% Corollary 1: range {T['cor1']['range']}, Gamma_base {T['cor1']['Gamma_base']:.5f}, max |deps/ds| {T['cor1']['L_eps_numeric']:.4f}")
P(f"% Theorem 2 on I=[0,1]: " + "; ".join(f"s={x['s']:+.1f}: G(I)=[{x['G_min']:.4f},{x['G_max']:.4f}] L={x['L']:.4f} rho={x['rho']:.4f} factor={x['factor']:.4f}" for x in T["thm2"]))
C = json.load(open(os.path.join(HERE, "results", "hamiltonian_layer", "two_mode_convergence.json")))
for op in ("H1", "H2"):
    for kind in ("complex", "hermitian"):
        rr = [r for r in C["runs"] if r["op"] == op and r["kind"] == kind]
        P(f"% two-mode convergence {op} {kind}: iterations {min(r['iters'] for r in rr)}-{max(r['iters'] for r in rr)}, "
          f"converged {sum(r['converged'] for r in rr)}/{len(rr)}, Delta* {sorted(set(round(r['delta_star'], 6) for r in rr))}, "
          f"factor {min(r['factor'] for r in rr):.4f}-{max(r['factor'] for r in rr):.4f}")
P(f"% reference solver (Study B stopping rule, tol 1e-6, <=100 iterations): E0(H1), E0(H2) = {C['reference_E0']}")

# --- bond study, PGD follower -------------------------------------------------------------------------
S = json.load(open(os.path.join(RG, "results_seeds.json"))); F = json.load(open(os.path.join(RG, "results_folds.json"))); R = S + F
nm = lambda r: f"Seed {r['seed']}" if r["split"] == "fixed" else "Fold " + r["split"][4:]
P("% ===== tab:bonddyn (seed 1) =====")
L = S[0]["had_log"]
for l in L:
    if l["round"] not in (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 29, 30): continue
    s = "--" if l["round"] == 0 else f"${l['s']:+.2f}$"; st = "--" if l["round"] < 2 else f"{l['stale']:.1f}"
    rr = "--" if l["round"] == 0 else f"{l['val_rrmse']:.3f}"
    P(f"{l['round']} & {s} & {l['eps']:.4f} & {l['train']:.1f} & {l['fresh']:.1f} & {st} & {l['strong']:.1f} & {rr}\\\\")
P("% ===== tab:bondruns =====")
for r in R:
    L = r["had_log"]; st = [(x["fresh"], x["stale"]) for x in L[1:] if x["stale"] == x["stale"]]; h = st[len(st) // 2:]
    fs = np.mean([p - q for p, q in h])
    P(f"{nm(r)} & {len(L)-1} ({r['had_best_round']}) & {min(x['eps'] for x in L):.4f}--{max(x['eps'] for x in L):.4f} & "
      f"{L[0]['train']:.1f}$\\to${L[-1]['train']:.1f} & {L[0]['strong']:.1f}$\\to${L[-1]['strong']:.1f} & ${fs:+.2f}$ & "
      f"{np.mean([x['strong'] - x['fresh'] for x in L[1:]]):.2f}\\\\")
P("% ===== tab:bondout =====")
added = lambda m, e: m[f"wcMAE@{e}"] - m["MAE"]
for grp, RR, title in (("folds", F, "Five walk-forward folds"), ("seeds", S, "Fixed split, three seeds")):
    P(f"% {title}")
    for k, lab in ((U, "Undefended Two-Head LSTM"), (H, "HAD-Stack"), (FB, "Fixed-budget AT")):
        q = lambda f: (np.mean([f(r["metrics"][k]) for r in RR]), np.std([f(r["metrics"][k]) for r in RR]))
        cells = [("MAE", 3), ("RMSE", 3), ("DA", 1), ("broken@0.5", 1), ("broken@1.0", 1), ("wcRMSE@0.5", 2), ("wcRMSE@1.0", 2)]
        txt = [f"{q(lambda m, x=x: m[x])[0]:.{d}f} $\\pm$ {q(lambda m, x=x: m[x])[1]:.{d}f}" for x, d in cells]
        txt += [f"{q(lambda m, e=e: added(m, e))[0]:.2f} $\\pm$ {q(lambda m, e=e: added(m, e))[1]:.2f}" for e in ("0.5", "1.0")]
        P(f"{lab} & " + " & ".join(txt) + "\\\\")
P("% ===== tab:bondci (broken = -robust) =====")
def sg(x, d):                                                  # signed number; zero printed as \phantom{+}0 like the paper
    t = f"{x:+.{d}f}"; return "\\phantom{+}" + t[1:] if float(t) == 0 else t
br = lambda c: f"${sg(-c['diff'], 1)}$ $[{sg(-c['hi'], 1)},{sg(-c['lo'], 1)}]$"
wr = lambda c: f"${sg(c['diff'], 2)}$ $[{sg(c['lo'], 2)},{sg(c['hi'], 2)}]$"
for r in R:
    c = r["ci"]; P(f"{nm(r)} & {br(c['robust@0.5'])} & {br(c['robust@1.0'])} & {wr(c['wcRMSE@0.5'])} & {wr(c['wcRMSE@1.0'])}\\\\")
P(f"% added error HAD < undefended: eps 0.5 {sum(added(r['metrics'][H], '0.5') < added(r['metrics'][U], '0.5') for r in R)}/8, "
  f"eps 1.0 {sum(added(r['metrics'][H], '1.0') < added(r['metrics'][U], '1.0') for r in R)}/8; "
  f"wcRMSE@0.5 HAD < undefended {sum(r['metrics'][H]['wcRMSE@0.5'] < r['metrics'][U]['wcRMSE@0.5'] for r in R)}/8, "
  f"intervals excluding zero {sum(r['ci']['wcRMSE@0.5']['hi'] < 0 for r in R)}/8; "
  f"wcRMSE@1.0 HAD < undefended {sum(r['metrics'][H]['wcRMSE@1.0'] < r['metrics'][U]['wcRMSE@1.0'] for r in R)}/8, "
  f"excluding zero {sum(r['ci']['wcRMSE@1.0']['hi'] < 0 for r in R)}/8; clean RMSE HAD < undefended {sum(r['metrics'][H]['RMSE'] < r['metrics'][U]['RMSE'] for r in R)}/8; "
  f"FB-AT wcRMSE@0.5 < HAD {sum(r['metrics'][FB]['wcRMSE@0.5'] < r['metrics'][H]['wcRMSE@0.5'] for r in R)}/8")

# --- bond study, QPSO followers -----------------------------------------------------------------------
Q = json.load(open(os.path.join(RG, "qpso_seeds.json"))) + json.load(open(os.path.join(RG, "qpso_folds_12.json")))
P("% ===== tab:bondqloop =====")
for key, lab, mk in (("had_log", "PGD-10", H), ("qpso_log", "QPSO", HQ), ("dual_log", "PGD+QPSO", HD)):
    for i, r in enumerate(Q):
        L = r[key]; st = [(x["fresh"], x["stale"]) for x in L[1:] if x["stale"] == x["stale"]]; h = st[len(st) // 2:]
        P(f"{lab if i == 0 else ''} & {nm(r)} & {len(L)-1} ({r['best_round'][mk]}) & {min(x['eps'] for x in L):.4f}--{max(x['eps'] for x in L):.4f} & "
          f"{L[0]['train']:.1f}$\\to${L[-1]['train']:.1f} & {L[0]['fresh']:.1f}$\\to${L[-1]['fresh']:.1f} & {L[0]['strong']:.1f}$\\to${L[-1]['strong']:.1f} & "
          f"${np.mean([p - q for p, q in h]):+.2f}$ & {np.mean([x['sec'] for x in L[1:]]):.1f}\\\\")
    P("\\midrule")
P("% ===== tab:bondqout =====")
for r in Q:
    for i, (k, lab) in enumerate(((U, "Undefended"), (H, "HAD-Stack, PGD"), (HQ, "HAD-Stack, QPSO"), (HD, "HAD-Stack, PGD+QPSO"), (FB, "Fixed-budget AT"))):
        m = r["metrics"][k]
        P(f"{nm(r) if i == 0 else ''} & {lab} & {m['RMSE']:.3f} & {m['wcRMSE@0.5']:.2f} & {m['wcRMSE@1.0']:.2f} & {m['q_wcRMSE@0.5']:.2f} & "
          f"{m['q_wcRMSE@1.0']:.2f} & {m['all_wcRMSE@0.5']:.2f} & {m['all_wcRMSE@1.0']:.2f} & {m['all_broken@0.5']:.1f} / {m['all_broken@1.0']:.1f} & "
          f"{m['all_wcMAE@0.5'] - m['MAE']:.2f} / {m['all_wcMAE@1.0'] - m['MAE']:.2f}\\\\")
    P("\\midrule")
P("% joint-attack CIs vs undefended (WC-RMSE 0.5 / 1.0):")
for k in (H, HQ, HD):
    P("% " + k + ": " + "; ".join(f"{nm(r)} {r['ci_all'][k]['wcRMSE@0.5']['diff']:+.2f} [{r['ci_all'][k]['wcRMSE@0.5']['lo']:+.2f},{r['ci_all'][k]['wcRMSE@0.5']['hi']:+.2f}]"
                                   f" / {r['ci_all'][k]['wcRMSE@1.0']['diff']:+.2f} [{r['ci_all'][k]['wcRMSE@1.0']['lo']:+.2f},{r['ci_all'][k]['wcRMSE@1.0']['hi']:+.2f}]" for r in Q))
os.makedirs(os.path.join(HERE, "results_adagb"), exist_ok=True)
open(os.path.join(HERE, "results_adagb", "new_tables.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
