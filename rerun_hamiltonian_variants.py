"""Re-run the Hamiltonian layer under the derivation's definitions and propagate every
quantity that enters the experiments.

Variants
  order   : 'ab'    -> solver as released: Delta_j <- <ab>, <ac>
            'adagb' -> derivation Eqs. (7)-(8): Delta_j <- Re<a^dag b>, Re<a^dag c>
  h29     : 'solver'-> Pauli sums of the released solver
            'blue'  -> Pauli sums typeset in the revised derivation (Eqs. 27-28), magnetic
                       term omitted (its parameters B, f00, f11 are not specified)
Outputs, per variant
  * controller table over s in {+1,+.5,0,-.5,-1}: E0, Gamma1, Gamma2, eps, lambda, phi
    (exactly as HamiltonianBudgetV3: couplings g1, g_2, eta_r scaled by 1 - 0.5 s)
  * attack step weights (floored v3 weights; unfloored base weights used by FB-AT)
  * H29-only ablation budget clip(0.5 phi/phi_base, 0.1, 0.9)
  * Study B targets: ground energies with the Hermitian treatment of the reference solver
Run from the repository root:  python rerun_hamiltonian_variants.py
"""
import io, json, contextlib, numpy as np, torch
import hamiltonian_v2 as hv
import os, sys; sys.path.insert(0, ".")
# Pauli sums of the solver and of the derivation (definitions only, not the comparison printout)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_hv_pauli_sums.py")).read().split("for k in code:")[0])

def ground(H):
    ev, V = np.linalg.eig(H); o = np.argsort(ev.real); ev, V = ev[o], V[:, o]
    return ev, V[:, 0] / np.linalg.norm(V[:, 0])

# ---------- two-mode operators with selectable order parameter ----------
class H1x(hv.H1v2):
    def __init__(self, p, order): super().__init__(p); self.order_kind = order
    def obs(self): return self.ab if self.order_kind == "ab" else self.a_dag @ self.b
    def solve(self, max_iter=2000, tol=1e-10, mixing=0.3):
        d = self.params["delta_init"]
        for it in range(max_iter):
            ev, psi = ground(self.construct_hamiltonian(d))
            dn = mixing * np.vdot(psi, self.obs() @ psi).real + (1 - mixing) * d
            if abs(dn - d) < tol: d = dn; break
            d = dn
        self.iters, self.order = it + 1, d
        return ground(self.construct_hamiltonian(d))[0]
class H2x(hv.H2v2):
    def __init__(self, p, order): super().__init__(p); self.order_kind = order
    def obs(self): return self.ac if self.order_kind == "ab" else self.a_dag @ self.c
    def solve(self, max_iter=2000, tol=1e-10, mixing=0.3):
        d = self.params["delta_2_init"]
        for it in range(max_iter):
            ev, psi = ground(self.construct_hamiltonian(d))
            dn = mixing * np.vdot(psi, self.obs() @ psi).real + (1 - mixing) * d
            if abs(dn - d) < tol: d = dn; break
            d = dn
        self.iters, self.order = it + 1, d
        return ground(self.construct_hamiltonian(d))[0]

# ---------- H29 with selectable Pauli sums ----------
BLUE = {k: build(v) for k, v in blue.items()}          # from compare_blue.py
class H29x(hv.H29v2):
    def __init__(self, p, kind): super().__init__(p); self.kind = kind
    def construct_full_hamiltonian(self, field):
        if self.kind == "solver": return super().construct_full_hamiltonian(field)
        p = self.params
        E = -p["epsilon_0"]/2*(self.s0z - self.s2z) - p["epsilon_1"]/2*(self.s1z@self.s0z + self.s3z@self.s2z@self.s1z)
        pre = 1j*p["g_1"]*p["h"]*field/4
        return (E + pre*BLUE["coupling_E1"]*p["E_1"] + pre*BLUE["coupling_E1_conj"]*np.conj(p["E_1"])
                + p["gamma_1"]/16*p["eta_r"]*BLUE["diss_eta"] + p["gamma_1"]/16*BLUE["diss_no_eta"])

def solve_all(s, order, h29):
    f = 1 - 0.5 * s                                       # HamiltonianBudgetV3 (signal negated)
    p1 = dict(hv.PARAMS_H1); p1["g1"] *= f
    p2 = dict(hv.PARAMS_H2); p2["g_2"] *= f
    p29 = dict(hv.PARAMS_EQ29); p29["eta_r"] *= f
    a, b, c = H1x(p1, order), H2x(p2, order), H29x(p29, h29)
    e1, e2, e29 = a.solve(), b.solve(), c.solve()
    return dict(e1=e1, e2=e2, E0=e1[0].real, G1=-e1[0].imag, G2=-e2[0].imag, d1=a.order, d2=b.order,
                field=c.field, mu29=e29[0])

def weights(r, floored, n=18):
    comb = np.concatenate([np.abs(r["e1"]), np.abs(r["e2"])])
    w = np.tile(comb, n // len(comb) + 1)[:n]; w = w / (w.sum() + 1e-12) * n
    return 0.5 + 0.5 * w / w.mean() if floored else w

# ---------- Study B: Hermitian treatment of the reference solver ----------
def herm_E0(order, h29):
    out = {}
    for name in ("H1", "H2"):
        S = H1x(dict(hv.PARAMS_H1), order) if name == "H1" else H2x(dict(hv.PARAMS_H2), order)
        d = S.params["delta_init" if name == "H1" else "delta_2_init"]
        for _ in range(2000):
            H = S.construct_hamiltonian(d); H = H if name == "H1" else H.real
            w, V = np.linalg.eigh(H); psi = V[:, 0]
            dn = 0.3 * np.vdot(psi, S.obs() @ psi).real + 0.7 * d
            if abs(dn - d) < 1e-10: d = dn; break
            d = dn
        H = S.construct_hamiltonian(d); H = H if name == "H1" else H.real
        out[name] = (float(np.linalg.eigh(H)[0][0]), float(d))
    S = H29x(dict(hv.PARAMS_EQ29), h29); f = S.params["field_init"]
    for _ in range(2000):
        w, V = np.linalg.eigh(S.construct_full_hamiltonian(f).real); psi = V[:, 0]
        fn = 0.5*np.hypot(np.vdot(psi, S.s0x@psi).real, np.vdot(psi, S.s0y@psi).real) + 0.5*f
        if abs(fn - f) < 1e-10: f = fn; break
        f = fn
    out["HV"] = (float(np.linalg.eigh(S.construct_full_hamiltonian(f).real)[0][0]), float(f))
    return out

RES = {}
for order in ("ab", "adagb"):
    for h29 in ("solver", "blue"):
        key = f"order={order}, h29={h29}"; R = {}
        base = solve_all(0.0, order, h29)
        rows = []
        for s in (1.0, 0.5, 0.0, -0.5, -1.0):
            r = solve_all(s, order, h29)
            rows.append(dict(s=s, E0=r["E0"], G1=r["G1"], G2=r["G2"], d1=r["d1"],
                             eps=0.5*r["G2"]/base["G2"], lam=float(np.clip(0.5*r["G1"]/base["G1"], .1, .9)),
                             phi=r["field"], eps_h29=float(np.clip(0.5*r["field"]/base["field"], .1, .9))))
        R["controller"] = rows
        R["mu29_base"] = [base["mu29"].real, base["mu29"].imag]
        R["weights_floored_s0"] = weights(base, True).round(4).tolist()
        R["weights_floored_s-1"] = weights(solve_all(-1.0, order, h29), True).round(4).tolist()
        wb = weights(base, False); R["weights_base_s0"] = wb.round(4).tolist(); R["zero_base_weights"] = int((wb < 1e-9).sum())
        R["studyB_hermitian_E0"] = herm_E0(order, h29)
        RES[key] = R
        print(f"\n=== {key} ===")
        print(" s      E0        Gamma1    Gamma2    Delta1    eps     lam     phi      eps_H29only")
        for x in rows:
            print(f"{x['s']:+.1f}  {x['E0']:+.5f}  {x['G1']:.5f}  {x['G2']:.5f}  {x['d1']:.4f}  {x['eps']:.4f}  {x['lam']:.4f}  {x['phi']:.5f}  {x['eps_h29']:.4f}")
        print(" mu0(H29) base:", np.round(complex(*R["mu29_base"]), 5))
        print(" floored weights s=0 :", R["weights_floored_s0"][:8], "... range", min(R["weights_floored_s0"]), max(R["weights_floored_s0"]))
        print(" base weights zeros  :", R["zero_base_weights"], "of 18")
        print(" Study B Hermitian E0 (H1, H2, HV):", {k: round(v[0], 4) for k, v in R["studyB_hermitian_E0"].items()})
os.makedirs("results/hamiltonian_layer", exist_ok=True)
json.dump(RES, open("results/hamiltonian_layer/hamiltonian_variants.json", "w"), indent=1, default=float)
