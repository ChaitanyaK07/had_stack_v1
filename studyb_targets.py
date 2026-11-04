"""Study B target ground states (Hermitian treatment of the reference solver) under both order parameters.

For each of H1, H2, H_V this writes the lowest eigenvector and energy used as the Q-VAE target, for
  * ab      : the original release (Delta_j <- <ab>, <ac>), reference solver as is;
  * adagb   : Delta_j <- Re<a^dag b>, Re<a^dag c> (Eqs. 7-8), two variants for H2:
      - "capped":      the reference solver's own stopping rule (tol 1e-6, at most 100 iterations), and
      - "fixed_point": the exact fixed point. For H2 under a^dag b, G(0) = 0 and G(Delta) < Delta on (0, 1],
                       so the iteration decreases monotonically to Delta = 0, but only sublinearly (G'(0) = 1).
H1 and H_V do not depend on the variant. Fidelities |<psi_ab|psi>|^2 are reported against the original targets.
Output: results/hamiltonian_layer/studyb_targets.json and studyb_targets.npz (arrays named <variant>_<op>).
Usage: python studyb_targets.py
"""
import io, os, json, contextlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "final_hamiltonian_solver.py")).read().split("def main():")[0]


def solver(order):
    os.environ["HCS_ORDER"] = order; ns = {}; exec(SRC, ns); return ns


def ground(H):
    w, V = np.linalg.eigh(H); v = V[:, 0]; k = np.argmax(np.abs(v)); v = v * np.exp(-1j * np.angle(v[k]))   # fix the phase
    return float(w[0]), v


def run(order, fixed_point):
    ns = solver(order); out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        S1 = ns["QuantumHamiltonianSolver"](dict(ns["PARAMS_H1"])); S1.solve()
        S2 = ns["H2Solver"](dict(ns["PARAMS_H2"])); H2c, _ = S2.solve()
        S3 = ns["Eq29Solver"](dict(ns["PARAMS_EQ29"])); H3, _ = S3.solve()
    # H1: rerun the reference iteration to read its final Delta (the solver does not return it)
    d = S1.params["delta_init"]
    for it in range(100):
        w, V = np.linalg.eigh(S1.construct_hamiltonian(d)); dn = 0.3 * np.vdot(V[:, 0], S1.order_op @ V[:, 0]).real + 0.7 * d
        if abs(dn - d) < 1e-6 and it > 5: d = dn; break
        d = dn
    out["H1"] = (*ground(S1.construct_hamiltonian(d)), d)
    d = S2.params["delta_2_init"]
    for it in range(100):
        w, V = np.linalg.eigh(S2.construct_hamiltonian(d)); dn = 0.3 * np.vdot(V[:, 0], S2.order_op @ V[:, 0]).real + 0.7 * d
        if abs(dn - d) < 1e-6 and it > 5: d = dn; break
        d = dn
    if fixed_point:
        grid = np.linspace(1e-6, 1.0, 2000)
        G = np.array([np.vdot(ground(S2.construct_hamiltonian(x))[1], S2.order_op @ ground(S2.construct_hamiltonian(x))[1]).real for x in grid])
        g0 = np.vdot(ground(S2.construct_hamiltonian(0.0))[1], S2.order_op @ ground(S2.construct_hamiltonian(0.0))[1]).real
        assert abs(g0) < 1e-12 and np.all(G < grid), "fixed point of the H2 iteration is not Delta = 0"
        d = 0.0
    out["H2"] = (*ground(S2.construct_hamiltonian(d)), d)
    out["HV"] = (*ground(H3), float("nan"))
    return out


V = {"ab": run("ab", False), "adagb_capped": run("adagb", False), "adagb_fixed_point": run("adagb", True)}
res, arr = {}, {}
for name, R in V.items():
    res[name] = {}
    for op, (E, v, d) in R.items():
        fid = abs(np.vdot(V["ab"][op][1], v)) ** 2
        res[name][op] = dict(E0=E, order_param=d, fidelity_vs_ab=float(fid), state=[[float(z.real), float(z.imag)] for z in v])
        arr[f"{name}_{op}"] = v
        print(f"{name:18s} {op}: E0 = {E:+.6f}  order param = {d:.3g}  fidelity vs ab = {fid:.6f}")
os.makedirs(os.path.join(HERE, "results", "hamiltonian_layer"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", "hamiltonian_layer", "studyb_targets.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "results", "hamiltonian_layer", "studyb_targets.npz"), **arr)
