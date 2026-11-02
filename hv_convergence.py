"""Convergence of the self-consistent field iteration for H_V (Eq. 29 / H29).

phi_{n+1} = F(phi_n) = 0.5 * sqrt(<s0x>^2 + <s0y>^2)_{psi0(phi_n)} + 0.5 * phi_n
under both spectral treatments used in the paper:
  complex   : ground state = right eigenvector of the eigenvalue with the smallest real part
              of the non-Hermitian H_V (numpy.linalg.eig; Study A, hamiltonian_v2.H29v2)
  hermitian : lowest eigenvector of Re(H_V) (numpy.linalg.eigh; reference solver, Study B)
for the controller's coupling scales eta_T * (1 - 0.5 s), s in {+1, 0, -1}, and several
initial values phi_0. Reports iterations to |phi_{n+1}-phi_n| < 1e-10, the fixed point, the
fixed-point residual |F(phi*) - phi*|, and the asymptotic contraction factor
|phi_{n+1}-phi*| / |phi_n-phi*| (median over the tail), which is < 1 for a convergent iteration.
Checks that the traced fixed point equals the solver's own .field (hamiltonian_v2.H29v2.solve).
"""
import io, json, contextlib, numpy as np
import hamiltonian_v2 as hv
src = open("final_hamiltonian_solver.py").read(); ns = {}; exec(src.split("def main():")[0], ns)

def F(H_of, S, phi, kind):
    H = H_of(phi)
    if kind == "complex":
        ev, V = np.linalg.eig(H); k = np.argmin(ev.real); psi = V[:, k] / np.linalg.norm(V[:, k])
    else:
        ev, V = np.linalg.eigh(H.real); psi = V[:, 0]
    sx = np.vdot(psi, S.s0x @ psi).real; sy = np.vdot(psi, S.s0y @ psi).real
    return 0.5 * np.hypot(sx, sy) + 0.5 * phi

def trace(kind, s, phi0, tol=1e-10, nmax=500):
    p = dict(hv.PARAMS_EQ29); p["eta_r"] *= (1 - 0.5 * s)
    S = hv.H29v2(p)                                   # complex operator (no .real projection)
    H_of = S.construct_full_hamiltonian
    phis = [phi0]
    for n in range(nmax):
        phis.append(F(H_of, S, phis[-1], kind))
        if abs(phis[-1] - phis[-2]) < tol: break
    phis = np.array(phis); star = phis[-1]
    err = np.abs(phis - star)
    tail = [(err[i + 1] / err[i]) for i in range(len(err) - 2) if 1e-13 < err[i] < 1e-3 and err[i + 1] > 1e-14]
    rate = float(np.median(tail)) if tail else float("nan")
    resid = abs(F(H_of, S, star, kind) - star)
    return dict(kind=kind, s=s, phi0=phi0, iters=len(phis) - 1, phi_star=float(star), residual=float(resid),
                contraction=rate, trace=phis.tolist())

R = []
for kind in ("complex", "hermitian"):
    for s in (1.0, 0.0, -1.0):
        for phi0 in (0.1, 0.5, 1.0, 1e-4):
            R.append(trace(kind, s, phi0))
# cross-check against the solver's own fixed point (complex treatment, base couplings)
c = hv.H29v2(dict(hv.PARAMS_EQ29)); c.solve()
with contextlib.redirect_stdout(io.StringIO()):
    h = ns["Eq29Solver"](dict(ns["PARAMS_EQ29"])); h.solve()
print("solver field (complex, base):", round(c.field, 6))
print(f"{'treatment':9s} {'s':>4s} {'phi0':>7s} {'iters':>5s} {'phi*':>10s} {'residual':>9s} {'contraction':>11s}")
for r in R:
    print(f"{r['kind']:9s} {r['s']:+4.1f} {r['phi0']:7.4g} {r['iters']:5d} {r['phi_star']:10.6f} {r['residual']:9.1e} {r['contraction']:11.4f}")
import os; os.makedirs("results/hamiltonian_layer", exist_ok=True)
json.dump(R, open("results/hamiltonian_layer/hv_convergence.json", "w"))
