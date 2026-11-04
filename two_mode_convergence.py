"""Convergence of the self-consistent two-mode iterations (H1, H2) under the order parameter
selected by HCS_ORDER (default "adagb": Delta_j = Re<a^dag b>, Re<a^dag c>, Eqs. 7-8).

Delta_{n+1} = T(Delta_n) = m G(Delta_n) + (1 - m) Delta_n,  m = 0.3,  G(Delta) = Re<psi0(Delta)| O_j |psi0(Delta)>
under both spectral treatments used in the paper:
  complex   : right eigenvector of the eigenvalue with the smallest real part of the non-Hermitian
              operator (numpy.linalg.eig; Study A, hamiltonian_v2.H1v2 / H2v2)
  hermitian : lowest eigenvector from numpy.linalg.eigh of the reference solver's matrix
              (final_hamiltonian_solver: H1 as built, H2 with .real; Study B)
for the controller's coupling scales g_j (1 - 0.5 s), s in {+1, 0, -1}, and initial values Delta_0.
Reports iterations to |Delta_{n+1} - Delta_n| < tol, the end point, the fixed-point residual |G - Delta|,
the slope G'(Delta*) (central difference) and the predicted asymptotic factor 1 - m + m G'(Delta*).
Output: results/hamiltonian_layer/two_mode_convergence.json
"""
import io, json, os, contextlib, numpy as np
import hamiltonian_v2 as hv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_hamiltonian_solver.py")).read()
ns = {}; exec(src.split("def main():")[0], ns)
M = 0.3


def make(which, kind, s):
    f = 1 - 0.5 * s
    if which == 1:
        p = dict(hv.PARAMS_H1); p["g1"] *= f
        S = hv.H1v2(p) if kind == "complex" else ns["QuantumHamiltonianSolver"](p)
    else:
        p = dict(hv.PARAMS_H2); p["g_2"] *= f
        S = hv.H2v2(p) if kind == "complex" else ns["H2Solver"](p)
    return S


def G(S, kind, d):
    H = S.construct_hamiltonian(d)
    if kind == "complex":
        ev, V = np.linalg.eig(H); k = int(np.argmin(ev.real)); psi = V[:, k] / np.linalg.norm(V[:, k])
    else:
        ev, V = np.linalg.eigh(H); psi = V[:, 0]
    return float(np.vdot(psi, S.order_op @ psi).real), ev


def trace(which, kind, s, d0, tol=1e-10, nmax=20000):
    S = make(which, kind, s); ds = [d0]
    for n in range(nmax):
        g, _ = G(S, kind, ds[-1]); ds.append(M * g + (1 - M) * ds[-1])
        if abs(ds[-1] - ds[-2]) < tol: break
    star = ds[-1]; g, ev = G(S, kind, star); h = 1e-6
    slope = (G(S, kind, star + h)[0] - G(S, kind, max(star - h, 0.0))[0]) / (star + h - max(star - h, 0.0))
    k = int(np.argmin(ev.real)) if kind == "complex" else 0
    return dict(op=f"H{which}", kind=kind, s=s, d0=d0, iters=len(ds) - 1, converged=bool(abs(ds[-1] - ds[-2]) < tol),
                delta_star=float(star), residual=float(abs(g - star)), slope=float(slope),
                factor=float(1 - M + M * slope), mu0=[float(np.real(ev[k])), float(np.imag(ev[k]))])


R = [trace(w, k, s, d0) for w in (1, 2) for k in ("complex", "hermitian") for s in (1.0, 0.0, -1.0)
     for d0 in (1e-4, 0.1, 0.5, 1.0)]
# the reference solver's own stopping rule (Study B): tol 1e-6, at most 100 iterations, start 0.1
with contextlib.redirect_stdout(io.StringIO()):
    r1 = ns["QuantumHamiltonianSolver"](dict(ns["PARAMS_H1"])).solve()[1].min()
    r2 = ns["H2Solver"](dict(ns["PARAMS_H2"])).solve()[1].min()
print(f"order = {hv.HCS_ORDER};  reference solver (tol 1e-6, <=100 iterations): E0(H1) = {r1:.6f}, E0(H2) = {r2:.6f}")
print(f"{'op':3s} {'treatment':9s} {'s':>4s} {'D0':>7s} {'iters':>6s} {'conv':>5s} {'Delta*':>12s} {'resid':>8s} {'Gprime':>8s} {'factor':>7s}  mu0")
for r in R:
    print(f"{r['op']:3s} {r['kind']:9s} {r['s']:+4.1f} {r['d0']:7.4g} {r['iters']:6d} {str(r['converged']):>5s} {r['delta_star']:12.6g} "
          f"{r['residual']:8.1e} {r['slope']:8.4f} {r['factor']:7.4f}  {r['mu0'][0]:+.5f}{r['mu0'][1]:+.5f}i")
os.makedirs("results/hamiltonian_layer", exist_ok=True)
json.dump(dict(order=hv.HCS_ORDER, runs=R, reference_E0=[r1, r2]),
          open("results/hamiltonian_layer/two_mode_convergence.json", "w"), indent=1)
