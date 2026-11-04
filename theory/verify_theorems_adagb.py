"""Numerical checks of Theorem 1, Theorem 2 (as applied to H1/H2) and Corollary 1 when the two-mode order
parameter is the transition field Delta = Re<a^dag b> of Eqs. (7)-(8) (HCS_ORDER = "adagb").
Same parameters as theory/verify_theorems.py: omega_a = omega_b = gamma = 0.1, E = 1, g_base = 0.1, m = 0.3.
"""
import numpy as np, json, os
w0 = w1 = gam = 0.1; E = 1.0; gb = 0.1; m = 0.3
A = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2)); B = np.kron(np.eye(2), np.array([[0, 1], [0, 0]]))
O = A.conj().T @ B                                                     # a^dag b


def Hmat(g, D):
    ca, cb, cab = w0 / 2j + gam / 2, w1 / 1j - gam / 2, g * E / 1j - gam * D
    return ca * (A.conj().T @ A) + cb * (B.conj().T @ B) + cab * (A.conj().T @ B + B.conj().T @ A)


def ground(g, D):
    ev, V = np.linalg.eig(Hmat(g, D)); k = int(np.argmin(ev.real)); return ev[k], V[:, k] / np.linalg.norm(V[:, k])


def Gfun(g, D): return float(np.real(np.vdot(ground(g, D)[1], O @ ground(g, D)[1])))


def closed(g, D):
    mu = -1j * (w0 + 2 * w1) / 4; d = gam / 2 + 1j * (w1 - w0 / 2) / 2; cab = -1j * g * E - gam * D
    w = d ** 2 + cab ** 2; s = np.sqrt(w + 0j)
    return mu - s, w


def fixed(g, D=0.1, tol=1e-13):
    for it in range(1, 5000):
        Dn = m * Gfun(g, D) + (1 - m) * D
        if abs(Dn - D) < tol: return Dn, it
        D = Dn
    raise RuntimeError("no convergence")


out = {}
# 1) Theorem 1 (i)-(ii) for Delta >= 0: closed form vs eig, Im w > 0, ground state in the one-excitation block
err, minImw, leak = 0, np.inf, 0
for g in np.linspace(0, 0.3, 31):
    for D in np.linspace(0, 1, 41):
        nu, w = closed(g, D); ev, psi = ground(g, D)
        err = max(err, abs(nu - ev)); minImw = min(minImw, w.imag); leak = max(leak, abs(psi[0]) + abs(psi[3]))
print(f"Thm 1: max |closed form - eig| = {err:.1e}; min Im w = {minImw:.4f} (b = 0.0025); max weight outside one-excitation block = {leak:.1e}")
out["thm1"] = dict(max_err=err, min_Im_w=minImw, leak=leak)

# 2) Theorem 2 applied to the complex-treatment iteration: G maps I = [0,1] into itself and is Lipschitz with L < 1
Dg = np.linspace(0, 1, 2001); rows = []
for s in (1.0, 0.5, 0.0, -0.5, -1.0):
    g = gb * (1 - 0.5 * s); Gv = np.array([Gfun(g, D) for D in Dg]); L = np.abs(np.diff(Gv) / np.diff(Dg)).max()
    Ds, it = fixed(g); slope = (Gfun(g, Ds + 1e-7) - Gfun(g, Ds - 1e-7)) / 2e-7
    rows.append(dict(s=s, g=g, G_min=Gv.min(), G_max=Gv.max(), L=L, rho=1 - m + m * L, Delta_star=Ds, slope=slope,
                     factor=1 - m + m * slope, iters_1e13=it))
    print(f"Thm 2, s={s:+.1f}: G([0,1]) = [{Gv.min():.4f}, {Gv.max():.4f}], L = {L:.4f}, rho = (1-m)+mL = {1-m+m*L:.4f}; "
          f"Delta* = {Ds:.6f}, G'(Delta*) = {slope:.4f}, asymptotic factor {1-m+m*slope:.4f}")
out["thm2"] = rows

# iterations of the solver's stopping rule |D_{n+1} - D_n| < 1e-8 from Delta_0 = 0.1 at the base coupling
D, it = 0.1, 0
while True:
    it += 1; Dn = m * Gfun(gb, D) + (1 - m) * D
    if abs(Dn - D) < 1e-8: D = Dn; break
    D = Dn
print(f"base coupling, Delta_0 = 0.1, tol 1e-8: {it} iterations, Delta = {D:.8f}")
out["iters_tol1e-8"] = it

# 3) Corollary 1 at the fixed point Delta*(g): eps(s) = 0.5 Gamma(s)/Gamma_base
Gam = lambda g: -closed(g, fixed(g)[0])[0].imag
Gb = Gam(gb); S = np.linspace(-1, 1, 801); eps = np.array([0.5 * Gam(gb * (1 - 0.5 * x)) / Gb for x in S])
de = np.gradient(eps, S)
print(f"Cor 1: Gamma_base = {Gb:.5f}; eps strictly decreasing: {bool(np.all(np.diff(eps) < 0))}; range [{eps[-1]:.4f}, {eps[0]:.4f}]; "
      f"max |deps/ds| = {np.abs(de).max():.4f} (at s = {S[np.abs(de).argmax()]:+.2f}), min |deps/ds| = {np.abs(de).min():.4f}")
# decomposition at s = 0: dGamma/dg = partial_g + partial_Delta * dDelta*/dg
h = 1e-6; Ds = fixed(gb)[0]
pg = (-closed(gb + h, Ds)[0].imag + closed(gb - h, Ds)[0].imag) / (2 * h)
pD = (-closed(gb, Ds + h)[0].imag + closed(gb, Ds - h)[0].imag) / (2 * h)
dD = (fixed(gb + h)[0] - fixed(gb - h)[0]) / (2 * h)
print(f"      at s=0: dGamma/dg = {pg:.4f} (direct) + {pD:.4f} x {dD:.4f} (via Delta*) = {pg + pD * dD:.4f}")
for s in (1.0, 0.5, 0.0, -0.5, -1.0):
    g = gb * (1 - 0.5 * s); Ds = fixed(g)[0]; nu = closed(g, Ds)[0]
    print(f"      s={s:+.1f}: Delta* = {Ds:.4f}, E0 = {nu.real:+.5f}, Gamma = {-nu.imag:.5f}, eps = lambda = {0.5*(-nu.imag)/Gb:.4f}")
out["cor1"] = dict(Gamma_base=Gb, decreasing=bool(np.all(np.diff(eps) < 0)), range=[eps[-1], eps[0]],
                   L_eps_numeric=float(np.abs(de).max()), dGdg_direct=pg, dGdDelta=pD, dDeltadg=dD)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hamiltonian_layer"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hamiltonian_layer",
                                 "verify_theorems_adagb.json"), "w"), indent=1, default=float)
