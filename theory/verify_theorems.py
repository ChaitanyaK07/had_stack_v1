"""Numerical checks of Theorems 1-3 and Corollary 1 of the paper (closed-form spectrum, Table VI, derivative and
Lipschitz bounds, the 43-iteration count, Hermitian-surrogate spectra, and the energy-fidelity inequality)."""
import numpy as np
np.random.seed(0)
w0 = w1 = w2 = gam = 0.1; E = 1.0
def Hmat(wa, wb, gam, g, E, D):
    ca, cb, cab = wa/(2j) + gam/2, wb/1j - gam/2, g*E/1j - gam*D
    H = np.zeros((4, 4), complex); H[1,1] = cb; H[2,2] = ca; H[3,3] = ca+cb; H[1,2] = H[2,1] = cab; return H
d = gam/2 + 1j*(w1 - w0/2)/2; m0 = -1j*(w0 + 2*w1)/4; b = (d**2).imag
print("d^2 =", d**2, " b = Im d^2 =", b, " (gamma(w_b-w_a/2)/2 =", gam*(w1-w0/2)/2, ")")
def closed(g):
    w = d**2 - (g*E)**2; s = np.sqrt(w + 0j)          # principal root
    G = (w0 + 2*w1)/4 + np.sqrt((abs(w) - w.real)/2); E0 = -np.sqrt((abs(w) + w.real)/2)
    return G, E0, s
# 1) closed form vs numerical eig over the visited range and beyond
err = 0
for g in np.linspace(0.0, 0.3, 61):
    ev = np.linalg.eigvals(Hmat(w0, w1, gam, g, E, 0.0)); k = np.argmin(ev.real)
    G, E0, s = closed(g); err = max(err, abs(-ev[k].imag - G), abs(ev[k].real - E0))
    assert s.real > 0 and ev[k].real < 0
print("max |closed-form - eig| over g in [0,0.3]:", err)
# 2) Table VI from the closed form
Gb = closed(0.1)[0]
for s_ in (1, .5, 0, -.5, -1):
    g = 0.1*(1 - 0.5*s_); G, E0, _ = closed(g); print(f"s={s_:+.1f} g={g:.3f} E0={E0:.5f} Gamma={G:.5f} eps={0.5*G/Gb:.4f}")
# 3) derivative formula and Lipschitz bounds
gs = np.linspace(0.05, 0.15, 2001); G = np.array([closed(g)[0] for g in gs])
num = np.gradient(G, gs); ana = np.array([g*E**2*closed(g)[2].imag/abs(closed(g)[2])**2 for g in gs])
print("max |dGamma/dg numeric - analytic|:", np.abs(num - ana)[5:-5].max(), " min dGamma/dg:", ana.min(), " max:", ana.max(),
      " bound gE^2/sqrt(b) at g=0.15:", 0.15*E**2/np.sqrt(b))
S = np.linspace(-1, 1, 2001); eps = np.array([0.5*closed(0.1*(1-0.5*x))[0]/Gb for x in S])
print("eps strictly decreasing in s:", bool(np.all(np.diff(eps) < 0)), " max |deps/ds| numeric:", np.abs(np.gradient(eps, S)).max(),
      " bound 0.375 g_base^2 E^2/(Gamma_base sqrt b):", 0.375*0.01*E**2/(Gb*np.sqrt(b)))
# 4) self-consistent iteration for H1 with O=ab: F == 0 exactly, contraction 0.7, 43 iterations
A = np.kron(np.array([[0,1],[0,0]]), np.eye(2)); B = np.kron(np.eye(2), np.array([[0,1],[0,0]])); O = A @ B
D = 0.1; m = 0.3; Fvals = []
for it in range(1, 200):
    ev, V = np.linalg.eig(Hmat(w0, w1, gam, 0.1, E, D)); k = np.argmin(ev.real); psi = V[:, k]/np.linalg.norm(V[:, k])
    F = (psi.conj() @ O @ psi).real; Fvals.append(abs(F)); Dn = m*F + (1-m)*D
    done = abs(Dn - D) < 1e-8; D = Dn
    if done: break
print("H1 iteration: max|F|", max(Fvals), " iterations", it, " predicted", 1 + int(np.ceil(np.log(1e-8/(m*0.1))/np.log(1-m))))
# Im w > 0 for Delta >= 0 along the iteration (ground state stays in the one-excitation block)
for Dv in np.linspace(0, 0.1, 11):
    cab = 0.1*E/1j - gam*Dv; w = d**2 + cab**2; assert w.imag > 0
print("Im w > 0 for Delta in [0, 0.1]: ok")
# 5) Hermitian surrogates used by Study B: norms
Hs1 = Hmat(w0, w1, gam, 0.1, E, 0.0); L = np.tril(Hs1); Hbar1 = L + L.conj().T - np.diag(np.diag(L)); Hbar1 = Hbar1 - np.diag(1j*np.diag(Hbar1).imag)
ev1 = np.linalg.eigvalsh(Hs1)               # eigh reads the lower triangle and the real diagonal
Hbar2 = Hmat(w0, w2, gam, 0.1, E, 0.0).real; ev2 = np.linalg.eigvalsh(Hbar2)
print("eigh(H1):", np.round(ev1, 4), " ||Hbar1|| =", abs(ev1).max(), "  eigh(Re H2):", np.round(ev2, 4), " ||Hbar2|| =", abs(ev2).max())
# 6) energy-fidelity inequality on random instances
worst = 0
for _ in range(20000):
    n = np.random.choice([4, 16]); M = np.random.randn(n, n) + 1j*np.random.randn(n, n); H = (M + M.conj().T)/2
    p = np.random.randn(n) + 1j*np.random.randn(n); p /= np.linalg.norm(p)
    q = p + np.random.rand()*(np.random.randn(n) + 1j*np.random.randn(n)); q /= np.linalg.norm(q)
    dE = abs((q.conj() @ H @ q - p.conj() @ H @ p).real); F = abs(p.conj() @ q); nh = np.linalg.norm(H, 2)
    r1 = dE / (2*nh*np.sqrt(max(1 - F**2, 0)) + 1e-15); r2 = 2*nh*np.sqrt(1-F**2) / (2*nh*np.linalg.norm(q - p) + 1e-15)
    worst = max(worst, r1, r2)
print("max ratio lhs/rhs over 20000 random cases (must be <= 1):", worst)
# 7) Theorem 3 as stated (spectral-spread form): |dE| <= (lmax - lmin) sqrt(1 - F^2)
worst = 0
for _ in range(20000):
    n = np.random.choice([4, 16]); M = np.random.randn(n, n) + 1j*np.random.randn(n, n); H = (M + M.conj().T)/2 + np.random.randn()*3*np.eye(n)
    p = np.random.randn(n) + 1j*np.random.randn(n); p /= np.linalg.norm(p)
    q = p + np.random.rand()*(np.random.randn(n) + 1j*np.random.randn(n)); q /= np.linalg.norm(q)
    ev = np.linalg.eigvalsh(H); dE = abs((q.conj() @ H @ q - p.conj() @ H @ p).real); F = abs(p.conj() @ q)
    worst = max(worst, dE / ((ev[-1] - ev[0]) * np.sqrt(max(1 - F**2, 1e-300))))
print("Theorem 3 (spread form): max lhs/rhs over 20000 random cases (must be <= 1):", worst)
