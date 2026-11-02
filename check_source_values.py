"""Independent checks of quantities reported in the two source papers."""
import numpy as np

def two_mode(ca, cb, ck):
    # basis |na nb>, nmax=1: |00>,|10>,|01>,|11>
    H = np.zeros((4, 4), complex)
    H[1, 1], H[2, 2], H[3, 3] = ca, cb, ca + cb
    H[1, 2] = H[2, 1] = ck
    return H

def ground(H):
    w, v = np.linalg.eig(H)
    i = np.argmin(w.real)
    return w, w[i], v[:, i] / np.linalg.norm(v[:, i])

def ab_expect(psi):
    # <psi| a b |psi>; a b maps |11> -> |00>
    return np.conj(psi[0]) * psi[3]

def solve_H1(g=0.1, eps0=0.1, eps1=0.1, gam=0.1, E=1.0, hbar=1.0, it=500):
    d = 0.0
    for _ in range(it):
        ca = eps0 / (2j * hbar) + gam / 2
        cb = eps1 / (1j * hbar) - gam / 2
        ck = g * E / (1j * hbar) - gam * d
        w, l0, psi = ground(two_mode(ca, cb, ck))
        d = 0.3 * ab_expect(psi) + 0.7 * d
    return w, l0, d

print("== Fraud study: H1/H2 spectrum at base parameters ==")
w, l0, d = solve_H1()
print("spectrum:", np.round(np.sort_complex(w), 5))
print("E0 = %.5f  Gamma = %.5f  order param = %.2e" % (l0.real, -l0.imag, abs(d)))

print("\n== Table XIX: controller response (couplings x (1-0.5 s)) ==")
Gb = -solve_H1()[1].imag
for s in [1.0, 0.5, 0.0, -0.5, -1.0]:
    f = 1 - 0.5 * s
    _, l0, _ = solve_H1(g=0.1 * f)
    G = -l0.imag
    print("s=%+.1f factor=%.2f E0=%.5f Gamma=%.5f eps=%.4f" % (s, f, l0.real, G, 0.5 * G / Gb))

print("\n== Quantum study, Table VI: E0(H1) = -0.1118 ? ==")
# real-symmetric counterpart of the same family (1/i factors dropped)
Hr = two_mode(0.05, -0.05, 0.1)
print("lowest eigenvalue, real-coefficient counterpart:", np.round(np.linalg.eigvalsh(Hr).min(), 4))

print("\n== Quantum study: Rob. = L_c / L_a ? ==")
rows = {"H1": (0.884, 0.134, 6.61), "H2": (1.263, 0.222, 5.70), "H3": (0.245, 0.514, 0.48),
        "No attack": (0.37, 0.37, 1.00), "ALS only": (0.49, 0.03, 16.0), "QPSO only": (0.76, 0.04, 17.3),
        "ALS+QPSO": (1.50, 0.61, 2.47), "Finite-diff": (0.31, 0.02, 17.2), "Plain VQC": (0.31, 0.13, 2.35),
        "Min-max": (0.52, 0.16, 3.28)}
for k, (c, a, r) in rows.items():
    lo, hi = (c - 0.005) / (a + 0.005), (c + 0.005) / max(a - 0.005, 1e-9)
    print("%-11s c/a=%.2f reported=%.2f  consistent-with-rounding=%s" % (k, c / a, r, lo <= r <= hi))

print("\n== Quantum study: speedups (Table I) ==")
for seq, par, sp in [(.005, .009, .56), (.017, .023, .74), (.042, .052, .81), (.102, .132, .77), (.431, .357, 1.21)]:
    print("%.2f vs reported %.2f" % (seq / par, sp))
print("P = 3qK (K=2) per block:", [3 * q * 2 for q in range(2, 7)], " total 3q^2K:", [3 * q * q * 2 for q in range(2, 7)])

print("\n== Quantum study: parameter-shift share of Table IV components ==")
for q, fc, ps, qp, qv in [(2, .001, .007, 0, .003), (3, .004, .018, 0, .006), (4, .007, .043, 0, .014),
                          (5, .017, .100, 0, .034), (6, .059, .319, 0, .107)]:
    print("q=%d share=%.1f%%" % (q, 100 * ps / (fc + ps + qp + qv)))
print("5-epoch totals:", [round(5 * t, 1) for t in (0.75, 4.36, 16.07, 52.88)])
print("fidelity ratio 0.658/0.174 = %.2f" % (0.658 / 0.174))

print("\n== Fraud study: arithmetic ==")
print("10-step vs 100x5 at eps=1: %.1fx; eps=0.5: %.1fx" % (36.5 / 3.5, 16.7 / 2.3))
print("pooled cost = 5 x per-fold:", [5 * c for c in (9358, 13981, 12903, 14221)])
print("Broken vs robust recall consistency (5-fold): baseline clean 89.7 * (1-0.427) = %.1f (rep 51.6)" % (89.7 * (1 - .427)))
print("HAD-Stack: 84.6*(1-0.169) = %.1f (rep 70.3); 84.6*(1-0.374)=%.1f (rep 53.0)" % (84.6 * .831, 84.6 * .626))
print("L2 radius for Linf 0.5 in 18 dims: %.2f" % (0.5 * np.sqrt(18)))
print("L_inf radii implied by L2 0.5, 1.0: %.3f %.3f" % (0.5 / np.sqrt(18), 1 / np.sqrt(18)))
print("leak-free fraud share: %.1f%%" % (100 * 112 / (112 + 3277)))
