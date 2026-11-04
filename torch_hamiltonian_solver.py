"""Device-agnostic (CPU or CUDA) batched self-consistent solver for H1, H2 and H_V.

Every operator is affine in its coupling and in its self-consistent variable, so it is written as
    H1(g1, Delta)   = B0 + g1 * Bg + Delta * BD            (4x4)
    H2(g2, Delta)   = same structure with mode c           (4x4)
    H_V(phi, eta_T) = A0 + phi * Aphi + eta_T * Aeta        (16x16)
with component matrices taken from the reference NumPy solver (exactness is asserted below).
A batch of coupling values is solved at once: every iteration builds a (B, d, d) complex128
tensor, calls torch.linalg.eig (complex treatment) or torch.linalg.eigh on the real part
(Hermitian treatment), selects the ground state, evaluates the observable and updates all
B self-consistent variables. The code is identical on CPU and GPU; only `device` changes.
"""
import time, json, io, contextlib, numpy as np, torch
import hamiltonian_v2 as hv

DT = torch.complex128

def _h1_components():
    s = hv.H1v2(dict(hv.PARAMS_H1)); p = dict(s.params)
    def H(g, d): q = dict(p); q["g1"] = g; s.params = q; return s.construct_hamiltonian(d)
    B0 = H(0.0, 0.0); Bg = H(1.0, 0.0) - B0; BD = H(0.0, 1.0) - B0
    s.params = p
    for g, d in [(0.137, 0.42), (0.05, -0.3)]: assert np.allclose(H(g, d), B0 + g * Bg + d * BD, atol=1e-14)
    return B0, Bg, BD, s.ab, s.a_dag @ s.b

def _h2_components():
    s = hv.H2v2(dict(hv.PARAMS_H2)); p = dict(s.params)
    def H(g, d): q = dict(p); q["g_2"] = g; s.params = q; return s.construct_hamiltonian(d)
    B0 = H(0.0, 0.0); Bg = H(1.0, 0.0) - B0; BD = H(0.0, 1.0) - B0
    s.params = p
    for g, d in [(0.137, 0.42), (0.05, -0.3)]: assert np.allclose(H(g, d), B0 + g * Bg + d * BD, atol=1e-14)
    return B0, Bg, BD, s.ac, s.a_dag @ s.c

def _hv_components():
    s = hv.H29v2(dict(hv.PARAMS_EQ29)); p = dict(s.params)
    def H(phi, eta): q = dict(p); q["eta_r"] = eta; s.params = q; return s.construct_full_hamiltonian(phi)
    A0 = H(0.0, 0.0); Aphi = H(1.0, 0.0) - A0; Aeta = H(0.0, 1.0) - A0
    s.params = p
    for phi, eta in [(0.00601, 1.0), (0.3, 0.42)]: assert np.allclose(H(phi, eta), A0 + phi * Aphi + eta * Aeta, atol=1e-13)
    return A0, Aphi, Aeta, s.s0x, s.s0y

class TorchSolver:
    def __init__(self, device="cpu"):
        self.dev = torch.device(device)
        t = lambda a: torch.as_tensor(a, dtype=DT, device=self.dev)
        self.h1 = [t(a) for a in _h1_components()]
        self.h2 = [t(a) for a in _h2_components()]
        self.hv = [t(a) for a in _hv_components()]

    def _ground(self, H, treatment, real_part=True):
        if treatment == "complex":
            ev, V = torch.linalg.eig(H)                                  # (B,d), (B,d,d)
            k = torch.argmin(ev.real, dim=-1)
            mu = ev.gather(-1, k[:, None]).squeeze(-1)
            psi = V.gather(-1, k[:, None, None].expand(-1, V.shape[-2], 1)).squeeze(-1)
        else:
            ev, V = torch.linalg.eigh(H.real.to(torch.float64) if real_part else H)   # lower triangle, as numpy
            mu = ev[:, 0].to(DT); psi = V[:, :, 0].to(DT)
        return mu, psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)

    @staticmethod
    def _expect(psi, O):                                                 # <psi|O|psi>, batched
        return torch.einsum("bi,ij,bj->b", psi.conj(), O, psi)

    def solve_two_mode(self, which, g, treatment="complex", order=None, delta0=0.1, mixing=0.3, tol=1e-8, nmax=200):
        B0, Bg, BD, O_ab, O_dag = self.h1 if which == 1 else self.h2
        O = O_ab if (order or hv.HCS_ORDER) == "ab" else O_dag
        g = torch.as_tensor(g, dtype=torch.float64, device=self.dev)
        d = torch.full_like(g, delta0)
        for it in range(nmax):
            H = B0 + g[:, None, None] * Bg + d[:, None, None] * BD
            mu, psi = self._ground(H, treatment, real_part=(which == 2))
            dn = mixing * self._expect(psi, O).real + (1 - mixing) * d
            done = bool((dn - d).abs().max() < tol); d = dn          # one host-device sync per iteration
            if done: break
        H = B0 + g[:, None, None] * Bg + d[:, None, None] * BD
        return self._ground(H, treatment, real_part=(which == 2))[0], d, it + 1

    def solve_hv(self, eta, treatment="complex", phi0=0.1, mixing=0.5, tol=1e-8, nmax=200):
        A0, Aphi, Aeta, sx, sy = self.hv
        eta = torch.as_tensor(eta, dtype=torch.float64, device=self.dev)
        phi = torch.full_like(eta, phi0)
        for it in range(nmax):
            H = A0 + phi[:, None, None] * Aphi + eta[:, None, None] * Aeta
            mu, psi = self._ground(H, treatment)
            fn = mixing * torch.hypot(self._expect(psi, sx).real, self._expect(psi, sy).real) + (1 - mixing) * phi
            done = bool((fn - phi).abs().max() < tol); phi = fn
            if done: break
        H = A0 + phi[:, None, None] * Aphi + eta[:, None, None] * Aeta
        return self._ground(H, treatment)[0], phi, it + 1

def numpy_round(s):
    """Reference: one controller round exactly as HamiltonianBudgetV3._solve (NumPy, one operator at a time)."""
    f = 1 - 0.5 * s
    p1 = dict(hv.PARAMS_H1); p1["g1"] *= f; p2 = dict(hv.PARAMS_H2); p2["g_2"] *= f
    p29 = dict(hv.PARAMS_EQ29); p29["eta_r"] *= f
    e1 = hv.H1v2(p1).solve(); e2 = hv.H2v2(p2).solve(); c = hv.H29v2(p29); e29 = c.solve()
    return e1[0], e2[0], c.field, e29[0]

if __name__ == "__main__":
    torch.set_num_threads(1)
    out = dict(device="cpu", torch=torch.__version__, cuda_available=torch.cuda.is_available())
    S = TorchSolver("cpu"); grid = np.array([1.0, 0.5, 0.0, -0.5, -1.0]); f = 1 - 0.5 * grid
    m1, d1, it1 = S.solve_two_mode(1, 0.1 * f); m2, d2, it2 = S.solve_two_mode(2, 0.1 * f); mv, ph, itv = S.solve_hv(1.0 * f)
    ref = [numpy_round(s) for s in grid]
    out["max_abs_diff_mu1"] = float(max(abs(complex(m1[i]) - ref[i][0]) for i in range(5)))
    out["max_abs_diff_mu2"] = float(max(abs(complex(m2[i]) - ref[i][1]) for i in range(5)))
    out["max_abs_diff_phi"] = float(max(abs(float(ph[i]) - ref[i][2]) for i in range(5)))
    out["max_abs_diff_muV"] = float(max(abs(complex(mv[i]) - ref[i][3]) for i in range(5)))
    # Hermitian treatment vs reference solver (Study B)
    src = open("final_hamiltonian_solver.py").read(); ns = {}; exec(src.split("def main():")[0], ns)
    with contextlib.redirect_stdout(io.StringIO()):
        r1 = ns["QuantumHamiltonianSolver"](dict(ns["PARAMS_H1"])).solve()[1].min()
        r29 = ns["Eq29Solver"](dict(ns["PARAMS_EQ29"])).solve()[1].min()
        r2 = ns["H2Solver"](dict(ns["PARAMS_H2"])).solve()[1].min()
    h1, _, _ = S.solve_two_mode(1, [0.1], treatment="hermitian", tol=1e-6)
    h2, _, _ = S.solve_two_mode(2, [0.1], treatment="hermitian", tol=1e-6)
    hvh, _, _ = S.solve_hv([1.0], treatment="hermitian", tol=1e-5)
    out["hermitian_E0_H1"] = [float(h1[0].real), float(r1)]; out["hermitian_E0_H2"] = [float(h2[0].real), float(r2)]; out["hermitian_E0_HV"] = [float(hvh[0].real), float(r29)]
    out["iterations"] = dict(H1=it1, H2=it2, HV=itv)

    def med(fn, k=20):
        ts = []
        for _ in range(k):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    out["numpy_one_round_ms"] = 1e3 * med(lambda: numpy_round(0.0))
    out["torch_cpu_one_round_ms"] = 1e3 * med(lambda: (S.solve_two_mode(1, [0.1]), S.solve_two_mode(2, [0.1]), S.solve_hv([1.0])))
    for B in (5, 100, 1000):
        ss = np.linspace(-1, 1, B); ff = 1 - 0.5 * ss
        out[f"torch_cpu_batch{B}_ms"] = 1e3 * med(lambda: (S.solve_two_mode(1, 0.1 * ff), S.solve_two_mode(2, 0.1 * ff), S.solve_hv(ff)), k=5)
        out[f"numpy_loop{B}_ms"] = 1e3 * med(lambda: [numpy_round(s) for s in ss], k=3 if B < 1000 else 1)
    import os; os.makedirs("results/hamiltonian_layer", exist_ok=True)
    json.dump(out, open("results/hamiltonian_layer/torch_solver_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
