"""
HAMILTONIAN SOLVERS v2 — non-Hermitian spectra (advisor decision: use the eigenvalues)
=====================================================================================
Uses the EXACT operators of final_hamiltonian_solver.py (H1, H2 and the FULL Eq. 29
including all dissipation terms), with two corrections:
  * np.linalg.eig (general, complex eigenvalues) instead of eigh, which is only valid
    for Hermitian matrices. Re(lambda) = energy, -Im(lambda) = decay rate.
  * No '.real' projection of H2 / H29: that projection deletes the epsilon and
    coupling terms (they enter through 1/i factors), which is why g had no effect.
Ground state = eigenvalue with the smallest real part; its RIGHT eigenvector
(unit-normalised) is used for the self-consistent order parameter.
The existing adversarial_game.py is NOT modified.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_hamiltonian_solver.py")).read()
_ns = {}; exec(_src.split("def main():")[0], _ns)          # user's exact operators + params
PARAMS_H1, PARAMS_H2, PARAMS_EQ29 = _ns["PARAMS_H1"], _ns["PARAMS_H2"], _ns["PARAMS_EQ29"]

def _ground(H):
    ev, V = np.linalg.eig(H)
    o = np.argsort(ev.real); ev, V = ev[o], V[:, o]
    psi = V[:, 0] / np.linalg.norm(V[:, 0])
    return ev, psi

class H1v2(_ns["QuantumHamiltonianSolver"]):
    def solve(self, max_iter=200, tol=1e-8, mixing=0.3):
        d = self.params["delta_init"]
        for it in range(max_iter):
            ev, psi = _ground(self.construct_hamiltonian(d))
            dn = mixing * np.vdot(psi, self.ab @ psi).real + (1 - mixing) * d
            if abs(dn - d) < tol: d = dn; break
            d = dn
        self.iters, self.order = it + 1, d
        return _ground(self.construct_hamiltonian(d))[0]

class H2v2(_ns["H2Solver"]):
    def construct_hamiltonian(self, delta_2):          # user's operator WITHOUT .real
        p = self.params; hbar = 1.0
        n_a = self.a_dag @ self.a; n_c = self.c_dag @ self.c
        return ((p["epsilon_0"]/(2j*hbar) + p["gamma_2"]/2) * n_a +
                (p["epsilon_2"]/(1j*hbar) - p["gamma_2"]/2) * n_c +
                (p["g_2"]*p["E_2"]/(1j*hbar) - p["gamma_2"]*delta_2) *
                (self.a_dag @ self.c + self.c_dag @ self.a))
    def solve(self, max_iter=200, tol=1e-8, mixing=0.3):
        d = self.params["delta_2_init"]
        for it in range(max_iter):
            ev, psi = _ground(self.construct_hamiltonian(d))
            dn = mixing * np.vdot(psi, self.ac @ psi).real + (1 - mixing) * d
            if abs(dn - d) < tol: d = dn; break
            d = dn
        self.iters, self.order = it + 1, d
        return _ground(self.construct_hamiltonian(d))[0]

class H29v2(_ns["Eq29Solver"]):
    """Full Eq. 29 (energy + both coupling blocks + both dissipation blocks), no .real."""
    def construct_full_hamiltonian(self, field):
        # the user's method ends with 'return H_full.real'; recover the complex operator
        # by evaluating it with the real-projection removed.
        import types
        code = _src[_src.index("    def construct_full_hamiltonian(self, field):"):
                    _src.index("    def solve(self, max_iter=50")]
        code = code.replace("return H_full.real", "return H_full")
        loc = {"np": np}; exec("class _T:\n" + code, loc)
        return loc["_T"].construct_full_hamiltonian(self, field)
    def solve(self, max_iter=200, tol=1e-8, mixing=0.5):
        f = self.params["field_init"]
        for it in range(max_iter):
            ev, psi = _ground(self.construct_full_hamiltonian(f))
            fn = mixing * np.hypot(np.vdot(psi, self.s0x @ psi).real,
                                   np.vdot(psi, self.s0y @ psi).real) + (1 - mixing) * f
            if abs(fn - f) < tol: f = fn; break
            f = fn
        self.iters, self.field = it + 1, f
        return _ground(self.construct_full_hamiltonian(f))[0]
