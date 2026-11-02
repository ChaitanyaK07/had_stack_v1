"""Compare the Pauli sums of H_V written in the derivation (Eqs. 27-28 of the revised fraud paper)
with the four Pauli sums built by the reference solver (final_hamiltonian_solver.Eq29Solver).
Run from the repository root:  python check_hv_pauli_sums.py
"""
import os, re, numpy as np, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
src = open(os.path.join(ROOT, "final_hamiltonian_solver.py")).read(); ns = {}
exec(src.split("def main():")[0], ns); S = ns["Eq29Solver"](dict(ns["PARAMS_EQ29"]))
body = src[src.index("    def construct_full_hamiltonian(self, field):"):src.index("return H_full.real")]
def inner(name):
    i = body.index(name + ' = ('); j = body.index('(\n', i) + 1; d = 1; k = j
    while d: k += 1; d += {'(': 1, ')': -1}.get(body[k], 0)
    return re.sub(r'\s+', ' ', body[j:k]).strip()
code = {k: eval(inner(k).replace('self.', 'S.').replace('np.eye(16, dtype=complex)', 'np.eye(16)'), {'S': S, 'np': np})
        for k in ['coupling_E1', 'coupling_E1_conj', 'diss_eta', 'diss_no_eta']}

def P(term):                       # "s2y s1z s0x" -> matrix product (as in the solver)
    M = np.eye(16, dtype=complex)
    for t in term.split():
        M = M @ getattr(S, t)
    return M
def build(terms):                  # list of (coef, "pauli string" or "I")
    return sum(c * (np.eye(16) if t == "I" else P(t)) for c, t in terms)

blue = {}
blue['coupling_E1'] = [(1,"s0x"),(-1j,"s0y"),(-1,"s2y"),(1j,"s2x"),(1j,"s1z s0y"),(-1,"s1z s0x"),(-1j,"s3z s2x s1z"),(1,"s3z s2y s1z")]
blue['coupling_E1_conj'] = [(1,"s2x"),(-1j,"s2y"),(1j,"s0x"),(-1,"s0y"),(-1,"s1y s0x"),(-1j,"s1y s0y"),(1j,"s3z s2y s1z"),(-1,"s3z s2x s1z")]
blue['diss_eta'] = [(4-8j,"I"),(5,"s0z"),(-4*(1+1j),"s1z"),(4j,"s1x"),(3,"s1y"),(4j,"s2z"),(-(5+3j),"s1z s0z"),(1,"s1z s0z"),
  (-3j,"s1x s0z"),(2j,"s2y s0x"),(2j,"s2x s0y"),(4j,"s3z s1z"),(-1,"s2y s1y s0x"),(-1j,"s2y s1y s0y"),(1j,"s2x s1y s0x"),
  (-1,"s2y s1z s0y"),(-1j,"s2y s1z s0x"),(-1j,"s2x s1z s0y"),(1,"s2x s1z s0x"),(-1,"s2x s1y s0y"),(1,"s3z s2y s0y"),
  (-2,"s3z s2x s0x"),(-2j,"s3z s2y s0x"),(-4j,"s3z s2y s1z"),(1-3j,"s3z s2y s1z s0x"),(-1j,"s3z s2y s1x s0x"),
  (1,"s3z s2y s1x s0y"),(1,"s3z s2x s1z s0x"),(-1j,"s3z s2x s1z s0y")]
blue['diss_no_eta'] = [(4-2j,"I"),(4+2j,"s0z"),(-(4-2j),"s1z"),(2j,"s1x"),(1,"s1y"),(-(5+1j),"s1z s0z"),(-2j,"s1x s0z"),
  (-1,"s1y s0z"),(-2,"s2y s0y"),(2,"s2x s0x"),(-1,"s2y s1y s0x"),(-1j,"s2y s1y s0y"),(1j,"s2x s1y s0x"),(1,"s2x s1z s0y"),
  (-1,"s2x s1y s0y"),(1,"s2y s1z s0x"),(1j,"s2y s1z s0x"),(-1,"s2x s1z s0x"),(-1,"s3z s2y s0y"),(-1j,"s3z s2y s0x"),
  (-1j,"s3z s2x s0y"),(1,"s3z s2x s0x"),(-2,"s3z s2x s1z s0x"),(-1,"s3z s2y s1x s0x"),(-1j,"s3z s2x s1x s0y"),
  (1-1j,"s3z s2y s1z s0x"),(2,"s3z s2y s1z s0y"),(-1j,"s3z s2y s1x s0x"),(1,"s3z s2y s1x s0y")]
for k in code:
    B = build(blue[k]); C = code[k]
    print(f"{k:17s} derivation==solver: {np.allclose(B, C)}   ||derivation-solver||_F = {np.linalg.norm(B-C):.3f}   ||solver||_F = {np.linalg.norm(C):.3f}")
