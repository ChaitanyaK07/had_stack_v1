"""Reproduce Table II (shared operators, both spectral treatments) and the phi column of the
controller-response table from the authors' solver.
Run from the root of https://github.com/ChaitanyaK07/had_stack_v1 :  python check_hamiltonians.py
"""
import io, contextlib, numpy as np
import hamiltonian_v2 as hv                      # complex-spectrum solver (Study A)

src = open("final_hamiltonian_solver.py").read(); ns = {}
exec(src.split("def main():")[0], ns)            # reference solver: eigh, .real for H2/H29 (Study B)
with contextlib.redirect_stdout(io.StringIO()):
    e1 = ns["QuantumHamiltonianSolver"](dict(ns["PARAMS_H1"])).solve()[1].min()
    e2 = ns["H2Solver"](dict(ns["PARAMS_H2"])).solve()[1].min()
    e3 = ns["Eq29Solver"](dict(ns["PARAMS_EQ29"])).solve()[1].min()
print("Hermitian treatment (Study B reports -0.1118, -0.0500, -2.0004):", round(e1, 4), round(e2, 4), round(e3, 4))

a = hv.H1v2(dict(hv.PARAMS_H1)); print("H1 complex mu0:", np.round(a.solve()[0], 5), "order", a.order)
b = hv.H2v2(dict(hv.PARAMS_H2)); print("H2 complex mu0:", np.round(b.solve()[0], 5), "order", b.order)
c = hv.H29v2(dict(hv.PARAMS_EQ29)); print("HV complex mu0:", np.round(c.solve()[0], 5), "field", round(c.field, 5))
for s in (1, 0.5, 0, -0.5, -1):                  # phi column of the controller-response table
    f = 1 - 0.5 * s; p = dict(hv.PARAMS_EQ29); p["g_1"] *= f; p["eta_r"] *= f
    h = hv.H29v2(p); h.solve(); print(f"s={s:+.1f} phi={h.field:.5f}")
