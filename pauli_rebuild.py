"""Independent rebuild of H_V from the Pauli listing printed in the paper (main.tex), not from stored coefficients.
The LaTeX source of the listing (C, C', D_eta, D_0) and of Eq. (HV) is parsed term by term, every Pauli string is
expanded with its own Kronecker-product code (qubit 0 = leftmost factor, operators on the same qubit multiplied in
the order written), and the result is compared with the reference solver (hamiltonian_v2.H29v2, complex operator) at
the self-consistent field. Usage: python pauli_rebuild.py path/to/main.tex"""
import io, contextlib, json, re, sys
import numpy as np

TEX = open(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex").read()
P = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]], complex),
     "z": np.array([[1, 0], [0, -1]], complex)}


def listing(name):
    m = re.search(r"\\noindent \$" + re.escape(name) + r"=\{\}\$\$(.*?)\$\.", TEX, re.S)
    return m.group(1).replace(r"\,", "").replace(" ", "")


def split_terms(s):
    terms, depth, cur = [], 0, ""
    for i, ch in enumerate(s):
        if ch == "(": depth += 1
        if ch == ")": depth -= 1
        if ch in "+-" and depth == 0 and cur:
            terms.append(cur); cur = ch
        else:
            cur += ch
    terms.append(cur); return terms


def coeff(c):
    sign = -1 if c.startswith("-") else 1; c = c.lstrip("+-")
    if c.startswith("("): c = c[1:-1]
    if c == "": return sign
    c = c.replace("i", "j")
    if c == "j": c = "1j"
    return sign * complex(c)


def term_matrix(t):
    m = re.match(r"([+-]?(?:\([^)]*\)|[0-9]*i?)?)(.*)", t); a = coeff(m.group(1)); rest = m.group(2)
    ops = [np.eye(2, dtype=complex) for _ in range(4)]
    if rest == "I": return a * np.eye(16)
    for k, ax in re.findall(r"\\sigma_\{(\d)\}\^\{([xyz])\}", rest):
        ops[int(k)] = ops[int(k)] @ P[ax]
    M = ops[0]
    for o in ops[1:]: M = np.kron(M, o)
    return a * M


def build(name):
    return sum(term_matrix(t) for t in split_terms(listing(name)))


def single(k, ax):
    ops = [np.eye(2, dtype=complex)] * 4; ops = list(ops); ops[k] = P[ax]; M = ops[0]
    for o in ops[1:]: M = np.kron(M, o)
    return M


if __name__ == "__main__":
    import hamiltonian_v2 as hv
    C, Cp, De, D0 = build("C"), build("C'"), build(r"D_\eta"), build("D_0")
    with contextlib.redirect_stdout(io.StringIO()):
        s = hv.H29v2(dict(hv.PARAMS_EQ29)); s.solve(); ref = s.construct_full_hamiltonian(s.field)
    p = hv.PARAMS_EQ29; phi = s.field
    z = lambda k: single(k, "z")
    Hen = -p["epsilon_0"] / 2 * (z(0) - z(2)) - p["epsilon_1"] / 2 * (z(1) @ z(0) + z(3) @ z(2) @ z(1))
    H = Hen + 1j * p["g_1"] * p["h"] * phi / 4 * (C * p["E_1"] + Cp * np.conj(p["E_1"])) + p["gamma_1"] / 16 * (p["eta_r"] * De + D0)
    res = dict(n_terms={k: len(split_terms(listing(k))) for k in ("C", "C'", r"D_\eta", "D_0")}, field=phi,
               frobenius_residual=float(np.linalg.norm(H - ref)), max_entry_error=float(np.abs(H - ref).max()),
               relative=float(np.linalg.norm(H - ref) / np.linalg.norm(ref)))
    print(json.dumps(res, indent=1)); json.dump(res, open("results/studyb_v2/pauli_rebuild.json", "w"), indent=1)
