"""Turn bench_methods.json / bench_scaling.json into LaTeX table rows (no hand-typed numbers)."""
import json, statistics as st
M = json.load(open("bench_methods.json")); S = json.load(open("bench_scaling.json"))
mean = lambda v: st.mean(v); sd = lambda v: st.stdev(v)
clean = mean(M["clean"])
rounds = {"clean": None, "had_stack": 30, "pgd_at": 30, "trades": 30, "mart": 30, "free_at": 8, "smoothing": 30}
names = {"clean": "Clean (baseline)", "had_stack": "HAD-Stack", "pgd_at": "PGD-AT / FB-AT", "trades": "TRADES",
         "mart": "MART", "free_at": "Free-AT ($m{=}4$)", "smoothing": "Gaussian aug.\\ ($\\sigma{=}0.5$)"}
rows = []
for k in ["clean", "had_stack", "pgd_at", "trades", "mart", "free_at", "smoothing"]:
    m = mean(M[k]); r = rounds[k]
    tot = "--" if r is None else f"{r * m / 60:.1f}"
    rows.append(f"{names[k]} & {m:.1f} $\\pm$ {sd(M[k]):.1f} & {m / clean:.1f} & {r if r else '--'} & {tot}\\\\")
print("\n".join(rows))
print("H1+H2 solve ms median", round(1e3 * st.median(M["hamiltonian_H1H2_per_round"]), 1))
print("eval", round(M["eval_pgd100x5x2_682rows"], 1), "certify/row", round(st.median(M["certify_per_row"]), 2))
for k, v in S.items():
    n = int(k.split("_")[-1]); print(k, round(v, 1), "rows/s", round(n / v))
