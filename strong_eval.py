"""Strong robustness evaluation (worst case over an attack ensemble).
For each target model and eps, a DETECTED fraud row counts as broken if ANY attack flips it:
  - white-box L-inf PGD, 100 steps, 5 random restarts, loss = BCE(true label)
  - white-box L-inf PGD, 100 steps, 5 random restarts, loss = logit margin (no saturation)
  - transfer: margin-PGD-100 crafted on every OTHER model, applied unchanged
Uniform step on all 18 features, step 2.5*eps/K, ||delta||_inf <= eps verified."""
import sys, json, numpy as np, torch, eval_extras as ex
torch.set_grad_enabled(True)
X, y = ex.load_split("test")
def pgd(m, x, eps, loss, K=100, R=5, seed=0):
    m.eval(); a = 2.5 * eps / K; best = x.clone(); bz = m(x).detach()
    for r in range(R):
        g = torch.Generator().manual_seed(seed + r)
        eta = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
        for _ in range(K):
            eta.requires_grad_(True); z = m(x + eta)
            L = (torch.nn.functional.softplus(-z) if loss == "bce" else -z).sum()
            gr, = torch.autograd.grad(L, eta)
            with torch.no_grad(): eta = torch.clamp(eta + a * gr.sign(), -eps, eps)
        with torch.no_grad():
            z = m(x + eta); better = z < bz; best[better] = (x + eta)[better]; bz = torch.minimum(bz, z)
    assert (best - x).abs().max() <= eps + 1e-5
    return best
models = dict(a.split("=") for a in sys.argv[1:])
M = {k: ex.load_transformer(v) for k, v in models.items()}
Xf = torch.tensor(X[y == 1])
crafted = {}
for e in (0.5, 1.0):
    for k, m in M.items(): crafted[(k, e)] = pgd(m, Xf, e, "margin", K=100, R=1)
rows = []
for k, m in M.items():
    pc = ex.probs(m, X); c = ex.metrics(y, pc)
    det = torch.tensor(pc[y == 1] >= 0.5)
    row = dict(model=k, clean_auc=round(c["auc"], 4), f1=round(c["f1"], 4), fp=c["fp"], fn=c["fn"], cost=c["cost"])
    for e in (0.5, 1.0):
        flip = {}
        adv = {loss: pgd(m, Xf, e, loss) for loss in ("bce", "margin")}
        with torch.no_grad():
            for loss in ("bce", "margin"):
                flip[loss] = (m(adv[loss]) < 0) & det
            flip["transfer"] = torch.zeros_like(det)
            for s in M:
                if s != k: flip["transfer"] |= (m(crafted[(s, e)]) < 0) & det
        worst = flip["bce"] | flip["margin"] | flip["transfer"]
        n = det.sum().item()
        for a, f in list(flip.items()) + [("WORST", worst)]:
            row[f"{a}@{e}"] = round(100 * f.sum().item() / n, 1)
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open("logs/strong_eval.json", "w"), indent=1)
