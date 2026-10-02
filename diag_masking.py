"""Gradient-masking diagnostics (Athalye et al., 2018) on a checkpoint:
 (a) strong PGD: 100 steps x 5 random restarts (worst case per row)
 (b) very large eps: ASR must approach 100% if gradients are informative
 (c) transfer: perturbations crafted on OTHER models, applied unchanged
 (d) loss must increase along the PGD path"""
import sys, numpy as np, torch
import adversarial_game as ag, eval_extras as ex
X, y = ex.load_split("test"); Xt, yt = torch.tensor(X), torch.tensor(y)
def asr(m, xa, det):
    pa = ex.probs(m, xa); return 100 * float(((pa < .5) & det).sum() / det.sum())
def pgd(m, eps, steps=10, mult=2.5, restarts=1, seed=0):
    att = ag.HamiltonianAttacker(ag.FixedPerturbationBudget(eps=eps), n_steps=steps, step_mult=mult)
    best = X.copy(); bestp = ex.probs(m, X)
    for r in range(restarts):
        torch.manual_seed(seed + r)
        xa = np.concatenate([att.evasion_attack(m, Xt[i:i+512], yt[i:i+512], eps=eps).numpy() for i in range(0, len(X), 512)])
        p = ex.probs(m, xa); better = p < bestp; best[better] = xa[better]; bestp[better] = p[better]
    return best
tag, ck = sys.argv[1].split("="); import functools; print = functools.partial(print, flush=True)
m = ex.load_transformer(ck); det = (ex.probs(m, X) >= .5) & (y == 1)
print(f"== {tag}: detected fraud = {det.sum()}")
for e in [0.5, 1.0]:
    print(f"  eps={e}: PGD-10 ASR {asr(m, pgd(m,e), det):5.1f}% | PGD-100x5 restarts ASR {asr(m, pgd(m,e,100,2.5,5), det):5.1f}%")
for e in [2.0, 4.0]:
    print(f"  eps={e} (large): PGD-100 ASR {asr(m, pgd(m,e,100), det):5.1f}%")
for stag, sck in [("baseline", "models/baseline_best.pt"), ("pgdat0.5", "models/v2_pgdat_best.pt")]:
    s = ex.load_transformer(sck)
    for e in [0.5, 1.0]:
        print(f"  transfer from {stag} eps={e}: ASR on {tag} {asr(m, pgd(s,e,100), det):5.1f}%")
# (d) loss along the path for detected fraud rows, eps=1.0
torch.manual_seed(0); idx = np.where(det)[0][:512]
x0 = Xt[idx]; eta = torch.empty_like(x0).uniform_(-1, 1); losses = []
for k in range(20):
    eta.requires_grad_(True)
    l = torch.nn.functional.binary_cross_entropy_with_logits(m(x0 + eta), torch.ones(len(idx)))
    g, = torch.autograd.grad(l, eta); losses.append(l.item())
    with torch.no_grad(): eta = torch.clamp(eta + 0.1 * g.sign(), -1, 1)
print("  loss along PGD path (eps=1):", " ".join(f"{v:.3f}" for v in losses[::4]), f"| mean |grad| at end {g.abs().mean():.2e}")
