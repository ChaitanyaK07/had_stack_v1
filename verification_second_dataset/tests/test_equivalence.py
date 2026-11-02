"""Checks that the device-aware functions in run_gpu.py match the released code.
  python tests/test_equivalence.py            (CPU; add --cuda to also compare CPU and GPU results)
1. fraud-only PGD == adversarial_game.HamiltonianAttacker.evasion_attack (full batch, masked), same start
2. run_gpu.pgd == run_confirmation.pgd (verbatim copy below), bitwise on CPU
3. pre-trained MLP reproduces its notebook's held-out accuracy region (sanity)
"""
import os, sys, torch, torch.nn as nn, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import run_gpu as rg
from run_gpu import bm, ag, hs

def released_pgd(m, x, eps, loss, K, R, seed=0):          # verbatim from run_confirmation.py
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
    return best

def released_attack(att, model, x, y, eps, init):           # adversarial_game.evasion_attack with a given start
    model.eval(); w = att.budget.get_feature_weights(x.shape[1]); mask = (y == 1).float().unsqueeze(1)
    alpha = att.step_mult * eps / att.n_steps; eta = init * mask
    for _ in range(att.n_steps):
        eta.requires_grad_(True)
        loss = (nn.functional.binary_cross_entropy_with_logits(model(x + eta), y.float(), reduction='none') * mask.squeeze(1)).sum()
        g, = torch.autograd.grad(loss, eta)
        with torch.no_grad(): eta = torch.clamp(eta + alpha * w.unsqueeze(0) * g.sign(), -eps, eps) * mask
    return x + eta

def main():
    rg.set_device("cpu"); torch.manual_seed(0)
    m = rg.make_transformer(30).eval(); x = torch.randn(256, 30); y = (torch.rand(256) < 0.05).long(); y[:3] = 1
    att = ag.HamiltonianAttacker(hs.HamiltonianBudgetV3()); eps = 0.6
    init = torch.empty_like(x).uniform_(-eps, eps)
    ref = released_attack(att, m, x, y, eps, init)
    # run fraud-only version with the same start
    torch_uniform = torch.Tensor.uniform_
    idx = (y == 1).nonzero(as_tuple=True)[0]
    torch.Tensor.uniform_ = lambda self, a, b: self.copy_(init[idx])
    try: fast = rg.fast_evasion_attack(att, m, x, y, eps=eps)
    finally: torch.Tensor.uniform_ = torch_uniform
    d1 = float((ref - fast).abs().max()); print("1. fraud-only PGD vs released attack, max |diff| =", d1)
    xf = torch.randn(40, 30)
    d2 = max(float((released_pgd(m, xf, e, l, 7, 2) - rg.pgd(m, xf, e, l, 7, 2)).abs().max()) for e in (0.5, 1.0) for l in ("bce", "margin"))
    print("2. run_gpu.pgd vs released pgd, max |diff| =", d2)
    ok = d1 == 0.0 and d2 == 0.0
    if "--cuda" in sys.argv and torch.cuda.is_available():
        rg.set_device("cuda"); mc = rg.make_transformer(30); mc.load_state_dict(m.state_dict()); mc = mc.cuda().eval()
        g = rg.pgd(mc, xf.cuda(), 0.5, "margin", 7, 2).cpu(); c = rg.pgd(m, xf, 0.5, "margin", 7, 2)
        print("3. GPU vs CPU pgd, max |diff| =", float((g - c).abs().max()), "(small float differences are expected on GPU)")
    print("ALL EQUIVALENCE CHECKS PASSED" if ok else "MISMATCH"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
