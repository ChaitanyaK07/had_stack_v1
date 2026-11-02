"""Runtime benchmark for Study A methods (reviewer items: runtime comparison, scaling).

Timing only: inputs are random tensors with the shape of the real data (18 standardised
features, 17.2% positives). Wall-clock cost of a training step does not depend on feature
values, so no accuracy or robustness number is produced or implied here.
FraudTransformer is re-implemented from the paper's specification (Sec. V-B); the
parameter count is checked against the reported 70,593.
"""
import json, sys, time, statistics
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)
D, BATCH, POS = 18, 256, 0.172


class FraudTransformer(nn.Module):
    def __init__(self, n=D, d=64):
        super().__init__()
        self.embed = nn.Linear(1, d)                       # shared R -> R^64
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d))   # 19 positions
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.4, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        t = self.embed(x.unsqueeze(-1))
        t = torch.cat([self.cls.expand(x.size(0), -1, -1), t], 1) + self.pos
        return self.head(self.enc(t)[:, 0]).squeeze(-1)


def focal(logit, y, g=2.0, a=0.75):
    p = torch.sigmoid(logit); pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.full_like(p, a), torch.full_like(p, 1 - a))
    return (-at * (1 - pt) ** g * torch.log(pt.clamp_min(1e-8))).mean()


def pgd(model, x, y, eps, steps=10, loss="bce", w=None, ref=None):
    """L_inf PGD on fraud rows only (Alg. 2). loss='kl' maximises KL to clean prediction."""
    was = model.training; model.eval()
    m = (y == 1).float().unsqueeze(1)
    a = 2.5 * eps / steps
    w = torch.ones(D) if w is None else w
    eta = m * (torch.rand_like(x) * 2 - 1) * eps
    for _ in range(steps):
        eta.requires_grad_(True)
        lo = model(x + eta)
        if loss == "bce":
            L = F.binary_cross_entropy_with_logits(lo, y, reduction="none")
        elif loss == "margin":
            L = -(2 * y - 1) * lo
        else:  # kl between Bernoulli(clean) and Bernoulli(adv)
            p, q = torch.sigmoid(ref), torch.sigmoid(lo)
            L = p * torch.log((p + 1e-8) / (q + 1e-8)) + (1 - p) * torch.log((1 - p + 1e-8) / (1 - q + 1e-8))
        g, = torch.autograd.grad((L * m.squeeze(1)).sum(), eta)
        eta = (m * torch.clamp(eta.detach() + a * w * g.sign(), -eps, eps))
    model.train(was)
    return (x + eta).detach()


def bkl(p_logit, q_logit):
    p, q = torch.sigmoid(p_logit), torch.sigmoid(q_logit)
    return p * torch.log((p + 1e-8) / (q + 1e-8)) + (1 - p) * torch.log((1 - p + 1e-8) / (1 - q + 1e-8))


def data(n):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, D, generator=g), (torch.rand(n, generator=g) < POS).float()


def epoch(method, X, Y, eps=0.6586, lam=0.6586, model=None):
    model = model or FraudTransformer()
    opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    model.train(); n = X.size(0); perm = torch.randperm(n)
    w = torch.ones(D); delta = torch.zeros(BATCH, D)
    t0 = time.perf_counter()
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]; x, y = X[idx], Y[idx]
        if method == "clean":
            loss = focal(model(x), y)
        elif method in ("pgd_at", "had_stack"):
            xa = pgd(model, x, y, eps, w=w)
            loss = (1 - lam) * focal(model(x), y) + lam * focal(model(xa), y)
        elif method == "trades":
            with torch.no_grad(): ref = model(x)
            xa = pgd(model, x, y, eps, loss="kl", ref=ref)
            lo = model(x); m = (y == 1).float()
            loss = focal(lo, y) + 6.0 * (bkl(lo, model(xa)) * m).mean()
        elif method == "mart":
            xa = pgd(model, x, y, eps)
            lo, la = model(x), model(xa)
            pyc = torch.where(y == 1, torch.sigmoid(lo), 1 - torch.sigmoid(lo))
            loss = 2 * F.binary_cross_entropy_with_logits(la, y) + 6.0 * (bkl(lo, la) * (1 - pyc)).mean()
        elif method == "free_at":
            m = (y == 1).float().unsqueeze(1); d = delta[:x.size(0)]
            for _ in range(4):
                d = d.detach().requires_grad_(True)
                loss = focal(model(x + m * d), y)
                opt.zero_grad(); loss.backward()
                d = torch.clamp(d + eps * d.grad.sign(), -eps, eps)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            delta[:x.size(0)] = d.detach(); continue
        elif method == "smoothing":
            loss = focal(model(x + 0.5 * torch.randn_like(x)), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return time.perf_counter() - t0


def solve_H(g, gam=0.1, w0=0.1, w1=0.1, E=1.0, it=500):
    d = 0.0
    for _ in range(it):
        H = np.zeros((4, 4), complex)
        ca, cb, ck = w0 / 2j + gam / 2, w1 / 1j - gam / 2, g * E / 1j - gam * d
        H[1, 1], H[2, 2], H[3, 3] = ca, cb, ca + cb; H[1, 2] = H[2, 1] = ck
        mu, V = np.linalg.eig(H); k = np.argmin(mu.real); v = V[:, k] / np.linalg.norm(V[:, k])
        d = 0.3 * np.conj(v[0]) * v[3] + 0.7 * d
    return -mu[k].imag


def main(part):
    out = {}
    if part == "methods":
        print("params:", sum(p.numel() for p in FraudTransformer().parameters()), flush=True)
        X, Y = data(18475)
        for meth in ["clean", "had_stack", "pgd_at", "trades", "mart", "free_at", "smoothing"]:
            ts = [epoch(meth, X, Y) for _ in range(3)]
            out[meth] = ts; print(meth, [round(t, 1) for t in ts], flush=True)
        # Hamiltonian controller cost per round (H1 and H2 self-consistent solves)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); solve_H(0.1); solve_H(0.1); ts.append(time.perf_counter() - t0)
        out["hamiltonian_H1H2_per_round"] = ts; print("H1+H2 solve ms", round(1e3 * statistics.median(ts), 2), flush=True)
        # Worst-case white-box evaluation: PGD-100 x 5 restarts x 2 losses on 682 fraud rows, one eps
        model = FraudTransformer().eval(); Xf = torch.randn(682, D); Yf = torch.ones(682)
        t0 = time.perf_counter()
        for loss in ("bce", "margin"):
            for _ in range(5):
                for i in range(0, 682, 256):
                    pgd(model, Xf[i:i + 256], Yf[i:i + 256], 1.0, steps=100, loss=loss)
        out["eval_pgd100x5x2_682rows"] = time.perf_counter() - t0; print("eval", round(out["eval_pgd100x5x2_682rows"], 1), flush=True)
        # CERTIFY cost per row: N0=100 + N=10,000 noisy forward passes
        ts = []
        with torch.no_grad():
            for r in range(10):
                x = torch.randn(1, D); t0 = time.perf_counter()
                for n in (100, 10000):
                    for j in range(0, n, 2000):
                        model(x.repeat(min(2000, n - j), 1) + 0.5 * torch.randn(min(2000, n - j), D))
                ts.append(time.perf_counter() - t0)
        out["certify_per_row"] = ts; print("certify/row", round(statistics.median(ts), 2), flush=True)
    else:
        # Scaling: inference to 1M rows, training epochs vs. number of rows
        model = FraudTransformer().eval()
        for n in (10_000, 100_000, 1_000_000):
            X, _ = data(n); t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(0, n, 4096): model(X[i:i + 4096])
            out[f"infer_{n}"] = time.perf_counter() - t0; print("infer", n, round(out[f'infer_{n}'], 1), flush=True)
        for meth, sizes in (("clean", (18475, 100_000, 1_000_000)), ("had_stack", (18475, 100_000, 300_000))):
            for n in sizes:
                X, Y = data(n); out[f"train_{meth}_{n}"] = epoch(meth, X, Y)
                print("train", meth, n, round(out[f'train_{meth}_{n}'], 1), flush=True)
    json.dump(out, open(f"bench_{part}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
