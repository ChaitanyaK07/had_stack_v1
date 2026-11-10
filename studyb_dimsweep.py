"""Study B dimension sweep: the latent-space Stackelberg game of Algorithm 6 (ALS+QPSO leader, follower step on the
adversarial ELBO) on FC-VQC Q-VAEs of input dimension d = q^2 in {4, 9, 16, 25}, several seeds.

Hamiltonian family (controlled across d): for each (d, seed), two real symmetric GOE matrices A, B are drawn and
H(g) = (1-g) A + g B is taken at N_STATES couplings g in [0, 1]; every H(g) is rescaled affinely to the spectrum
[-1, 1], so that energies are comparable across d. The data are the N_STATES ground states (real, sign-continuous).

FC-VQC (Sec. VI-B): B = q blocks of q qubits; input angles encoded by RY; K entangling layers per FC layer
(RZ RY RZ on every qubit, then a cyclic CNOT ring); L FC layers, the readouts <Z> of all blocks mixed between FC
layers by the parameter-free transpose (qubit i of block b feeds block i), so that every block sees every block.
Q-VAE: e = enc(x), mu = W_mu e + b, logvar = W_s e + b, z = mu + sigma*eps, phi = Expand(z) (linear, d x q),
x_hat = W_out dec(phi) + b_out; loss ||x - x_hat||^2 + beta_KL KL.

Gradients are exact: autodiff through the state-vector simulator, which equals Algorithm 3 (parameter-shift rule
plus chain rule). check_parameter_shift() verifies this on one instance.

Game (Algorithm 6), per epoch and state: leader with theta fixed plays the better of ALS (3 sweeps, probe step
0.02) and QPSO (20 particles, 15 iterations, beta_Q 1.0 -> 0.3) on J_L = E(x_hat(mu+dz)) - E(x_hat(mu)),
||dz||_inf <= eps = 0.1; follower then takes one step of size eta on J_F = ELBO(x_hat(mu+dz*)).
Controls: the same Q-VAE and update with dz = 0 (no attack), and the standard Q-VAE trained on the
sampled clean ELBO, both with the same number of steps.
Test-time response: the same ALS+QPSO attack on the trained models at eps in {0.05, 0.1, 0.2}.

Usage: python studyb_dimsweep.py [--dims 4 9 16 25] [--seeds 1 2 3 4 5] [--epochs 30] [--out results/studyb_dimsweep]
"""
import argparse, json, math, os, time
import numpy as np, torch

torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
CD = torch.complex128


# ---------------------------------------------------------------- state-vector simulator of FC-VQC
class FCVQC(torch.nn.Module):
    """B = q blocks of q qubits, L FC layers of K entangling layers. Input: angles (..., d). Output: <Z> (..., d)."""

    def __init__(self, q, K=2, L=2, gen=None):
        super().__init__()
        self.q, self.K, self.L, self.n = q, K, L, 2 ** q
        self.theta = torch.nn.Parameter((torch.rand(L, q, K, q, 3, generator=gen) * 2 - 1) * math.pi)
        idx = torch.arange(self.n); bits = ((idx[:, None] >> (q - 1 - torch.arange(q))) & 1)          # (n, q), qubit 0 = MSB
        self.register_buffer("zsign", (1 - 2 * bits).double())                                          # (n, q)
        perms = []
        for c in range(q):                                                                              # cyclic CNOT ring c -> c+1
            t = (c + 1) % q
            if q == 1 or t == c: continue
            flip = bits.clone(); flip[:, t] ^= bits[:, c]
            perms.append((flip * (2 ** (q - 1 - torch.arange(q)))).sum(1))
        self.perms = perms

    @staticmethod
    def _rot(axis, a):
        c, s = torch.cos(a / 2), torch.sin(a / 2)
        if axis == "y":
            U = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CD)
        else:  # z
            e = torch.exp(-0.5j * a.to(CD)); z = torch.zeros_like(e)
            U = torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)
        return U

    def _apply(self, psi, U, i):
        """psi (..., B, n); U broadcastable to (..., B, 2, 2); acts on qubit i."""
        q = self.q; sh = psi.shape
        p = psi.reshape(*sh[:-1], 2 ** i, 2, 2 ** (q - i - 1))
        p = torch.einsum("...ab,...xbz->...xaz", U, p)
        return p.reshape(sh)

    def _layer(self, psi, th):  # th (B, K, q, 3)
        for k in range(self.K):
            for i in range(self.q):
                psi = self._apply(psi, self._rot("z", th[:, k, i, 2]), i)
                psi = self._apply(psi, self._rot("y", th[:, k, i, 1]), i)
                psi = self._apply(psi, self._rot("z", th[:, k, i, 0]), i)
            for p in self.perms:
                psi = psi[..., p]
        return psi

    def forward(self, ang, theta=None):
        th = self.theta if theta is None else theta
        q = self.q; lead = ang.shape[:-1]
        for l in range(self.L):
            a = ang.reshape(*lead, q, q)                                   # (..., block, qubit)
            psi = torch.zeros(*lead, q, self.n, dtype=CD); psi[..., 0] = 1
            for i in range(q):
                psi = self._apply(psi, self._rot("y", a[..., :, i]), i)
            psi = self._layer(psi, th[l])
            o = (psi.abs() ** 2) @ self.zsign                              # (..., block, qubit) = <Z>
            if l < self.L - 1:
                ang = (math.pi * o.transpose(-1, -2)).reshape(*lead, q * q)  # transpose mixing across blocks
        return o.reshape(*lead, q * q)


class QVAE(torch.nn.Module):
    def __init__(self, d, K=2, L=2, beta=0.01, gen=None):
        super().__init__()
        q = int(round(math.sqrt(d))); assert q * q == d
        self.d, self.q, self.beta = d, q, beta
        self.enc, self.dec = FCVQC(q, K, L, gen), FCVQC(q, K, L, gen)
        lin = lambda o, i: torch.nn.Parameter(torch.randn(o, i, generator=gen) / math.sqrt(i))
        self.Wm, self.bm = lin(q, d), torch.nn.Parameter(torch.zeros(q))
        self.Ws, self.bs = lin(q, d), torch.nn.Parameter(torch.zeros(q))
        self.We = lin(d, q)                                                # Expand(z)
        self.Wo, self.bo = lin(d, d), torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        e = self.enc(math.pi * x)
        return e @ self.Wm.T + self.bm, e @ self.Ws.T + self.bs

    def decode(self, z):
        return self.dec(z @ self.We.T) @ self.Wo.T + self.bo

    def elbo(self, x, mu, lv, z):
        xh = self.decode(z)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)
        return ((x - xh) ** 2).sum(-1) + self.beta * kl


def energy(xh, H):
    psi = xh / xh.norm(dim=-1, keepdim=True)
    return torch.einsum("...i,ij,...j->...", psi, H, psi)


# ---------------------------------------------------------------- leader: ALS and QPSO on J_L
@torch.no_grad()
def payoff(model, mu, H, dz):
    """J_L for a batch of perturbations dz (P, q) of one latent mean mu (q,)."""
    e0 = energy(model.decode(mu[None]), H)[0]
    return energy(model.decode(mu[None] + dz), H) - e0


@torch.no_grad()
def als(model, mu, H, eps, sweeps=3, h=0.02):
    q = mu.numel(); dz = torch.zeros(q); probes = torch.arange(-eps, eps + 1e-12, h)
    for _ in range(sweeps):
        for i in range(q):
            cand = dz.repeat(len(probes), 1); cand[:, i] = probes
            j = payoff(model, mu, H, cand); dz = cand[j.argmax()].clone()
    return dz, float(payoff(model, mu, H, dz[None])[0])


@torch.no_grad()
def qpso(model, mu, H, eps, Np=20, Nit=15, b0=1.0, b1=0.3, gen=None):
    q = mu.numel()
    x = (torch.rand(Np, q, generator=gen) * 2 - 1) * eps
    pb = x.clone(); fpb = payoff(model, mu, H, pb); g = pb[fpb.argmax()].clone(); fg = fpb.max()
    for t in range(Nit):
        bq = b0 - (b0 - b1) * t / max(Nit - 1, 1)
        mb = pb.mean(0)
        ph = torch.rand(Np, q, generator=gen); u = torch.rand(Np, q, generator=gen).clamp_min(1e-12)
        p = ph * pb + (1 - ph) * g
        sgn = torch.where(torch.rand(Np, q, generator=gen) < 0.5, -1.0, 1.0)
        x = (p + sgn * bq * (mb - x).abs() * torch.log(1 / u)).clamp(-eps, eps)
        f = payoff(model, mu, H, x)
        better = f > fpb; pb[better] = x[better]; fpb[better] = f[better]
        if fpb.max() > fg: fg = fpb.max(); g = pb[fpb.argmax()].clone()
    return g, float(fg)


def leader(model, mu, H, eps, gen):
    a, ja = als(model, mu, H, eps); b, jb = qpso(model, mu, H, eps, gen=gen)
    return (a, ja, "ALS") if ja >= jb else (b, jb, "QPSO")


# ---------------------------------------------------------------- data
def family(d, seed, n_states):
    rng = np.random.default_rng(1000 * d + seed)
    def goe():
        M = rng.normal(size=(d, d)); return (M + M.T) / 2
    A, B = goe(), goe(); Hs, xs, prev = [], [], None
    for g in np.linspace(0, 1, n_states):
        H = (1 - g) * A + g * B; w = np.linalg.eigvalsh(H)
        H = (2 * H - (w[-1] + w[0]) * np.eye(d)) / (w[-1] - w[0])          # spectrum -> [-1, 1]
        w, V = np.linalg.eigh(H); v = V[:, 0]
        if prev is None: v = v * np.sign(v[np.argmax(np.abs(v))])
        elif v @ prev < 0: v = -v
        prev = v; Hs.append(H); xs.append(v)
    return torch.tensor(np.array(Hs)), torch.tensor(np.array(xs))


# ---------------------------------------------------------------- evaluation
def evaluate(model, X, Hs, eps, gen):
    out = dict(dE=[], fid_adv=[], fid_clean=[], fid_adv_target=[], Lc=[], La=[])
    with torch.no_grad():
        for x, H in zip(X, Hs):
            mu, lv = model.encode(x[None]); mu, lv = mu[0], lv[0]
            dz, j, _ = leader(model, mu, H, eps, gen)
            xc, xa = model.decode(mu[None])[0], model.decode((mu + dz)[None])[0]
            pc, pa = xc / xc.norm(), xa / xa.norm()
            out["dE"].append(j); out["fid_adv"].append(float((pc @ pa) ** 2)); out["fid_clean"].append(float((pc @ x) ** 2)); out["fid_adv_target"].append(float((pa @ x) ** 2))
            out["Lc"].append(float(model.elbo(x, mu, lv, mu))); out["La"].append(float(model.elbo(x, mu, lv, mu + dz)))
    r = {k: float(np.mean(v)) for k, v in out.items()}; r["rob"] = r["Lc"] / r["La"]
    return r


# ---------------------------------------------------------------- one run
def run(d, seed, epochs, n_states, eps, eta, game=True, log=None, mode=None):
    mode = mode or ("game" if game else "standard")
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    Hs, X = family(d, seed, n_states)
    model = QVAE(d, gen=torch.Generator().manual_seed(10_000 + seed))       # same init for game and control
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    traj = []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter(); dEs, la, lc, used = [], [], [], []
        for k in torch.randperm(n_states, generator=gen).tolist():
            x, H = X[k], Hs[k]
            mu, lv = model.encode(x[None]); mu, lv = mu[0], lv[0]
            if mode == "game":
                dz, j, who = leader(model, mu.detach(), H, eps, gen); dEs.append(j); used.append(who)
                loss = model.elbo(x, mu, lv, mu + dz)                      # J_F: decoder input mu + dz*, eps = 0
            elif mode == "noattack":                                       # same update with dz = 0
                loss = model.elbo(x, mu, lv, mu)
            else:
                noise = torch.randn(model.q, generator=gen)
                loss = model.elbo(x, mu, lv, mu + (0.5 * lv).exp() * noise)
            opt.zero_grad(); loss.backward(); opt.step(); la.append(float(loss.detach()))
        row = dict(epoch=ep, dE=float(np.mean(dEs)) if dEs else float("nan"), loss=float(np.mean(la)),
                   qpso=used.count("QPSO"), als=used.count("ALS"), sec=time.perf_counter() - t0)
        traj.append(row)
        if log: print(f"[d={d} seed={seed} {mode}] ep {ep:3d} dE {row['dE']:+.4f} loss {row['loss']:.4f} "
                      f"QPSO {row['qpso']:2d} ({row['sec']:.1f}s)", flush=True)
    ev = {f"{e}": evaluate(model, X, Hs, e, torch.Generator().manual_seed(77 + seed)) for e in (0.05, 0.1, 0.2)}
    return dict(d=d, q=model.q, seed=seed, game=game, mode=mode, traj=traj, eval=ev,
                n_quantum=sum(p.numel() for n, p in model.named_parameters() if "theta" in n),
                n_classical=sum(p.numel() for n, p in model.named_parameters() if "theta" not in n))


def stationarity(traj, w=5, thr=0.01):
    dE = np.array([r["dE"] for r in traj]); rs = np.array([dE[max(0, i - w + 1):i + 1].std() for i in range(len(dE))])
    below = np.where(rs < thr)[0]; first = None
    for i in below:                                                        # first epoch from which it stays below
        if (rs[i:] < thr).all() and i >= w - 1: first = int(i + 1); break
    return float(rs[-1]), first, rs.tolist()


def check_parameter_shift(d=9, seed=0):
    """Autodiff gradient of a decoder readout vs. the two-term parameter-shift rule, Eq. (ps), for the rotation angles
    of the circuit that produces the readout (the last FC layer; earlier layers reach it through the classical
    mixing and are covered by the chain rule, Eq. (psgrad))."""
    m = QVAE(d, gen=torch.Generator().manual_seed(seed)); z = torch.randn(1, m.q, generator=torch.Generator().manual_seed(1))
    phi = (z @ m.We.T).detach()
    o = m.dec(phi)[0, 0]; g = torch.autograd.grad(o, m.dec.theta)[0]
    th = m.dec.theta.detach().clone(); worst = 0.0
    with torch.no_grad():
        for idx in [tuple(int(i) for i in np.unravel_index(k, th.shape)) for k in range(th[-1].numel() * (m.dec.L - 1), th.numel(), max(1, th[-1].numel() // 40))]:
            tp, tm = th.clone(), th.clone(); tp[idx] += math.pi / 2; tm[idx] -= math.pi / 2
            ps = 0.5 * (m.dec(phi, tp)[0, 0] - m.dec(phi, tm)[0, 0])
            worst = max(worst, abs(float(ps - g[idx])))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 9, 16, 25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--out", default="results/studyb_dimsweep"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ps = check_parameter_shift(); print(f"parameter-shift vs autodiff, max |diff| = {ps:.2e}")
    for d in a.dims:
        for s in a.seeds:
            f = f"{a.out}/d{d}_seed{s}.json"
            if os.path.exists(f): print("done:", f); continue
            res = {}
            for mode in ("game", "noattack", "standard"):
                r = run(d, s, a.epochs, a.states, a.eps, a.eta, mode == "game", log=not a.quiet, mode=mode)
                if mode == "game": r["stationarity"] = stationarity(r["traj"])
                res[mode] = r
            res["param_shift_check"] = ps
            json.dump(res, open(f, "w"))
            g, c = res["game"]["eval"]["0.1"], res["noattack"]["eval"]["0.1"]
            print(f"== d={d} seed={s}: fid_adv game {g['fid_adv']:.3f} std {c['fid_adv']:.3f} | dE game {g['dE']:.4f} std {c['dE']:.4f}"
                  f" | clean fid {g['fid_clean']:.3f}/{c['fid_clean']:.3f} | rob {g['rob']:.2f} | rollstd {res['game']['stationarity'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
