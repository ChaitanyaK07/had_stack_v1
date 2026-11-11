"""Study B, version 2: the latent-space Stackelberg game with
  * Route A targets: the ground state of a non-Hermitian H is the right eigenvector of the eigenvalue with the smallest
    real part (numpy.linalg.eig); the payoff energy of a state is E(psi) = Re<psi|H|psi>/<psi|psi> = <psi|H_re|psi>,
    H_re = (H + H^dag)/2;
  * complex amplitudes encoded as [Re x, Im x] zero-padded to the next square D = q^2 >= 2d (FC-VQC needs d = q^2);
  * the exact follower gradient (Algorithm 9). Training uses automatic differentiation through the state-vector
    simulator; ps_checks.py shows that it equals the explicit parameter-shift chain rule and a finite difference;
  * variants for the ablations: model (fcvqc | mlp | plain), leader (both | als | qpso), follower (exact | fd | old |
    simultaneous | none), payoff (energy | recon | norm | linear), and the standard Q-VAE (sampled ELBO).

Families (16 states each, see family()):
  goe        : real symmetric GOE interpolation H(g) = (1-g)A + gB, spectrum rescaled to [-1, 1] (the first sweep);
  nh:<r>     : non-Hermitian, Hermitian part as goe, anti-Hermitian part i*r*(|H_h|_F/|S|_F)*S with S a GOE
               interpolation, so that |H_anti|_F / |H_herm|_F = r;
  H1, HV     : the benchmark operators (complex treatment, transition field / field at its self-consistent fixed point)
               at 16 coupling scales 0.5-1.5 of the controller (g1 for H1, eta_r for H_V). H2 equals H1 (Section 4).

Usage (one run): python studyb_v2.py --family H1 --d 4 --seed 1 --variant game --out results/studyb_v2/...
"""
import argparse, contextlib, io, json, math, os, time
import numpy as np, torch

torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
CD = torch.complex128
HERE = os.path.dirname(os.path.abspath(__file__))
EPS_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


# ============================================================ circuits
def _rot(axis, a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    if axis == "y":
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CD)
    e = torch.exp(-0.5j * a.to(CD)); z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


class Register:
    """B registers of n qubits simulated in parallel; psi (..., B, 2^n)."""

    def __init__(self, n):
        self.n, self.N = n, 2 ** n
        idx = torch.arange(self.N); bits = ((idx[:, None] >> (n - 1 - torch.arange(n))) & 1)
        self.zsign = (1 - 2 * bits).double()
        self.perms = []
        for c in range(n):
            t = (c + 1) % n
            if n == 1 or t == c: continue
            flip = bits.clone(); flip[:, t] ^= bits[:, c]
            self.perms.append((flip * (2 ** (n - 1 - torch.arange(n)))).sum(1))

    def apply(self, psi, U, i):
        sh = psi.shape; p = psi.reshape(*sh[:-1], 2 ** i, 2, 2 ** (self.n - i - 1))
        return torch.einsum("...ab,...xbz->...xaz", U, p).reshape(sh)

    def run(self, ang, th):
        """ang (..., B, n) input angles (RY); th (..., B, K, n, 3) rotations; returns <Z> (..., B, n)."""
        lead = ang.shape[:-1]
        psi = torch.zeros(*lead, self.N, dtype=CD); psi[..., 0] = 1
        for i in range(self.n):
            psi = self.apply(psi, _rot("y", ang[..., i]), i)
        for k in range(th.shape[-3]):
            U = _rot("z", th[..., k, :, 0]) @ _rot("y", th[..., k, :, 1]) @ _rot("z", th[..., k, :, 2])   # Rz Ry Rz fused
            for i in range(self.n):
                psi = self.apply(psi, U[..., i, :, :], i)
            for p in self.perms:
                psi = psi[..., p]
        return (psi.abs() ** 2) @ self.zsign


class FCVQC(torch.nn.Module):
    """B = q blocks of q qubits, L FC layers; between layers the readout of qubit i of block b feeds block i."""

    def __init__(self, D, K=2, L=2, gen=None):
        super().__init__()
        q = int(round(math.sqrt(D))); assert q * q == D
        self.q, self.K, self.L, self.D = q, K, L, D; self.reg = Register(q)
        self.theta = torch.nn.Parameter((torch.rand(L, q, K, q, 3, generator=gen) * 2 - 1) * math.pi)

    def layer(self, l, ang, th):  # ang (..., D) ; th (..., q, K, q, 3)
        q = self.q
        return self.reg.run(ang.reshape(*ang.shape[:-1], q, q), th).reshape(*ang.shape[:-1], q * q)

    def mix(self, o):
        q = self.q
        return math.pi * o.reshape(*o.shape[:-1], q, q).transpose(-1, -2).reshape(*o.shape[:-1], q * q)

    def forward(self, ang, theta=None):
        th = self.theta if theta is None else theta
        for l in range(self.L):
            o = self.layer(l, ang, th[..., l, :, :, :, :])
            if l < self.L - 1: ang = self.mix(o)
        return o


class PlainVQC(torch.nn.Module):
    """One register of D qubits (no blocks, no mixing), K entangling layers."""

    def __init__(self, D, K=2, gen=None):
        super().__init__(); self.D, self.reg = D, Register(D)
        self.theta = torch.nn.Parameter((torch.rand(1, K, D, 3, generator=gen) * 2 - 1) * math.pi)

    def forward(self, ang, theta=None):
        th = self.theta if theta is None else theta
        return self.reg.run(ang[..., None, :], th)[..., 0, :]


class MLP(torch.nn.Module):
    """Classical replacement of one FC-VQC with matched parameter count: D -> h -> D, tanh (outputs in [-1, 1])."""

    def __init__(self, D, n_match, gen=None):
        super().__init__(); h = max(1, round((n_match - D) / (2 * D + 1))); self.h = h
        self.W1 = torch.nn.Parameter(torch.randn(h, D, generator=gen) / math.sqrt(D)); self.b1 = torch.nn.Parameter(torch.zeros(h))
        self.W2 = torch.nn.Parameter(torch.randn(D, h, generator=gen) / math.sqrt(h)); self.b2 = torch.nn.Parameter(torch.zeros(D))

    def forward(self, ang, theta=None):
        return torch.tanh(torch.tanh(ang / math.pi @ self.W1.T + self.b1) @ self.W2.T + self.b2)


class QVAE(torch.nn.Module):
    def __init__(self, D, m, kind="fcvqc", beta=0.01, gen=None):
        super().__init__()
        self.D, self.m, self.beta, self.kind = D, m, beta, kind
        if kind == "fcvqc": self.enc, self.dec = FCVQC(D, gen=gen), FCVQC(D, gen=gen)
        elif kind == "plain": self.enc, self.dec = PlainVQC(D, gen=gen), PlainVQC(D, gen=gen)
        else:
            n = FCVQC(D).theta.numel(); self.enc, self.dec = MLP(D, n, gen), MLP(D, n, gen)
        lin = lambda o, i: torch.nn.Parameter(torch.randn(o, i, generator=gen) / math.sqrt(i))
        self.Wm, self.bm = lin(m, D), torch.nn.Parameter(torch.zeros(m))
        self.Ws, self.bs = lin(m, D), torch.nn.Parameter(torch.zeros(m))
        self.We = lin(D, m)
        self.Wo, self.bo = lin(D, D), torch.nn.Parameter(torch.zeros(D))

    def encode(self, v):
        e = self.enc(math.pi * v); return e @ self.Wm.T + self.bm, e @ self.Ws.T + self.bs

    def decode(self, z):
        return self.dec(z @ self.We.T) @ self.Wo.T + self.bo

    def elbo(self, v, mu, lv, z):
        vh = self.decode(z); kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(-1)
        return ((v - vh) ** 2).sum(-1) + self.beta * kl

    def counts(self):
        circ = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("enc.", "dec.")))
        return circ, sum(p.numel() for p in self.parameters()) - circ


# ============================================================ encoding and data
class Codec:
    def __init__(self, d, complex_):
        self.d, self.cx = d, complex_
        need = 2 * d if complex_ else d; q = math.ceil(math.sqrt(need)); self.D = q * q

    def enc(self, x):
        v = np.zeros(self.D)
        if self.cx: v[:self.d], v[self.d:2 * self.d] = x.real, x.imag
        else: v[:self.d] = x.real
        return v

    def state(self, vh):  # torch (..., D) -> complex normalised (..., d)
        psi = torch.complex(vh[..., :self.d], vh[..., self.d:2 * self.d]) if self.cx else vh[..., :self.d].to(CD)
        return psi / psi.abs().pow(2).sum(-1, keepdim=True).sqrt()


def _ground_right(H):
    w, V = np.linalg.eig(H); k = np.argmin(w.real); v = V[:, k]; return w[k], v / np.linalg.norm(v)


def _align(vs):
    out, prev = [], None
    for v in vs:
        if prev is None: v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        else: v = v * np.exp(-1j * np.angle(np.vdot(prev, v)))
        out.append(v); prev = v
    return out


def family(name, d, seed, n=16):
    """Returns (Hs complex (n,d,d), xs complex (n,d), info)."""
    gs = np.linspace(0, 1, n)
    if name in ("goe",) or name.startswith("nh:"):
        r = 0.0 if name == "goe" else float(name.split(":")[1])
        rng = np.random.default_rng(1000 * d + seed)
        goe = lambda: (lambda M: (M + M.T) / 2)(rng.normal(size=(d, d)))
        A, B = goe(), goe(); SA, SB = goe(), goe()
        Hs, xs, nu = [], [], []
        for g in gs:
            Hh = (1 - g) * A + g * B; w = np.linalg.eigvalsh(Hh); Hh = (2 * Hh - (w[-1] + w[0]) * np.eye(d)) / (w[-1] - w[0])
            S = (1 - g) * SA + g * SB
            H = Hh + 1j * r * np.linalg.norm(Hh) / np.linalg.norm(S) * S
            e, v = _ground_right(H); Hs.append(H); xs.append(v); nu.append(e)
        return np.array(Hs), np.array(_align(xs)), dict(r=r, nu0=[complex(z) for z in nu])
    import hamiltonian_v2 as hv
    Hs, xs, nu = [], [], []
    for f in np.linspace(0.5, 1.5, n):
        with contextlib.redirect_stdout(io.StringIO()):
            if name == "H1":
                p = dict(hv.PARAMS_H1); p["g1"] *= f; s = hv.H1v2(p); s.solve(); H = s.construct_hamiltonian(s.order)
            elif name == "H2":
                p = dict(hv.PARAMS_H2); p["g_2"] *= f; s = hv.H2v2(p); s.solve(); H = s.construct_hamiltonian(s.order)
            else:
                p = dict(hv.PARAMS_EQ29); p["eta_r"] *= f; s = hv.H29v2(p); s.solve(); H = s.construct_full_hamiltonian(s.field)
        e, v = _ground_right(H); Hs.append(H); xs.append(v); nu.append(e)
    Hs = np.array(Hs); Hh = (Hs + Hs.conj().transpose(0, 2, 1)) / 2
    r = float(np.mean([np.linalg.norm(H - h) / np.linalg.norm(h) for H, h in zip(Hs, Hh)]))
    return Hs, np.array(_align(xs)), dict(r=r, nu0=[complex(z) for z in nu])


# ============================================================ payoffs and leader
class Game:
    def __init__(self, model, codec, payoff="energy", gen=None):
        self.m, self.c, self.payoff = model, codec, payoff
        if payoff == "linear":
            u = torch.randn(codec.D, generator=gen); self.u = u / u.norm()

    def energy(self, vh, Hr):
        psi = self.c.state(vh); return torch.einsum("...i,ij,...j->...", psi.conj(), Hr, psi).real

    @torch.no_grad()
    def J(self, mu, dz, v, Hr, kind=None):
        kind = kind or self.payoff
        vc = self.m.decode(mu[None]); va = self.m.decode(mu[None] + dz)
        if kind == "energy": return self.energy(va, Hr) - self.energy(vc, Hr)[0]
        if kind == "recon": return ((va - v) ** 2).sum(-1) - ((vc[0] - v) ** 2).sum()
        if kind == "norm": return dz.norm(dim=-1)
        return (va - vc) @ self.u

    def als(self, mu, v, Hr, eps, kind=None, sweeps=3, h=0.02):
        q = mu.numel(); dz = torch.zeros(q); probes = torch.arange(-eps, eps + 1e-12, h)
        for _ in range(sweeps):
            for i in range(q):
                cand = dz.repeat(len(probes), 1); cand[:, i] = probes
                dz = cand[self.J(mu, cand, v, Hr, kind).argmax()].clone()
        return dz, float(self.J(mu, dz[None], v, Hr, kind)[0])

    def qpso(self, mu, v, Hr, eps, gen, kind=None, Np=20, Nit=15, b0=1.0, b1=0.3):
        q = mu.numel(); x = (torch.rand(Np, q, generator=gen) * 2 - 1) * eps
        pb = x.clone(); fpb = self.J(mu, pb, v, Hr, kind); g = pb[fpb.argmax()].clone(); fg = fpb.max()
        for t in range(Nit):
            bq = b0 - (b0 - b1) * t / max(Nit - 1, 1); mb = pb.mean(0)
            ph = torch.rand(Np, q, generator=gen); u = torch.rand(Np, q, generator=gen).clamp_min(1e-12)
            p = ph * pb + (1 - ph) * g; sg = torch.where(torch.rand(Np, q, generator=gen) < 0.5, -1.0, 1.0)
            x = (p + sg * bq * (mb - x).abs() * torch.log(1 / u)).clamp(-eps, eps)
            f = self.J(mu, x, v, Hr, kind); b = f > fpb; pb[b] = x[b]; fpb[b] = f[b]
            if fpb.max() > fg: fg = fpb.max(); g = pb[fpb.argmax()].clone()
        return g, float(fg)

    def leader(self, mu, v, Hr, eps, gen, which="both", kind=None):
        if which == "als": a, ja = self.als(mu, v, Hr, eps, kind); return a, ja, "ALS"
        if which == "qpso": b, jb = self.qpso(mu, v, Hr, eps, gen, kind); return b, jb, "QPSO"
        a, ja = self.als(mu, v, Hr, eps, kind); b, jb = self.qpso(mu, v, Hr, eps, gen, kind)
        return (a, ja, "ALS") if ja >= jb else (b, jb, "QPSO")


# ============================================================ follower gradients other than autodiff
def circuit_params(model):
    return [p for n, p in model.named_parameters() if n.startswith(("enc.", "dec."))]


def grad_fd_or_old(model, lossfn, mode, delta=1e-5):
    """Gradient of the circuit parameters by a central finite difference (mode 'fd') or by the two-term rule applied to
    the loss itself (mode 'old', the earlier implementation); classical parameters by backprop."""
    out = []
    with torch.no_grad():
        for p in circuit_params(model):
            g = torch.zeros_like(p); flat = p.view(-1); h = delta if mode == "fd" else math.pi / 2
            for i in range(flat.numel()):
                o = flat[i].item()
                flat[i] = o + h; lp = float(lossfn()); flat[i] = o - h; lm = float(lossfn()); flat[i] = o
                g.view(-1)[i] = (lp - lm) / (2 * h) if mode == "fd" else 0.5 * (lp - lm)
            out.append(g)
    return out


# ============================================================ one run
def run(family_name, d, seed, variant="game", model_kind="fcvqc", leader="both", payoff="energy", follower="exact",
        epochs=30, eps=0.1, eta=0.05, n_states=16, complex_enc=True, log=False):
    """variant: game (Algorithm 6), noattack (dz = 0), standard (sampled clean ELBO)."""
    Hs, xs, info = family(family_name, d, seed, n_states)
    codec = Codec(d, complex_enc); m = max(2, int(round(math.sqrt(d))))
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = QVAE(codec.D, m, model_kind, gen=torch.Generator().manual_seed(10_000 + seed))
    game = Game(model, codec, payoff, gen=torch.Generator().manual_seed(20_000 + seed))
    V = torch.tensor(np.array([codec.enc(x) for x in xs])); Hr = torch.tensor((Hs + Hs.conj().transpose(0, 2, 1)) / 2)
    X = torch.tensor(xs)
    opt = torch.optim.SGD(model.parameters(), lr=eta); traj = []; stale = {}
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter(); dEs, ls, used = [], [], []
        for k in torch.randperm(n_states, generator=gen).tolist():
            v, H = V[k], Hr[k]
            mu, lv = model.encode(v[None]); mu, lv = mu[0], lv[0]
            if variant == "game":
                dz, j, who = game.leader(mu.detach(), v, H, eps, gen, leader)
                dEs.append(float(game.J(mu.detach(), dz[None], v, H, "energy")[0])); used.append(who)
                dz_use = stale.get(k, torch.zeros_like(dz)) if follower == "simultaneous" else dz
                stale[k] = dz
                lossfn = lambda: model.elbo(v, *[t[0] for t in model.encode(v[None])], model.encode(v[None])[0][0] + dz_use)
            elif variant == "noattack":
                lossfn = lambda: model.elbo(v, *[t[0] for t in model.encode(v[None])], model.encode(v[None])[0][0])
            else:
                noise = torch.randn(m, generator=gen)
                lossfn = lambda: (lambda mu_, lv_: model.elbo(v, mu_, lv_, mu_ + (0.5 * lv_).exp() * noise))(*[t[0] for t in model.encode(v[None])])
            loss = lossfn(); opt.zero_grad(); loss.backward()
            if follower in ("fd", "old"):
                for p, g in zip(circuit_params(model), grad_fd_or_old(model, lossfn, follower)): p.grad = g
            opt.step(); ls.append(float(loss.detach()))
        traj.append(dict(epoch=ep, dE=float(np.mean(dEs)) if dEs else float("nan"), loss=float(np.mean(ls)),
                         qpso=used.count("QPSO"), als=used.count("ALS"), sec=time.perf_counter() - t0))
        if log: print(f"[{family_name} d={d} s={seed} {variant}/{model_kind}/{leader}/{payoff}/{follower}] ep {ep} dE {traj[-1]['dE']:+.4f} loss {traj[-1]['loss']:.4f} ({traj[-1]['sec']:.1f}s)", flush=True)
    ev = evaluate(model, game, V, X, Hr, seed)
    circ, cls = model.counts()
    wre = [float(np.ptp(np.linalg.eigvalsh(h))) for h in Hr.numpy()]
    return dict(family=family_name, d=d, D=codec.D, m=m, seed=seed, variant=variant, model=model_kind, leader=leader,
                payoff=payoff, follower=follower, n_circuit=circ, n_classical=cls, r=info["r"], w_re=wre,
                nu0=[[z.real, z.imag] for z in info["nu0"]], traj=traj, eval=ev)


def evaluate(model, game, V, X, Hr, seed):
    gen = torch.Generator().manual_seed(77 + seed); out = {}
    jac = []
    for v in V:                                     # decoder Jacobian norm ||d x_hat / d z||_F at the latent mean
        mu = model.encode(v[None])[0][0].detach()
        jac.append(float(torch.autograd.functional.jacobian(lambda z: model.decode(z[None])[0], mu).norm()))
    out["jacobian_F"] = jac
    with torch.no_grad():
        for e in EPS_GRID:
            rows = []
            for v, x, H in zip(V, X, Hr):
                mu, lv = model.encode(v[None]); mu, lv = mu[0], lv[0]
                dz, j, _ = game.leader(mu, v, H, e, gen, "both", "energy")
                vc, va = model.decode(mu[None])[0], model.decode((mu + dz)[None])[0]
                pc, pa = game.c.state(vc), game.c.state(va)
                rows.append(dict(dE=j, fid_cc=float(abs(torch.vdot(pc, pa)) ** 2), fid_clean=float(abs(torch.vdot(x, pc)) ** 2),
                                 fid_adv=float(abs(torch.vdot(x, pa)) ** 2), Lc=float(model.elbo(v, mu, lv, mu)),
                                 La=float(model.elbo(v, mu, lv, mu + dz)), dz_inf=float(dz.abs().max())))
            out[str(e)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--variant", default="game")
    ap.add_argument("--model", default="fcvqc"); ap.add_argument("--leader", default="both")
    ap.add_argument("--payoff", default="energy"); ap.add_argument("--follower", default="exact")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out): print("done:", a.out); return
    t = time.time()
    r = run(a.family, a.d, a.seed, a.variant, a.model, a.leader, a.payoff, a.follower, a.epochs, complex_enc=not a.real, log=a.log)
    r["wall_s"] = time.time() - t
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(r, open(a.out, "w"))
    e = r["eval"]["0.1"]; f = lambda k: np.mean([x[k] for x in e])
    print(f"{a.out}: fid_clean {f('fid_clean'):.3f} fid_adv {f('fid_adv'):.3f} fid_cc {f('fid_cc'):.4f} dE {f('dE'):.4f} ({r['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
