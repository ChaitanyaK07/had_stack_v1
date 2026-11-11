"""Verification of the follower gradient (Algorithm 9): the parameter-shift rule composed with the chain rule through
the classical head, the decoder input angles and the inter-layer mixing of FC-VQC, against central finite differences.

Algorithm 9 is implemented here explicitly, with circuits evaluated only forward (no autodiff):
  per FC layer l with input angles a^(l) and readouts o^(l):
    dL/dtheta^(l)_i = sum_k dL/do^(l)_k * 1/2 [o^(l)_k(theta + pi/2 e_i) - o^(l)_k(theta - pi/2 e_i)]       (2 P_l shifts)
    dL/da^(l)_n     = sum_k dL/do^(l)_k * 1/2 [o^(l)_k(a + pi/2 e_n) - o^(l)_k(a - pi/2 e_n)]               (2 D shifts)
    dL/do^(l-1)     = pi * T^T dL/da^(l)          (a^(l) = pi * T o^(l-1), T the block transpose)
  decoder: dL/do_dec = 2 W_out^T (x_hat - x); after its first layer dL/dz = W_e^T dL/dphi;
  encoder: dL/de = W_mu^T (dL/dz + beta mu) + W_s^T (dL/dz * sigma * eps / 2 - beta/2 (1 - sigma^2)).

Checks (eps drawn once and held fixed):
  A  shift rule vs finite difference on the readouts of one FC layer (exact for rotation gates);
  A' the same two-term rule applied to the final readouts of the two-layer FC-VQC for first-layer angles (not exact:
     the classical mixing between layers makes the readout non-sinusoidal in those angles);
  B  classical factors dL/do_dec, dL/de, dL/dz vs finite differences with the readouts / z treated as free variables;
  C  full gradient of Algorithm 9 vs central finite difference of L for every encoder and decoder angle; also vs autodiff;
  D  the earlier loss-level rule g_old = 1/2 [L(theta + pi/2 e_i) - L(theta - pi/2 e_i)] vs the same finite difference.
Output: results/studyb_v2/ps_checks.json and a printed table.  Usage: python ps_checks.py
"""
import json, math, os
import numpy as np, torch
import studyb_v2 as S

torch.set_default_dtype(torch.float64)
DELTA = 1e-6


def layer_out(m, l, ang, th):
    return m.layer(l, ang, th)


def shift_param_jac(m, l, ang, th):
    """d o^(l) / d theta^(l): (P_l, D) by the two-term rule, batched."""
    P = th.numel(); E = torch.eye(P).reshape(P, *th.shape) * (math.pi / 2)
    o = layer_out(m, l, ang.expand(2 * P, -1), torch.cat([th + E, th - E]))
    return 0.5 * (o[:P] - o[P:])


def shift_angle_jac(m, l, ang, th):
    D = ang.numel(); E = torch.eye(D) * (math.pi / 2)
    o = layer_out(m, l, torch.cat([ang + E, ang - E]), th.expand(2 * D, *th.shape))
    return 0.5 * (o[:D] - o[D:])


def T_back(m, g):  # adjoint of a = pi * T o
    q = m.q; return math.pi * g.reshape(q, q).transpose(0, 1).reshape(-1)


def forward(model, v, eps, dz=None):
    """Returns loss and the cache. With dz given: adversarial loss, decoder input mu + dz (eps = 0)."""
    with torch.no_grad():
        E, Dc = model.enc, model.dec; c = {}
        a = [math.pi * v]; o = []
        for l in range(E.L):
            o.append(E.layer(l, a[-1], E.theta[l]))
            if l < E.L - 1: a.append(E.mix(o[-1]))
        e = o[-1]; mu = e @ model.Wm.T + model.bm; lv = e @ model.Ws.T + model.bs; sig = (0.5 * lv).exp()
        z = mu + (dz if dz is not None else sig * eps)
        b = [z @ model.We.T]; p = []
        for l in range(Dc.L):
            p.append(Dc.layer(l, b[-1], Dc.theta[l]))
            if l < Dc.L - 1: b.append(Dc.mix(p[-1]))
        xh = p[-1] @ model.Wo.T + model.bo
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum()
        L = ((v - xh) ** 2).sum() + model.beta * kl
        c.update(a=a, o=o, e=e, mu=mu, lv=lv, sig=sig, z=z, b=b, p=p, xh=xh)
    return float(L), c


def alg9(model, v, eps, dz=None):
    """Gradient of L w.r.t. encoder and decoder angles by Algorithm 9 (shifts + chain rule), plus classical factors."""
    L, c = forward(model, v, eps, dz); E, Dc = model.enc, model.dec; beta = model.beta
    g_dec = torch.zeros_like(Dc.theta); g_enc = torch.zeros_like(E.theta)
    dL_dxh = 2 * (c["xh"] - v); g = model.Wo.T @ dL_dxh; f = {"dL_do_dec": g.clone()}
    for l in reversed(range(Dc.L)):
        g_dec[l] = (shift_param_jac(Dc, l, c["b"][l], Dc.theta[l]) @ g).reshape(Dc.theta[l].shape)
        g_ang = shift_angle_jac(Dc, l, c["b"][l], Dc.theta[l]) @ g
        g = T_back(Dc, g_ang) if l > 0 else g_ang
    dL_dz = model.We.T @ g; f["dL_dz"] = dL_dz.clone()
    if dz is None:
        dmu = dL_dz + beta * c["mu"]; dlv = dL_dz * 0.5 * c["sig"] * eps - 0.5 * beta * (1 - c["sig"] ** 2)
    else:
        dmu = dL_dz + beta * c["mu"]; dlv = -0.5 * beta * (1 - c["sig"] ** 2)
    g = model.Wm.T @ dmu + model.Ws.T @ dlv; f["dL_de"] = g.clone()
    for l in reversed(range(E.L)):
        g_enc[l] = (shift_param_jac(E, l, c["a"][l], E.theta[l]) @ g).reshape(E.theta[l].shape)
        if l > 0: g = T_back(E, shift_angle_jac(E, l, c["a"][l], E.theta[l]) @ g)
    return L, torch.cat([g_enc.reshape(-1), g_dec.reshape(-1)]), f, c


def loss_at(model, v, eps, dz, which, i, h):
    th = (model.enc if which == "enc" else model.dec).theta; flat = th.data.view(-1); o = flat[i].item()
    flat[i] = o + h; lp = forward(model, v, eps, dz)[0]; flat[i] = o - h; lm = forward(model, v, eps, dz)[0]; flat[i] = o
    return lp, lm


def check(D, complex_, seed=0, adversarial=False):
    gen = torch.Generator().manual_seed(seed); m = max(2, int(round(math.sqrt(D))) - (1 if complex_ else 0))
    model = S.QVAE(D, m, "fcvqc", gen=gen)
    v = torch.randn(D, generator=gen); v = v / v.norm()
    eps = torch.randn(m, generator=gen); dz = (torch.rand(m, generator=gen) * 0.2 - 0.1) if adversarial else None
    out = dict(D=D, q=model.enc.q, m=m, P=model.enc.theta.numel() + model.dec.theta.numel(), loss="adversarial" if adversarial else "sampled ELBO")
    # A: one FC layer, shift vs FD on readouts
    E = model.enc; l = 1; ang = E.mix(E.layer(0, math.pi * v, E.theta[0]).detach())
    Jps = shift_param_jac(E, l, ang, E.theta[l].detach())
    th = E.theta[l].detach(); P = th.numel(); Ed = torch.eye(P).reshape(P, *th.shape) * DELTA
    o = E.layer(l, ang.expand(2 * P, -1), torch.cat([th + Ed, th - Ed]).detach()); Jfd = (o[:P] - o[P:]) / (2 * DELTA)
    out["A_rel"] = float((Jps - Jfd).abs().max() / Jfd.abs().max())
    # A': two-term rule on the final readouts of the two-layer circuit, first-layer angles
    th0 = E.theta[0].detach(); P0 = th0.numel(); Es = torch.eye(P0).reshape(P0, *th0.shape)
    def full(th0b):
        a2 = E.mix(E.layer(0, (math.pi * v).expand(th0b.shape[0], -1), th0b)); return E.layer(1, a2, E.theta[1].detach().expand(th0b.shape[0], *E.theta[1].shape))
    with torch.no_grad():
        o = full(torch.cat([th0 + Es * math.pi / 2, th0 - Es * math.pi / 2])); Jold = 0.5 * (o[:P0] - o[P0:])
        o = full(torch.cat([th0 + Es * DELTA, th0 - Es * DELTA])); Jfd0 = (o[:P0] - o[P0:]) / (2 * DELTA)
    out["A2_rel"] = float((Jold - Jfd0).abs().max() / Jfd0.abs().max())
    # C: full gradient
    L, g_an, f, c = alg9(model, v, eps, dz)
    g_fd, g_old = [], []
    for which, th in (("enc", model.enc.theta), ("dec", model.dec.theta)):
        for i in range(th.numel()):
            lp, lm = loss_at(model, v, eps, dz, which, i, DELTA); g_fd.append((lp - lm) / (2 * DELTA))
            lp, lm = loss_at(model, v, eps, dz, which, i, math.pi / 2); g_old.append(0.5 * (lp - lm))
    g_fd, g_old = torch.tensor(g_fd), torch.tensor(g_old)
    # autodiff
    model.zero_grad(); mu, lv = model.encode(v[None]); mu, lv = mu[0], lv[0]
    zz = mu + (dz if dz is not None else (0.5 * lv).exp() * eps); Lt = model.elbo(v, mu, lv, zz); Lt.backward()
    g_ad = torch.cat([model.enc.theta.grad.reshape(-1), model.dec.theta.grad.reshape(-1)])
    nE = model.enc.theta.numel(); big = g_fd.abs() > 1e-6
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    out.update(C_rel=rel(g_an, g_fd), C_rel_enc=rel(g_an[:nE], g_fd[:nE]), C_rel_dec=rel(g_an[nE:], g_fd[nE:]),
               C_elem_rel_max=float(((g_an - g_fd).abs() / g_fd.abs())[big].max()), C_vs_autodiff=rel(g_an, g_ad),
               D_rel=float((g_old - g_fd).norm() / g_fd.norm()), D_elem_median=float(((g_old - g_fd).abs() / g_fd.abs())[big].median()),
               D_cos=float(g_old @ g_fd / g_old.norm() / g_fd.norm()), loss_check=abs(L - float(Lt)))
    # B: classical factors with readouts / z free
    def L_from_p2(p2):
        xh = p2 @ model.Wo.T + model.bo; kl = -0.5 * (1 + c["lv"] - c["mu"] ** 2 - c["lv"].exp()).sum()
        return float(((v - xh) ** 2).sum() + model.beta * kl)
    def L_from_z(z):
        b = z @ model.We.T
        with torch.no_grad():
            for l2 in range(model.dec.L):
                pp = model.dec.layer(l2, b, model.dec.theta[l2])
                if l2 < model.dec.L - 1: b = model.dec.mix(pp)
        return L_from_p2(pp)
    def L_from_e(e):
        mu = e @ model.Wm.T + model.bm; lv = e @ model.Ws.T + model.bs
        z = mu + (dz if dz is not None else (0.5 * lv).exp() * eps)
        b = z @ model.We.T
        with torch.no_grad():
            for l2 in range(model.dec.L):
                pp = model.dec.layer(l2, b, model.dec.theta[l2])
                if l2 < model.dec.L - 1: b = model.dec.mix(pp)
        xh = pp @ model.Wo.T + model.bo; kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum()
        return float(((v - xh) ** 2).sum() + model.beta * kl)
    def fdvec(fn, x0):
        r = torch.zeros_like(x0)
        for i in range(x0.numel()):
            xp, xm = x0.clone(), x0.clone(); xp[i] += DELTA; xm[i] -= DELTA; r[i] = (fn(xp) - fn(xm)) / (2 * DELTA)
        return r
    with torch.no_grad():
        out["B_rel_do_dec"] = rel(f["dL_do_dec"], fdvec(L_from_p2, c["p"][-1]))
        out["B_rel_dz"] = rel(f["dL_dz"], fdvec(L_from_z, c["z"]))
        out["B_rel_de"] = rel(f["dL_de"], fdvec(L_from_e, c["e"]))
    out["shifts_per_step"] = dict(params=2 * out["P"], angles=2 * D * (model.dec.L + model.enc.L - 1))
    return out


if __name__ == "__main__":
    rows = []
    for D, cx in ((4, False), (16, False), (9, True)):
        for adv in (False, True):
            r = check(D, cx, adversarial=adv); rows.append(r)
            print(f"D={D:2d} q={r['q']} P={r['P']:3d} {r['loss']:13s} | A {r['A_rel']:.1e} | A' {r['A2_rel']:.2f} | B dec {r['B_rel_do_dec']:.1e} z {r['B_rel_dz']:.1e} e {r['B_rel_de']:.1e} "
                  f"| C {r['C_rel']:.1e} (enc {r['C_rel_enc']:.1e}, dec {r['C_rel_dec']:.1e}, elem {r['C_elem_rel_max']:.1e}; autodiff {r['C_vs_autodiff']:.1e}) "
                  f"| D {r['D_rel']:.2f} (median elem {r['D_elem_median']:.2f}, cos {r['D_cos']:.2f})", flush=True)
    os.makedirs("results/studyb_v2", exist_ok=True); json.dump(rows, open("results/studyb_v2/ps_checks.json", "w"), indent=1)
