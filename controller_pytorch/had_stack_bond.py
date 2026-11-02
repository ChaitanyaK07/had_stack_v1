#!/usr/bin/env python3
"""
HAD-Stack for regression: Hamiltonian-controlled Stackelberg adversarial training of the
Two-Head LSTM (TH-LSTM) 10-year bond-yield forecaster.

This is Study A of "Hamiltonian-Channel Stackelberg Games for Adversarial Learning"
(Defs. 1-8, Algs. 1-5) with the fraud detector replaced by the TH-LSTM of
"Hybrid Deep Learning Architectures for Forecasting 10-Year Indian Government Bond Yields".

Game (Table I of the HCS paper, regression instantiation)
  Scenario   : an adversary corrupts the quotes in the forecaster's look-back window
               (erroneous/stale/manipulated ticks) so that the next-day forecast is wrong.
  Leader     : defender = TH-LSTM weights theta (commits; attacker probes it).
  Follower   : attacker  = delta on the (L+1) x d window of observed levels,
               ||delta||_inf <= eps, in units of the TRAIN std of daily changes (sigma_Delta).
  u_att      : sum_i ((yhat_i(x_i+delta_i) - y_i)/sigma_Delta)^2      (squared forecast error)
  u_def      : -[(1-lam) L_TH(x) + lam L_TH(x+delta)],  L_TH = a*MSE(level) + (1-a)*MSE(delta)
               (bond paper Eq. 4) -> the game is not zero-sum (different losses, same rows).
  Control ch.: eps_r = eps_ref * G2(k_r)/G2_base,  lam_r = clip(0.5*G1(k_r)/G1_base, .1, .9)  (Eq. 19)
  Feedback   : zeta_r = (L_adv-L_clean)/L_clean on validation; s_r = clip(zeta_r/zeta_1-1,-1,1);
               k_{r+1} = k_base (1-0.5 s_r)                                                  (Eq. 20)
  Diagnostics: train flip, fresh vs stale, strong (exploitability gap)               (Def. 8)
  Test time  : worst-case ensemble (white-box PGD-100x5 with SE and directional-margin losses,
               plus transfer from the other detectors), budgets {0.5, 1.0}               (Alg. 5)
  "Broken"   : a test day whose next-day DIRECTION the clean model forecasts correctly and that
               any attack flips (regression analogue of "detected fraud flipped").

Usage
  python had_stack_bond.py --check-hamiltonian
  python had_stack_bond.py --sweep                     # eps sweep on the undefended model
  python had_stack_bond.py --mode all --out results    # 3 seeds + 5 walk-forward folds
  python had_stack_bond.py --mode seeds --device cuda
"""
import argparse, copy, json, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =====================================================================================
# 1. Hamiltonian layer (Sec. IV; Alg. 1 complex treatment; Alg. 2 controller step)
# =====================================================================================
BASE = dict(w0=0.1, w1=0.1, w2=0.1, gam1=0.1, gam2=0.1, g1=0.1, g2=0.1, E1=1.0, E2=1.0)

# mode operators on C^4, basis index = 2*n_a + n_b  (|00>,|01>,|10>,|11>), n_max = 1
_sm = np.array([[0, 1], [0, 0]], complex)          # annihilation on one hard-core mode
_A = np.kron(_sm, np.eye(2))
_B = np.kron(np.eye(2), _sm)
_O_AB = _A @ _B                                      # released solver evaluates O = ab


def two_mode_H(wa, wb, gam, g, E, D, hbar=1.0):
    """Eq. (9): H = c_a n_a + c_b n_b + c_ab (a^dag b + b^dag a)."""
    ca = wa / (2j * hbar) + gam / 2
    cb = wb / (1j * hbar) - gam / 2
    cab = g * E / (1j * hbar) - gam * D
    na, nb = _A.conj().T @ _A, _B.conj().T @ _B
    hop = _A.conj().T @ _B + _B.conj().T @ _A
    return ca * na + cb * nb + cab * hop


def ground(H):
    """Complex treatment: ground = eigenvalue with smallest real part (numpy.linalg.eig)."""
    ev, V = np.linalg.eig(H)
    k = int(np.argmin(ev.real))
    psi = V[:, k] / np.linalg.norm(V[:, k])
    return ev, ev[k], psi


def solve_self_consistent(wa, wb, gam, g, E, D0=0.1, m=0.3, tol=1e-8, max_iter=500):
    """Alg. 1 (single operator): D <- m Re<psi0|O|psi0> + (1-m) D."""
    D = D0
    for it in range(1, max_iter + 1):
        ev, nu0, psi = ground(two_mode_H(wa, wb, gam, g, E, D))
        Dn = m * float(np.real(psi.conj() @ _O_AB @ psi)) + (1 - m) * D
        if abs(Dn - D) < tol:
            D = Dn
            break
        D = Dn
    ev, nu0, psi = ground(two_mode_H(wa, wb, gam, g, E, D))
    return dict(spec=ev, nu0=nu0, D=D, iters=it)


def _sorted_spec(ev):
    return ev[np.lexsort((-ev.imag, ev.real))]


class HamiltonianController:
    """Alg. 2. Couplings kappa = (g1, g2[, eta_T of H_V]) are scaled by (1 - 0.5 s)."""

    def __init__(self, eps_ref=0.5, n_coords=1):
        self.eps_ref, self.n_coords = eps_ref, n_coords
        b = self._solve(1.0)
        self.G1b, self.G2b = b["G1"], b["G2"]
        self.base_w = self._weights(b)

    def _solve(self, factor):
        p = BASE
        h1 = solve_self_consistent(p["w0"], p["w1"], p["gam1"], p["g1"] * factor, p["E1"])
        h2 = solve_self_consistent(p["w0"], p["w2"], p["gam2"], p["g2"] * factor, p["E2"])
        return dict(h1=h1, h2=h2, G1=-h1["nu0"].imag, G2=-h2["nu0"].imag)

    def _weights(self, sol):
        mags = np.concatenate([np.abs(_sorted_spec(sol["h1"]["spec"])),
                               np.abs(_sorted_spec(sol["h2"]["spec"]))])
        wH = np.resize(mags, self.n_coords)              # tile over input coordinates
        return 0.5 + 0.5 * wH / wH.mean()                 # floor 0.5, mean 1

    def step(self, s):
        factor = 1.0 - 0.5 * s                                         # Eq. (20)
        sol = self._solve(factor)
        eps = self.eps_ref * sol["G2"] / self.G2b                      # Eq. (19)
        lam = float(np.clip(0.5 * sol["G1"] / self.G1b, 0.1, 0.9))
        w = self._weights(sol)
        info = dict(factor=factor, E0=float(sol["h1"]["nu0"].real), G1=float(sol["G1"]),
                    G2=float(sol["G2"]), D1=sol["h1"]["D"], D_iters=sol["h1"]["iters"],
                    w_min=float(w.min()), w_max=float(w.max()))
        return float(eps), lam, w, info


TABLE_VI = {+1.0: (-0.03124, 0.11501, 0.3461), +0.5: (-0.01945, 0.13925, 0.4190),
            0.0: (-0.01371, 0.16618, 0.5000), -0.5: (-0.01062, 0.19274, 0.5799),
            -1.0: (-0.00869, 0.21888, 0.6586)}


def check_hamiltonian(verbose=True):
    c = HamiltonianController(eps_ref=0.5, n_coords=18)
    rows, ok = [], True
    for s, (E0, G, eps_t) in TABLE_VI.items():
        eps, lam, w, info = c.step(s)
        good = abs(info["E0"] - E0) < 1e-5 and abs(info["G1"] - G) < 1e-5 and abs(eps - eps_t) < 1e-4
        ok &= good
        rows.append((s, info["factor"], info["E0"], info["G1"], eps, lam, info["D1"], info["D_iters"], good))
    if verbose:
        print("Controller response vs. Table VI of the HCS paper")
        print("   s  factor        E0        Gamma      eps      lam   Delta*  iters  match")
        for r in rows:
            print(f"{r[0]:+5.1f}  {r[1]:5.2f}  {r[2]:+.5f}  {r[3]:.5f}  {r[4]:.4f}  {r[5]:.4f}  {r[6]:.1e}  {r[7]:4d}   {r[8]}")
        print(f"step weights (18 features, base): min {c.base_w.min():.3f} max {c.base_w.max():.3f} mean {c.base_w.mean():.3f}")
    assert ok, "Hamiltonian layer does not reproduce Table VI"
    return rows


# =====================================================================================
# 2. Data: windows of L+1 observed levels -> next-day level and change
# =====================================================================================
def load_windows(csv, inputs, target, L):
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    cols = list(dict.fromkeys(inputs + [target]))
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).reset_index(drop=True)
    Y = df[inputs].values.astype(np.float32)
    yt = df[target].values.astype(np.float32)
    tgt_in = inputs.index(target)
    idx = np.arange(L, len(df) - 1)                                   # window ends at t
    X = np.stack([Y[t - L:t + 1] for t in idx])                       # (N, L+1, d)
    y_next, y_last = yt[idx + 1], yt[idx]
    dates = df["Date"].values[idx + 1]
    return dict(X=X, y=y_next, ylast=y_last, dates=dates, tgt=tgt_in, levels=Y,
                n_rows=len(df), first=str(df["Date"].iloc[0].date()), last=str(df["Date"].iloc[-1].date()))


def make_splits(N, mode, n_test=250, val_frac=0.15, n_folds=5, block=150, val_block=150):
    if mode == "fixed":
        te = np.arange(N - n_test, N)
        n_val = int(round(val_frac * (N - n_test)))
        va = np.arange(N - n_test - n_val, N - n_test)
        return [dict(name="fixed", train=np.arange(0, va[0]), val=va, test=te)]
    out = []
    for k in range(n_folds):
        t0 = N - (n_folds - k) * block
        te = np.arange(t0, t0 + block)
        va = np.arange(t0 - val_block, t0)
        out.append(dict(name=f"fold{k + 1}", train=np.arange(0, va[0]), val=va, test=te))
    return out


class Data:
    """Scalers fit on the training range only (levels and daily changes)."""

    def __init__(self, W, split, L, device):
        tr = split["train"]
        lev = W["X"][tr][:, 1:, :].reshape(-1, W["X"].shape[2])
        dif = np.diff(W["X"][tr], axis=1).reshape(-1, W["X"].shape[2])
        self.mu, self.sdL = lev.mean(0), lev.std(0) + 1e-8
        self.sdD = dif.std(0) + 1e-8
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self.sets = {k: dict(X=T(W["X"][split[k]]), y=T(W["y"][split[k]]), ylast=T(W["ylast"][split[k]]),
                             dates=W["dates"][split[k]]) for k in ("train", "val", "test")}
        self.tgt, self.device = W["tgt"], device
        self.mu_t, self.sdL_t, self.sdD_t = T(self.mu), T(self.sdL), T(self.sdD)


# =====================================================================================
# 3. Defender: Two-Head LSTM (bond paper Sec. 3.3 item 6, Eqs. 4-5)
# =====================================================================================
class THLSTM(nn.Module):
    def __init__(self, data, hidden=50, head=32, drop=0.2, beta=0.7):
        super().__init__()
        d = data.sets["train"]["X"].shape[2]
        self.register_buffer("mu", data.mu_t.clone())
        self.register_buffer("sdL", data.sdL_t.clone())
        self.register_buffer("sdD", data.sdD_t.clone())
        self.tgt, self.beta = data.tgt, beta
        self.lstm = nn.LSTM(2 * d, hidden, batch_first=True)
        self.headA = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Dropout(drop), nn.Linear(head, 1))
        self.headB = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Dropout(drop), nn.Linear(head, 1))

    def forward(self, X):                      # X: (B, L+1, d) observed levels (possibly perturbed)
        lev = (X[:, 1:] - self.mu) / self.sdL
        dlt = (X[:, 1:] - X[:, :-1]) / self.sdD    # Delta features recomputed from the (perturbed) levels
        h, _ = self.lstm(torch.cat([lev, dlt], -1))
        h = h[:, -1]
        lvl = self.headA(h).squeeze(-1)            # standardised level of y_{t+1}
        dd = self.headB(h).squeeze(-1)             # standardised change
        t = self.tgt
        final = self.beta * (lvl * self.sdL[t] + self.mu[t]) + (1 - self.beta) * (X[:, -1, t] + dd * self.sdD[t])
        return lvl, dd, final                      # final: Eq. (5)


class RandomWalk(nn.Module):
    """No-change forecast y_hat_{t+1} = y_t (reference only; it has no parameters to defend)."""

    def __init__(self, data):
        super().__init__()
        self.register_buffer("sdD", data.sdD_t.clone())
        self.tgt = data.tgt

    def forward(self, X):
        f = X[:, -1, self.tgt]
        return f, f, f


def th_loss(model, X, y, ylast, alpha=0.6, reduce=True):
    """Bond paper Eq. (4) in standardised target units (defender's loss L_TH)."""
    lvl, dd, _ = model(X)
    t = model.tgt
    ys = (y - model.mu[t]) / model.sdL[t]
    ds = (y - ylast) / model.sdD[t]
    l = alpha * (lvl - ys) ** 2 + (1 - alpha) * (dd - ds) ** 2
    return l.mean() if reduce else l


def att_obj(final, y, ylast, sdD, kind):
    """Attacker objectives, per row. 'se' = squared forecast error (u_att, Eq. 16 analogue);
    'margin' = non-saturating directional margin (pushes the forecast against the true move)."""
    if kind == "se":
        return ((final - y) / sdD) ** 2
    sgn = torch.sign(y - ylast)
    return -sgn * (final - ylast) / sdD


# =====================================================================================
# 4. Follower: weighted L_inf PGD on the look-back window (Alg. 3) and its strong variants
# =====================================================================================
def pgd(model, X, y, ylast, eps, w=None, steps=10, restarts=1, kind="se", step_mult=2.5, bs=1024):
    """Returns perturbation delta in sigma_Delta units, ||delta||_inf <= eps exactly.
    Restarts keep, per row, the perturbation with the highest attacker objective."""
    was_training = model.training
    model.eval()
    sd = model.sdD
    W = torch.ones_like(X[0]) if w is None else torch.as_tensor(w, dtype=X.dtype, device=X.device).reshape(X.shape[1:])
    a = step_mult * eps / steps
    out = torch.zeros_like(X)
    for i0 in range(0, len(X), bs):
        x, yy, yl = X[i0:i0 + bs], y[i0:i0 + bs], ylast[i0:i0 + bs]
        best_d = torch.zeros_like(x)
        best_o = torch.full((len(x),), -float("inf"), device=X.device)
        for _ in range(restarts):
            d = (torch.rand_like(x) * 2 - 1) * eps
            for _ in range(steps):
                d.requires_grad_(True)
                with torch.backends.cudnn.flags(enabled=False):          # cuDNN RNN backward needs train mode
                    o = att_obj(model(x + d * sd)[2], yy, yl, sd[model.tgt], kind).sum()
                g, = torch.autograd.grad(o, d)
                d = (d.detach() + a * W * torch.sign(g)).clamp(-eps, eps)
            with torch.no_grad():
                o = att_obj(model(x + d * sd)[2], yy, yl, sd[model.tgt], kind)
                better = o > best_o
                best_o = torch.where(better, o, best_o)
                best_d[better] = d[better]
        out[i0:i0 + bs] = best_d
    model.train(was_training)
    return out


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X)[2]


def dir_correct(final, y, ylast):
    tr = torch.sign(y - ylast)
    return (torch.sign(final - ylast) == tr) & (tr != 0), tr != 0


def hit(final, y, sdt, tol):
    """Forecast 'hit': |error| <= tol * sigma_Delta (regression analogue of 'fraud detected')."""
    return (final - y).abs() <= tol * sdt


def flip_rate(model, S, delta, base_final=None, tol=1.0):
    """% of clean hits that the perturbation turns into misses (analogue of '% detected fraud flipped').
    Also returns the % of correct directions flipped (secondary)."""
    sdt = model.sdD[model.tgt]
    f0 = predict(model, S["X"]) if base_final is None else base_final
    fa = predict(model, S["X"] + delta * model.sdD)
    h0, ha = hit(f0, S["y"], sdt, tol), hit(fa, S["y"], sdt, tol)
    c0, _ = dir_correct(f0, S["y"], S["ylast"])
    ca, _ = dir_correct(fa, S["y"], S["ylast"])
    brk = 100.0 * float((h0 & ~ha).sum()) / max(int(h0.sum()), 1)
    dflip = 100.0 * float((c0 & ~ca).sum()) / max(int(c0.sum()), 1)
    return brk, fa, dflip


def rmse(f, y):
    return float(torch.sqrt(((f - y) ** 2).mean()))


# =====================================================================================
# 5. Training: undefended baseline (bond paper Alg. 1) and the game loop (HCS Alg. 4)
# =====================================================================================
def batches(n, bs, gen):
    p = torch.randperm(n, generator=gen)
    for i in range(0, n, bs):
        yield p[i:i + bs]


def train_baseline(data, args, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = THLSTM(data).to(data.device)
    opt = torch.optim.Adam(m.parameters(), lr=args.base_lr)
    tr, va = data.sets["train"], data.sets["val"]
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.base_epochs):
        m.train()
        for b in batches(len(tr["X"]), args.batch, gen):
            b = b.to(data.device)
            loss = th_loss(m, tr["X"][b], tr["y"][b], tr["ylast"][b], args.alpha)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(th_loss(m, va["X"], va["y"], va["ylast"], args.alpha))
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= args.base_patience:
                break
    m.load_state_dict(best_state)
    return m, dict(epochs=ep + 1, val_loss=best)


def game_loop(theta0, data, args, seed, controller="hamiltonian", fixed=(0.6586, 0.5), log_prefix=""):
    """HCS Alg. 4 (controller='hamiltonian') or fixed-budget PGD-AT (controller='fixed')."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(theta0)
    n_coords = int(np.prod(data.sets["train"]["X"].shape[1:]))
    ctrl = HamiltonianController(args.eps_ref, n_coords)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.rounds)
    tr, va = data.sets["train"], data.sets["val"]
    s, zeta1, stale = 0.0, None, None
    best, best_state, best_round, bad, log = float("inf"), None, 0, 0, []
    # round 0: the undefended starting point theta_0, attacked at the base budget (no update)
    eps0, lam0, w0, info0 = ctrl.step(0.0) if controller == "hamiltonian" else (fixed[0], fixed[1], None, {})
    d_tr = pgd(m, tr["X"], tr["y"], tr["ylast"], eps0, w0, steps=10)
    tr_brk = flip_rate(m, tr, d_tr, None, args.tol)[0]
    f_clean = predict(m, va["X"])
    d0 = pgd(m, va["X"], va["y"], va["ylast"], eps0, w0, steps=10)
    fr0, _, fd0 = flip_rate(m, va, d0, f_clean, args.tol)
    st0 = flip_rate(m, va, pgd(m, va["X"], va["y"], va["ylast"], eps0, None, steps=50, restarts=2), f_clean, args.tol)[0]
    log.append(dict(round=0, s=0.0, eps=eps0, lam=lam0, train_flip=tr_brk, fresh=fr0, stale=float("nan"), strong=st0,
                    gap=st0 - fr0, fresh_dir=fd0, zeta=float("nan"), val_rmse=rmse(f_clean, va["y"]),
                    val_rrmse=float("nan"), sec=0.0, **info0))
    if args.verbose:
        print(f"{log_prefix} r00 (theta_0, no update) eps={eps0:.4f} train={tr_brk:5.1f} fresh={fr0:5.1f} strong={st0:5.1f} "
              f"valRMSE={log[0]['val_rmse']:.4f}", flush=True)
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        if controller == "hamiltonian":
            eps, lam, w, info = ctrl.step(s)                                     # Hamiltonian step
        else:
            eps, lam, w, info = fixed[0], fixed[1], None, {}
        n_corr = n_flip = 0
        for b in batches(len(tr["X"]), args.batch, gen):
            b = b.to(data.device)
            x, y, yl = tr["X"][b], tr["y"][b], tr["ylast"][b]
            d = pgd(m, x, y, yl, eps, w, steps=10)                               # follower (Alg. 3)
            xa = x + d * m.sdD
            with torch.no_grad():                                                # follower payoff (Def. 8a)
                m.eval()
                h0 = hit(m(x)[2], y, m.sdD[m.tgt], args.tol)
                ha = hit(m(xa)[2], y, m.sdD[m.tgt], args.tol)
                n_corr += int(h0.sum()); n_flip += int((h0 & ~ha).sum())
            m.train()
            loss = (1 - lam) * th_loss(m, x, y, yl, args.alpha) + lam * th_loss(m, xa, y, yl, args.alpha)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()                                                           # leader (AdamW step)
        sch.step()
        # ---- validation: checkpoint criterion, feedback, diagnostics ----
        f_clean = predict(m, va["X"])
        d_ref = pgd(m, va["X"], va["y"], va["ylast"], args.eps_ref, ctrl.base_w, steps=10)
        r_rmse = rmse(predict(m, va["X"] + d_ref * m.sdD), va["y"])               # robust val RMSE
        d_fresh = pgd(m, va["X"], va["y"], va["ylast"], eps, w, steps=10)
        fresh, f_adv, fresh_dir = flip_rate(m, va, d_fresh, f_clean, args.tol)
        stale_rate = flip_rate(m, va, stale, f_clean, args.tol)[0] if stale is not None else float("nan")
        d_strong = pgd(m, va["X"], va["y"], va["ylast"], eps, None, steps=50, restarts=2)
        strong = flip_rate(m, va, d_strong, f_clean, args.tol)[0]
        sdt = m.sdD[m.tgt]
        L_clean = float((((f_clean - va["y"]) / sdt) ** 2).mean())
        L_adv = float((((f_adv - va["y"]) / sdt) ** 2).mean())
        zeta = (L_adv - L_clean) / max(L_clean, 1e-12)
        if r == 1:
            zeta1, s_next = zeta, 0.0
        else:
            s_next = float(np.clip(zeta / zeta1 - 1, -1, 1))                       # Eq. (20)
        row = dict(round=r, s=s, eps=eps, lam=lam, train_flip=100.0 * n_flip / max(n_corr, 1),
                   fresh=fresh, stale=stale_rate, strong=strong, gap=strong - fresh, fresh_dir=fresh_dir,
                   zeta=zeta, val_rmse=rmse(f_clean, va["y"]), val_rrmse=r_rmse, sec=time.time() - t0, **info)
        log.append(row)
        if args.verbose:
            print(f"{log_prefix} r{r:02d} s={s:+.2f} eps={eps:.4f} lam={lam:.4f} train={row['train_flip']:5.1f} "
                  f"fresh={fresh:5.1f} stale={stale_rate:5.1f} strong={strong:5.1f} zeta={zeta:.3f} "
                  f"valRMSE={row['val_rmse']:.4f} rRMSE={r_rmse:.4f} "
                  f"({row['sec']:.1f}s)", flush=True)
        s, stale = s_next, d_fresh
        if r_rmse < best - 1e-6:
            best, best_state, best_round, bad = r_rmse, copy.deepcopy(m.state_dict()), r, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    m.load_state_dict(best_state)
    return m, log, best_round


# =====================================================================================
# 6. Worst-case test-time response (HCS Alg. 5) and statistics
# =====================================================================================
def evaluate(models, S, budgets, args, sources=None):
    """models: dict name -> model (same split); sources: names used as transfer sources.
    Returns per-model metrics and per-row arrays."""
    sources = list(models) if sources is None else sources
    res, rows = {}, {}
    for name, m in models.items():
        sdt = m.sdD[m.tgt]
        f0 = predict(m, S["X"])
        c0, nz = dir_correct(f0, S["y"], S["ylast"])
        h0 = hit(f0, S["y"], sdt, args.tol)
        out = dict(clean_MAE=float((f0 - S["y"]).abs().mean()), clean_RMSE=rmse(f0, S["y"]),
                   clean_hit=100.0 * float(h0.float().mean()),
                   clean_DA=100.0 * float(c0.sum()) / int(nz.sum()))
        rows[name] = dict(h0=h0.cpu().numpy(), c0=c0.cpu().numpy(), nz=nz.cpu().numpy(),
                          err0=(f0 - S["y"]).abs().cpu().numpy())
        for eps in budgets:
            finals = []
            for kind in ("se", "margin"):                                         # white-box
                d = pgd(m, S["X"], S["y"], S["ylast"], eps, None, steps=args.eval_steps,
                        restarts=args.eval_restarts, kind=kind)
                assert float(d.abs().max()) <= eps + 1e-6
                finals.append(predict(m, S["X"] + d * m.sdD))
            for other in sources:                                                 # transfer
                mo = models[other]
                if other == name:
                    continue
                for kind in ("se", "margin"):
                    d = pgd(mo, S["X"], S["y"], S["ylast"], eps, None, steps=args.eval_steps, kind=kind)
                    assert float(d.abs().max()) <= eps + 1e-6
                    finals.append(predict(m, S["X"] + d * m.sdD))
            broken = torch.zeros_like(h0)
            dflip = torch.zeros_like(c0)
            worst = torch.zeros_like(S["y"])
            for fa in finals:
                broken |= h0 & ~hit(fa, S["y"], sdt, args.tol)
                ca, _ = dir_correct(fa, S["y"], S["ylast"])
                dflip |= c0 & ~ca
                worst = torch.maximum(worst, (fa - S["y"]).abs())
            out[f"broken@{eps}"] = 100.0 * float(broken.sum()) / max(int(h0.sum()), 1)
            out[f"robHit@{eps}"] = 100.0 * float((h0 & ~broken).float().mean())
            out[f"wcRMSE@{eps}"] = float(torch.sqrt((worst ** 2).mean()))
            out[f"wcMAE@{eps}"] = float(worst.mean())
            out[f"dirFlip@{eps}"] = 100.0 * float(dflip.sum()) / max(int(c0.sum()), 1)
            out[f"robDA@{eps}"] = 100.0 * float((c0 & ~dflip).sum()) / int(nz.sum())
            rows[name][f"broken@{eps}"] = broken.cpu().numpy()
            rows[name][f"worst@{eps}"] = worst.cpu().numpy()
        if isinstance(m, RandomWalk):            # a no-change forecast has no direction to score
            for k in list(out):
                if "DA" in k or "dirFlip" in k:
                    out[k] = float("nan")
        res[name] = out
    return res, rows


def block_bootstrap_diff(ra, rb, budgets, n_boot=1000, block=10, seed=0):
    """Paired moving-block bootstrap (95%) of model a minus model b on the same test rows."""
    rng = np.random.default_rng(seed)
    n = len(ra["c0"])
    nb = int(math.ceil(n / block))
    out = {}
    stats = {}
    for eps in budgets:
        stats[f"robHit@{eps}"] = lambda r, i, e=eps: 100.0 * (r["h0"][i] & ~r[f"broken@{e}"][i]).mean()
        stats[f"wcRMSE@{eps}"] = lambda r, i, e=eps: float(np.sqrt((r[f"worst@{e}"][i] ** 2).mean()))
    for k, f in stats.items():
        full = np.arange(n)
        point = f(ra, full) - f(rb, full)
        bs = []
        for _ in range(n_boot):
            starts = rng.integers(0, n - block + 1, nb)
            i = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:n]
            bs.append(f(ra, i) - f(rb, i))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = dict(diff=float(point), lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))
    return out


# =====================================================================================
# 7. Experiment drivers
# =====================================================================================
def run_split(W, split, args, seeds, base_seed, tag):
    data = Data(W, split, args.L, args.device)
    print(f"\n=== {tag}: train {len(split['train'])}  val {len(split['val'])}  test {len(split['test'])}  "
          f"test {str(data.sets['test']['dates'][0])[:10]} .. {str(data.sets['test']['dates'][-1])[:10]}", flush=True)
    base, binfo = train_baseline(data, args, base_seed)
    print(f"{tag} baseline: {binfo['epochs']} epochs, val L_TH {binfo['val_loss']:.4f}", flush=True)
    outs = []
    for sd in seeds:
        had, log, br = game_loop(base, data, args, sd, "hamiltonian", log_prefix=f"{tag}/s{sd} HAD")
        pat, plog, pbr = game_loop(base, data, args, sd, "fixed", fixed=(args.fixed_eps, 0.5), log_prefix=f"{tag}/s{sd} PGD-AT")
        models = {"random-walk (ref)": RandomWalk(data).to(data.device), "undefended": base,
                  "PGD-AT(fixed)": pat, "HAD-Stack": had}
        res, rows = evaluate(models, data.sets["test"], args.budgets, args,
                             sources=["undefended", "PGD-AT(fixed)", "HAD-Stack"])
        ci = block_bootstrap_diff(rows["HAD-Stack"], rows["undefended"], args.budgets)
        outs.append(dict(split=tag, seed=sd, baseline=binfo, had_best_round=br, pgdat_best_round=pbr,
                         metrics=res, ci_had_minus_undef=ci, had_log=log, pgdat_log=plog))
        print(f"{tag}/s{sd} test: " + "  ".join(
            f"{k}: hit {v['clean_hit']:.1f} RMSE {v['clean_RMSE']:.4f} | " +
            " ".join(f"robHit@{e} {v[f'robHit@{e}']:.1f} brk@{e} {v[f'broken@{e}']:.1f}" for e in args.budgets)
            for k, v in res.items()), flush=True)
        if args.plots and tag == "fixed" and sd == seeds[0]:
            make_plots(log, data, models, args)
    return outs


def make_plots(log, data, models, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(args.out, exist_ok=True)
    r = [l["round"] for l in log]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(r, [l["eps"] for l in log], "o-", label=r"$\varepsilon_r$ (= budget)")
    ax[0].plot(r, [l["lam"] for l in log], "s--", label=r"$\lambda_r$")
    ax[0].set_xlabel("round"); ax[0].set_title("Hamiltonian controller"); ax[0].legend()
    for k, st in [("train_flip", "o-"), ("fresh", "s-"), ("stale", "^--"), ("strong", "d:")]:
        ax[1].plot(r, [l[k] for l in log], st, label=k)
    ax[1].set_xlabel("round"); ax[1].set_ylabel("% of clean hits broken")
    ax[1].set_title("Loop diagnostics (Def. 8)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "round_dynamics.png"), dpi=150); plt.close(fig)

    S = data.sets["test"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for a, name in zip(ax, ["undefended", "HAD-Stack"]):
        m = models[name]
        d = pgd(m, S["X"], S["y"], S["ylast"], 0.5, None, steps=100, restarts=1, kind="se")
        a.plot(S["dates"], S["y"].cpu(), "k-", lw=1, label="actual")
        a.plot(S["dates"], predict(m, S["X"]).cpu(), lw=1, label="forecast (clean)")
        a.plot(S["dates"], predict(m, S["X"] + d * m.sdD).cpu(), lw=1, alpha=.8, label=r"forecast (PGD-100, $\varepsilon$=0.5)")
        a.set_title(name); a.legend(fontsize=8); a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "test_forecasts.png"), dpi=150); plt.close(fig)


def summarize(all_out, budgets):
    lines = []
    for group in sorted(set(o["split"] == "fixed" for o in all_out), reverse=True):
        sub = [o for o in all_out if (o["split"] == "fixed") == group]
        lines.append(f"\n## {'Fixed split, seeds' if group else 'Walk-forward folds'} (n={len(sub)})")
        keys = ["clean_RMSE", "clean_hit", "clean_DA"] + [f"{p}@{e}" for e in budgets for p in ("broken", "robHit", "wcRMSE", "robDA")]
        lines.append("| model | " + " | ".join(keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        for name in sub[0]["metrics"]:
            vals = np.array([[o["metrics"][name][k] for k in keys] for o in sub])
            lines.append(f"| {name} | " + " | ".join(f"{m:.3f}±{s:.3f}" if "RMSE" in k or "MAE" in k else f"{m:.1f}±{s:.1f}"
                                                   for k, m, s in zip(keys, vals.mean(0), vals.std(0))) + " |")
        lines.append("\nHAD-Stack minus undefended, paired block bootstrap 95% CI:")
        for o in sub:
            lines.append(f"- {o['split']}/seed{o['seed']}: " + "; ".join(
                f"{k} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]{'*' if v['sig'] else ''}" for k, v in o["ci_had_minus_undef"].items()))
    return "\n".join(lines)


def sweep(W, args):
    split = make_splits(len(W["X"]), "fixed", args.n_test)[0]
    data = Data(W, split, args.L, args.device)
    base, _ = train_baseline(data, args, 0)
    S = data.sets["val"]
    f0 = predict(base, S["X"])
    print(f"undefended, validation: RMSE {rmse(f0, S['y']):.4f}  DA {100 * float(dir_correct(f0, S['y'], S['ylast'])[0].sum()) / int(dir_correct(f0, S['y'], S['ylast'])[1].sum()):.1f}%")
    rw = S["ylast"]
    print(f"random walk, validation: RMSE {rmse(rw, S['y']):.4f}  hit {100 * float(hit(rw, S['y'], base.sdD[base.tgt], args.tol).float().mean()):.1f}%  "
          f"undefended hit {100 * float(hit(f0, S['y'], base.sdD[base.tgt], args.tol).float().mean()):.1f}%")
    print("eps (x sigma_Delta)  PGD-10 hits broken%   adv RMSE / clean RMSE")
    for eps in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]:
        d = pgd(base, S["X"], S["y"], S["ylast"], eps, None, steps=10)
        fr, fa, dfl = flip_rate(base, S, d, f0, args.tol)
        print(f"   {eps:5.2f}             {fr:5.1f} (dir {dfl:5.1f})   {rmse(fa, S['y']) / rmse(f0, S['y']):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "india_bond_yields_2010_2025.csv"))
    p.add_argument("--inputs", default="10Y", help="comma list of input tenors, e.g. 1Y,5Y,10Y,30Y")
    p.add_argument("--target", default="10Y")
    p.add_argument("--L", type=int, default=20, help="look-back window (bond paper: 20)")
    p.add_argument("--mode", choices=["seeds", "folds", "all"], default="all")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--n-test", type=int, default=250)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold-block", type=int, default=150)
    p.add_argument("--alpha", type=float, default=0.6, help="TH-LSTM loss weight (Eq. 4)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--base-epochs", type=int, default=200)
    p.add_argument("--base-patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW lr inside the game loop (HCS Alg. 4 uses 1e-4 at batch 256; 1e-3 = bond-paper lr at batch 32)")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--eps-ref", type=float, default=0.5, help="reference budget in Eq. (19), sigma_Delta units")
    p.add_argument("--fixed-eps", type=float, default=0.6586)
    p.add_argument("--budgets", default="0.5,1.0")
    p.add_argument("--tol", type=float, default=1.0, help="hit tolerance in sigma_Delta units")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-restarts", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default="results_bond")
    p.add_argument("--check-hamiltonian", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args()
    args.budgets = [float(b) for b in args.budgets.split(",")]
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)

    check_hamiltonian(verbose=True)
    if args.check_hamiltonian:
        return
    inputs = [c.strip() for c in args.inputs.split(",")]
    W = load_windows(args.csv, inputs, args.target, args.L)
    print(f"\ndata: {args.csv} inputs={inputs} target={args.target}  rows {W['n_rows']} ({W['first']} .. {W['last']}), "
          f"windows {len(W['X'])}, device {args.device}")
    lv = W["levels"][:, W["tgt"]]
    print(f"target level range {lv.min():.2f}..{lv.max():.2f}; zero-change days {100 * np.mean(np.diff(lv) == 0):.1f}%")
    if args.sweep:
        sweep(W, args)
        return
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    all_out, t0 = [], time.time()
    if args.mode in ("seeds", "all"):
        sp = make_splits(len(W["X"]), "fixed", args.n_test)[0]
        all_out += run_split(W, sp, args, seeds, base_seed=0, tag="fixed")
    if args.mode in ("folds", "all"):
        for sp in make_splits(len(W["X"]), "folds", n_folds=args.n_folds, block=args.fold_block, val_block=args.fold_block):
            all_out += run_split(W, sp, args, [seeds[0]], base_seed=0, tag=sp["name"])
    summ = summarize(all_out, args.budgets)
    print(summ)
    print(f"\ntotal time {time.time() - t0:.0f}s")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), runs=all_out), f, indent=1, default=str)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summ + "\n")
    pd.DataFrame([dict(split=o["split"], seed=o["seed"], **l) for o in all_out for l in o["had_log"]]).to_csv(
        os.path.join(args.out, "had_rounds.csv"), index=False)


if __name__ == "__main__":
    main()
