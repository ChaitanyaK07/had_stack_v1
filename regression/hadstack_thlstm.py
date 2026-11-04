#!/usr/bin/env python3
"""
HAD-Stack (Study A of "Hamiltonian-Channel Stackelberg Games for Adversarial Learning") applied,
unchanged, to the regression model of "Hybrid Deep Learning Architectures for Forecasting 10-Year
Indian Government Bond Yields": the Two-Head LSTM (level + Delta) hybrid forecaster, taken AS IS
from cell 13 of India_Bonds_DL_MLP.ipynb (same data pipeline, features, window, split, architecture,
loss weights, optimiser, early stopping and hybrid combination).

Mapping of the building blocks (HCS Table I, Study A column -> this regression instantiation)
  Scenario  : an adversary corrupts the market data the forecaster reads (the standardised technical
              indicators of the 10-day look-back window) so that the next-day yield forecast is wrong.
              Calendar and regime dummies are not market data and are not perturbed (as legitimate rows
              are not perturbed in Eq. 15).
  Leader    : defender = Two-Head LSTM weights theta (commits; the attacker probes it)
  Follower  : attacker = delta on the window, ||delta||_inf <= eps_r on market features   (Eq. 15)
  u_att     : squared error of the hybrid forecast on the true yield                        (Eq. 16)
  u_def     : -[(1-lam_r) L(x) + lam_r L(x+delta)], L = the model's own compiled loss
              0.5*MSE(level) + 0.5*MSE(delta)                                              (Eq. 17)
  Controller: eps_r = 0.5 G2/G2_base, lam_r = clip(0.5 G1/G1_base, .1, .9)                (Eq. 19)
  Feedback  : zeta_r on validation, s_r = clip(zeta_r/zeta_1 - 1, -1, 1), k = k_base(1-0.5 s) (Eq. 20)
  Algorithms: 1-2 (Hamiltonian step), 3 (weighted PGD-10), 4 (loop), 5 (worst-case ensemble)
  Follower  : PGD-10 (default, as in the paper's Study A), QPSO (derivative-free, Study B's Alg. 10 / Eq. 36,
              20 particles x 15 iterations, beta annealed 1.0 -> 0.3), or the better of the two per window
  "Broken"  : a test day whose absolute forecast error the attack raises by more than tau = one
              training-set standard deviation of daily yield changes (regression analogue of
              "detected fraud flipped"; Def. 6: robustness = share of cases where the response fails).
All HAD-Stack hyperparameters are the paper's: AdamW(1e-4, wd 1e-4), cosine over 30 rounds,
clip-norm 1.0, PGD-10 with alpha = 2.5 eps/10, eps_ref = 0.5, checkpoint on robust validation metric
at eps = 0.5 (patience 6), fixed-budget comparison (eps = controller maximum, lam 0.5), evaluation budgets {0.5, 1.0}.
"""
import argparse, copy, json, math, os, time
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# =============================================================================================
# 1. Hamiltonian layer (HCS Sec. IV, Alg. 1 complex treatment, Alg. 2)
# =============================================================================================
BASE = dict(w0=0.1, w1=0.1, w2=0.1, gam1=0.1, gam2=0.1, g1=0.1, g2=0.1, E1=1.0, E2=1.0)
_sm = np.array([[0, 1], [0, 0]], complex)
_A, _B = np.kron(_sm, np.eye(2)), np.kron(np.eye(2), _sm)
HCS_ORDER = os.environ.get("HCS_ORDER", "adagb")   # "adagb": Delta = Re<a^dag b> (Eqs. 7-8); "ab": <ab> (original release)
assert HCS_ORDER in ("ab", "adagb"), HCS_ORDER
_O_AB = _A @ _B if HCS_ORDER == "ab" else _A.conj().T @ _B


def two_mode_H(wa, wb, gam, g, E, D):                       # Eq. (9), hbar = 1
    ca, cb, cab = wa / 2j + gam / 2, wb / 1j - gam / 2, g * E / 1j - gam * D
    return ca * (_A.conj().T @ _A) + cb * (_B.conj().T @ _B) + cab * (_A.conj().T @ _B + _B.conj().T @ _A)


def solve_sc(wa, wb, gam, g, E, D=0.1, m=0.3, tol=1e-8):     # Def. 4 / Alg. 1
    for it in range(1, 1000):
        ev, V = np.linalg.eig(two_mode_H(wa, wb, gam, g, E, D)); k = int(np.argmin(ev.real))
        psi = V[:, k] / np.linalg.norm(V[:, k])
        Dn = m * float(np.real(psi.conj() @ _O_AB @ psi)) + (1 - m) * D
        done = abs(Dn - D) < tol; D = Dn
        if done: break
    ev = np.linalg.eigvals(two_mode_H(wa, wb, gam, g, E, D))
    return ev, ev[int(np.argmin(ev.real))], D, it


class Controller:                                             # Alg. 2
    def __init__(self, d):
        self.d = d
        _, self.G1b, self.G2b, self.base_w, _ = self._at(1.0)

    def _at(self, f):
        p = BASE
        e1, n1, D1, it = solve_sc(p["w0"], p["w1"], p["gam1"], p["g1"] * f, p["E1"])
        e2, n2, _, _ = solve_sc(p["w0"], p["w2"], p["gam2"], p["g2"] * f, p["E2"])
        srt = lambda e: e[np.lexsort((-e.imag, e.real))]
        wH = np.resize(np.r_[np.abs(srt(e1)), np.abs(srt(e2))], self.d)       # tile over d features
        return n1, -n1.imag, -n2.imag, 0.5 + 0.5 * wH / wH.mean(), (D1, it)

    def step(self, s):
        n1, G1, G2, w, (D1, it) = self._at(1 - 0.5 * s)                        # Eq. (20)
        return 0.5 * G2 / self.G2b, float(np.clip(0.5 * G1 / self.G1b, .1, .9)), w, \
            dict(E0=float(n1.real), G1=float(G1), G2=float(G2), w_max=float(w.max()))  # Eq. (19)


def check_table_vi():
    c = Controller(18)
    tab = {"ab": {+1.0: (0.11501, .3461), +.5: (0.13925, .4190), 0.: (0.16618, .5), -.5: (0.19274, .5799), -1.: (0.21888, .6586)},
           "adagb": {+1.0: (0.12730, .3686), +.5: (0.14906, .4317), 0.: (0.17266, .5), -.5: (0.19713, .5708), -1.: (0.22198, .6428)}}[HCS_ORDER]
    for s, (G, e) in tab.items():
        eps, lam, w, info = c.step(s)
        assert abs(info["G1"] - G) < 1e-5 and abs(eps - e) < 1e-4, (s, info, eps)
    print(f"Hamiltonian layer (order {HCS_ORDER}) reproduces Table VI (Gamma, eps), eps max {EPS_MAX:.4f}; step weights in [{c.base_w.min():.3f}, {c.base_w.max():.3f}], mean {c.base_w.mean():.3f}")


EPS_MAX = Controller(18).step(-1.0)[0]          # controller's largest budget (s = -1); fixed-budget comparison uses it
FB_LABEL = "FB-AT (eps max)"


# =============================================================================================
# 2. The bond paper's model AS IS: notebook cell 13 (data pipeline, model, training, hybrid)
#    Only change: the train/test split index can be set (fold experiments); default = cell 13's 0.8.
# =============================================================================================
def cell13_data(split_at=None):
    tech_df = pd.read_csv(os.path.join(DATA_DIR, 'bond_yield_technical_indicators.csv'), parse_dates=['Date'])
    y_df = pd.read_csv(os.path.join(DATA_DIR, 'india_bond_yields_2010_2025.csv'), parse_dates=['Date']).set_index('Date')
    y_df = y_df[pd.to_numeric(y_df['10Y'], errors='coerce').notnull()]
    tech_df['yield'] = tech_df['Date'].map(y_df['10Y'])
    tech_df.dropna(subset=['yield'], inplace=True)
    df = tech_df.copy()
    df['d_yield'] = df['yield'].diff()
    for lag in range(1, 6):
        df[f'd_lag{lag}'] = df['d_yield'].shift(lag)
    df['vol_20'] = df['yield'].rolling(20).std()
    df['ma5'] = df['yield'].rolling(5).mean()
    df['ma20'] = df['yield'].rolling(20).mean()
    df['momentum'] = df['ma5'] - df['ma20']
    df['roc5'] = df['yield'].pct_change(5)
    df['dow'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df.dropna(inplace=True)
    for c in ['vol_20', 'momentum', 'roc5']:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)
    df = pd.get_dummies(df, columns=['dow', 'month'], prefix=['dow', 'm'])
    df['regime'] = (df['vol_20'] > df['vol_20'].median()).astype(int)
    df = pd.get_dummies(df, columns=['regime'], prefix='regime')
    df['y_next'] = df['yield'].shift(-1)
    df.dropna(subset=['y_next', 'd_yield'], inplace=True)
    base_feats = ['Yield_Curve_Slope', 'Yield_Curvature', 'Level_Shift', 'Roll_Down', 'Duration', 'Convexity',
                  '30y_5y_Spread', '10y_1y_Spread', 'NS_Level', 'NS_Slope', 'NS_Curvature', 'vol_20', 'momentum', 'roc5']
    delta_lags = [f'd_lag{i}' for i in range(1, 6)]
    dummy_feats = [c for c in df.columns if c.startswith('dow_') or c.startswith('m_') or c.startswith('regime_')]
    all_feats = base_feats + delta_lags + dummy_feats
    X = df[all_feats].astype(float).values
    y_level = df['y_next'].values
    y_delta = df['d_yield'].values
    levels = df['yield'].values
    X_mean, X_std = X.mean(0), X.std(0)
    X_scaled = (X - X_mean) / X_std
    split = int(0.8 * len(df)) if split_at is None else split_at
    window = 10

    def build_seq_multi(X, y1, y2, w):
        Xs, ys1, ys2 = [], [], []
        for i in range(w, len(y1)):
            Xs.append(X[i - w:i]); ys1.append(y1[i]); ys2.append(y2[i])
        return np.array(Xs), np.array(ys1), np.array(ys2)

    D = {}
    for name, sl in (("tr", slice(0, split)), ("te", slice(split, None))):
        Xs, lv, dl = build_seq_multi(X_scaled[sl], y_level[sl], y_delta[sl], window)
        lev = levels[sl]
        D[name] = dict(X=Xs.astype(np.float32), lvl=lv.astype(np.float32), dlt=dl.astype(np.float32),
                       prev=lev[window - 1:-1].astype(np.float32), true=lev[window:].astype(np.float32),
                       dates=df['Date'].values[sl][window:])
    # Keras validation_split=0.1 holds out the LAST 10% of the training sequences
    n = len(D["tr"]["X"]); n_val = int(n * 0.1)
    D["fit"] = {k: v[:n - n_val] for k, v in D["tr"].items()}
    D["val"] = {k: v[n - n_val:] for k, v in D["tr"].items()}
    market = np.array([not (f.startswith('dow_') or f.startswith('m_') or f.startswith('regime_')) for f in all_feats])
    D.update(feats=all_feats, market=market.astype(np.float32), window=window, n_feats=X_scaled.shape[1],
             split=split, n_rows=len(df), sd_delta=float(np.std(y_delta[:split])),
             test_span=(str(D["te"]["dates"][0])[:10], str(D["te"]["dates"][-1])[:10]))
    return D


def cell13_model(window, n_feats):
    inp = Input((window, n_feats))
    x = LSTM(32, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(16)(x)
    x = Dropout(0.2)(x)
    level_out = Dense(1, name='level')(x)
    delta_out = Dense(1, name='delta')(x)
    model = Model(inp, [level_out, delta_out])
    model.compile(optimizer=Adam(1e-3), loss={'level': 'mse', 'delta': 'mse'}, loss_weights={'level': 0.5, 'delta': 0.5})
    return model


def cell13_train(D, seed=0):
    tf.keras.utils.set_random_seed(seed)
    model = cell13_model(D["window"], D["n_feats"])
    es = EarlyStopping(monitor='val_level_loss', mode='min', patience=5, restore_best_weights=True)
    model.fit(D["tr"]["X"], {'level': D["tr"]["lvl"], 'delta': D["tr"]["dlt"]}, epochs=50, batch_size=16,
              validation_split=0.1, callbacks=[es], verbose=0)
    return model


def hybrid(model, X, prev, training=False):                   # cell 13: 0.5*level + 0.5*(prev + delta)
    lvl, dlt = model(X, training=training)
    return 0.5 * tf.reshape(lvl, [-1]) + 0.5 * (prev + tf.reshape(dlt, [-1]))


def th_loss(model, X, lvl_t, dlt_t, training):                # the model's compiled loss
    lvl, dlt = model(X, training=training)
    return 0.5 * tf.reduce_mean((tf.reshape(lvl, [-1]) - lvl_t) ** 2) + 0.5 * tf.reduce_mean((tf.reshape(dlt, [-1]) - dlt_t) ** 2)


# =============================================================================================
# 3. Follower: weighted L_inf PGD on market features (Alg. 3) and its strong variants
# =============================================================================================
def att_obj(f, true, prev, kind):
    if kind == "se":                                          # Eq. (16) analogue
        return (f - true) ** 2
    return -tf.sign(true - prev) * (f - prev)                 # directional margin (Alg. 5, line 4)


def make_pgd(model):
    @tf.function(reduce_retracing=True)
    def step(X, d, true, prev, W, a, eps, kind_se):
        with tf.GradientTape() as t:
            t.watch(d)
            f = hybrid(model, X + d, prev)
            o = tf.cond(kind_se, lambda: att_obj(f, true, prev, "se"), lambda: att_obj(f, true, prev, "m"))
            o = tf.reduce_sum(o)
        g = t.gradient(o, d)
        return tf.clip_by_value(d + a * W * tf.sign(g), -eps, eps)

    @tf.function(reduce_retracing=True)
    def objective(X, d, true, prev, kind_se):
        f = hybrid(model, X + d, prev)
        return tf.cond(kind_se, lambda: att_obj(f, true, prev, "se"), lambda: att_obj(f, true, prev, "m"))

    def pgd(X, true, prev, eps, w, mask, steps=10, restarts=1, kind="se"):
        W = tf.constant((np.ones(X.shape[-1]) if w is None else w) * mask, tf.float32)   # zero on non-market features
        M = tf.constant(mask, tf.float32)
        a = tf.constant(2.5 * eps / steps, tf.float32); e = tf.constant(eps, tf.float32)
        X = tf.constant(X); true = tf.constant(true); prev = tf.constant(prev); kse = tf.constant(kind == "se")
        best_d, best_o = tf.zeros_like(X), tf.fill([X.shape[0]], -np.inf)
        for _ in range(restarts):
            d = tf.random.uniform(X.shape, -eps, eps) * M
            for _ in range(steps):
                d = step(X, d, true, prev, W, a, e, kse)
            o = objective(X, d, true, prev, kse)
            better = o > best_o
            best_o = tf.where(better, o, best_o)
            best_d = tf.where(better[:, None, None], d, best_d)
        assert float(tf.reduce_max(tf.abs(best_d))) <= eps + 1e-6
        assert float(tf.reduce_max(tf.abs(best_d * (1 - M)))) == 0.0
        return best_d.numpy()
    return pgd


def make_qpso(model):
    """Derivative-free follower: QPSO (Sun et al. 2004) as in Study B (Alg. 10, Eq. 36), one swarm per window, all
    swarms of a batch evaluated in one forward pass and the whole search compiled into one graph.
    Search space: one offset per market feature, held over the T days of the window, i.e. the 19-dimensional
    subset {delta_{i,t,j} = c_{i,j}, |c_{i,j}| <= eps, j in M} of D_eps (Eq. DR). Over all 190 coordinates the
    same budget of 20 x 15 evaluations leaves the training error unchanged (0.0% broken vs 9.5% for PGD-10).
    Particles start at random vertices of the box; the Hamiltonian step weights w scale the per-feature spread,
    as they scale the per-feature PGD step."""
    @tf.function(reduce_retracing=True)
    def core(X, true, prev, W, M, eps, n_p, n_it, kse, b0, b1):
        n, T, d = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        Xr = tf.repeat(X, n_p, axis=0); tr = tf.repeat(true, n_p); pr = tf.repeat(prev, n_p)

        def F(c):                                                                   # attacker's utility per particle
            z = tf.reshape(tf.broadcast_to(c, [n, n_p, T, d]), [n * n_p, T, d])
            f = hybrid(model, Xr + z, pr)
            o = tf.cond(kse, lambda: att_obj(f, tr, pr, "se"), lambda: att_obj(f, tr, pr, "m"))
            return tf.reshape(o, [n, n_p])
        sh = [n, n_p, 1, d]
        xi = eps * tf.sign(tf.random.uniform(sh, -1.0, 1.0)) * M                    # random vertices of D_eps
        pb, pf = xi, F(xi)
        gb = tf.gather(pb, tf.argmax(pf, 1), batch_dims=1)
        nf = tf.cast(tf.maximum(n_it - 1, 1), tf.float32)
        for t in tf.range(n_it):
            b = b0 - (b0 - b1) * tf.cast(t, tf.float32) / nf                          # contraction-expansion, annealed
            mb = tf.reduce_mean(pb, 1, keepdims=True)
            phi = tf.random.uniform(sh); u = tf.random.uniform(sh, 1e-7, 1.0)
            pj = phi * pb + (1 - phi) * gb[:, None]
            sgn = tf.where(tf.random.uniform(sh) < 0.5, -1.0, 1.0)
            xi = tf.clip_by_value(pj + sgn * b * W * tf.abs(mb - xi) * tf.math.log(1.0 / u), -eps, eps) * M
            f = F(xi); better = f > pf
            pb = tf.where(better[:, :, None, None], xi, pb); pf = tf.maximum(pf, f)
            gb = tf.gather(pb, tf.argmax(pf, 1), batch_dims=1)
        return tf.broadcast_to(gb, [n, T, d])

    def qpso(X, true, prev, eps, w, mask, n_p=20, n_it=15, kind="se", beta=(1.0, 0.3)):
        d = X.shape[-1]
        W = tf.constant(((np.ones(d) if w is None else w) * mask).reshape(1, 1, 1, d), tf.float32)
        M = tf.constant(mask.reshape(1, 1, 1, d), tf.float32)
        out = core(tf.constant(X), tf.constant(true), tf.constant(prev), W, M, tf.constant(eps, tf.float32),
                   tf.constant(n_p), tf.constant(n_it), tf.constant(kind == "se"),
                   tf.constant(beta[0], tf.float32), tf.constant(beta[1], tf.float32)).numpy()
        assert np.abs(out).max() <= eps + 1e-6 and np.abs(out * (1 - mask)).max() == 0.0
        return out
    return qpso


def make_follower(model, kind):
    """In-loop attacker best response: 'pgd' (Alg. 3, unchanged), 'qpso', or 'dual' (better of the two per window,
    scored by the attacker's utility, as Study B's leader plays the better of ALS and QPSO)."""
    pgd = make_pgd(model)
    if kind == "pgd":
        return lambda X, true, prev, eps, w, mask: pgd(X, true, prev, eps, w, mask, steps=10)
    qp = make_qpso(model)
    if kind == "qpso":
        return lambda X, true, prev, eps, w, mask: qp(X, true, prev, eps, w, mask)

    def dual(X, true, prev, eps, w, mask):
        dp, dq = pgd(X, true, prev, eps, w, mask, steps=10), qp(X, true, prev, eps, w, mask)
        sp, sq = [(predict(model, X + z, prev) - true) ** 2 for z in (dp, dq)]
        return np.where((sq > sp)[:, None, None], dq, dp)
    return dual


def predict(model, X, prev):
    return hybrid(model, tf.constant(X), tf.constant(prev)).numpy()


def degraded(e0, ea, tau):
    return np.abs(ea) - np.abs(e0) > tau


# =============================================================================================
# 4. Game loop (Alg. 4) and fixed-budget comparison
# =============================================================================================
def game(theta0, D, seed, rounds=30, patience=6, controller=True, fixed=None, batch=16, tag="", verbose=True,
         follower="pgd"):
    fixed = (EPS_MAX, 0.5) if fixed is None else fixed
    tf.keras.utils.set_random_seed(seed)
    m = cell13_model(D["window"], D["n_feats"]); m.set_weights(theta0.get_weights())
    pgd = make_pgd(m)
    att = make_follower(m, follower) if follower != "pgd" else (lambda X, t, p, e, w, mk: pgd(X, t, p, e, w, mk, steps=10))
    ctrl = Controller(D["n_feats"])
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0)
    tr, va, tau, mask = D["fit"], D["val"], D["sd_delta"], D["market"]

    @tf.function(reduce_retracing=True)
    def train_step(x, xa, lt, dt, lam):
        with tf.GradientTape() as t:
            loss = (1 - lam) * th_loss(m, x, lt, dt, True) + lam * th_loss(m, xa, lt, dt, True)
        opt.apply_gradients(zip(t.gradient(loss, m.trainable_variables), m.trainable_variables))
        return loss

    def diag_val(eps, w, stale):
        f0 = predict(m, va["X"], va["prev"]); e0 = f0 - va["true"]
        d = att(va["X"], va["true"], va["prev"], eps, w, mask)
        fa = predict(m, va["X"] + d, va["prev"]); ea = fa - va["true"]
        fresh = 100 * degraded(e0, ea, tau).mean()
        st = 100 * degraded(e0, predict(m, va["X"] + stale, va["prev"]) - va["true"], tau).mean() if stale is not None else float("nan")
        ds = pgd(va["X"], va["true"], va["prev"], eps, None, mask, steps=50, restarts=2)
        strong = 100 * degraded(e0, predict(m, va["X"] + ds, va["prev"]) - va["true"], tau).mean()
        return fresh, st, strong, d, float(np.mean(e0 ** 2)), float(np.mean(ea ** 2)), float(np.sqrt(np.mean(e0 ** 2)))

    def train_payoff(eps, w):
        f0 = predict(m, tr["X"], tr["prev"])
        d = att(tr["X"], tr["true"], tr["prev"], eps, w, mask)
        return 100 * degraded(f0 - tr["true"], predict(m, tr["X"] + d, tr["prev"]) - tr["true"], tau).mean()

    log, s, zeta1, stale = [], 0.0, None, None
    e0_, l0_, w0_, _ = ctrl.step(0.0) if controller else (fixed[0], fixed[1], None, {})
    fr, st, sg, _, _, _, vr = diag_val(e0_, w0_, None)
    log.append(dict(round=0, s=0.0, eps=e0_, lam=l0_, train=train_payoff(e0_, w0_), fresh=fr, stale=st, strong=sg, val_rmse=vr))
    if verbose: print(f"{tag} r00 theta0  eps={e0_:.4f} train={log[0]['train']:5.1f} fresh={fr:5.1f} strong={sg:5.1f} valRMSE={vr:.3f}", flush=True)
    best, best_w, best_r, bad = np.inf, m.get_weights(), 0, 0
    n = len(tr["X"]); rng = np.random.default_rng(seed)
    for r in range(1, rounds + 1):
        t0 = time.time()
        eps, lam, w, info = ctrl.step(s) if controller else (fixed[0], fixed[1], None, {})
        opt.learning_rate.assign(1e-4 * 0.5 * (1 + math.cos(math.pi * (r - 1) / rounds)))   # cosine over 30 rounds
        n_hit = n_brk = 0
        for b in np.array_split(rng.permutation(n), math.ceil(n / batch)):
            x, lt, dt, tv, pv = tr["X"][b], tr["lvl"][b], tr["dlt"][b], tr["true"][b], tr["prev"][b]
            d = att(x, tv, pv, eps, w, mask)                                             # follower, Alg. 3 (or QPSO)
            f0, fa = predict(m, x, pv), predict(m, x + d, pv)
            n_brk += int(degraded(f0 - tv, fa - tv, tau).sum()); n_hit += len(b)
            train_step(tf.constant(x), tf.constant(x + d), tf.constant(lt), tf.constant(dt), tf.constant(lam, tf.float32))  # leader
        # validation: checkpoint criterion (robust RMSE at eps=0.5, base weights), feedback, diagnostics
        dref = att(va["X"], va["true"], va["prev"], 0.5, ctrl.base_w, mask)             # checkpoint: in-loop attack, eps 0.5
        r_rmse = float(np.sqrt(np.mean((predict(m, va["X"] + dref, va["prev"]) - va["true"]) ** 2)))
        fr, st, sg, dfresh, Lc, La, vr = diag_val(eps, w, stale)
        zeta = (La - Lc) / max(Lc, 1e-12)
        if r == 1: zeta1, s_next = zeta, 0.0
        else: s_next = float(np.clip(zeta / zeta1 - 1, -1, 1))                           # Eq. (20)
        log.append(dict(round=r, s=s, eps=eps, lam=lam, train=100 * n_brk / n_hit, fresh=fr, stale=st, strong=sg,
                        zeta=zeta, val_rmse=vr, val_rrmse=r_rmse, sec=time.time() - t0, **info))
        if verbose:
            print(f"{tag} r{r:02d} s={s:+.2f} eps={eps:.4f} lam={lam:.4f} train={log[-1]['train']:5.1f} fresh={fr:5.1f} "
                  f"stale={st:5.1f} strong={sg:5.1f} zeta={zeta:.3f} valRMSE={vr:.3f} rRMSE={r_rmse:.3f} ({time.time()-t0:.0f}s)", flush=True)
        s, stale = s_next, dfresh
        if r_rmse < best - 1e-9: best, best_w, best_r, bad = r_rmse, m.get_weights(), r, 0
        else:
            bad += 1
            if bad >= patience: break
    m.set_weights(best_w)
    return m, log, best_r


# =============================================================================================
# 5. Worst-case test-time response (Alg. 5) and statistics
# =============================================================================================
def evaluate(models, D, budgets=(0.5, 1.0), steps=100, restarts=5):
    te, tau, mask = D["te"], D["sd_delta"], D["market"]
    pg = {k: make_pgd(m) for k, m in models.items()}
    res, rows = {}, {}
    for name, m in models.items():
        f0 = predict(m, te["X"], te["prev"]); e0 = f0 - te["true"]
        nz = np.sign(te["true"] - te["prev"]) != 0
        c0 = (np.sign(f0 - te["prev"]) == np.sign(te["true"] - te["prev"])) & nz
        out = dict(MAE=float(np.abs(e0).mean()), RMSE=float(np.sqrt((e0 ** 2).mean())), DA=float(100 * c0.sum() / nz.sum()))
        rows[name] = dict(e0=e0, c0=c0, nz=nz)
        for eps in budgets:
            finals = []
            for kind in ("se", "margin"):                                        # white-box, 100 steps x 5 restarts
                d = pg[name](te["X"], te["true"], te["prev"], eps, None, mask, steps, restarts, kind)
                finals.append(predict(m, te["X"] + d, te["prev"]))
            for other in models:                                                 # transfer (margin-PGD on the others)
                if other != name:
                    d = pg[other](te["X"], te["true"], te["prev"], eps, None, mask, steps, 1, "margin")
                    finals.append(predict(m, te["X"] + d, te["prev"]))
            worst = np.max([np.abs(f - te["true"]) for f in finals], 0)
            brk = np.zeros(len(e0), bool); dflip = np.zeros(len(e0), bool)
            for f in finals:
                brk |= degraded(e0, f - te["true"], tau)
                dflip |= c0 & ~((np.sign(f - te["prev"]) == np.sign(te["true"] - te["prev"])) & nz)
            out[f"broken@{eps}"] = float(100 * brk.mean())
            out[f"robust@{eps}"] = float(100 * (~brk).mean())
            out[f"wcMAE@{eps}"] = float(worst.mean()); out[f"wcRMSE@{eps}"] = float(np.sqrt((worst ** 2).mean()))
            out[f"robDA@{eps}"] = float(100 * (c0 & ~dflip).sum() / nz.sum())
            rows[name][f"brk@{eps}"] = brk; rows[name][f"worst@{eps}"] = worst
        res[name] = out
    return res, rows


def bootstrap(ra, rb, budgets, n_boot=1000, block=10, seed=0):
    """Paired moving-block bootstrap (95%) of a minus b on the same test days."""
    rng = np.random.default_rng(seed); n = len(ra["e0"]); out = {}
    for eps in budgets:
        for key, f in ((f"robust@{eps}", lambda r, i, e=eps: 100 * (~r[f"brk@{e}"][i]).mean()),
                       (f"wcRMSE@{eps}", lambda r, i, e=eps: np.sqrt((r[f"worst@{e}"][i] ** 2).mean()))):
            pt = f(ra, np.arange(n)) - f(rb, np.arange(n)); bs = []
            for _ in range(n_boot):
                i = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, math.ceil(n / block))])[:n]
                bs.append(f(ra, i) - f(rb, i))
            lo, hi = np.percentile(bs, [2.5, 97.5]); out[key] = dict(diff=float(pt), lo=float(lo), hi=float(hi))
    return out


def run(D, tag, seeds, args):
    print(f"\n=== {tag}: rows {D['n_rows']}, split {D['split']}, fit {len(D['fit']['X'])} val {len(D['val']['X'])} "
          f"test {len(D['te']['X'])} ({D['test_span'][0]}..{D['test_span'][1]}), features {D['n_feats']} "
          f"({int(D['market'].sum())} market), tau = {D['sd_delta']:.4f}", flush=True)
    base = cell13_train(D, seed=0)
    out = []
    for sd in seeds:
        had, hlog, hbr = game(base, D, sd, rounds=args.rounds, controller=True, tag=f"{tag}/s{sd} HAD", verbose=args.verbose)
        fb, flog, fbr = game(base, D, sd, rounds=args.rounds, controller=False, tag=f"{tag}/s{sd} FB-AT", verbose=args.verbose)
        models = {"Two-Head LSTM (undefended)": base, FB_LABEL: fb, "HAD-Stack": had}
        res, rows = evaluate(models, D, steps=args.eval_steps, restarts=args.eval_restarts)
        ci = bootstrap(rows["HAD-Stack"], rows["Two-Head LSTM (undefended)"], (0.5, 1.0))
        out.append(dict(split=tag, seed=sd, order=HCS_ORDER, eps_max=EPS_MAX, had_best_round=hbr, fb_best_round=fbr, metrics=res, ci=ci, had_log=hlog, fb_log=flog))
        for k, v in res.items():
            print(f"{tag}/s{sd} {k:28s} MAE {v['MAE']:.3f} RMSE {v['RMSE']:.3f} DA {v['DA']:.1f} | robust@0.5 {v['robust@0.5']:.1f} "
                  f"robust@1.0 {v['robust@1.0']:.1f} | wcRMSE {v['wcRMSE@0.5']:.3f}/{v['wcRMSE@1.0']:.3f}", flush=True)
        print(f"{tag}/s{sd} HAD-Stack minus undefended: " + "; ".join(f"{k} {c['diff']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}]" for k, c in ci.items()), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seeds", "folds", "all"], default="all")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-restarts", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args()
    check_table_vi()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    allout = []
    if args.mode in ("seeds", "all"):
        allout += run(cell13_data(), "fixed", seeds, args)                      # cell 13's own 80/20 split
    if args.mode in ("folds", "all"):
        N = cell13_data()["n_rows"]
        for k in range(5):                                                        # walk-forward: test on 10% blocks of the last half
            D = cell13_data(split_at=int((0.5 + 0.1 * k) * N))
            nte = int(0.1 * N) + D["window"]
            D["te"] = {kk: v[:nte - D["window"]] for kk, v in D["te"].items()}
            D["test_span"] = (str(D["te"]["dates"][0])[:10], str(D["te"]["dates"][-1])[:10])
            allout += run(D, f"fold{k+1}", seeds[:1], args)
        json.dump(allout, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    json.dump(allout, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
