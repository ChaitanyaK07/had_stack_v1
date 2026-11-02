"""Re-implementation of the six models of the bond paper (Sec. 3) under its stated protocol:
univariate 10Y series, look-back L=20, strictly held-out last 250 observations, the rest split
80/20 into train/validation (chronological), one-step-ahead forecasts, MAE/RMSE/DA on the test
block.  Keras/TF as in the original code.  Adds the naive persistence forecast y_hat_{t+1}=y_t.
Usage: python paper_protocol.py <csv> [seeds]"""
import sys, json, os, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "india_bond_yields_2010_2025.csv")
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
L, N_TEST = 20, 250

df = pd.read_csv(CSV); df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date")
df["10Y"] = pd.to_numeric(df["10Y"], errors="coerce"); df = df.dropna(subset=["10Y"]).reset_index(drop=True)
y = df["10Y"].values.astype(float); T = len(y)
idx = np.arange(L, T)                                     # target index t, inputs y[t-L:t]
Xlev = np.stack([y[t - L:t] for t in idx]); tgt = y[idx]; prev = y[idx - 1]
dy = np.diff(y, prepend=np.nan)
Xd = np.stack([dy[t - L:t] for t in idx]) if L < T else None   # first window contains nan -> drop
ok = ~np.isnan(Xd).any(1)
Xlev, Xd, tgt, prev, idx = Xlev[ok], Xd[ok], tgt[ok], prev[ok], idx[ok]
N = len(tgt); te = np.arange(N - N_TEST, N); n_trv = N - N_TEST
n_tr = int(0.8 * n_trv); tr = np.arange(0, n_tr); va = np.arange(n_tr, n_trv)
mu_l, sd_l = y[:idx[n_trv - 1]].mean(), y[:idx[n_trv - 1]].std()
sd_d = np.nanstd(dy[1:idx[n_tr - 1]])
print(f"rows {T} ({df.Date.iloc[0].date()}..{df.Date.iloc[-1].date()}), samples {N}: train {len(tr)} val {len(va)} test {len(te)}; "
      f"test {df.Date.iloc[idx[te[0]]].date()}..{df.Date.iloc[idx[te[-1]]].date()}; level range {y.min():.2f}..{y.max():.2f}")


def metrics(pred, s=te):
    t, p, q = tgt[s], pred, prev[s]
    nz = np.sign(t - q) != 0
    return dict(MAE=float(np.mean(np.abs(t - p))), RMSE=float(np.sqrt(np.mean((t - p) ** 2))),
                DA=float(100 * np.mean(np.sign(p - q)[nz] == np.sign(t - q)[nz])))


def es(): return [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


def lstm_reg(inp_dim, out_names, units=50):
    i = keras.Input((L, inp_dim)); h = keras.layers.LSTM(units)(i)
    outs = [keras.layers.Dense(1, name=n)(h) for n in out_names]
    return keras.Model(i, outs if len(outs) > 1 else outs[0])


res = {}
# 1) AR(1) on levels: y_t = c + theta*y_{t-1}
A = np.c_[np.ones(n_trv), prev[:n_trv]]; c, th = np.linalg.lstsq(A, tgt[:n_trv], rcond=None)[0]
res["Baseline AR(1)"] = [metrics(c + th * prev[te])]
res["Naive persistence"] = [metrics(prev[te])]
for seed in SEEDS:
    np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    # 2) GBM on 20 lagged levels, grid search on validation
    best = None
    for lr in (0.03, 0.1):
        for dpt in (2, 3, 4):
            for ne in (100, 300):
                m = GradientBoostingRegressor(learning_rate=lr, max_depth=dpt, n_estimators=ne, random_state=seed).fit(Xlev[tr], tgt[tr])
                v = np.mean((m.predict(Xlev[va]) - tgt[va]) ** 2)
                if best is None or v < best[0]: best = (v, dict(learning_rate=lr, max_depth=dpt, n_estimators=ne))
    gbm = GradientBoostingRegressor(random_state=seed, **best[1]).fit(Xlev[:n_trv], tgt[:n_trv])
    res.setdefault("GBM", []).append(metrics(gbm.predict(Xlev[te])))
    # 3) Ensemble (GBM + ExtraTrees) + residual MLP (2 hidden ReLU layers) on the ensemble's residuals
    et = ExtraTreesRegressor(n_estimators=300, random_state=seed).fit(Xlev[:n_trv], tgt[:n_trv])
    g_tr = GradientBoostingRegressor(random_state=seed, **best[1]).fit(Xlev[tr], tgt[tr])
    e_tr = ExtraTreesRegressor(n_estimators=300, random_state=seed).fit(Xlev[tr], tgt[tr])
    ens_va = 0.5 * (g_tr.predict(Xlev[va]) + e_tr.predict(Xlev[va]))          # out-of-sample residuals for the MLP
    Zs = lambda X: (X - mu_l) / sd_l
    mlp = keras.Sequential([keras.Input((L,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(32, activation="relu"), keras.layers.Dense(1)])
    mlp.compile(keras.optimizers.Adam(1e-3), "mse")
    mlp.fit(Zs(Xlev[va]), (tgt[va] - ens_va) / sd_d, epochs=200, batch_size=32, validation_split=0.2, callbacks=es(), verbose=0)
    ens_te = 0.5 * (gbm.predict(Xlev[te]) + et.predict(Xlev[te]))
    res.setdefault("Ensemble + Residual MLP", []).append(metrics(ens_te + sd_d * mlp.predict(Zs(Xlev[te]), verbose=0).ravel()))
    # 4) ARIMA(1,0,0) on Delta + LSTM on its residuals; integrate one step
    dt = tgt - prev
    B = np.c_[np.ones(n_tr), Xd[tr, -1]]; c1, ph = np.linalg.lstsq(B, dt[tr], rcond=None)[0]
    ar_d = c1 + ph * Xd[:, -1]; resid = dt - ar_d
    m4 = lstm_reg(1, ["r"]); m4.compile(keras.optimizers.Adam(1e-3), "mse")
    m4.fit(Xd[tr, :, None] / sd_d, resid[tr] / sd_d, validation_data=(Xd[va, :, None] / sd_d, resid[va] / sd_d),
           epochs=200, batch_size=32, callbacks=es(), verbose=0)
    res.setdefault("Hybrid ARIMA-LSTM (on Delta)", []).append(metrics(prev[te] + ar_d[te] + sd_d * m4.predict(Xd[te, :, None] / sd_d, verbose=0).ravel()))
    # 5) LSTM on Delta-yields with Delta-lag features
    m5 = lstm_reg(1, ["d"]); m5.compile(keras.optimizers.Adam(1e-3), "mse")
    m5.fit(Xd[tr, :, None] / sd_d, dt[tr] / sd_d, validation_data=(Xd[va, :, None] / sd_d, dt[va] / sd_d),
           epochs=200, batch_size=32, callbacks=es(), verbose=0)
    res.setdefault("LSTM on Delta-yields", []).append(metrics(prev[te] + sd_d * m5.predict(Xd[te, :, None] / sd_d, verbose=0).ravel()))
    # 6) Two-Head LSTM (level + Delta), alpha=0.6, beta=0.7 (Eqs. 4-5)
    m6 = lstm_reg(1, ["level", "delta"]); m6.compile(keras.optimizers.Adam(1e-3), loss={"level": "mse", "delta": "mse"}, loss_weights={"level": 0.6, "delta": 0.4})
    Xl = lambda s: ((Xlev[s] - mu_l) / sd_l)[:, :, None]
    m6.fit(Xl(tr), {"level": (tgt[tr] - mu_l) / sd_l, "delta": dt[tr] / sd_d},
           validation_data=(Xl(va), {"level": (tgt[va] - mu_l) / sd_l, "delta": dt[va] / sd_d}),
           epochs=200, batch_size=32, callbacks=es(), verbose=0)
    lv, dd = m6.predict(Xl(te), verbose=0)
    res.setdefault("Two-Head LSTM (Hybrid)", []).append(metrics(0.7 * (lv.ravel() * sd_l + mu_l) + 0.3 * (prev[te] + dd.ravel() * sd_d)))
    print(f"seed {seed} done", flush=True)

PAPER = {"Baseline AR(1)": (0.45, 0.58, 62.4), "GBM": (0.31, 0.41, 74.8), "Ensemble + Residual MLP": (0.22, 0.29, 85.6),
         "Hybrid ARIMA-LSTM (on Delta)": (0.35, 0.47, 70.2), "LSTM on Delta-yields": (0.28, 0.39, 79.1),
         "Two-Head LSTM (Hybrid)": (0.18, 0.24, 89.5), "Naive persistence": (None, None, None)}
print(f"\n| Model | MAE | RMSE | DA (%) | Paper MAE | Paper RMSE | Paper DA |\n|---|---|---|---|---|---|---|")
out = {}
for k, v in res.items():
    a = {m: (np.mean([r[m] for r in v]), np.std([r[m] for r in v])) for m in ("MAE", "RMSE", "DA")}
    out[k] = a; p = PAPER[k]
    print(f"| {k} | {a['MAE'][0]:.3f}±{a['MAE'][1]:.3f} | {a['RMSE'][0]:.3f}±{a['RMSE'][1]:.3f} | {a['DA'][0]:.1f}±{a['DA'][1]:.1f} | "
          + " | ".join("–" if x is None else str(x) for x in p) + " |")
json.dump(out, open(f"paper_protocol_{os.path.basename(CSV)}.json", "w"), indent=1)
