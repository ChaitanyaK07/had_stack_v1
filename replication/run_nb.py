"""Run the bond paper's notebook (data/India_Bonds_DL_MLP.ipynb) cell by cell, as given, and score every
forecasting model with MAE, RMSE and directional accuracy, plus the no-change forecast on the same test targets.
Cell 7 (a 243-model grid search that only reports a configuration) is skipped unless --grid.
  python run_nb.py [--save-plots] [--grid]      -> nb_scores.json (+ plots/ with --save-plots)"""
import json, os, sys, time, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "nb_scores.json"); PLOTS = os.path.join(HERE, "plots")
SAVE = "--save-plots" in sys.argv; SKIP = set() if "--grid" in sys.argv else {7}
CUR, NPLOT = [0], [0]
def _show(*a, **k):
    if SAVE:
        os.makedirs(PLOTS, exist_ok=True); NPLOT[0] += 1
        plt.savefig(os.path.join(PLOTS, f"cell{CUR[0]:02d}_{NPLOT[0]}.png"), dpi=80)
    plt.close("all")
plt.show = _show
nb = json.load(open(os.path.join(DATA, "India_Bonds_DL_MLP.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
os.chdir(DATA)        # the notebook reads its CSV files by bare name

def score(name, true, pred, prev):
    true, pred, prev = (np.asarray(a, float).ravel() for a in (true, pred, prev))
    n = min(len(true), len(pred), len(prev)); true, pred, prev = true[-n:], pred[-n:], prev[-n:]
    nz = np.sign(true - prev) != 0
    r = dict(model=name, n=n, MAE=float(np.mean(np.abs(true - pred))), RMSE=float(np.sqrt(np.mean((true - pred) ** 2))),
             DA=float(100 * np.mean(np.sign(pred - prev)[nz] == np.sign(true - prev)[nz])),
             DA_seq=float(100 * np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(true)))),
             naive_MAE=float(np.mean(np.abs(true - prev))), naive_RMSE=float(np.sqrt(np.mean((true - prev) ** 2))))
    print("SCORE", json.dumps(r), flush=True); return r

EXTRACT = {
 5:  ("LSTM on 3 PCs (cell 5)",                   lambda g: (g["y_test"], g["y_pred"], g["y_seq"][g["split"]-1:-1])),
 6:  ("LSTM+dropout on 3 PCs (cell 6)",           lambda g: (g["y_test_orig"], g["y_pred"], g["y_seq"][g["split"]-1:-1])),
 8:  ("Final LSTM (cell 8)",                      lambda g: (g["y_te_orig"], g["y_pred"], g["ys_orig"][int(.8*len(g["ys_orig"]))-1:-1])),
 10: ("Ensemble LSTMs + residual MLP (cell 10)",  lambda g: (g["y_te_o"], g["y_pred_final"], g["ys_o"][int(.8*len(g["ys_o"]))-1:-1])),
 11: ("Hybrid AR(1)+LSTM on Delta (cell 11)",     lambda g: (g["y_level_true"], g["y_level_pred"], g["df"]["yield"].values[g["split"]+g["pad"]-1:-1])),
 12: ("LSTM on Delta-yields (cell 12)",           lambda g: (g["y_true_level"], g["y_pred_level"], g["df"]["yield"].values[g["split"]+g["window"]-1:-1])),
 13: ("Two-Head LSTM hybrid (cell 13)",           lambda g: (g["true_levels"], g["hybrid_pred"], g["prev_levels"])),
 14: ("Residual LSTM on Delta (cell 14)",         lambda g: (g["true_levels"], g["level_pred"], g["prev_levels"])),
}
g, results = {"__name__": "__nb__"}, []
for i, src in enumerate(cells):
    if not src.strip() or i in SKIP:
        print(f"--- cell {i}: skipped"); continue
    CUR[0], NPLOT[0], t0 = i, 0, time.time()
    try:
        exec(compile(src, f"cell{i}", "exec"), g); print(f"--- cell {i}: ok ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"--- cell {i}: ERROR {type(e).__name__}: {e}", flush=True); continue
    if i in EXTRACT:
        try: results.append(score(EXTRACT[i][0], *EXTRACT[i][1](g)))
        except Exception as e: print(f"score error cell {i}: {e}")
json.dump(results, open(OUT, "w"), indent=1)
