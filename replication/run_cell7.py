import json, numpy as np, matplotlib, time
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: plt.close("all")
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
nb = json.load(open(os.path.join(DATA, "India_Bonds_DL_MLP.ipynb")))
os.chdir(DATA); cells = ["".join(c["source"]) for c in nb["cells"]]
g = {}
for i in (0, 6, 7):            # cell 7 uses `data` defined in cell 6, as when run in order
    t = time.time(); exec(compile(cells[i], f"cell{i}", "exec"), g); print(f"cell {i} ok {time.time()-t:.0f}s", flush=True)
yt, yp = g["y_te_orig"], g["y_pred"]; prev = g["ys_orig"][int(.8*len(g["ys_orig"]))-1:-1][-len(yt):]
nz = np.sign(yt-prev) != 0
print("BEST", g["best"]); print(f"retrained best: MAE {np.mean(np.abs(yt-yp)):.3f} RMSE {np.sqrt(np.mean((yt-yp)**2)):.3f} "
      f"DA {100*np.mean(np.sign(yp-prev)[nz]==np.sign(yt-prev)[nz]):.1f}%  naive MAE {np.mean(np.abs(yt-prev)):.3f}", flush=True)
