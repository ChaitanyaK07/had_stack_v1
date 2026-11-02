import numpy as np, json
import os
from hadstack_thlstm import *
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, '..', 'paper', 'figs')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
D = cell13_data(); base = cell13_train(D, 0)
had, _, hbr = game(base, D, 1, verbose=False)
fb, _, fbr = game(base, D, 1, controller=False, verbose=False)
models = {"Two-Head LSTM (undefended)": base, "FB-AT (eps 0.6586)": fb, "HAD-Stack": had}
res, rows = evaluate(models, D)
out = {"best": [hbr, fbr], "tau": D["sd_delta"], "metrics": res,
       "rows": {k: {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in r.items()} for k, r in rows.items()}}
# per-day error increase of the worst member = worst - |e0|
json.dump(out, open(os.path.join(RES, "tau_sens.json"), "w"))
for k, v in res.items(): print(k, round(v["broken@0.5"], 2), round(v["broken@1.0"], 2), round(v["wcRMSE@0.5"], 3))
