"""HAD-Stack verification on the ULB credit-card data, on GPU or CPU.

  python run_gpu.py --track scratch    --unit seed1     # seed1..seed3 (fixed split) or fold0..fold4
  python run_gpu.py --track pretrained --unit seed1     # seed1..seed3 (fixed split only)
  options: --device auto|cuda|cpu   --rounds 30   --smoke (tiny subset, for testing only)

Per unit it (1) obtains the starting detector theta_0 -- a FraudTransformer baseline trained as in
baseline_model.train_baseline (track A) or the pre-trained MLP (track B); (2) trains HAD-Stack (Algorithm 4)
and a fixed-budget detector (eps 0.6586, lambda 0.5) used only as an extra transfer source, both from theta_0,
exactly as had_stack_v2.train (AdamW 1e-4, wd 1e-4, cosine over 30 rounds, robust-AUC selection at eps 0.5,
patience 6); (3) logs the loop diagnostics of Definition 8 every round; (4) evaluates every detector with
the worst-case ensemble of run_confirmation.worst_case (Algorithm 5). Outputs go to results/.
The attack runs on fraud rows only, which gives exactly the same perturbation as the released code
(verified in tests/test_equivalence.py).
"""
import os, sys, json, csv, time, argparse
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
# released HAD-Stack modules: a local hadstack/ copy if present, otherwise the repository root
sys.path.insert(0, os.path.join(HERE, "hadstack") if os.path.isdir(os.path.join(HERE, "hadstack")) else os.path.dirname(HERE))
import baseline_model as bm, adversarial_game as ag, had_stack_v2 as hs
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# ----------------------------------------------------------------------------------- device
def set_device(name):
    dev = torch.device("cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name))
    bm.DEVICE = dev; ag.DEVICE = dev        # the released modules read DEVICE at call time
    return dev

# ----------------------------------------------------------------------------------- attacker (Algorithm 3)
def fast_evasion_attack(self, model, x, y, eps=None):
    """Identical to adversarial_game.HamiltonianAttacker.evasion_attack, run on fraud rows only."""
    eps = self.eps if eps is None else eps
    idx = (y == 1).nonzero(as_tuple=True)[0]
    if len(idx) == 0: return x.detach().clone()
    was = model.training; model.eval()
    xf, yf = x[idx], y[idx]
    w = self.budget.get_feature_weights(x.shape[1]).to(x.device)
    alpha = self.step_mult * eps / self.n_steps
    eta = torch.empty_like(xf).uniform_(-eps, eps)
    for _ in range(self.n_steps):
        eta.requires_grad_(True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(xf + eta), yf.float(), reduction="none").sum()
        g, = torch.autograd.grad(loss, eta)
        with torch.no_grad(): eta = torch.clamp(eta + alpha * w.unsqueeze(0) * g.sign(), -eps, eps)
    if was: model.train()
    out = x.detach().clone(); out[idx] = xf + eta.detach(); return out
ag.HamiltonianAttacker.evasion_attack = fast_evasion_attack

def train_epoch_diag(self, loader):
    """StackelbergAdversarialTrainer.train_epoch + follower payoff (share of detected training fraud flipped)."""
    total, det, flip = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(bm.DEVICE), yb.to(bm.DEVICE)
        xa = self.attacker.evasion_attack(self.model, xb, yb)
        f = yb == 1
        if f.any():
            self.model.eval()
            with torch.no_grad(): zc, za = self.model(xb[f]), self.model(xa[f])
            d = zc >= 0; det += int(d.sum()); flip += int((d & (za < 0)).sum())
        total += self._defender_step(xb, xa, yb) * len(yb)
    self.scheduler.step(); self.diag_train_flip = 100 * flip / max(det, 1)
    return total / len(loader.dataset)
ag.StackelbergAdversarialTrainer.train_epoch = train_epoch_diag

def robust_auc(model, loader, attacker, eps):
    """had_stack_v2.robust_auc, device-aware."""
    ps, ys = [], []
    for xb, yb in loader:
        xb, yb = xb.to(bm.DEVICE), yb.to(bm.DEVICE)
        xa = attacker.evasion_attack(model, xb, yb, eps=eps)
        with torch.no_grad(): ps.append(torch.sigmoid(model(xa)).cpu().numpy()); ys.append(yb.cpu().numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

# ----------------------------------------------------------------------------------- models
def make_transformer(nf):
    return bm.TabularTransformer(nf, d_model=bm.HP["d_model"], n_heads=bm.HP["n_heads"], n_layers=bm.HP["n_layers"],
                                 d_ffn=bm.HP["d_ffn"], dropout=bm.HP["dropout"])

def make_model(track, nf):
    if track == "scratch": return make_transformer(nf)
    sys.path.insert(0, os.path.join(HERE, "pretrained")); from pretrained_model import PretrainedMLP
    return PretrainedMLP(nf)

def loaders(d, smoke):
    def ld(n):
        X = np.load(f"{d}/X_{n}.npy").astype(np.float32); y = np.load(f"{d}/y_{n}.npy").astype(np.int64)
        if smoke:                                   # keep all fraud, subsample legitimate rows (testing only)
            rng = np.random.RandomState(0); keep = (y == 1) | (rng.rand(len(y)) < 0.03); X, y = X[keep], y[keep]
        return X, y
    S = {n: ld(n) for n in ("train", "val", "test")}
    mk = lambda n, sh: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(S[n][0]), torch.tensor(S[n][1])),
                                                   batch_size=bm.HP["batch_size"], shuffle=sh, num_workers=0)
    return mk("train", True), mk("val", False), mk("test", False), S

def train_baseline(nf, tr, va, path):
    """baseline_model.train_baseline (seed 42, AdamW 3e-4, wd 1e-2, cosine T_max 60, focal, val-AUC early stop)."""
    bm.set_seed(42); m = make_transformer(nf).to(bm.DEVICE)
    crit = bm.FocalLoss(gamma=bm.HP["focal_gamma"], alpha=bm.HP["focal_alpha"])
    opt = torch.optim.AdamW(m.parameters(), lr=bm.HP["lr"], weight_decay=bm.HP["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bm.HP["epochs"], eta_min=1e-5)
    best, wait = 0.0, 0
    for ep in range(1, bm.HP["epochs"] + 1):
        bm.train_epoch(m, tr, opt, crit); vm = bm.evaluate(m, va); sch.step()
        print(f"[baseline] epoch {ep:2d} val AUC {vm['auc']:.4f}", flush=True)
        if vm["auc"] > best: best, wait = vm["auc"], 0; torch.save(m.state_dict(), path)
        else:
            wait += 1
            if wait >= bm.HP["patience"]: break
    return path

# ----------------------------------------------------------------------------------- HAD-Stack loop (Algorithm 4)
def train_adv(method, seed, track, nf, tr, va, S, base_ckpt, out, rounds):
    bm.set_seed(seed)
    budget, adapt = {"v3": (hs.HamiltonianBudgetV3(), True), "fbat05": (hs.FixedBudget(hs.EPS_MAX, 0.5), False)}[method]
    model = make_model(track, nf).to(bm.DEVICE); model.load_state_dict(torch.load(base_ckpt, map_location=bm.DEVICE))
    T = ag.StackelbergAdversarialTrainer(model, budget, adapt=adapt, epochs=rounds)
    sel = ag.HamiltonianAttacker(hs.FixedBudget(hs.SEL_EPS, 0.5))
    Xv, yv = S["val"]; Xvf = torch.tensor(Xv[yv == 1]).to(bm.DEVICE); yvf = torch.ones(len(Xvf), dtype=torch.long, device=bm.DEVICE)
    def flip_rate(x0, xa):
        model.eval()
        with torch.no_grad(): d = model(x0) >= 0; return 100 * float((d & (model(xa) < 0)).sum()) / max(int(d.sum()), 1)
    ck = f"{out}/{method}.pt"; best, wait, sig, ref, traj, stale = -1.0, 0, None, None, [], None
    for r in range(1, rounds + 1):
        t0 = time.time()
        budget.resolve(sig if adapt else None); T.attacker.refresh(); T.adv_weight = budget.derived_lambda()
        loss = T.train_epoch(tr)
        torch.manual_seed(seed + r)
        rauc = robust_auc(model, va, sel, hs.SEL_EPS)
        sig, gap = ag.robustness_gap_signal(model, T.attacker, va, ref)
        if ref is None: ref = gap
        rng = torch.get_rng_state(); torch.manual_seed(10_000 + r)
        fresh_x = T.attacker.evasion_attack(model, Xvf, yvf); fresh = flip_rate(Xvf, fresh_x)
        stale_r = flip_rate(Xvf, stale) if stale is not None else float("nan")
        sa = ag.HamiltonianAttacker(ag.FixedPerturbationBudget(T.attacker.eps, 0.5), n_steps=50)
        model.eval()
        with torch.no_grad(): det = model(Xvf) >= 0
        flipped = torch.zeros(len(Xvf), dtype=torch.bool, device=bm.DEVICE)
        for rs in range(2):
            torch.manual_seed(20_000 + 10 * r + rs); xa = sa.evasion_attack(model, Xvf, yvf)
            with torch.no_grad(): flipped |= (model(xa) < 0)
        strong = 100 * float((det & flipped).sum()) / max(int(det.sum()), 1)
        torch.set_rng_state(rng); stale = fresh_x
        row = dict(round=r, eps=budget.eps_global, lam=T.adv_weight, field=getattr(budget, "field", 0.0), robust_val_auc=rauc,
                   rel_gap=gap, next_signal=sig, train_flip=T.diag_train_flip, fresh=fresh, stale=stale_r, strong=strong,
                   loss=loss, seconds=time.time() - t0)
        traj.append(row)
        print(f"[{method}] r{r:2d} eps {row['eps']:.4f} lam {row['lam']:.3f} rAUC {rauc:.4f} s {sig:+.3f} "
              f"train {row['train_flip']:.1f} fresh {fresh:.1f} stale {stale_r:.1f} strong {strong:.1f} ({row['seconds']:.0f}s)", flush=True)
        if rauc > best: best, wait = rauc, 0; torch.save(model.state_dict(), ck)
        else:
            wait += 1
            if wait >= 6: break
    with open(f"{out}/traj_{method}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=traj[0].keys()); w.writeheader(); w.writerows(traj)
    return ck

# ----------------------------------------------------------------------------------- worst case (Algorithm 5)
def pgd(m, x, eps, loss, K, R, seed=0):
    """run_confirmation.pgd, device-aware; random starts drawn on the CPU so they match the released code."""
    m.eval(); a = 2.5 * eps / K; best = x.clone(); bz = m(x).detach()
    for r in range(R):
        g = torch.Generator().manual_seed(seed + r)
        eta = ((torch.rand(x.shape, generator=g) * 2 - 1) * eps).to(x.device)
        for _ in range(K):
            eta.requires_grad_(True); z = m(x + eta)
            L = (torch.nn.functional.softplus(-z) if loss == "bce" else -z).sum()
            gr, = torch.autograd.grad(L, eta)
            with torch.no_grad(): eta = torch.clamp(eta + a * gr.sign(), -eps, eps)
        with torch.no_grad():
            z = m(x + eta); better = z < bz; best[better] = (x + eta)[better]; bz = torch.minimum(bz, z)
    assert (best - x).abs().max() <= eps + 1e-5
    return best

def metrics(y, p):
    pred = (p >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
                fp=int(fp), fn=int(fn), tp=int(tp), cost=int(100 * fn + fp))

def worst_case(models, X, y, K=100, R=5, eps_list=(0.5, 1.0)):
    """run_confirmation.worst_case: white-box PGD (BCE, margin; K steps, R restarts) + margin-PGD transfer."""
    Xt = torch.tensor(X).to(bm.DEVICE); Xf = Xt[torch.tensor(y == 1)]; n_fraud = int((y == 1).sum())
    for m in models.values(): m.eval()
    crafted = {(k, e): pgd(m, Xf, e, "margin", K, 1) for k, m in models.items() for e in eps_list}
    rows = {}
    for k, m in models.items():
        with torch.no_grad(): pc = torch.cat([torch.sigmoid(m(Xt[i:i + 8192])) for i in range(0, len(Xt), 8192)]).cpu().numpy()
        row = metrics(y, pc); det = torch.tensor(pc[y == 1] >= 0.5, device=bm.DEVICE)
        for e in eps_list:
            adv = {l: pgd(m, Xf, e, l, K, R) for l in ("bce", "margin")}
            with torch.no_grad():
                broken = torch.zeros_like(det)
                for l in adv: broken |= (m(adv[l]) < 0) & det
                for s in models:
                    if s != k: broken |= (m(crafted[(s, e)]) < 0) & det
            row[f"broken@{e}"] = 100 * broken.sum().item() / max(det.sum().item(), 1)
            row[f"robust_recall@{e}"] = 100 * (det & ~broken).sum().item() / n_fraud
            row[f"robust_detected_rows@{e}"] = (det & ~broken).cpu().numpy().astype(int).tolist()   # for paired bootstrap
        rows[k] = row
    return rows

# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scratch", "pretrained"], required=True)
    ap.add_argument("--unit", required=True, help="seed1|seed2|seed3|fold0..fold4")
    ap.add_argument("--device", default="auto"); ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--data", default=os.path.join(HERE, "data")); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = set_device(a.device); torch.set_num_threads(max(1, os.cpu_count() // 2))
    if a.track == "pretrained" and a.unit.startswith("fold"): sys.exit("pretrained track uses seeds only (see prep.py)")
    split = "fixed" if a.unit.startswith("seed") else a.unit
    seed = int(a.unit[4:]) if a.unit.startswith("seed") else 42
    d = f"{a.data}/{a.track}_{split}"; tag = f"{a.track}_{a.unit}" + ("_SMOKE" if a.smoke else "")
    out = os.path.join(HERE, "results", tag); os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/results.json"): print("already done:", out); return
    tr, va, te, S = loaders(d, a.smoke); nf = S["train"][0].shape[1]
    print(f"[{tag}] device {dev}, features {nf}, rows train/val/test {len(S['train'][1])}/{len(S['val'][1])}/{len(S['test'][1])}, "
          f"fraud {int(S['train'][1].sum())}/{int(S['val'][1].sum())}/{int(S['test'][1].sum())}", flush=True)
    t0 = time.time()
    if a.track == "scratch":
        base_dir = os.path.join(HERE, "results", f"scratch_baseline_{split}" + ("_SMOKE" if a.smoke else "")); os.makedirs(base_dir, exist_ok=True)
        base = f"{base_dir}/baseline.pt"
        if not os.path.exists(base):
            if a.smoke: bm.HP["epochs"] = 2
            train_baseline(nf, tr, va, base)
    else:
        sys.path.insert(0, os.path.join(HERE, "pretrained")); from pretrained_model import load_pretrained
        base = f"{out}/pretrained.pt"; torch.save(load_pretrained().state_dict(), base)
    ck = {"baseline": base}
    for meth in ("v3", "fbat05"):
        ck[meth] = train_adv(meth, seed, a.track, nf, tr, va, S, base, out, 2 if a.smoke else a.rounds)
    models = {}
    for k, p in ck.items():
        m = make_model(a.track, nf).to(dev); m.load_state_dict(torch.load(p, map_location=dev)); models[k] = m.eval()
    K, R = (5, 1) if a.smoke else (100, 5)
    rows = worst_case(models, *S["test"], K=K, R=R)
    res = dict(track=a.track, unit=a.unit, device=str(dev), torch=torch.__version__, smoke=a.smoke, minutes=(time.time() - t0) / 60,
               n_test=len(S["test"][1]), n_test_fraud=int(S["test"][1].sum()), models=rows)
    json.dump(res, open(f"{out}/results.json", "w"))
    for k, r in rows.items():
        print(f"[{tag}] {k:8s} AUC {r['auc']:.4f} recall {100*r['recall']:.1f} FP {r['fp']} FN {r['fn']} cost {r['cost']} | "
              f"robust recall 0.5: {r['robust_recall@0.5']:.1f}%  1.0: {r['robust_recall@1.0']:.1f}%", flush=True)
    print(f"[{tag}] done in {res['minutes']:.1f} min -> {out}", flush=True)

if __name__ == "__main__":
    main()
