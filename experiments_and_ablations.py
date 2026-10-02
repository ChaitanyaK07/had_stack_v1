"""
ADVISOR-REQUESTED EXPERIMENTS (5-fold cross-validation)
=======================================================
Run in the SAME folder as the earlier runs (needs models/, models_cv/, logs/, fraud_dataset.csv and the code).
  python experiments_and_ablations.py --part train      # 10 training runs per fold (resumable)
  python experiments_and_ablations.py --part eval       # extended + constrained attacks on 13 models per fold (resumable)
  python experiments_and_ablations.py --part summary    # decision rules, bootstrap, LaTeX-ready json
  add --quick for a smoke test (fold 0, 1 round, weak attacks) -> separate files, NOT for reporting.

Every new method uses the same pipeline as HAD-Stack (had_stack_v2.train): same fold baseline, optimiser, focal loss,
30 rounds, robust-AUC checkpoint selection (PGD, eps 0.5, identical selector), patience 6, seed 42.

NEW METHODS
  fbat_floor   fixed eps = lambda = 0.6586, HAD-Stack's floored step weights (only difference to HAD-Stack: no adaptation)
  signal_lin   non-Hamiltonian controller: same validation signal s and same eps range as HAD-Stack, linear in s,
               lambda = eps, floored weights (tests whether the Hamiltonian mapping itself is needed)
  ramp_floor   eps rises linearly 0.3461 -> 0.6586 over 30 rounds, lambda = eps, floored weights (scheduled eps)
  grid_035, grid_050   fixed eps = lambda in {0.35, 0.50}, floored weights; with fbat_floor (0.6586) they form the
               grid; "grid_sel" = the one with the best robust validation AUC in each fold (grid-searched eps)
  pgdat_uniform  standard PGD-AT: uniform step weights, eps = 0.6586, lambda = 0.5
  trades_b1, trades_b6   TRADES: focal loss on clean data + beta * KL(p(x) || p(x')), x' maximises the KL
               (10 steps, uniform weights, eps = 0.6586). Same threat model as every other method: only fraud rows are
               perturbed (perturbing all rows at this budget collapses training). "trades_sel" = the beta in {1, 6}
               with the best robust validation AUC in each fold (tuned on validation only, like the eps grid).
  h2_only      eps from H2 (as HAD-Stack), lambda fixed 0.5
  h1_only      lambda from H1 (as HAD-Stack), eps fixed 0.5
  h29_only     eps = lambda = clip(0.5 * phi / phi_base, 0.1, 0.9), phi = H29 equilibrium field (couplings driven as HAD-Stack)

EVALUATION (every model of the fold, eps in {0.5, 1.0}; a DETECTED fraud row is broken if ANY attack flips it)
  standard ensemble (as in the paper): PGD-100 x 5 restarts (BCE, margin) + margin-PGD transfer from all other models
  extended: + APGD-CE and APGD-margin (100 iterations, AutoAttack step-size rule) + Square attack (1000 queries, no gradients)
  constrained (realistic): only the 14 raw features are perturbed (|delta| <= eps in standardised units), values are
               clipped to the training range, integer-valued features are rounded, and the 4 engineered features are
               recomputed from the perturbed raw values; white-box PGD-100 x 2 restarts (BCE, margin)

PRE-REGISTERED DECISION RULES (fixed before running)
  The Hamiltonian controller is credited against a control only if HAD-Stack's worst-case robust recall at eps = 1.0
  (extended ensemble) exceeds that control's in at least 4 of 5 folds. Applied to: fbat_floor, signal_lin, ramp_floor,
  grid_sel. Paired bootstrap CIs (pooled folds, B = 1000) are reported alongside but do not decide.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, csv, json, math, time
import numpy as np, torch
import baseline_model as bm, adversarial_game as ag, had_stack_v2 as hs, eval_extras as ex, run_confirmation as rc

EPS_EVAL = (0.5, 1.0)
EXISTING = ("baseline", "v3", "pgdat066", "pgdat066lam")
NEW = ("fbat_floor", "signal_lin", "ramp_floor", "grid_035", "grid_050", "pgdat_uniform", "trades_b1", "trades_b6",
       "h2_only", "h1_only", "h29_only")
EVAL_NEW = ("fbat_floor", "signal_lin", "ramp_floor", "grid_sel", "pgdat_uniform", "trades_sel", "h2_only", "h1_only", "h29_only")
CONTROLS = ("fbat_floor", "signal_lin", "ramp_floor", "grid_sel")
GRID, TRADES = ("grid_035", "grid_050", "fbat_floor"), ("trades_b1", "trades_b6")
Q = dict(epochs=30, K=100, R=5, KC=100, RC=2, APGD=100, SQ=1000, B=1000, folds=range(5), tag="")


def floor_weights():
    return hs.HamiltonianBudgetV3().get_feature_weights(18)          # HAD-Stack's weights at base parameters


# ---------------------------------------------------------------- budgets
class FixedFloor:
    def __init__(self, eps, lam): self.eps_global, self._lam, self.field, self._w = eps, lam, 0.0, floor_weights()
    def resolve(self, signal=None): return self
    def derived_lambda(self, *a, **k): return self._lam
    def get_feature_weights(self, n): return self._w

class SignalLinear(FixedFloor):
    """Same signal and same eps range as HAD-Stack (0.5 at s=0, 0.6586 at s=-1, 0.3461 at s=+1), linear in s."""
    def __init__(self): super().__init__(0.5, 0.5)
    def resolve(self, signal=None):
        s = 0.0 if signal is None else float(np.clip(signal, -1, 1))
        self.eps_global = 0.5 - (0.1586 if s <= 0 else 0.1539) * s; self._lam = self.eps_global; return self

class RampFloor(FixedFloor):
    def __init__(self, epochs): super().__init__(0.3461, 0.3461); self.r, self.E = 0, epochs
    def resolve(self, signal=None):
        self.eps_global = 0.3461 + (0.6586 - 0.3461) * self.r / max(self.E - 1, 1); self._lam = self.eps_global
        self.r += 1; return self

class H2Only(hs.HamiltonianBudgetV3):
    def derived_lambda(self, *a, **k): return 0.5

class H1Only(hs.HamiltonianBudgetV3):
    def resolve(self, signal=None):
        super().resolve(signal); self.eps_global = 0.5; return self

class H29Only(hs.HamiltonianBudgetV3):
    def resolve(self, signal=None):
        super().resolve(signal)
        self.eps_global = float(np.clip(0.5 * self.field / self._base["field"], 0.1, 0.9)); self.lam = self.eps_global
        return self


class TradesTrainer(ag.StackelbergAdversarialTrainer):
    BETA = 6.0
    def train_epoch(self, loader):
        total = 0.0; eps = self.attacker.eps; K = 10; a = 2.5 * eps / K
        for xb, yb in loader:
            self.model.eval()
            with torch.no_grad(): p_clean = torch.sigmoid(self.model(xb))
            fr = (yb == 1).float().unsqueeze(1)                     # threat model: only fraud rows are perturbed
            delta = 0.001 * torch.randn_like(xb) * fr
            for _ in range(K):
                delta.requires_grad_(True)
                p = torch.sigmoid(self.model(xb + delta)).clamp(1e-6, 1 - 1e-6)
                kl = (p_clean * torch.log(p_clean.clamp(1e-6) / p) + (1 - p_clean) * torch.log((1 - p_clean).clamp(1e-6) / (1 - p))).sum()
                g, = torch.autograd.grad(kl, delta)
                with torch.no_grad(): delta = torch.clamp(delta + a * g.sign(), -eps, eps) * fr
            xa = (xb + delta).detach()
            self.model.train(); self.optimizer.zero_grad()
            z = self.model(xb); pc = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
            pa = torch.sigmoid(self.model(xa)).clamp(1e-6, 1 - 1e-6)
            kl = ((pc * torch.log(pc / pa) + (1 - pc) * torch.log((1 - pc) / (1 - pa))) * fr.squeeze(1)).sum() / fr.sum().clamp(min=1)
            loss = self.criterion(z, yb) + self.BETA * kl
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
            total += loss.item() * len(yb)
        self.scheduler.step()
        return total / len(loader.dataset)


def make_budget(m, epochs):
    return {"fbat_floor": lambda: FixedFloor(0.6586, 0.6586), "signal_lin": SignalLinear,
            "ramp_floor": lambda: RampFloor(epochs), "grid_035": lambda: FixedFloor(0.35, 0.35),
            "grid_050": lambda: FixedFloor(0.5, 0.5), "pgdat_uniform": lambda: ag.FixedPerturbationBudget(eps=0.6586, lam=0.5),
            "trades_b1": lambda: ag.FixedPerturbationBudget(eps=0.6586, lam=0.5),
            "trades_b6": lambda: ag.FixedPerturbationBudget(eps=0.6586, lam=0.5), "h2_only": H2Only,
            "h1_only": H1Only, "h29_only": H29Only}[m]()

ADAPTIVE = ("signal_lin", "h2_only", "h1_only", "h29_only")


def train_method(m, base_ckpt, tag, epochs, seed=42, patience=6):
    """Same loop as had_stack_v2.train, with the budget object / trainer of method m."""
    bm.set_seed(seed)
    tr, va, te, nf = bm.load_processed_data()
    budget = make_budget(m, epochs)
    model = bm.TabularTransformer(nf, d_model=bm.HP['d_model'], n_heads=bm.HP['n_heads'], n_layers=bm.HP['n_layers'],
                                  d_ffn=bm.HP['d_ffn'], dropout=bm.HP['dropout'])
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    Trainer = TradesTrainer if m.startswith("trades") else ag.StackelbergAdversarialTrainer
    T = Trainer(model, budget, adapt=(m in ADAPTIVE), epochs=epochs)
    if m.startswith("trades"): T.BETA = {"trades_b1": 1.0, "trades_b6": 6.0}[m]
    sel = ag.HamiltonianAttacker(hs.FixedBudget(hs.SEL_EPS, 0.5))
    best, wait, sig, ref, traj = -1.0, 0, None, None, []
    ck = f"models/adv_{tag}_best.pt"; t0 = time.perf_counter()
    for r in range(1, epochs + 1):
        budget.resolve(sig if m in ADAPTIVE else None); T.attacker.refresh(); T.adv_weight = budget.derived_lambda()
        T.train_epoch(tr)
        torch.manual_seed(seed + r)
        rauc = hs.robust_auc(model, va, sel, hs.SEL_EPS)
        sig, gap = ag.robustness_gap_signal(model, T.attacker, va, ref)
        if ref is None: ref = gap
        traj.append(dict(round=r, eps=budget.eps_global, lam=T.adv_weight, robust_val_auc=rauc, rel_gap=gap,
                         next_signal=sig, seconds=time.perf_counter() - t0))
        print(f"[{tag}] round {r:2d} eps {budget.eps_global:.4f} lam {T.adv_weight:.3f} robustAUC {rauc:.4f}", flush=True)
        if rauc > best:
            best, wait = rauc, 0; torch.save(model.state_dict(), ck)
        else:
            wait += 1
            if wait >= patience: break
    with open(f"logs/adv_traj_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=traj[0].keys()); w.writeheader(); w.writerows(traj)
    return ck


# ---------------------------------------------------------------- folds
def fold_data():
    import data_processing as dp
    from sklearn.model_selection import StratifiedKFold
    df = dp.engineer_features(dp.select_and_clean(dp.load_data(dp.DATA_PATH)))
    feats = [c for c in df.columns if c != dp.TARGET_COL]
    X = df[feats].values.astype(np.float32); y = df[dp.TARGET_COL].values.astype(np.int64)
    return feats, X, y, list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def use_fold(k, splits, X, y):
    tr, te = splits[k]; d = rc.make_fold(k, tr, te, X, y)
    bm.OUTPUT_DIR, ex.DATA = d, d
    return d


def part_train():
    feats, X, y, splits = fold_data(); orig = (bm.OUTPUT_DIR, ex.DATA)
    try:
        for k in Q["folds"]:
            use_fold(k, splits, X, y); base = rc._need(f"models_cv/fold{k}/baseline_best.pt")
            for m in NEW:
                tag = f"{m}_fold{k}{Q['tag']}"
                if os.path.exists(f"models/adv_{tag}_best.pt") and os.path.exists(f"logs/adv_traj_{tag}.csv"): continue
                train_method(m, base, tag, Q["epochs"])
    finally:
        bm.OUTPUT_DIR, ex.DATA = orig


# ---------------------------------------------------------------- attacks
def _loss(z, kind): return torch.nn.functional.softplus(-z) if kind == "bce" else -z

def apgd(m, x, eps, kind, n_iter, seed=0):
    """APGD (Croce & Hein, 2020), L-inf, per-sample step size, momentum 0.75, checkpoint halving rule."""
    m.eval(); g = torch.Generator().manual_seed(seed)
    proj = lambda v: torch.max(torch.min(v, x + eps), x - eps)
    p = [0, 0.22]
    while p[-1] < 1: p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    ckpts = sorted({math.ceil(q * n_iter) for q in p[1:] if q <= 1})
    def lg(v):
        v = v.clone().requires_grad_(True); L = _loss(m(v), kind); gr, = torch.autograd.grad(L.sum(), v); return L.detach(), gr
    xa = proj(x + eps * (2 * torch.rand(x.shape, generator=g) - 1))
    L, gr = lg(xa); best_x, best_L = xa.clone(), L.clone()
    step = torch.full((len(x), 1), 2 * eps); prev = xa.clone(); succ = torch.zeros(len(x))
    last_ck, L_at_ck, reduced_at_ck = 0, best_L.clone(), torch.zeros(len(x), dtype=torch.bool)
    for k in range(n_iter):
        z = proj(xa + step * gr.sign())
        z = proj(xa + 0.75 * (z - xa) + 0.25 * (xa - prev)) if k > 0 else z
        prev = xa; xa = z
        Ln, gr = lg(xa); succ += (Ln > L).float(); L = Ln
        better = L > best_L; best_x[better] = xa[better]; best_L = torch.where(better, L, best_L)
        if (k + 1) in ckpts:
            h = (k + 1) - last_ck
            cond = (succ < 0.75 * h) | ((~reduced_at_ck) & (best_L <= L_at_ck))
            step[cond] /= 2; xa[cond] = best_x[cond]
            if cond.any(): L_c, g_c = lg(xa); L, gr = torch.where(cond, L_c, L), torch.where(cond[:, None], g_c, gr)
            reduced_at_ck = cond; succ.zero_(); L_at_ck = best_L.clone(); last_ck = k + 1
    return best_x

def square(m, x, eps, n_q, seed=0):
    """Square attack (Andriushchenko et al., 2020) adapted to tabular data: random +-eps on feature subsets,
    accepted when the fraud logit decreases. Uses no gradients."""
    m.eval(); g = torch.Generator().manual_seed(seed); d = x.shape[1]
    with torch.no_grad():
        delta = eps * (2 * torch.randint(0, 2, x.shape, generator=g) - 1).float(); best = m(x + delta)
        for i in range(n_q):
            k = max(1, int(round(d * 0.3 * 0.5 ** (i // max(n_q // 5, 1)))))
            ranks = torch.rand(x.shape, generator=g).argsort(dim=1).argsort(dim=1)
            mask = ranks < k                                   # k random features per row
            signs = eps * (2 * torch.randint(0, 2, x.shape, generator=g) - 1).float()
            cand = torch.where(mask, signs, delta)
            z = m(x + cand); acc = z < best
            delta[acc] = cand[acc]; best = torch.where(acc, z, best)
    return x + delta


class Constrained:
    """Maps a perturbation of the 14 raw (standardised) features to a valid transaction in model space."""
    def __init__(self, feats, mean, scale, raw_domain):
        self.base = list(range(14)); self.f = feats
        self.mean = torch.tensor(mean, dtype=torch.float32); self.scale = torch.tensor(scale, dtype=torch.float32)
        # valid domain = each raw feature's observed range over the whole dataset (domain limits, no label information)
        self.lo = torch.tensor(raw_domain[:, :14].min(0)); self.hi = torch.tensor(raw_domain[:, :14].max(0))
        ints = ("session_duration", "receiver_transaction_history", "transaction_time_of_day", "app_switching_frequency", "screen_active_time")
        self.int_mask = torch.tensor([f in ints for f in feats[:14]])
        i = {f: j for j, f in enumerate(feats)}; self.i = i
        assert feats[14:] == ["log_amount", "typing_anomaly", "amount_vs_history_ratio", "engagement_ratio"], feats[14:]
    def to_model(self, zb):
        raw = zb * self.scale[:14] + self.mean[:14]
        raw = torch.max(torch.min(raw, self.hi), self.lo)
        raw = torch.where(self.int_mask, raw + (torch.round(raw) - raw).detach(), raw)   # straight-through rounding
        i = self.i
        eng = torch.stack([torch.log1p(raw[:, i["amount"]]),
                           raw[:, i["input_pause_patterns"]] / (raw[:, i["keyboard_input_speed"]] + 1e-6),
                           raw[:, i["amount"]] / (raw[:, i["transaction_amount_vs_sender_history"]] + 1e-6),
                           raw[:, i["screen_active_time"]] / (raw[:, i["session_duration"]] + 1.0)], 1)
        full = torch.cat([raw, eng], 1)
        return (full - self.mean) / self.scale

def constrained_pgd(m, C, x, eps, kind, K, R, seed=0):
    m.eval(); zb0 = x[:, :14]; a = 2.5 * eps / K
    best = C.to_model(zb0).detach(); bz = m(best).detach()
    for r in range(R):
        g = torch.Generator().manual_seed(seed + r); eta = (torch.rand(zb0.shape, generator=g) * 2 - 1) * eps
        for _ in range(K):
            eta.requires_grad_(True); L = _loss(m(C.to_model(zb0 + eta)), kind).sum()
            gr, = torch.autograd.grad(L, eta)
            with torch.no_grad(): eta = torch.clamp(eta + a * gr.sign(), -eps, eps)
        with torch.no_grad():
            xa = C.to_model(zb0 + eta); z = m(xa); b = z < bz; best[b] = xa[b]; bz = torch.minimum(bz, z)
    return best


# ---------------------------------------------------------------- evaluation
def choose(k, candidates):
    """Pick the candidate with the best robust validation AUC in fold k (validation data only)."""
    best, choice = -1, None
    for m in candidates:
        f = f"logs/adv_traj_{m}_fold{k}{Q['tag']}.csv"
        v = max(float(r["robust_val_auc"]) for r in csv.DictReader(open(f)))
        if v > best: best, choice = v, m
    return choice

def ckpts(k):
    c = {"baseline": f"models_cv/fold{k}/baseline_best.pt", "v3": f"models/v2_v3_fold{k}_best.pt",
         "pgdat066": f"models/v2_pgdat066_fold{k}_best.pt", "pgdat066lam": f"models/v2_pgdat066lam_fold{k}_best.pt"}
    for m in EVAL_NEW:
        src = {"grid_sel": lambda: choose(k, GRID), "trades_sel": lambda: choose(k, TRADES)}.get(m, lambda: m)()
        c[m] = f"models/adv_{src}_fold{k}{Q['tag']}_best.pt"
    for p in c.values(): rc._need(p)
    return c

def part_eval():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    feats, X, y, splits = fold_data(); os.makedirs(f"results_advisor{Q['tag']}", exist_ok=True); orig = ex.DATA
    try:
        for k in Q["folds"]:
            out = f"results_advisor{Q['tag']}/fold{k}.npz"
            if os.path.exists(out): continue
            use_fold(k, splits, X, y); tr, te = splits[k]
            Xtr, _, _, _ = train_test_split(X[tr], y[tr], test_size=0.15, random_state=42, stratify=y[tr])
            sc = StandardScaler().fit(Xtr); C = Constrained(feats, sc.mean_, sc.scale_, X)
            Xt, yt = ex.load_split("test"); Xf = torch.tensor(Xt[yt == 1])
            assert torch.allclose(C.to_model(Xf[:, :14]), Xf, atol=2e-3), "constrained map does not reproduce clean rows"
            paths = ckpts(k); M = {n: ex.load_transformer(p) for n, p in paths.items()}
            save = {"y": yt, "grid_choice": np.array(choose(k, GRID)), "trades_choice": np.array(choose(k, TRADES))}
            crafted = {(n, e): rc.pgd(mm, Xf, e, "margin", Q["K"], 1) for n, mm in M.items() for e in EPS_EVAL}
            for n, mm in M.items():
                t0 = time.perf_counter()
                pc = ex.probs(mm, Xt); save[f"{n}__clean"] = pc; det = torch.tensor(pc[yt == 1] >= 0.5)
                for e in EPS_EVAL:
                    advs = {"pgd_bce": rc.pgd(mm, Xf, e, "bce", Q["K"], Q["R"]), "pgd_margin": rc.pgd(mm, Xf, e, "margin", Q["K"], Q["R"]),
                            "apgd_ce": apgd(mm, Xf, e, "bce", Q["APGD"]), "apgd_margin": apgd(mm, Xf, e, "margin", Q["APGD"]),
                            "square": square(mm, Xf, e, Q["SQ"]),
                            "cons_bce": constrained_pgd(mm, C, Xf, e, "bce", Q["KC"], Q["RC"]),
                            "cons_margin": constrained_pgd(mm, C, Xf, e, "margin", Q["KC"], Q["RC"])}
                    with torch.no_grad():
                        for a, xa in advs.items():
                            if not a.startswith("cons"): assert (xa - Xf).abs().max() <= e + 1e-4
                            save[f"{n}__{a}__{e}"] = ((mm(xa) < 0) & det).numpy()
                        tr_ = torch.zeros_like(det)
                        for s in M:
                            if s != n: tr_ |= (mm(crafted[(s, e)]) < 0) & det
                        save[f"{n}__transfer__{e}"] = tr_.numpy()
                print(f"[fold {k}] {n:14s} evaluated in {time.perf_counter()-t0:6.0f}s", flush=True)
            np.savez_compressed(out, **save)
    finally:
        ex.DATA = orig


# ---------------------------------------------------------------- summary
STD = ("pgd_bce", "pgd_margin", "transfer"); EXT = STD + ("apgd_ce", "apgd_margin", "square"); CONS = ("cons_bce", "cons_margin")

def part_summary():
    D = {k: dict(np.load(f"results_advisor{Q['tag']}/fold{k}.npz")) for k in Q["folds"]}
    models = list(EXISTING) + list(EVAL_NEW)
    def rr(d, m, e, attacks, idx=None):
        y = d["y"]; det = d[f"{m}__clean"][y == 1] >= 0.5; br = np.zeros_like(det)
        for a in attacks: br |= d[f"{m}__{a}__{e}"]
        kept = det & ~br
        return 100 * (kept.mean() if idx is None else kept[idx].mean())
    def clean(d, m):
        y, p = d["y"], d[f"{m}__clean"]; pr = p >= 0.5
        tp, fp, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
        return dict(recall=100 * tp / (tp + fn), precision=100 * tp / max(tp + fp, 1), fp=fp, fn=fn, cost=100 * fn + fp)
    S = {"per_fold": {}, "mean": {}, "std": {}, "rules": {}, "paired": {}, "grid_choice": {k: str(D[k]["grid_choice"]) for k in D},
         "eval_strength": {}, "trades_choice": {k: str(D[k]["trades_choice"]) for k in D}}
    for m in models:
        rows = []
        for k, d in D.items():
            r = dict(clean(d, m)); r.update({f"{t}@{e}": rr(d, m, e, A) for t, A in (("std", STD), ("ext", EXT), ("cons", CONS)) for e in EPS_EVAL})
            rows.append(r)
        S["per_fold"][m] = rows
        S["mean"][m] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        S["std"][m] = {key: float(np.std([r[key] for r in rows])) for key in rows[0]}
    for c in CONTROLS:
        wins = sum(S["per_fold"]["v3"][i]["ext@1.0"] > S["per_fold"][c][i]["ext@1.0"] for i in range(len(D)))
        S["rules"][c] = dict(wins=int(wins), of=len(D), passed=bool(wins >= 4 and len(D) == 5))
    rng = np.random.default_rng(0)
    def pooled(m, e, attacks, draws):
        return np.array([np.mean([rr(D[k], m, e, attacks, draws[k][b]) for k in D]) for b in range(Q["B"])])
    draws = {k: [rng.integers(0, int((D[k]["y"] == 1).sum()), int((D[k]["y"] == 1).sum())) for _ in range(Q["B"])] for k in D}
    for c in [x for x in models if x != "v3"]:
        S["paired"][c] = {}
        for e in EPS_EVAL:
            diff = pooled("v3", e, EXT, draws) - pooled(c, e, EXT, draws)
            pt = S["mean"]["v3"][f"ext@{e}"] - S["mean"][c][f"ext@{e}"]
            S["paired"][c][str(e)] = dict(point=float(pt), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
    for m in models:
        S["eval_strength"][m] = {str(e): float(S["mean"][m][f"std@{e}"] - S["mean"][m][f"ext@{e}"]) for e in EPS_EVAL}
    # timing per method from training logs
    S["train_minutes"] = {}
    for m in NEW:
        mins = [float(list(csv.DictReader(open(f"logs/adv_traj_{m}_fold{k}{Q['tag']}.csv")))[-1]["seconds"]) / 60 for k in D]
        S["train_minutes"][m] = float(np.mean(mins))
    json.dump(S, open(f"results_advisor{Q['tag']}/advisor_summary.json", "w"), indent=1)
    print("\nFive-fold means, worst-case robust recall (extended ensemble) and constrained attack:")
    for m in models:
        a = S["mean"][m]
        print(f"  {m:14s} ext@0.5 {a['ext@0.5']:5.1f}  ext@1.0 {a['ext@1.0']:5.1f}  cons@1.0 {a['cons@1.0']:5.1f}  cost {a['cost']:7.0f}")
    print("\nPre-registered rules (HAD-Stack must beat the control at eps=1.0 in >= 4 of 5 folds):")
    for c, r in S["rules"].items(): print(f"  vs {c:12s}: {r['wins']}/{r['of']} -> {'PASSED' if r['passed'] else 'not passed'}")
    print(f"\nWritten results_advisor{Q['tag']}/advisor_summary.json  -> send this file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["train", "eval", "summary"], required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        Q.update(epochs=1, K=3, R=1, KC=3, RC=1, APGD=6, SQ=10, B=50, folds=range(1), tag="_QUICK")
    {"train": part_train, "eval": part_eval, "summary": part_summary}[a.part]()
