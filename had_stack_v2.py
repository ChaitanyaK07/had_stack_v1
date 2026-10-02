"""
HAD-Stack v2 — Hamiltonian controller from COMPLEX eigenvalues (pilot)
=====================================================================
New file; adversarial_game.py and baseline_model.py are NOT modified.

Pre-registered design (fixed before any v2 result was seen):
  * Spectra: hamiltonian_v2 (user's exact H1, H2, full Eq.29; np.linalg.eig, no .real).
  * Game coupling: validation robustness-gap signal s in [-1, 1] (same definition as
    v1) scales the attacker-detector couplings g1 (H1), g_2 (H2) and the H29
    dissipation eta_r by (1 + 0.5 s).
  * eps    = EPS_REF * Gamma2 / Gamma2_base   (Gamma = decay rate of ground state)
    lambda = clip(LAM_REF * Gamma1 / Gamma1_base, 0.1, 0.9)
    EPS_REF = 0.5 chosen from the undefended-baseline check (attacks start to bite).
  * Checkpoint selection for EVERY method: best ROBUST validation AUC under PGD at
    eps = 0.5 (clean-AUC selection picked round 2 in v1, before any adaptation).
  * Comparison: PGD-AT with fixed eps = 0.5, lambda = 0.5, identical optimiser,
    rounds, attacker and selection rule.
  * Success criterion: lower attack-success rate at eps in {0.5, 1.0} than PGD-AT,
    without higher clean cost (100*FN + FP).
Usage:  python had_stack_v2.py --method v2        (or --method pgdat)
"""
import argparse, csv, json, os
import numpy as np, torch
import baseline_model as bm
import adversarial_game as ag
import hamiltonian_v2 as hv

EPS_REF, LAM_REF, KAPPA, SEL_EPS = 0.5, 0.5, 0.5, 0.5
EVAL_EPS = (0.0, 0.15, 0.3, 0.5, 1.0)


class HamiltonianBudgetV2:
    """Same interface as ag.HamiltonianPerturbationBudget."""
    def __init__(self):
        self._base = self._solve(0.0)
        self.resolve(None)

    @staticmethod
    def _solve(s):
        f = 1.0 + KAPPA * s
        p1 = dict(hv.PARAMS_H1);   p1["g1"] *= f
        p2 = dict(hv.PARAMS_H2);   p2["g_2"] *= f
        p29 = dict(hv.PARAMS_EQ29); p29["eta_r"] *= f
        e1 = hv.H1v2(p1).solve(); e2 = hv.H2v2(p2).solve()
        s29 = hv.H29v2(p29); e29 = s29.solve()
        return dict(E0=e1[0].real, G1=-e1[0].imag, G2=-e2[0].imag, e1=e1, e2=e2,
                    field=s29.field)

    def resolve(self, signal=None):
        r = self._solve(0.0 if signal is None else float(np.clip(signal, -1, 1)))
        self.cur = r
        self.eps_global = float(EPS_REF * r["G2"] / self._base["G2"])
        self.lam = float(np.clip(LAM_REF * r["G1"] / self._base["G1"], 0.1, 0.9))
        self.field = r["field"]
        return self

    def derived_lambda(self, *a, **k):
        return self.lam

    def get_feature_weights(self, n_features):
        # identical construction to v1, using |eigenvalue| magnitudes of H1, H2
        comb = np.concatenate([np.abs(self.cur["e1"]), np.abs(self.cur["e2"])])
        w = np.tile(comb, n_features // len(comb) + 1)[:n_features]
        return torch.tensor(w / (w.sum() + 1e-12) * n_features, dtype=torch.float32)


class HamiltonianBudgetV3(HamiltonianBudgetV2):
    """v3 (hypothesis stated after the v2 pilot): (i) curriculum direction — the
    Hamiltonian couplings respond to -s, so eps/lambda RISE as the defender hardens;
    (ii) step weights floored: w = 0.5 + 0.5 * w_H / mean(w_H), so every feature is
    attacked (mean weight 1, same as PGD-AT)."""
    def resolve(self, signal=None):
        return super().resolve(None if signal is None else -float(signal))
    def get_feature_weights(self, n_features):
        w = super().get_feature_weights(n_features)
        return 0.5 + 0.5 * w / w.mean()


class RampBudget(ag.FixedPerturbationBudget):
    """Non-Hamiltonian curriculum control: eps rises linearly 0.3461 -> 0.6586 (v3's
    range) over the rounds, lambda = 0.5, uniform weights."""
    def __init__(self, epochs):
        super().__init__(eps=0.3461, lam=0.5); self.r, self.E = 0, epochs
    def resolve(self, signal=None):
        self.eps_global = 0.3461 + (0.6586 - 0.3461) * self.r / max(self.E - 1, 1)
        self.r += 1; return self


class FixedBudget(ag.FixedPerturbationBudget):
    """PGD-AT reference; uses the v2 base feature weights so the attacker's step
    geometry is identical to v2's."""
    def __init__(self, eps, lam):
        super().__init__(eps=eps, lam=lam)
        self._w = HamiltonianBudgetV2()
    def get_feature_weights(self, n):
        return self._w.get_feature_weights(n)


def robust_auc(model, loader, attacker, eps):
    ps, ys = [], []
    for xb, yb in loader:
        xa = attacker.evasion_attack(model, xb, yb, eps=eps)
        with torch.no_grad():
            ps.append(torch.sigmoid(model(xa)).numpy()); ys.append(yb.numpy())
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


def train(method, epochs=30, patience=6, seed=42,
          base_ckpt="models/baseline_best.pt", out_tag=None):
    """base_ckpt / out_tag: optional; defaults reproduce the original behaviour."""
    out_tag = out_tag or method
    bm.set_seed(seed) if hasattr(bm, "set_seed") else torch.manual_seed(seed)
    tr, va, te, nf = bm.load_processed_data()
    budget = {"v2": lambda: HamiltonianBudgetV2(), "v3": lambda: HamiltonianBudgetV3(),
              "ramp": lambda: RampBudget(epochs),
              "pgdat066": lambda: FixedBudget(0.6586, 0.5),
              "pgdat066lam": lambda: FixedBudget(0.6586, 0.6586)}.get(method, lambda: FixedBudget(EPS_REF, 0.5))()
    model = bm.TabularTransformer(nf, d_model=bm.HP['d_model'], n_heads=bm.HP['n_heads'],
                                  n_layers=bm.HP['n_layers'], d_ffn=bm.HP['d_ffn'],
                                  dropout=bm.HP['dropout'])
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    T = ag.StackelbergAdversarialTrainer(model, budget, adapt=(method in ("v2", "v3")), epochs=epochs)
    sel_attacker = ag.HamiltonianAttacker(FixedBudget(SEL_EPS, 0.5))   # same selector for all
    best, wait, sig, ref, traj = -1.0, 0, None, None, []
    ck = f"models/v2_{out_tag}_best.pt"
    for r in range(1, epochs + 1):
        budget.resolve(sig if method in ("v2", "v3") else None); T.attacker.refresh()
        T.adv_weight = budget.derived_lambda()
        loss = T.train_epoch(tr)
        torch.manual_seed(seed + r)
        rauc = robust_auc(model, va, sel_attacker, SEL_EPS)
        sig, gap = ag.robustness_gap_signal(model, T.attacker, va, ref)
        if ref is None: ref = gap
        traj.append(dict(round=r, eps=budget.eps_global, lam=T.adv_weight,
                         field=budget.field, robust_val_auc=rauc, rel_gap=gap, next_signal=sig))
        print(f"[{method}] round {r:2d} eps {budget.eps_global:.4f} lam {T.adv_weight:.3f} "
              f"field {budget.field:.5f} robustAUC {rauc:.4f} gap {gap:.4f} s {sig:+.3f}", flush=True)
        if rauc > best:
            best, wait = rauc, 0; torch.save(model.state_dict(), ck)
        else:
            wait += 1
            if wait >= patience: break
    with open(f"logs/v2_traj_{out_tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=traj[0].keys()); w.writeheader(); w.writerows(traj)
    return ck


def evaluate(ckpt, tag, seed=42):
    import eval_extras as ex
    X, y = ex.load_split("test"); m = ex.load_transformer(ckpt)
    row = dict(model=tag); pc = ex.probs(m, X); det = (pc >= .5) & (y == 1)
    cm = ex.metrics(y, pc); row.update(clean_auc=cm["auc"], f1=cm["f1"], fp=cm["fp"],
                                       fn=cm["fn"], cost=cm["cost"])
    att = ag.HamiltonianAttacker(FixedBudget(0.5, 0.5))                 # identical attacker
    for e in EVAL_EPS[1:]:
        torch.manual_seed(seed)
        out = [];
        for i in range(0, len(X), 512):
            out.append(att.evasion_attack(m, torch.tensor(X[i:i+512]),
                                          torch.tensor(y[i:i+512]), eps=e).numpy())
        pa = ex.probs(m, np.concatenate(out))
        row[f"asr@{e}"] = float(((pa < .5) & det).sum() / det.sum())
        row[f"recall@{e}"] = ex.metrics(y, pa)["recall"]
    with open("logs/v2_results.jsonl", "a") as f: f.write(json.dumps(row, default=float) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", choices=["v2", "v3", "ramp", "pgdat", "pgdat066", "pgdat066lam", "baseline"])
    a = ap.parse_args()
    if a.method == "baseline":
        evaluate("models/baseline_best.pt", "baseline")
    else:
        evaluate(train(a.method), a.method)
