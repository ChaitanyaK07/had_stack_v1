"""
=============================================================================
FILE 3: ADVERSARIAL GAME FRAMEWORK — Stackelberg Game + Hamiltonian Equations
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions
Advisor: Aneesh Chivukula

THEORETICAL FRAMEWORK:
  We model the fraud detection problem as a two-player Stackelberg game:
    • Leader  (Defender): the fraud detector (our transformer)
    • Follower (Attacker): an adversarial perturbation generator

  The attacker generates adversarial transaction features that attempt to
  evade the detector. The detector is trained to be robust against these.

  HAMILTONIAN INTEGRATION (REVISED — every item below was verified by
  direct test on this code):

    1. SOLVER MATH. H1 and the Eq29 real part are non-Hermitian /
       non-symmetric; np.linalg.eigh silently read only the lower
       triangle. All spectra now come from the Hermitian part
       (H + H^dag)/2. Consequences, verified numerically:
         E0(H1) = -gamma_1/2      (g1, eps0, eps1, E1 have no effect)
         gap(H2) = gamma_2        (H.real drops eps0, eps2, g_2, E_2)
         gap(Eq29) = min(eps0, eps1) = 1.0, field -> ~1e-5 (constant)
    2. Hamiltonians are re-solved EVERY ROUND. The feedback signal is the
       validation robustness gap (L_adv - L_clean)/L_clean relative to its
       round-1 value, clip(gap/gap_1 - 1, -1, 1), which scales the damping
       rates gamma_1 and gamma_2 by up to +-30% — the only parameters that
       move the spectra. (Perturbing the solvers' initial guess has no
       effect; 1 - val_AUC as a signal moved eps only in the 5th decimal.)
    3. eps    = clip( cbrt(gamma_1/4 * gamma_2/4 * 0.1), 0.02, 0.3 )
       lambda = 0.1 + 0.8 * clip(gamma_1, 0, 1)
       (lambda previously a hardcoded 0.5 never connected to H1.)
    4. Eq29 is a constant factor in eps and is NOT a stopping rule.
       Stopping is val-AUC early stopping for every configuration; the
       round at which (eps, lambda) become stationary is logged only.
    5. Attack is L-inf PGD that lowers P(fraud) on fraud rows (the
       original ascended BCE toward label 0, which RAISED P(fraud)), and
       ||x_adv - x||_inf <= eps holds exactly (original exceeded eps by
       up to 2.65x via feature weights).
    6. Seeds fixed (42); adversarial model uses the baseline's dropout
       (0.4, originally silently 0.2); per-run tagged checkpoints.

  GAME OBJECTIVE:
    min_θ max_δ  L(f_θ(x + δ), y)
    subject to:  ‖δ‖∞ ≤ ε_H  (Hamiltonian-derived, per-round budget)
=============================================================================
"""

import argparse
import csv
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline_model import (TabularTransformer, FocalLoss, load_processed_data,
                            evaluate, set_seed, HP)

warnings.filterwarnings('ignore')

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
LOG_DIR   = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HAMILTONIAN SOLVER BASE PARAMETERS
# ─────────────────────────────────────────────────────────────

PARAMS_H1 = {
    'gamma_1': 0.1, 'W': 1.5, 'delta_init': 0.1,
    'ħ': 1.0, 'ε0': 0.1, 'ε1': 0.1, 'g1': 0.1, 'E1': 1
}
PARAMS_H2 = {
    'epsilon_0': 0.1, 'epsilon_2': 0.1, 'gamma_2': 0.1,
    'g_2': 0.1, 'E_2': 1, 'delta_2_init': 0.1, 'delta_init': 0.095
}
PARAMS_EQ29 = {
    'epsilon_0': 1.0, 'epsilon_1': 1.0, 'gamma_1': 0.1,
    'g_1': 0.15, 'h': 1.0, 'E_1': 0.5 + 0.3j,
    'eta_r': 1.0, 'field_init': 0.1
}


def _herm(H):
    """Hermitian part (H + H^dag)/2."""
    return 0.5 * (H + H.conj().T)


class QuantumHamiltonianSolver:
    """H1 Solver — adapted from final_hamiltonian_solver.py (spectrum now
    taken from the Hermitian part; see solve())."""
    def __init__(self, params):
        self.params = params
        self.dim = 4
        self.a, self.a_dag, self.b, self.b_dag = self._create_operators()
        self.ab = self.a @ self.b

    def _create_operators(self):
        a_s = np.zeros((2, 2), dtype=complex); a_s[0, 1] = 1.0
        I   = np.eye(2, dtype=complex)
        return (np.kron(a_s, I), np.kron(a_s.conj().T, I),
                np.kron(I, a_s), np.kron(I, a_s.conj().T))

    def construct_hamiltonian(self, delta):
        W, g1, E1 = self.params['W'], self.params['g1'], self.params['E1']
        hbar, eps0, eps1 = self.params['ħ'], self.params['ε0'], self.params['ε1']
        gamma = self.params['gamma_1']
        n_a = self.a_dag @ self.a; n_b = self.b_dag @ self.b
        H = ((eps0 / (2j * hbar) + gamma / 2) * n_a +
             (eps1 / (1j * hbar) - gamma / 2) * n_b +
             (g1 * E1 / (1j * hbar) - gamma * delta) *
             (self.a_dag @ self.b + self.b_dag @ self.a))
        return H

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        # CHANGE: H1 as constructed is NON-Hermitian (||H - H^dag|| = 0.3).
        # np.linalg.eigh assumes a Hermitian input and silently reads only
        # the lower triangle, so the original "E0 = -0.1118" was not an
        # eigenvalue of H1. We now diagonalise the Hermitian part
        # H_h = (H + H^dag)/2 explicitly (well-defined, real spectrum).
        # Verified consequence: the spectrum of H_h depends on gamma_1 only
        # (E0 = -gamma_1/2); g1, eps0, eps1, E1 enter anti-Hermitian terms
        # and cancel.
        delta = self.params['delta_init']
        for _ in range(max_iter):
            evals, evecs = np.linalg.eigh(_herm(self.construct_hamiltonian(delta)))
            psi0 = evecs[:, 0]
            dc   = np.vdot(psi0, self.ab @ psi0).real
            dn   = mixing * dc + (1 - mixing) * delta
            if abs(dn - delta) < tol:
                break
            delta = dn
        return np.linalg.eigvalsh(_herm(self.construct_hamiltonian(delta)))


class H2Solver:
    """H2 Solver — adapted from final_hamiltonian_solver.py (unchanged).

    NOTE (verified): construct_hamiltonian returns H.real, which discards
    every term containing epsilon_0, epsilon_2, g_2 and E_2 (they are
    purely imaginary). The retained matrix is real-symmetric, so eigh is
    valid here, and its spectral gap (max - min) equals gamma_2 exactly.
    """
    def __init__(self, params):
        self.params = params
        self.dim = 4
        self.a, self.a_dag, self.c, self.c_dag = self._create_operators()
        self.ac = self.a @ self.c

    def _create_operators(self):
        a_s = np.zeros((2, 2), dtype=complex); a_s[0, 1] = 1.0
        I   = np.eye(2, dtype=complex)
        return (np.kron(a_s, I), np.kron(a_s.conj().T, I),
                np.kron(I, a_s), np.kron(I, a_s.conj().T))

    def construct_hamiltonian(self, delta_2):
        e0, e2 = self.params['epsilon_0'], self.params['epsilon_2']
        g2, E2 = self.params['g_2'], self.params['E_2']
        gamma  = self.params['gamma_2']
        n_a = self.a_dag @ self.a; n_c = self.c_dag @ self.c
        H = ((e0 / (2j) + gamma / 2) * n_a +
             (e2 / (1j) - gamma / 2) * n_c +
             (g2 * E2 / (1j) - gamma * delta_2) *
             (self.a_dag @ self.c + self.c_dag @ self.a))
        return H.real

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        d2 = self.params['delta_2_init']
        for _ in range(max_iter):
            H = self.construct_hamiltonian(d2)
            evals, evecs = np.linalg.eigh(H)
            psi0 = evecs[:, 0]
            dc   = np.vdot(psi0, self.ac @ psi0).real
            dn   = mixing * dc + (1 - mixing) * d2
            if abs(dn - d2) < tol:
                break
            d2 = dn
        return np.linalg.eigh(self.construct_hamiltonian(d2))[0]


class Eq29Solver:
    """Eq29 Solver — adapted from final_hamiltonian_solver.py.

    CHANGE: solve() can now optionally return the converged equilibrium
    field alongside the eigenvalues, so callers can track field
    convergence round-to-round as a logged diagnostic
    (previously this value was computed and discarded).
    """
    def __init__(self, params):
        self.params = params
        self.dim = 16
        self._create_pauli_matrices()

    def _create_pauli_matrices(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I  = np.eye(2, dtype=complex)
        for i, s in enumerate([[sx, sy, sz]] * 4):
            for j, (mat, name) in enumerate(zip(s, ['x', 'y', 'z'])):
                parts = [I] * 4; parts[i] = mat
                full  = parts[0]
                for p in parts[1:]:
                    full = np.kron(full, p)
                setattr(self, f"s{i}{name}", full)

    def construct_full_hamiltonian(self, field):
        e0, e1 = self.params['epsilon_0'], self.params['epsilon_1']
        g1, h  = self.params['g_1'], self.params['h']
        E1     = self.params['E_1']
        gamma  = self.params['gamma_1']
        eta_r  = self.params['eta_r']
        H_en   = (-e0 / 2 * (self.s0z - self.s2z) -
                  e1 / 2 * (self.s1z @ self.s0z + self.s3z @ self.s2z @ self.s1z))
        cp_E1  = (1j * g1 * h * field / 4 * (
            self.s0x - 1j * self.s0y - self.s2y + 1j * self.s2x +
            1j * self.s1z @ self.s0y - self.s1z @ self.s0x) * E1)
        cp_Ec  = (1j * g1 * h * field / 4 * (
            self.s2x - 1j * self.s2y + 1j * self.s0x - self.s0y) * np.conj(E1))
        return (H_en + cp_E1 + cp_Ec).real

    def solve(self, max_iter=50, tol=1e-5, mixing=0.5, return_field=False):
        field = self.params['field_init']
        # CHANGE: the real part returned by construct_full_hamiltonian is
        # not symmetric, so eigh was again reading only the lower triangle
        # (the old evals -2, -1 were that artefact). We diagonalise the
        # symmetric part. Verified: gap = min(epsilon_0, epsilon_1) = 1.0 and
        # the field collapses to ~1e-5 for every coupling tested.
        for _ in range(max_iter):
            H     = _herm(self.construct_full_hamiltonian(field))
            evals, evecs = np.linalg.eigh(H)
            psi0  = evecs[:, 0]
            sx_e  = np.vdot(psi0, self.s0x @ psi0).real
            sy_e  = np.vdot(psi0, self.s0y @ psi0).real
            fn    = mixing * np.sqrt(sx_e ** 2 + sy_e ** 2) + (1 - mixing) * field
            if abs(fn - field) < tol:
                break
            field = fn
        evals = np.linalg.eigvalsh(_herm(self.construct_full_hamiltonian(field)))
        if return_field:
            return evals, field
        return evals


# ─────────────────────────────────────────────────────────────
# HAMILTONIAN PERTURBATION BUDGET (REVISED)
# ─────────────────────────────────────────────────────────────
class HamiltonianPerturbationBudget:
    """
    Solves H1/H2/Eq29 and derives structured perturbation constraints and
    the defender's loss-blend weight.

    CHANGES from the original:
      • resolve() is called once per round. `signal` in [-1, 1] scales
        the damping rates gamma_1 / gamma_2 by (1 + 0.3*signal); these are
        the only parameters that move the correctly computed spectra.
      • use_h1 / use_h2 select which Hamiltonians ADAPT to the signal; a
        disabled one stays frozen at its base parameters, so all
        ablations share the same starting eps/lambda. use_eq29 is kept
        for interface compatibility but is a NO-OP: Eq29 has no
        parameter that moves its spectrum (gap = min(eps0, eps1)).
      • derived_lambda() exposes H1's ground-state energy as the
        clean/adversarial blend weight, replacing the old hardcoded 0.5.
    """
    def __init__(self, use_h1=True, use_h2=True, use_eq29=True, signal=None):
        self.use_h1, self.use_h2, self.use_eq29 = use_h1, use_h2, use_eq29
        self.resolve(signal=signal)

    def resolve(self, signal=None):
        """Re-solve the enabled Hamiltonians. Call once per epoch.

        `signal` in [-1, 1] scales the damping rates gamma_1 (H1) and
        gamma_2 (H2) by (1 + 0.3*signal), i.e. +-30%.
        Verified facts this relies on:
          * perturbing a solver's initial guess (delta_init etc.) has no
            effect — the self-consistent loop returns to the same fixed point;
          * once spectra are computed correctly (Hermitian part), the
            coupling constants g1 / g_2 / g_1 have NO effect; only gamma_1
            and gamma_2 move E0(H1) = -gamma_1/2 and gap(H2) = gamma_2.
        Resulting closed forms (checked numerically in the test suite):
            eps_H1 = gamma_1/4, eps_H2 = gamma_2/4, eps_Eq29 = 0.1*min(eps0, eps1)
            eps    = clip((eps_H1 * eps_H2 * eps_Eq29)^(1/3), 0.02, 0.3)
            lambda = 0.1 + 0.8 * clip(gamma_1, 0, 1)
        """
        # signal in [-1, 1]  ->  gamma scaled by (1 + 0.3*signal), i.e. +-30%
        bump_frac = 0.0 if signal is None else float(np.clip(signal, -1.0, 1.0)) * 0.3

        # A disabled Hamiltonian is FROZEN at its base parameters (solved
        # with no signal) rather than removed, so every ablation starts from
        # the same eps/lambda and differs only in WHICH solvers adapt.
        s1 = bump_frac if self.use_h1 else 0.0
        s2 = bump_frac if self.use_h2 else 0.0
        s3 = bump_frac if self.use_eq29 else 0.0

        # The signal scales the DAMPING rates gamma_1 (H1) and gamma_2 (H2):
        # verified to be the only parameters that move the correctly
        # computed spectra. Eq29 has no parameter coupled to the signal
        # (its gap is min(eps0, eps1)); s3 is kept for interface symmetry.
        p = dict(PARAMS_H1);   p['gamma_1'] = PARAMS_H1['gamma_1'] * (1.0 + s1)
        self.evals_H1 = QuantumHamiltonianSolver(p).solve()
        p = dict(PARAMS_H2);   p['gamma_2'] = PARAMS_H2['gamma_2'] * (1.0 + s2)
        self.evals_H2 = H2Solver(p).solve()
        p = dict(PARAMS_EQ29)
        self.evals_Eq29, self.field = Eq29Solver(p).solve(return_field=True)

        self.eps_H1   = abs(self.evals_H1[1]  - self.evals_H1[0])   * 0.5
        self.eps_H2   = abs(self.evals_H2[1]  - self.evals_H2[0])   * 0.5
        self.eps_Eq29 = abs(self.evals_Eq29[1] - self.evals_Eq29[0]) * 0.1

        self.eps_global = float((self.eps_H1 * self.eps_H2 * self.eps_Eq29) ** (1 / 3))
        self.eps_global = float(np.clip(self.eps_global, 0.02, 0.3))

        # Ground-state energy of H1 -> defender regularisation weight (paper Eq. 8)
        self.E0_H1 = float(self.evals_H1[0])
        return self

    def derived_lambda(self, lo=0.1, hi=0.9, scale=0.5):
        """lambda = lo + (hi - lo) * clip(|E0(H1)| / scale, 0, 1).
        With E0 = -gamma_1/2 this is 0.1 + 0.8*gamma_1 (base: 0.18)."""
        frac = np.clip(abs(self.E0_H1) / scale, 0.0, 1.0)
        return float(lo + frac * (hi - lo))

    def get_feature_weights(self, n_features):
        """Project H1+H2 eigenvalue spreads onto feature dimension as
        per-feature perturbation weights (unchanged logic)."""
        h1_spread = np.abs(self.evals_H1)
        h2_spread = np.abs(self.evals_H2)
        combined  = np.concatenate([h1_spread, h2_spread])
        weights = np.tile(combined, n_features // len(combined) + 1)[:n_features]
        weights = weights / (weights.sum() + 1e-8) * n_features
        return torch.tensor(weights, dtype=torch.float32)


class FixedPerturbationBudget:
    """Non-Hamiltonian baseline with the same interface as
    HamiltonianPerturbationBudget: a constant eps, a fixed lambda, and
    uniform feature weights. This is the "ordinary PGD adversarial
    training with fixed eps" comparison reviewers asked for — pass an
    instance of this to StackelbergAdversarialTrainer in place of
    HamiltonianPerturbationBudget and nothing else needs to change."""
    def __init__(self, eps=0.01, lam=0.5):
        self.eps_global = eps
        self.E0_H1 = 0.0
        self._lam = lam
        self.field = 0.0

    def resolve(self, signal=None):
        return self  # nothing to resolve; budget stays fixed by design

    def derived_lambda(self, lo=0.1, hi=0.9, scale=0.5):
        return self._lam

    def get_feature_weights(self, n_features):
        return torch.ones(n_features, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# ATTACKER: PROJECTED GRADIENT DESCENT (PGD), L-infinity
#
# FIXES (verified by direct test, see accompanying notes):
#   (a) DIRECTION. The original loss was BCE(logits, 0) and the update
#       was gradient ASCENT on it, which pushes fraud samples AWAY from
#       the legitimate class (P(fraud) rose from 0.908 to 0.947 on a
#       test model). That is why "the attack improves both models" in
#       the paper. The attacker now maximises the loss w.r.t. the TRUE
#       label (y=1) on fraud rows, which lowers P(fraud) = evasion.
#   (b) L-INF BOUND. The original returned x + delta * w with
#       ||delta||_inf <= eps and feature weights w up to 2.65, so the
#       real perturbation reached 2.65*eps. The Hamiltonian feature
#       weights now scale the per-feature STEP size, and the final
#       perturbation itself is projected onto ||eta||_inf <= eps, so
#       the stated constraint holds exactly.
#   (c) STEP SIZE is 2.5*eps/n_steps (Madry et al. 2018 convention), so
#       the attack can reach the boundary of the eps-ball for any eps.
#   (d) Gradients are taken w.r.t. the perturbation only
#       (torch.autograd.grad), so no gradient leaks into model params.
# ─────────────────────────────────────────────────────────────
class HamiltonianAttacker:
    def __init__(self, budget, n_steps=10, step_mult=2.5):
        self.budget    = budget
        self.n_steps   = n_steps
        self.step_mult = step_mult
        self.eps       = budget.eps_global

    def refresh(self):
        """Pick up the current budget's eps after budget.resolve()."""
        self.eps = self.budget.eps_global

    def evasion_attack(self, model: nn.Module, x: torch.Tensor,
                       y: torch.Tensor, eps: float = None) -> torch.Tensor:
        """L-inf PGD evasion on fraud rows only: maximise loss w.r.t. the
        true label so fraud is pushed toward 'legitimate'.
        ||x_adv - x||_inf <= eps holds exactly."""
        eps = self.eps if eps is None else eps
        was_training = model.training
        model.eval()
        w     = self.budget.get_feature_weights(x.shape[1]).to(x.device)
        mask  = (y == 1).float().unsqueeze(1)
        alpha = self.step_mult * eps / self.n_steps
        eta   = torch.empty_like(x).uniform_(-eps, eps) * mask

        for _ in range(self.n_steps):
            eta.requires_grad_(True)
            logits = model(x + eta)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, y.float(), reduction='none') * mask.squeeze(1)).sum()
            grad, = torch.autograd.grad(loss, eta)
            with torch.no_grad():
                eta = eta + alpha * w.unsqueeze(0) * grad.sign()
                eta = torch.clamp(eta, -eps, eps) * mask
        if was_training:
            model.train()
        return (x + eta.detach()).detach()

    def poisoning_attack(self, x: torch.Tensor,
                         poison_rate: float = 0.05) -> torch.Tensor:
        """Uniform-noise feature poisoning on a random `poison_rate`
        subset of rows, bounded by ||.||_inf <= eps (weights now scale
        the noise inside the bound instead of exceeding it).
        NOTE: this is UNIFORM noise, not Gaussian, and it is NOT used
        inside adversarial training — only as an evaluation attack."""
        w        = self.budget.get_feature_weights(x.shape[1]).to(x.device)
        w        = w / w.max()
        n_poison = max(1, int(len(x) * poison_rate))
        idx      = torch.randperm(len(x))[:n_poison]
        noise    = (torch.rand(n_poison, x.shape[1], device=x.device) * 2 - 1)
        x_poison = x.clone()
        x_poison[idx] += noise * self.eps * w.unsqueeze(0)
        return x_poison


# ─────────────────────────────────────────────────────────────
# STACKELBERG ADVERSARIAL TRAINING
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _mean_bce(model, x, y):
    model.eval()
    return nn.functional.binary_cross_entropy_with_logits(
        model(x), y.float()).item()


def robustness_gap_signal(model, attacker, val_loader, ref_gap=None):
    """Stackelberg feedback signal in [-1, 1].

    rel_gap = (L_adv - L_clean) / L_clean on the validation set, where
    L_adv uses this round's PGD evasion attack. The signal compares it
    with the round-1 gap `ref_gap`:
        signal = clip(rel_gap / ref_gap - 1, -1, 1)
    Gap shrinking relative to round 1 (defender hardening) -> signal < 0
    -> damping rates gamma_1, gamma_2 shrink -> smaller eps and lambda. Gap growing
    (attacker gaining) -> signal > 0 -> larger eps and lambda.
    Verified failure modes of earlier choices: 1 - val_AUC moved eps only
    in the 5th decimal at AUC ~0.99; the absolute mapping 2*rel_gap - 1
    pinned the signal at -1 (rel_gap ~0.006) after round 1.
    Returns (signal, rel_gap); signal is 0.0 when ref_gap is None.
    """
    l_clean, l_adv, n = 0.0, 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xa = attacker.evasion_attack(model, xb, yb)
        l_clean += _mean_bce(model, xb, yb) * len(yb)
        l_adv   += _mean_bce(model, xa, yb) * len(yb)
        n += len(yb)
    l_clean, l_adv = l_clean / n, l_adv / n
    rel_gap = (l_adv - l_clean) / max(l_clean, 1e-8)
    if ref_gap is None or ref_gap <= 1e-8:
        return 0.0, rel_gap
    return float(np.clip(rel_gap / ref_gap - 1.0, -1.0, 1.0)), rel_gap


class StackelbergAdversarialTrainer:
    """
    Per round:
      1. Re-solve Hamiltonians with the previous round's robustness-gap
         signal -> eps (attacker budget), lambda (defender weight).
      2. Attacker: PGD evasion within eps.
      3. Defender: minimise (1-lambda)*L_clean + lambda*L_adv.
      4. Stop on validation-AUC early stopping ONLY (same rule for every
         configuration, so ablations/baselines are comparable). The round
         at which (eps, lambda) become stationary — relative change < tol
         for `stat_patience` rounds — is recorded as `budget_converged_round`
         (diagnostic, not a stopping rule).

    Why not the H29-field criterion: verified that H29's equilibrium
    field collapses to ~1.2e-5 for every coupling value tested
    (ground-state <sigma_x>, <sigma_y> are zero), so a field-based
    stop fires trivially after 4 rounds regardless of training. The
    field is still logged as a diagnostic.
    Why stationarity is not a stopping rule: a fixed-eps baseline is
    trivially stationary, so using it would stop baselines at round 4
    and confound every comparison (observed in testing).
    """
    def __init__(self, model, budget, adapt=True, lr=1e-4, epochs=30):
        self.model     = model
        self.budget    = budget
        self.attacker  = HamiltonianAttacker(budget)
        self.criterion = FocalLoss(gamma=HP['focal_gamma'], alpha=HP['focal_alpha'])
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.adapt     = adapt
        self.adv_weight = budget.derived_lambda()
        self.traj = []   # per-round log rows
        self.budget_converged_round = None

    def _defender_step(self, x_clean, x_adv, y):
        self.model.train()
        self.optimizer.zero_grad()
        loss = ((1 - self.adv_weight) * self.criterion(self.model(x_clean), y) +
                self.adv_weight * self.criterion(self.model(x_adv), y))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, loader):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xa = self.attacker.evasion_attack(self.model, xb, yb)
            total += self._defender_step(xb, xa, yb) * len(yb)
        self.scheduler.step()
        return total / len(loader.dataset)

    def train(self, train_loader, val_loader, epochs=30, patience=6,
              stat_patience=3, stat_tol=1e-3, ckpt_path=None):
        ckpt_path = ckpt_path or f"{MODEL_DIR}/adversarial_best.pt"
        print("\n" + "=" * 70)
        print("STACKELBERG ADVERSARIAL TRAINING (Hamiltonian-Constrained)")
        print("=" * 70)
        best_auc, patience_c, stable_c = 0.0, 0, 0
        prev_eps = prev_lam = None
        signal, rel_gap, ref_gap = None, float('nan'), None
        history = []

        for rnd in range(1, epochs + 1):
            self.budget.resolve(signal=signal if self.adapt else None)
            self.attacker.refresh()
            self.adv_weight = self.budget.derived_lambda()
            eps, lam = self.budget.eps_global, self.adv_weight

            train_loss = self.train_epoch(train_loader)
            vm = evaluate(self.model, val_loader)
            signal, rel_gap = robustness_gap_signal(self.model, self.attacker,
                                                    val_loader, ref_gap)
            if ref_gap is None:
                ref_gap = rel_gap
            vm.update(epoch=rnd, train_loss=train_loss, eps=eps, lam=lam,
                      rel_gap=rel_gap, signal=signal)
            history.append(vm)
            self.traj.append(dict(round=rnd, eps=eps, lam=lam, field=self.budget.field,
                                  val_auc=vm['auc'], rel_gap=rel_gap, next_signal=signal))

            print(f"Round {rnd:3d}/{epochs} | Adv Loss {train_loss:.4f} | "
                  f"Val AUC {vm['auc']:.4f} | Val F1 {vm['f1']:.4f} | "
                  f"eps {eps:.5f} | lambda {lam:.5f} | rel_gap {rel_gap:.4f}")

            if vm['auc'] > best_auc:
                best_auc, patience_c = vm['auc'], 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_c += 1
                if patience_c >= patience:
                    print(f"[Early Stop] no val-AUC improvement for {patience} rounds.")
                    break

            if prev_eps is not None and \
               abs(eps - prev_eps) / max(abs(prev_eps), 1e-12) < stat_tol and \
               abs(lam - prev_lam) / max(abs(prev_lam), 1e-12) < stat_tol:
                stable_c += 1
            else:
                stable_c = 0
            prev_eps, prev_lam = eps, lam
            if stable_c >= stat_patience and self.budget_converged_round is None:
                self.budget_converged_round = rnd - stat_patience
        return history

    def dump_trajectory(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.traj[0].keys()))
            w.writeheader(); w.writerows(self.traj)
        print(f"[LOG] trajectory -> {path}")


# ─────────────────────────────────────────────────────────────
# ROBUSTNESS EVALUATION (test set)
# ─────────────────────────────────────────────────────────────
def evaluate_under_attack(model, budget, test_loader, eps):
    """Evaluate on a test set where every fraud row is PGD-attacked with
    budget `eps`. Also returns the attack success rate: fraction of
    fraud rows correctly flagged on clean data that flip to 'legit'."""
    attacker = HamiltonianAttacker(budget)
    xs, xas, ys = [], [], []
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xas.append(attacker.evasion_attack(model, xb, yb, eps=eps).cpu())
        xs.append(xb.cpu()); ys.append(yb.cpu())
    X, XA, Y = torch.cat(xs), torch.cat(xas), torch.cat(ys)
    adv_loader = DataLoader(torch.utils.data.TensorDataset(XA, Y), batch_size=256)
    m = evaluate(model, adv_loader)
    with torch.no_grad():
        model.eval()
        p_clean = torch.sigmoid(model(X.to(DEVICE))).cpu()
        p_adv   = torch.sigmoid(model(XA.to(DEVICE))).cpu()
    caught = (Y == 1) & (p_clean >= 0.5)
    m['attack_success_rate'] = float(((p_adv < 0.5) & caught).sum() / max(caught.sum(), 1))
    m['linf'] = float((XA - X).abs().max())
    return m


def cost(cm, c_fn=100, c_fp=1):
    return int(c_fn * cm[1, 0] + c_fp * cm[0, 1])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
EVAL_EPS = (0.01, 0.03, 0.05, 0.10, 0.15)


def run_adversarial_training(use_h1=True, use_h2=True, use_eq29=True,
                             fixed_eps=None, fixed_lam=0.5, adapt=True,
                             tag="full", epochs=30, seed=42):
    """
    fixed_eps : float -> FixedPerturbationBudget(eps, fixed_lam) (plain PGD-AT)
    fixed_lam : blend weight for fixed_eps runs; fixed_lam=0.0 gives the
                CLEAN FINE-TUNE control (same optimiser/epochs, no adversarial
                loss) that isolates the effect of extra fine-tuning itself.
    adapt     : False -> Hamiltonians solved at base params every round
                (static budget; the 'no adaptation' ablation)
    tag       : filename suffix so runs never overwrite each other
    """
    set_seed(seed)
    print("=" * 70 + f"\nADVERSARIAL GAME — run tag: {tag}\n" + "=" * 70)
    train_loader, val_loader, test_loader, n_features = load_processed_data()

    if fixed_eps is not None:
        budget = FixedPerturbationBudget(eps=fixed_eps, lam=fixed_lam)
        print(f"[INFO] FixedPerturbationBudget(eps={fixed_eps}, lam={fixed_lam})")
    else:
        budget = HamiltonianPerturbationBudget(use_h1, use_h2, use_eq29)
        print(f"[INFO] Hamiltonian budget h1={use_h1} h2={use_h2} eq29={use_eq29} "
              f"adapt={adapt} initial eps={budget.eps_global:.6f} "
              f"lambda={budget.derived_lambda():.4f}")

    model = TabularTransformer(n_features=n_features, d_model=HP['d_model'],
                               n_heads=HP['n_heads'], n_layers=HP['n_layers'],
                               d_ffn=HP['d_ffn'], dropout=HP['dropout']).to(DEVICE)
    base_ckpt = f"{MODEL_DIR}/baseline_best.pt"
    if os.path.exists(base_ckpt):
        model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))
        print("[INFO] Defender initialised from baseline checkpoint.")
    else:
        print("[WARN] No baseline checkpoint — training from scratch.")

    ckpt = f"{MODEL_DIR}/adversarial_best_{tag}.pt"
    trainer = StackelbergAdversarialTrainer(model, budget, adapt=adapt, epochs=epochs)
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=6,
                            ckpt_path=ckpt)
    trainer.dump_trajectory(f"{LOG_DIR}/trajectory_{tag}.csv")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    clean = evaluate(model, test_loader)
    print(f"\n[{tag}] CLEAN  AUC {clean['auc']:.4f} F1 {clean['f1']:.4f} "
          f"P {clean['precision']:.4f} R {clean['recall']:.4f} "
          f"cost(100FN+FP) {cost(clean['cm'])}\n{clean['cm']}")

    row = dict(tag=tag, rounds=len(history),
               budget_converged_round=trainer.budget_converged_round,
               eps_min=min(t['eps'] for t in trainer.traj),
               eps_max=max(t['eps'] for t in trainer.traj),
               lam_min=min(t['lam'] for t in trainer.traj),
               lam_max=max(t['lam'] for t in trainer.traj),
               auc=clean['auc'], f1=clean['f1'],
               precision=clean['precision'], recall=clean['recall'],
               fn=int(clean['cm'][1, 0]), fp=int(clean['cm'][0, 1]), cost=cost(clean['cm']))
    # Identical attacker (base-parameter feature weights) for EVERY model,
    # so robustness rows are directly comparable.
    eval_budget = HamiltonianPerturbationBudget()
    for e in EVAL_EPS:
        am = evaluate_under_attack(model, eval_budget, test_loader, e)
        print(f"[{tag}] PGD eps={e:.2f}: AUC {am['auc']:.4f} F1 {am['f1']:.4f} "
              f"R {am['recall']:.4f} ASR {am['attack_success_rate']:.4f} "
              f"Linf {am['linf']:.4f}")
        row.update({f"auc@{e}": am['auc'], f"recall@{e}": am['recall'],
                    f"asr@{e}": am['attack_success_rate']})

    np.save(f"{MODEL_DIR}/adv_test_probs_{tag}.npy", clean['probs'])
    np.save(f"{MODEL_DIR}/adv_test_labels_{tag}.npy", clean['labels'])
    with open(f"{MODEL_DIR}/adv_history_{tag}.pkl", "wb") as f:
        pickle.dump(history, f)
    return model, row


def evaluate_baseline_row():
    """Same metrics row for the non-adversarial baseline checkpoint."""
    set_seed(42)
    _, _, test_loader, n_features = load_processed_data()
    model = TabularTransformer(n_features=n_features, d_model=HP['d_model'],
                               n_heads=HP['n_heads'], n_layers=HP['n_layers'],
                               d_ffn=HP['d_ffn'], dropout=HP['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/baseline_best.pt", map_location=DEVICE))
    clean = evaluate(model, test_loader)
    row = dict(tag="baseline", rounds=0, budget_converged_round=None,
               eps_min=None, eps_max=None, lam_min=None, lam_max=None,
               auc=clean['auc'], f1=clean['f1'],
               precision=clean['precision'], recall=clean['recall'],
               fn=int(clean['cm'][1, 0]), fp=int(clean['cm'][0, 1]), cost=cost(clean['cm']))
    budget = HamiltonianPerturbationBudget()
    for e in EVAL_EPS:
        am = evaluate_under_attack(model, budget, test_loader, e)
        row.update({f"auc@{e}": am['auc'], f"recall@{e}": am['recall'],
                    f"asr@{e}": am['attack_success_rate']})
    return row


def run_full_suite(epochs=30):
    """Every row the revised paper needs, in one CSV:
    baseline, full HAD-Stack, static (no-adapt) Hamiltonian, H1-only, H2-only
    (an Eq29-only row would be identical to static: Eq29 cannot adapt),
    and fixed-eps PGD-AT at 0.01 and at HAD-Stack's initial eps."""
    rows = [evaluate_baseline_row()]
    init_eps = HamiltonianPerturbationBudget().eps_global
    configs = [
        dict(tag="full"),
        dict(tag="static_hamiltonian", adapt=False),
        dict(tag="h1_only",   use_h2=False, use_eq29=False),
        dict(tag="h2_only",   use_h1=False, use_eq29=False),
        dict(tag="pgdat_eps0.01", fixed_eps=0.01),
        dict(tag=f"pgdat_eps{init_eps:.4f}", fixed_eps=init_eps),
        dict(tag="clean_finetune", fixed_eps=0.01, fixed_lam=0.0),
    ]
    for cfg in configs:
        _, row = run_adversarial_training(epochs=epochs, **cfg)
        rows.append(row)
    keys = list(rows[0].keys())
    with open(f"{LOG_DIR}/results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[DONE] {LOG_DIR}/results_summary.csv")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "suite", "fixed_eps"], default="full",
                    help="full: HAD-Stack only | suite: every table row | "
                         "fixed_eps: plain PGD-AT at --eps")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    if a.mode == "full":
        run_adversarial_training(epochs=a.epochs)
    elif a.mode == "suite":
        run_full_suite(epochs=a.epochs)
    else:
        run_adversarial_training(fixed_eps=a.eps, tag=f"pgdat_eps{a.eps}", epochs=a.epochs)
