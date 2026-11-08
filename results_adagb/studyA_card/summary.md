# HAD-Stack verification on the ULB credit-card data


## Track A (FraudTransformer from scratch) — seeds (fixed split): seed1, seed2, seed3

Test set per unit: 42722 rows, 74 fraud. Device: cpu.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9604 ± 0.0000 | 77.0 ± 0.0 | 24.0 ± 0.0 | 17.0 ± 0.0 | 1724.0 ± 0.0 | 64.9 ± 0.0 | 54.1 ± 0.0 | 15.8 ± 0.0 | 29.8 ± 0.0 |
| HAD-Stack | 0.9630 ± 0.0006 | 78.4 ± 1.9 | 25.7 ± 2.5 | 16.0 ± 1.4 | 1625.7 ± 139.1 | 71.2 ± 4.2 | 64.4 ± 3.2 | 9.3 ± 3.1 | 17.9 ± 2.0 |

HAD-Stack above the baseline at both budgets in **3 of 3** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+6.3** points (95% CI [+3.2, +9.5]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+10.4** points (95% CI [+6.8, +14.9]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| seed1 | 8 | 0.5000→0.5742 (8) | 9.7→5.1 | 18.3→8.3 | +0.0 | 0.0 | 334 |
| seed2 | 7 | 0.5000→0.5663 (7) | 11.5→6.5 | 20.6→10.0 | +1.6 | 0.0 | 367 |
| seed3 | 16 | 0.5000→0.5860 (15) | 12.3→5.0 | 14.5→3.2 | +0.6 | 0.0 | 321 |

## Track A (FraudTransformer from scratch) — five-fold CV: fold0, fold1, fold2, fold3, fold4

Test set per unit: 56962 rows, 99 fraud. Device: cpu.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9700 ± 0.0071 | 75.6 ± 5.3 | 39.2 ± 44.5 | 24.0 ± 5.2 | 2439.2 ± 551.4 | 54.7 ± 8.0 | 36.6 ± 3.7 | 27.8 ± 7.7 | 51.6 ± 3.8 |
| HAD-Stack | 0.9740 ± 0.0042 | 80.3 ± 0.5 | 23.2 ± 4.7 | 19.4 ± 0.5 | 1963.2 ± 48.2 | 75.0 ± 2.2 | 58.4 ± 7.6 | 6.6 ± 2.9 | 27.3 ± 9.5 |

HAD-Stack above the baseline at both budgets in **5 of 5** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+20.3** points (95% CI [+16.7, +23.8]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+21.7** points (95% CI [+17.9, +25.4]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| fold0 | 11 | 0.5000→0.5556 (7) | 16.7→8.6 | 14.6→4.2 | +0.3 | 0.0 | 269 |
| fold1 | 10 | 0.5000→0.5615 (7) | 16.7→6.3 | 21.7→6.8 | +0.0 | 0.0 | 269 |
| fold2 | 18 | 0.5000→0.5751 (16) | 17.8→4.4 | 6.2→4.2 | +0.5 | 0.0 | 272 |
| fold3 | 11 | 0.5000→0.5493 (7) | 11.0→3.2 | 20.4→10.2 | -0.3 | 0.0 | 290 |
| fold4 | 16 | 0.5000→0.5653 (15) | 33.0→10.4 | 11.5→3.9 | -0.2 | 0.0 | 300 |
