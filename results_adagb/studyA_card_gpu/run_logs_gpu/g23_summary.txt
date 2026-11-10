# HAD-Stack verification on the ULB credit-card data


## Track A (FraudTransformer from scratch) — seeds (fixed split): seed1, seed2, seed3

Test set per unit: 42722 rows, 74 fraud. Device: cuda.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9702 ± 0.0000 | 75.7 ± 0.0 | 22.0 ± 0.0 | 18.0 ± 0.0 | 1822.0 ± 0.0 | 64.9 ± 0.0 | 43.2 ± 0.0 | 14.3 ± 0.0 | 42.9 ± 0.0 |
| HAD-Stack | 0.9714 ± 0.0008 | 77.5 ± 1.3 | 22.3 ± 1.2 | 16.7 ± 0.9 | 1689.0 ± 93.3 | 67.1 ± 1.7 | 57.7 ± 4.6 | 13.4 ± 1.0 | 25.7 ± 4.8 |

HAD-Stack above the baseline at both budgets in **2 of 3** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+2.3** points (95% CI [-0.9, +5.0]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+14.4** points (95% CI [+9.9, +19.8]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| seed1 | 8 | 0.5000→0.5738 (7) | 11.6→8.6 | 12.7→6.5 | +0.4 | 0.0 | 40 |
| seed2 | 9 | 0.5000→0.5784 (8) | 12.5→6.8 | 11.1→9.5 | +0.6 | 1.6 | 40 |
| seed3 | 8 | 0.5000→0.5611 (7) | 12.5→6.7 | 12.7→12.7 | +0.4 | 0.0 | 42 |

## Track A (FraudTransformer from scratch) — five-fold CV: fold0, fold1, fold2, fold3, fold4

Test set per unit: 56962 rows, 99 fraud. Device: cuda.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9677 ± 0.0086 | 71.3 ± 6.2 | 17.8 ± 2.1 | 28.2 ± 6.1 | 2837.8 ± 611.5 | 47.6 ± 9.2 | 27.2 ± 8.4 | 33.9 ± 7.6 | 62.6 ± 9.2 |
| HAD-Stack | 0.9747 ± 0.0046 | 78.7 ± 4.2 | 20.8 ± 4.9 | 21.0 ± 4.1 | 2120.8 ± 406.3 | 68.7 ± 3.7 | 50.0 ± 6.5 | 12.6 ± 3.9 | 36.3 ± 8.1 |

HAD-Stack above the baseline at both budgets in **5 of 5** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+21.1** points (95% CI [+17.5, +24.6]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+22.8** points (95% CI [+19.3, +26.4]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| fold0 | 8 | 0.5000→0.5502 (8) | 24.3→10.2 | 20.8→15.2 | +1.6 | 4.3 | 40 |
| fold1 | 16 | 0.5000→0.5815 (10) | 18.6→4.5 | 20.5→12.2 | +0.0 | 0.0 | 39 |
| fold2 | 8 | 0.5000→0.5412 (6) | 24.7→7.7 | 12.2→8.0 | +0.0 | 0.0 | 39 |
| fold3 | 8 | 0.5000→0.5608 (6) | 12.9→4.0 | 14.6→8.2 | +0.0 | 0.0 | 39 |
| fold4 | 11 | 0.5000→0.5890 (11) | 36.1→7.8 | 19.1→0.0 | -0.0 | 0.0 | 38 |
