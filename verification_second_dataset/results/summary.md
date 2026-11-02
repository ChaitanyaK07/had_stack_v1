# HAD-Stack verification on the ULB credit-card data


## Track A (FraudTransformer from scratch) — seeds (fixed split): seed1, seed2, seed3

Test set per unit: 42722 rows, 74 fraud. Device: cuda.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9771 ± 0.0000 | 73.0 ± 0.0 | 16.0 ± 0.0 | 20.0 ± 0.0 | 2016.0 ± 0.0 | 45.9 ± 0.0 | 10.8 ± 0.0 | 37.0 ± 0.0 | 85.2 ± 0.0 |
| HAD-Stack | 0.9729 ± 0.0011 | 81.1 ± 1.1 | 29.3 ± 0.5 | 14.0 ± 0.8 | 1429.3 ± 81.2 | 77.0 ± 0.0 | 65.3 ± 0.6 | 5.0 ± 1.3 | 19.4 ± 0.6 |

HAD-Stack above the baseline at both budgets in **3 of 3** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+31.1** points (95% CI [+24.8, +36.9]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+54.5** points (95% CI [+47.7, +61.3]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| seed1 | 14 | 0.5000→0.5954 (13) | 14.2→4.9 | 11.1→4.8 | +0.2 | 0.0 | 38 |
| seed2 | 13 | 0.5000→0.6003 (12) | 12.1→5.3 | 12.7→4.8 | -0.0 | 0.0 | 38 |
| seed3 | 12 | 0.5000→0.5960 (11) | 10.7→5.0 | 14.3→4.8 | +0.5 | 0.0 | 38 |

## Track A (FraudTransformer from scratch) — five-fold CV: fold0, fold1, fold2, fold3, fold4

Test set per unit: 56962 rows, 99 fraud. Device: cuda.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9680 ± 0.0064 | 73.5 ± 8.4 | 25.6 ± 17.0 | 26.0 ± 8.2 | 2625.6 ± 830.4 | 52.0 ± 14.2 | 32.5 ± 11.3 | 30.2 ± 13.4 | 56.7 ± 12.0 |
| HAD-Stack | 0.9751 ± 0.0081 | 78.2 ± 4.2 | 20.0 ± 5.4 | 21.4 ± 4.1 | 2160.0 ± 407.3 | 67.7 ± 9.5 | 48.6 ± 14.7 | 13.9 ± 8.8 | 38.4 ± 17.6 |

HAD-Stack above the baseline at both budgets in **5 of 5** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+15.7** points (95% CI [+12.2, +19.1]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+16.1** points (95% CI [+12.6, +19.7]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| fold0 | 7 | 0.5000→0.5150 (6) | 10.0→8.6 | 6.2→8.3 | -0.5 | 2.1 | 37 |
| fold1 | 21 | 0.5000→0.5879 (17) | 17.0→5.6 | 19.1→4.7 | -0.2 | 0.0 | 37 |
| fold2 | 8 | 0.5000→0.5582 (6) | 24.0→9.3 | 10.6→8.2 | +0.5 | 0.0 | 36 |
| fold3 | 19 | 0.5000→0.5897 (10) | 20.6→5.5 | 28.6→10.2 | -0.4 | 0.0 | 37 |
| fold4 | 8 | 0.5000→0.5495 (3) | 49.1→14.9 | 54.8→12.0 | -1.1 | 0.0 | 37 |

## Track B (pre-trained MLP) — seeds (fixed split): seed1, seed2, seed3

Test set per unit: 55133 rows, 87 fraud. Device: cpu.

| Model | AUC | Recall (%) | FP | FN | Cost | RR 0.5 (%) | RR 1.0 (%) | Broken 0.5 (%) | Broken 1.0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Pre-trained (no game) | 0.9784 ± 0.0000 | 83.9 ± 0.0 | 11.0 ± 0.0 | 14.0 ± 0.0 | 1411.0 ± 0.0 | 73.6 ± 0.0 | 62.1 ± 0.0 | 12.3 ± 0.0 | 26.0 ± 0.0 |
| HAD-Stack | 0.9798 ± 0.0000 | 83.9 ± 0.0 | 16.7 ± 0.9 | 14.0 ± 0.0 | 1416.7 ± 0.9 | 75.5 ± 0.5 | 65.9 ± 0.5 | 10.0 ± 0.6 | 21.5 ± 0.6 |

HAD-Stack above the baseline at both budgets in **3 of 3** units.
Paired bootstrap, robust recall HAD-Stack − baseline at ε=0.5: **+1.9** points (95% CI [+0.4, +3.8]).
Paired bootstrap, robust recall HAD-Stack − baseline at ε=1.0: **+3.8** points (95% CI [+1.5, +6.1]).

**Loop diagnostics of HAD-Stack (Definition 8), per unit:**

| Unit | Rounds | ε first→max (round) | Train flip first→last (%) | Strong first→last (%) | Fresh−stale, 2nd half | Exploit. gap last | s/round |
|---|---|---|---|---|---|---|---|
| seed1 | 9 | 0.5000→0.6553 (9) | 6.6→10.3 | 8.5→12.8 | +0.9 | 2.1 | 11 |
| seed2 | 8 | 0.5000→0.6545 (8) | 7.3→11.5 | 8.5→12.8 | +1.1 | 2.1 | 11 |
| seed3 | 8 | 0.5000→0.6545 (8) | 9.0→9.8 | 8.5→14.9 | +1.6 | 2.1 | 8 |
