## 1. Benchmark systems, exact gradient, Route A targets (5 seeds; H2 = H1 under the complex treatment)

| System | Model | Variant | n | clean fid | fid under attack | clean-adv fid | dE | dE/w_re | Rob | Atk % | QPSO/ALS | stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | fcvqc | game | 5 | 0.986 ± 0.005 | 0.998 ± 0.001 | 0.9860 ± 0.0040 | 0.0045 ± 0.0006 | 0.0341 ± 0.0048 | 7.41 ± 6.11 | 100.0 ± 0.0 | 129/351 | 5/5 [5, 6, 6, 6, 5] |
| H1 | fcvqc | noattack | 5 | 0.998 ± 0.001 | 0.976 ± 0.011 | 0.9790 ± 0.0100 | 0.0086 ± 0.0028 | 0.0647 ± 0.0211 | 0.12 ± 0.06 | 100.0 ± 0.0 | 0/0 |  |
| H1 | fcvqc | standard | 5 | 0.992 ± 0.002 | 0.992 ± 0.002 | 0.9999 ± 0.0001 | 0.0003 ± 0.0000 | 0.0024 ± 0.0003 | 1.01 ± 0.07 | 0.0 ± 0.0 | 0/0 |  |
| H1 | mlp | game | 3 | 0.992 ± 0.003 | 0.996 ± 0.002 | 0.9948 ± 0.0051 | 0.0023 ± 0.0015 | 0.0170 ± 0.0110 | 3.42 ± 2.79 | 66.7 ± 47.1 | 100/380 | 3/3 [6, 6, 6] |
| H1 | mlp | noattack | 3 | 0.996 ± 0.002 | 0.990 ± 0.004 | 0.9943 ± 0.0052 | 0.0033 ± 0.0024 | 0.0248 ± 0.0179 | 0.46 ± 0.29 | 66.7 ± 47.1 | 0/0 |  |
| HV | fcvqc | game | 5 | 1.000 ± 0.000 | 0.998 ± 0.000 | 0.9982 ± 0.0003 | 0.0041 ± 0.0005 | 0.0010 ± 0.0001 | 0.35 ± 0.07 | 0.0 ± 0.0 | 79/401 | 5/5 [5, 5, 5, 5, 6] |
| HV | fcvqc | noattack | 5 | 1.000 ± 0.000 | 0.986 ± 0.005 | 0.9859 ± 0.0055 | 0.0321 ± 0.0097 | 0.0080 ± 0.0024 | 0.02 ± 0.01 | 30.0 ± 40.0 | 0/0 |  |
| HV | fcvqc | standard | 5 | 0.995 ± 0.003 | 0.993 ± 0.004 | 0.9993 ± 0.0002 | 0.0038 ± 0.0019 | 0.0009 ± 0.0005 | 0.78 ± 0.12 | 0.0 ± 0.0 | 0/0 |  |
| HV | mlp | game | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.9998 ± 0.0001 | 0.0008 ± 0.0005 | 0.0002 ± 0.0001 | 0.37 ± 0.22 | 0.0 ± 0.0 | 50/430 | 3/3 [6, 6, 6] |
| HV | mlp | noattack | 3 | 1.000 ± 0.000 | 0.957 ± 0.012 | 0.9572 ± 0.0126 | 0.0994 ± 0.0237 | 0.0248 ± 0.0059 | 0.00 ± 0.00 | 100.0 ± 0.0 | 0/0 |  |

H1: game vs no attack, dE lower in 5/5, fidelity under attack higher in 5/5, clean-adv fidelity higher in 5/5
H1: game vs standard, dE lower in 0/5, fidelity under attack higher in 5/5

HV: game vs no attack, dE lower in 5/5, fidelity under attack higher in 5/5, clean-adv fidelity higher in 5/5
HV: game vs standard, dE lower in 2/5, fidelity under attack higher in 5/5

## 2. H1: fidelity under the ALS+QPSO attack vs budget (seeds 1-5; mean [min, max])

game      fid_adv : 0.02: 0.990 [0.985, 0.994]  0.05: 0.995 [0.993, 0.997]  0.1: 0.998 [0.995, 0.999]  0.15: 0.992 [0.989, 0.996]  0.2: 0.979 [0.972, 0.987]  0.3: 0.922 [0.887, 0.955]
game      fid_cc  : 0.02: 0.999 [0.999, 1.000]  0.05: 0.997 [0.995, 0.998]  0.1: 0.986 [0.979, 0.990]  0.15: 0.968 [0.950, 0.978]  0.2: 0.942 [0.909, 0.959]  0.3: 0.866 [0.802, 0.914]
game      decoder Jacobian norm 1.18 [0.77, 1.58]
standard  fid_adv : 0.02: 0.992 [0.990, 0.994]  0.05: 0.992 [0.990, 0.994]  0.1: 0.992 [0.990, 0.995]  0.15: 0.992 [0.989, 0.995]  0.2: 0.992 [0.989, 0.995]  0.3: 0.992 [0.989, 0.995]
standard  fid_cc  : 0.02: 1.000 [1.000, 1.000]  0.05: 1.000 [1.000, 1.000]  0.1: 1.000 [1.000, 1.000]  0.15: 1.000 [1.000, 1.000]  0.2: 1.000 [0.999, 1.000]  0.3: 0.999 [0.998, 1.000]
standard  decoder Jacobian norm 0.13 [0.11, 0.17]
noattack  fid_adv : 0.02: 0.997 [0.996, 0.998]  0.05: 0.992 [0.987, 0.995]  0.1: 0.976 [0.954, 0.984]  0.15: 0.949 [0.902, 0.968]  0.2: 0.911 [0.834, 0.947]  0.3: 0.806 [0.681, 0.887]
noattack  fid_cc  : 0.02: 0.999 [0.998, 0.999]  0.05: 0.995 [0.990, 0.997]  0.1: 0.979 [0.959, 0.986]  0.15: 0.953 [0.908, 0.968]  0.2: 0.914 [0.842, 0.947]  0.3: 0.809 [0.690, 0.886]
noattack  decoder Jacobian norm 1.50 [0.91, 1.99]

## 3. Ablations on H1 (seeds 1-3, 30 epochs, eps 0.1)

| # | Variant | n | Lc | La | Rob | dE | clean-adv fid | fid under attack | clean fid | Atk % |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | No attack (dz=0) | 3 | 0.003 | 0.035 | 0.13 | 0.0093 | 0.9767 | 0.973 | 0.998 | 100.0 |
| 2 | Standard Q-VAE (clean-trained) | 3 | 0.011 | 0.011 | 1.03 | 0.0003 | 0.9999 | 0.991 | 0.991 | 0.0 |
| 3 | ALS only | 3 | 0.024 | 0.003 | 9.77 | 0.0041 | 0.9842 | 0.998 | 0.982 | 100.0 |
| 4 | QPSO only | 3 | 0.019 | 0.004 | 7.41 | 0.0044 | 0.9860 | 0.997 | 0.986 | 100.0 |
| 5 | ALS+QPSO | 3 | 0.020 | 0.004 | 8.50 | 0.0044 | 0.9849 | 0.997 | 0.985 | 100.0 |
| 6 | Finite-difference follower | 1 | 0.016 | 0.003 | 4.65 | 0.0040 | 0.9859 | 0.998 | 0.986 | 100.0 |
| 7 | Earlier loss-level rule | 1 | 0.021 | 0.026 | 0.83 | 0.0036 | 0.9906 | 0.985 | 0.990 | 100.0 |
| 8 | Plain VQC | 3 | 0.005 | 0.005 | 1.03 | 0.0003 | 0.9999 | 0.995 | 0.995 | 0.0 |
| 9 | Min-max (simultaneous) | 3 | 0.020 | 0.004 | 8.35 | 0.0044 | 0.9862 | 0.997 | 0.986 | 100.0 |
| 10 | Classical MLP VAE (matched) | 3 | 0.009 | 0.004 | 3.42 | 0.0023 | 0.9948 | 0.996 | 0.992 | 66.7 |
| 11 | Payoff: reconstruction error | 3 | 0.006 | 0.007 | 0.90 | 0.0007 | 0.9993 | 0.994 | 0.995 | 31.2 |
| 12 | Payoff: ||dz||_2 | 3 | 0.039 | 0.154 | 0.25 | 0.0155 | 0.9641 | 0.886 | 0.972 | 100.0 |
| 13 | Payoff: random linear | 3 | 0.044 | 0.056 | 8.35 | 0.0060 | 0.9818 | 0.954 | 0.970 | 100.0 |

H_V payoff variants (seed 1):
  energy : dE 0.0037  fid under attack 0.9982  clean-adv 0.9984  clean 0.9999
  recon  : dE 0.0036  fid under attack 0.9984  clean-adv 0.9986  clean 0.9999
  norm   : dE 0.0542  fid under attack 0.9749  clean-adv 0.9900  clean 0.9939
  linear : dE 0.0723  fid under attack 0.9643  clean-adv 0.9905  clean 0.9912

## 4. FC-VQC (first sweep, real encoding) vs matched classical MLP VAE (5 seeds)

| d | FC-VQC dE na/game | MLP dE na/game | FC-VQC fid att na/game | MLP fid att na/game | FC-VQC clean fid | MLP clean fid | MLP hidden |
|---|---|---|---|---|---|---|---|
| 4 | 0.046 / 0.005 | 0.012 / 0.004 | 0.954 / 0.965 | 0.959 / 0.966 | 0.971 | 0.968 | |
| 9 | 0.048 / 0.012 | 0.027 / 0.008 | 0.933 / 0.969 | 0.944 / 0.950 | 0.982 | 0.958 | |
| 16 | 0.052 / 0.011 | 0.032 / 0.010 | 0.919 / 0.967 | 0.948 / 0.944 | 0.982 | 0.955 | |
| 25 | 0.066 / 0.015 | 0.055 / 0.009 | 0.900 / 0.932 | 0.911 / 0.945 | 0.952 | 0.957 | |

Paired per seed (game-trained models, eps 0.1): FC-VQC vs MLP
  d=4: FC-VQC higher fidelity under attack in 3/5, lower dE in 0/5, higher clean fidelity in 3/5; MLP parameters per VAE (both networks) 98
  d=9: FC-VQC higher fidelity under attack in 4/5, lower dE in 0/5, higher clean fidelity in 4/5; MLP parameters per VAE (both networks) 208
  d=16: FC-VQC higher fidelity under attack in 3/5, lower dE in 2/5, higher clean fidelity in 5/5; MLP parameters per VAE (both networks) 362
  d=25: FC-VQC higher fidelity under attack in 2/5, lower dE in 0/5, higher clean fidelity in 2/5; MLP parameters per VAE (both networks) 560
  H1: FC-VQC higher fidelity under attack in 2/3, lower dE in 1/3; circuit params 216, MLP params 208
  HV: FC-VQC higher fidelity under attack in 0/3, lower dE in 0/3; circuit params 864, MLP params 802

Reproduction of the first sweep (d=4, seed 1) by studyb_v2.py: max |dE_epoch difference| = 2.50e-16

## 5. Non-Hermitian sweep (complex encoding; r = |H_anti|_F/|H_herm|_F; 0 = Hermitian arm; 0.105 = H_V; 2.531 = H1)

| d | r | n | clean fid na/game | dE na/game | fid under attack na/game | dE lower | fid higher | Rob game | stationary |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 5 | 0.995 / 0.970 | 0.0349 / 0.0065 | 0.967 / 0.965 | 5/5 | 3/5 | 0.78 | 5/5 |
| 4 | 0.105 | 5 | 0.995 / 0.978 | 0.0379 / 0.0071 | 0.962 / 0.971 | 5/5 | 5/5 | 0.76 | 5/5 |
| 4 | 2.531 | 5 | 0.989 / 0.979 | 0.0631 / 0.0437 | 0.971 / 0.987 | 5/5 | 5/5 | 1.54 | 5/5 |
| 9 | 0 | 5 | 0.976 / 0.949 | 0.0424 / 0.0127 | 0.924 / 0.936 | 5/5 | 3/5 | 0.79 | 5/5 |
| 9 | 0.105 | 5 | 0.974 / 0.924 | 0.0419 / 0.0116 | 0.929 / 0.910 | 5/5 | 1/5 | 0.81 | 5/5 |
| 9 | 2.531 | 5 | 0.954 / 0.951 | 0.0563 / 0.0281 | 0.913 / 0.950 | 5/5 | 5/5 | 1.12 | 5/5 |
| 16 | 0 | 5 | 0.965 / 0.918 | 0.0396 / 0.0151 | 0.921 / 0.902 | 5/5 | 3/5 | 0.82 | 5/5 |
| 16 | 0.105 | 5 | 0.952 / 0.907 | 0.0508 / 0.0130 | 0.897 / 0.894 | 5/5 | 4/5 | 0.85 | 5/5 |
| 16 | 2.531 | 5 | 0.881 / 0.864 | 0.0453 / 0.0292 | 0.861 / 0.871 | 5/5 | 4/5 | 1.05 | 5/5 |

d = 25 with the complex encoding (D = 64, 7 runs): clean fidelity 0.433-0.609; not fitted in 30 epochs, excluded.
Family variation H1 d=4: min pairwise fidelity between its 16 targets 0.9484; r = 2.531
Family variation HV d=16: min pairwise fidelity between its 16 targets 0.9997; r = 0.105
Family variation goe d=4: min pairwise fidelity between its 16 targets 0.9198; r = 0.000
Family variation goe d=25: min pairwise fidelity between its 16 targets 0.0000; r = 0.000

## 6. Energy-fidelity bound |dE| <= w_re sqrt(1 - F_cc): 19776 (dE, F) pairs, max ratio 0.879
