| Split | Model | RMSE | wcRMSE@0.5 | wcRMSE@1.0 | broken@0.5 | broken@1.0 | vs HAD wcRMSE@0.5 [CI] | vs HAD wcRMSE@1.0 [CI] |
|---|---|---|---|---|---|---|---|---|
| fixed | Two-Head LSTM (undefended) | 8.163 | 13.14 | 14.15 | 90.9 | 94.5 |  |  |
| fixed | HAD-Stack | 6.268 | 11.97 | 13.71 | 87.8 | 91.7 |  |  |
| fixed | Linear ramp | 6.560 | 12.29 | 13.92 | 88.6 | 93.7 | +0.32 [+0.18, +0.48] | +0.21 [+0.07, +0.30] |
| fixed | Cosine ramp | 6.522 | 12.28 | 13.79 | 88.6 | 92.1 | +0.30 [+0.15, +0.45] | +0.08 [+0.03, +0.14] |
| fixed | Tuned constant | 4.683 | 8.29 | 12.61 | 79.9 | 88.2 | -3.69 [-5.12, -2.54] | -1.09 [-1.70, -0.61] |
| fixed | tuned eps 0.6428 (val rRMSE {'0.3686': 6.1139349937438965, '0.5000': 5.605128288269043, '0.6428': 3.3419241905212402}) | | | | | | | |
| fold1 | Two-Head LSTM (undefended) | 6.687 | 10.94 | 12.22 | 71.0 | 77.9 |  |  |
| fold1 | HAD-Stack | 4.813 | 10.62 | 11.80 | 65.6 | 72.5 |  |  |
| fold1 | Linear ramp | 5.173 | 10.61 | 11.95 | 65.6 | 74.8 | -0.00 [-0.02, +0.01] | +0.15 [-0.02, +0.44] |
| fold1 | Cosine ramp | 5.393 | 10.63 | 11.95 | 65.6 | 74.8 | +0.02 [+0.00, +0.03] | +0.15 [-0.02, +0.45] |
| fold1 | Tuned constant | 1.639 | 9.03 | 10.80 | 51.1 | 64.1 | -1.58 [-2.55, -0.57] | -1.00 [-2.04, -0.03] |
| fold1 | tuned eps 0.6428 (val rRMSE {'0.3686': 9.854514122009277, '0.5000': 9.212824821472168, '0.6428': 4.4208664894104}) | | | | | | | |
| fold2 | Two-Head LSTM (undefended) | 0.703 | 8.73 | 11.95 | 64.9 | 77.9 |  |  |
| fold2 | HAD-Stack | 0.716 | 2.17 | 7.62 | 7.6 | 44.3 |  |  |
| fold2 | Linear ramp | 0.718 | 3.36 | 8.71 | 22.9 | 48.9 | +1.19 [+0.44, +2.50] | +1.09 [+0.56, +1.90] |
| fold2 | Cosine ramp | 0.718 | 3.03 | 8.58 | 20.6 | 48.1 | +0.87 [+0.31, +1.90] | +0.96 [+0.49, +1.69] |
| fold2 | Tuned constant | 0.692 | 1.50 | 6.11 | 3.1 | 35.9 | -0.67 [-1.25, -0.03] | -1.51 [-2.24, -1.07] |
| fold2 | tuned eps 0.6428 (val rRMSE {'0.3686': 5.48471736907959, '0.5000': 3.3245389461517334, '0.6428': 0.9623400568962097}) | | | | | | | |
| fold3 | Two-Head LSTM (undefended) | 4.664 | 9.70 | 12.28 | 67.9 | 77.9 |  |  |
| fold3 | HAD-Stack | 1.704 | 6.41 | 10.60 | 41.2 | 69.5 |  |  |
| fold3 | Linear ramp | 3.685 | 8.25 | 11.36 | 44.3 | 69.5 | +1.85 [+0.75, +3.15] | +0.76 [-0.14, +1.70] |
| fold3 | Cosine ramp | 3.580 | 8.33 | 11.34 | 43.5 | 69.5 | +1.93 [+0.75, +3.27] | +0.74 [-0.17, +1.68] |
| fold3 | Tuned constant | 0.836 | 5.72 | 10.04 | 36.6 | 67.9 | -0.69 [-1.29, -0.20] | -0.56 [-1.03, -0.29] |
| fold3 | tuned eps 0.6428 (val rRMSE {'0.3686': 1.24861478805542, '0.5000': 0.6281371712684631, '0.6428': 0.4380872845649719}) | | | | | | | |
| fold4 | Two-Head LSTM (undefended) | 4.330 | 13.16 | 14.55 | 100.0 | 100.0 |  |  |
| fold4 | HAD-Stack | 1.504 | 11.23 | 13.87 | 92.4 | 95.4 |  |  |
| fold4 | Linear ramp | 1.718 | 11.75 | 14.21 | 93.1 | 98.5 | +0.52 [+0.29, +0.75] | +0.34 [+0.10, +0.48] |
| fold4 | Cosine ramp | 1.662 | 11.73 | 13.98 | 93.9 | 95.4 | +0.50 [+0.26, +0.73] | +0.11 [+0.04, +0.24] |
| fold4 | Tuned constant | 0.683 | 5.86 | 12.17 | 80.2 | 93.1 | -5.37 [-7.36, -3.58] | -1.70 [-2.95, -0.85] |
| fold4 | tuned eps 0.6428 (val rRMSE {'0.3686': 6.1139349937438965, '0.5000': 5.605128288269043, '0.6428': 3.3419241905212402}) | | | | | | | |
| fold5 | Two-Head LSTM (undefended) | 6.084 | 11.32 | 13.43 | 81.1 | 85.2 |  |  |
| fold5 | HAD-Stack | 5.427 | 9.58 | 13.21 | 77.9 | 82.0 |  |  |
| fold5 | Linear ramp | 5.128 | 9.20 | 13.12 | 76.2 | 80.3 | -0.38 [-0.62, -0.18] | -0.10 [-0.24, -0.01] |
| fold5 | Cosine ramp | 4.925 | 8.90 | 13.01 | 76.2 | 80.3 | -0.68 [-1.14, -0.35] | -0.20 [-0.34, -0.06] |
| fold5 | Tuned constant | 3.741 | 7.25 | 11.99 | 59.0 | 77.9 | -2.33 [-3.88, -1.38] | -1.22 [-1.73, -0.68] |
| fold5 | tuned eps 0.6428 (val rRMSE {'0.3686': 6.458596706390381, '0.5000': 5.41443395614624, '0.6428': 4.076925754547119}) | | | | | | | |

Counts over splits (schedule minus HAD-Stack; negative = schedule more robust):
  Linear ramp     eps 0.5: lower wcRMSE than HAD in 2/6, CI below 0 in 1, CI above 0 in 4; walk-forward mean 8.64 (HAD 8.00)
  Linear ramp     eps 1.0: lower wcRMSE than HAD in 1/6, CI below 0 in 1, CI above 0 in 3; walk-forward mean 11.87 (HAD 11.42)
  Cosine ramp     eps 0.5: lower wcRMSE than HAD in 1/6, CI below 0 in 1, CI above 0 in 5; walk-forward mean 8.53 (HAD 8.00)
  Cosine ramp     eps 1.0: lower wcRMSE than HAD in 1/6, CI below 0 in 1, CI above 0 in 3; walk-forward mean 11.77 (HAD 11.42)
  Tuned constant  eps 0.5: lower wcRMSE than HAD in 6/6, CI below 0 in 6, CI above 0 in 0; walk-forward mean 5.87 (HAD 8.00)
  Tuned constant  eps 1.0: lower wcRMSE than HAD in 6/6, CI below 0 in 6, CI above 0 in 0; walk-forward mean 10.22 (HAD 11.42)
