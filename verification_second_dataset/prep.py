"""Prepare the ULB credit-card fraud data (creditcard.csv) for both verification tracks.

Track A (scratch): the paper's protocol. Features Time, V1..V28, log(1+Amount) (30 features).
  * fixed split 70/15/15, stratified, random_state 42; StandardScaler fit on the training split
  * five-fold stratified CV (shuffle, random_state 42); per fold, 15% of the training part is held
    out for validation (stratified, random_state 42); scaler fit on the fold's training part
Track B (pretrained): the pre-trained model's own protocol (PHNPR/Credit-Card-Fraud-Detection,
  1_Data_Cleaning.ipynb): drop Time, drop duplicate rows, StandardScaler on all rows, 80/20 split with
  random_state 4. Its 20% test split is held out from the pre-trained model; validation is 15% of the
  80% part (stratified, random_state 42). Only the fixed split is used, because other folds would put
  the model's own training rows into the test set.

Usage:  python prep.py --csv /path/to/creditcard.csv
"""
import argparse, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def save(d, parts):
    os.makedirs(d, exist_ok=True)
    for n, a in parts.items():
        np.save(f"{d}/{n}.npy", a.astype(np.float32 if n[0] == "X" else np.int64))
    info = {k: [int(len(v)), int(v.sum())] for k, v in parts.items() if k[0] == "y"}
    json.dump(info, open(f"{d}/info.json", "w")); print(d, info)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    assert df.shape == (284807, 31) and int(df["Class"].sum()) == 492, "unexpected creditcard.csv"
    # ---------- Track A ----------
    feats = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feats].values.astype(np.float64); X[:, -1] = np.log1p(X[:, -1]); y = df["Class"].values
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.15 / 0.85, random_state=42, stratify=ytv)
    sc = StandardScaler().fit(Xtr)
    save(f"{a.out}/scratch_fixed", dict(X_train=sc.transform(Xtr), X_val=sc.transform(Xva), X_test=sc.transform(Xte),
                                        y_train=ytr, y_val=yva, y_test=yte))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        Xtr, Xva, ytr, yva = train_test_split(X[tr], y[tr], test_size=0.15, random_state=42, stratify=y[tr])
        sc = StandardScaler().fit(Xtr)
        save(f"{a.out}/scratch_fold{k}", dict(X_train=sc.transform(Xtr), X_val=sc.transform(Xva), X_test=sc.transform(X[te]),
                                              y_train=ytr, y_val=yva, y_test=y[te]))
    # ---------- Track B ----------
    d = df.drop(columns=["Time"]).drop_duplicates(keep="first")
    X = StandardScaler().fit_transform(d.drop(columns=["Class"])); y = d["Class"].values
    xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.8, random_state=4)
    xfit, xva, yfit, yva = train_test_split(xtr, ytr, test_size=0.15, random_state=42, stratify=ytr)
    save(f"{a.out}/pretrained_fixed", dict(X_train=xfit, X_val=xva, X_test=xte, y_train=yfit, y_val=yva, y_test=yte))

if __name__ == "__main__":
    main()
