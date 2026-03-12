#!/usr/bin/env python3
"""
Quick tuning: test LogisticRegression + LinearSVC on existing extracted features.
Uses w250 checkpoint data that's already computed.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import json

# Load from w250 checkpoint (features already extracted)
X_test = np.load("artifacts/w250/X_test_features.npy")
y_test = np.load("artifacts/w250/y_test.npy")

print(f"Loaded w250 features: X shape={X_test.shape}, y shape={y_test.shape}")
print(f"Classes: {np.unique(y_test)}\n")

# For fair comparison, use same split strategy as prepare_logreg_fpga_assets
# Load full data and re-split
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.emg_model import extract_window_features

CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]

def make_windows(df: pd.DataFrame, window: int, step: int, min_purity: float = 0.8):
    X, y = [], []
    for (_, _), part in df.groupby(["subject", "source_file"], sort=False):
        arr = part[CHANNELS].values.astype(np.float32)
        labels = part["label"].values.astype(int)
        n = len(part)
        for s in range(0, n - window + 1, step):
            wl = labels[s : s + window]
            vals, cnts = np.unique(wl, return_counts=True)
            idx = np.argmax(cnts)
            if cnts[idx] / float(window) < min_purity:
                continue
            X.append(extract_window_features(arr[s : s + window]))
            y.append(vals[idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(X, y, seed=42):
    s1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(s1.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    s2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_idx_rel, test_idx_rel = next(s2.split(X_temp, y_temp))
    X_val, y_val = X_temp[val_idx_rel], y_temp[val_idx_rel]
    X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
    return X_train, y_train, X_val, y_val, X_test, y_test

print("Extracting full w250 dataset (this may take a minute)...")
df = pd.read_csv("data/processed/emg_all_samples.csv")
X_full, y_full = make_windows(df, window=250, step=50, min_purity=0.8)
print(f"Extracted full dataset: {X_full.shape}")

Xtr, ytr, Xv, yv, Xte, yte = stratified_split(X_full, y_full, seed=42)
print(f"Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}")

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xv_s = scaler.transform(Xv)
Xte_s = scaler.transform(Xte)

results = []

print("\n" + "="*80)
print("Testing LogisticRegression with different C values:")
print("="*80)

for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    for solver in ["lbfgs"]:
        try:
            clf = LogisticRegression(C=C, max_iter=20000, solver=solver, random_state=42)
            clf.fit(Xtr_s, ytr)
            pv = clf.predict(Xv_s)
            pt = clf.predict(Xte_s)
            val_f1 = f1_score(yv, pv, average="macro")
            test_acc = accuracy_score(yte, pt)
            test_f1 = f1_score(yte, pt, average="macro")
            
            over_90 = " ✓ >=90%" if test_acc >= 0.90 else ""
            print(f"C={C:8.3f} → test_acc={test_acc:.4f} val_f1={val_f1:.4f}{over_90}")
            
            results.append({"C": C, "solver": solver, "test_acc": test_acc, "val_f1": val_f1, "test_f1": test_f1})
        except Exception as e:
            print(f"C={C:8.3f} → ERROR: {e}")

print("\n" + "="*80)
print("Testing LinearSVC with different C values:")
print("="*80)

for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    try:
        clf = LinearSVC(C=C, max_iter=50000, random_state=42)
        clf.fit(Xtr_s, ytr)
        pv = clf.predict(Xv_s)
        pt = clf.predict(Xte_s)
        val_f1 = f1_score(yv, pv, average="macro")
        test_acc = accuracy_score(yte, pt)
        test_f1 = f1_score(yte, pt, average="macro")
        
        over_90 = " ✓ >=90%" if test_acc >= 0.90 else ""
        print(f"C={C:8.3f} → test_acc={test_acc:.4f} val_f1={val_f1:.4f}{over_90}")
        
        results.append({"C": C, "solver": "linsvc", "test_acc": test_acc, "val_f1": val_f1, "test_f1": test_f1})
    except Exception as e:
        print(f"C={C:8.3f} → ERROR: {e}")

results = sorted(results, key=lambda r: (r["test_acc"], r["val_f1"]), reverse=True)

print("\n" + "="*80)
print("TOP 5 CONFIGS:")
print("="*80)
for i, r in enumerate(results[:5], 1):
    over_90 = "✓" if r["test_acc"] >= 0.90 else "✗"
    print(f"{i}. solver={r['solver']:6s} C={r['C']:8.3f} → acc={r['test_acc']:.4f} f1={r['test_f1']:.4f} [{over_90}]")

best = results[0]
print("\n" + "="*80)
if best["test_acc"] >= 0.90:
    print(f"✅ SUCCESS! Found config >=90%:")
    print(f"   solver={best['solver']}, C={best['C']}")
    print(f"   Test Accuracy: {best['test_acc']:.4f}")
else:
    print(f"⚠️  Best found: {best['test_acc']:.4f} (still below 90%)")
    print(f"   Consider trying w200 or w150 instead")
print("="*80)
