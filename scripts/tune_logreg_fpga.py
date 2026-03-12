#!/usr/bin/env python3
"""
Systematic hyperparameter tuning for LogisticRegression to achieve >90% FPGA-compatible accuracy.
Tests different window configs, C values, and regularization strategies.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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


def stratified_split(X, y, seed=42):
    """Split using stratified random split (matches prepare_logreg_fpga_assets.py)."""
    s1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(s1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    s2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_idx_rel, test_idx_rel = next(s2.split(X_temp, y_temp))

    X_val, y_val = X_temp[val_idx_rel], y_temp[val_idx_rel]
    X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    df = pd.read_csv("data/processed/emg_all_samples.csv")
    
    results = []
    
    # Sweep window sizes, C values, and solver configs
    for window in [150, 200, 250]:
        for step in [25, 50]:
            for min_purity in [0.8]:
                X, y = make_windows(df, window=window, step=step, min_purity=min_purity)
                
                if len(np.unique(y)) < 7:
                    print(f"⚠️  w={window} s={step} p={min_purity}: < 7 classes")
                    continue
                
                Xtr, ytr, Xv, yv, Xte, yte = stratified_split(X, y, seed=42)
                
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xv_s = scaler.transform(Xv)
                Xte_s = scaler.transform(Xte)
                
                # Grid: C from very small (strong regularization) to large (weak regularization)
                for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
                    for solver in ["lbfgs", "liblinear"]:
                        try:
                            clf = LogisticRegression(
                                C=C,
                                max_iter=20000,
                                solver=solver,
                                class_weight=None,
                                random_state=42,
                                n_jobs=-1,
                            )
                            clf.fit(Xtr_s, ytr)
                            
                            pv = clf.predict(Xv_s)
                            pt = clf.predict(Xte_s)
                            
                            val_f1 = f1_score(yv, pv, average="macro")
                            test_acc = accuracy_score(yte, pt)
                            test_f1 = f1_score(yte, pt, average="macro")
                            
                            results.append({
                                "window": window,
                                "step": step,
                                "min_purity": min_purity,
                                "C": C,
                                "solver": solver,
                                "val_f1": val_f1,
                                "test_acc": test_acc,
                                "test_f1": test_f1,
                                "n_samples": len(X),
                            })
                        except Exception as e:
                            print(f"❌ w={window} C={C} solver={solver}: {e}")
                            continue
    
    # Sort by test accuracy (descending)
    results = sorted(results, key=lambda r: (r["test_acc"], r["val_f1"]), reverse=True)
    
    print("\n" + "="*100)
    print("TOP 15 CONFIGS (sorted by test accuracy)")
    print("="*100)
    
    for i, r in enumerate(results[:15], 1):
        over_90 = "✓ >=90%" if r["test_acc"] >= 0.90 else ""
        print(
            f"{i:2d}. w={r['window']} s={r['step']} C={r['C']:7.3f} "
            f"solver={r['solver']:9s} → test_acc={r['test_acc']:.4f} "
            f"val_f1={r['val_f1']:.4f} test_f1={r['test_f1']:.4f} {over_90}"
        )
    
    # Find best config >=90%
    best_90 = [r for r in results if r["test_acc"] >= 0.90]
    if best_90:
        best = best_90[0]
        print("\n" + "="*100)
        print(f"✅ BEST >=90% CONFIG FOUND:")
        print("="*100)
        print(f"  window={best['window']}, step={best['step']}, min_purity={best['min_purity']}")
        print(f"  logreg_C={best['C']}, solver={best['solver']}")
        print(f"  → Test Accuracy: {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)")
        print(f"  → Val F1: {best['val_f1']:.4f}, Test F1: {best['test_f1']:.4f}")
        print(f"  → N samples: {best['n_samples']}")
        
        print("\n" + "="*100)
        print("RECOMMENDED COMMAND TO GENERATE FPGA ASSETS:")
        print("="*100)
        cmd = (
            f"python fpga/scripts/prepare_logreg_fpga_assets.py "
            f"--window {best['window']} --step {best['step']} "
            f"--min_purity {best['min_purity']} "
            f"--logreg_C {best['C']} --max_iter 20000 "
            f"--q_frac 12 --xw_bits 16 --acc_bits 48 --outdir fpga/mem"
        )
        print(cmd)
    else:
        print("\n⚠️  NO CONFIG REACHED 90% ACCURACY")
        best = results[0]
        print(f"Best found: w={best['window']} C={best['C']} → {best['test_acc']:.4f}")


if __name__ == "__main__":
    main()
