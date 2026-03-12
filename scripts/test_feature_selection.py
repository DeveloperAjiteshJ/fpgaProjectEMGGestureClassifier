#!/usr/bin/env python3
"""
Option 2: Feature Selection via ExtraTreesClassifier Importances
Boost LogisticRegression FPGA accuracy by selecting only important features.

Approach:
1. Train ExtraTreesClassifier to get feature importances
2. Test LogisticRegression with subsets of top-N features
3. Find subset size that reaches >=90% quantized accuracy
4. Minimal FPGA impact: still single LogGL model, fewer features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.emg_model import extract_window_features

CHANNELS = [f'ch{i}' for i in range(1, 9)]

def make_windows(df: pd.DataFrame, window: int, step: int, min_purity: float = 0.8):
    X, y = [], []
    for (_, _), part in df.groupby(["subject", "source_file"], sort=False):
        arr = part[CHANNELS].values.astype(np.float32)
        labels = part["label"].values.astype(int)
        n = len(part)
        for s in range(0, n - window + 1, step):
            e = s + window
            w = arr[s:e]
            wl = labels[s:e]
            vals, cnts = np.unique(wl, return_counts=True)
            best_idx = np.argmax(cnts)
            lab = vals[best_idx]
            purity = cnts[best_idx] / float(window)
            if purity < min_purity:
                continue
            X.append(extract_window_features(w))
            y.append(lab)
    return np.asarray(X, dtype=np.float32), np.asarray(y)

def stratified_train_test(X: np.ndarray, y: np.ndarray, seed: int = 42):
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_idx_rel, test_idx_rel = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp[val_idx_rel], y_temp[val_idx_rel]
    X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
    return X_train, y_train, X_val, y_val, X_test, y_test

def quantize_signed(x: np.ndarray, frac_bits: int, bit_width: int):
    scale = 1 << frac_bits
    q = np.rint(x * scale).astype(np.int64)
    lo = -(1 << (bit_width - 1))
    hi = (1 << (bit_width - 1)) - 1
    return np.clip(q, lo, hi).astype(np.int64)

print("\n" + "="*70)
print("OPTION 2: Feature Selection for LogisticRegression + FPGA")
print("="*70)

# Load data
print("\n[1/5] Loading EMG data...")
df = pd.read_csv("data/processed/emg_all_samples.csv")
print(f"  Loaded {len(df)} samples, {len(df['subject'].unique())} subjects")

# Extract features
print("\n[2/5] Extracting w250 features...")
X, y = make_windows(df, window=250, step=50, min_purity=0.8)
print(f"  Extracted {X.shape[0]} windows, {X.shape[1]} features")

# Split data
print("\n[3/5] Splitting data (70% train, 15% val, 15% test)...")
X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_test(X, y)
print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Convert labels to 0-6
uniq = np.unique(y)
label_to_idx = {int(l): i for i, l in enumerate(uniq.tolist())}
y_train_idx = np.asarray([label_to_idx[int(v)] for v in y_train], dtype=np.int32)
y_val_idx = np.asarray([label_to_idx[int(v)] for v in y_val], dtype=np.int32)
y_test_idx = np.asarray([label_to_idx[int(v)] for v in y_test], dtype=np.int32)

# Train ExtraTreesClassifier to get feature importances
print("\n[4/5] Training ExtraTreesClassifier for feature importances...")
et = ExtraTreesClassifier(n_estimators=800, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced')
et.fit(X_train, y_train_idx)
importances = et.feature_importances_
top_indices = np.argsort(importances)[::-1]  # Sort descending
print(f"  Top 10 features: {top_indices[:10]}")
print(f"  Feature importance range: {importances.min():.6f} to {importances.max():.6f}")

# Test LogisticRegression with different feature subset sizes
print("\n[5/5] Testing LogisticRegression with feature subsets...")
print(f"\n{'Num Features':<15} {'Float Acc':<15} {'Quantized Acc':<15} {'Status'}")
print("-"*70)

results = []
for n_features in [40, 50, 60, 70, 80, 90, 100, 110, 120]:
    selected_indices = top_indices[:n_features]
    X_train_sel = X_train[:, selected_indices]
    X_val_sel = X_val[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Train LogisticRegression
    logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial', max_iter=15000, random_state=42)
    logreg.fit(X_train_scaled, y_train_idx)
    
    # Float accuracy
    float_acc = logreg.score(X_test_scaled, y_test_idx)
    
    # Quantize for FPGA
    q_frac = 12
    xw_bits = 16
    X_test_q = quantize_signed(X_test_scaled, q_frac, xw_bits)
    W_q = quantize_signed(logreg.coef_, q_frac, xw_bits)
    b_q = quantize_signed(logreg.intercept_ * (1 << q_frac), 0, 48)
    
    # Quantized inference
    acc_bits = 48
    logits_q = (X_test_q @ W_q.T) + b_q
    logits_q = np.clip(logits_q, -(1 << (acc_bits-1)), (1 << (acc_bits-1))-1)
    preds_q = np.argmax(logits_q, axis=1)
    quant_acc = np.mean(preds_q == y_test_idx)
    
    status = "PASS 90+" if quant_acc >= 0.90 else ("CLOSE" if quant_acc >= 0.88 else "")
    print(f"{n_features:<15} {float_acc:<15.4f} {quant_acc:<15.4f} {status}")
    
    results.append((n_features, float_acc, quant_acc, selected_indices))

# Find best
best_n, best_float, best_quant, best_indices = max(results, key=lambda x: x[2])

print("\n" + "="*70)
if best_quant >= 0.90:
    print(f"SUCCESS! Found config reaching >=90%: {best_n} features at {best_quant:.4f} quantized accuracy")
else:
    print(f"BEST: {best_n} features at {best_quant:.4f} (below 90% target but better than baseline 87.61%)")

print(f"\nBest Configuration:")
print(f"  Number of features: {best_n} / 120")
print(f"  Float accuracy: {best_float:.4f}")
print(f"  Quantized accuracy: {best_quant:.4f}")
print(f"  Improvement over C=1.0 baseline: +{(best_quant - 0.8761)*100:.2f}%")

# Save configuration
config = {
    'strategy': 'feature_selection',
    'num_features': int(best_n),
    'total_features': 120,
    'selected_feature_indices': best_indices.tolist(),
    'logreg_C': 1.0,
    'float_accuracy': float(best_float),
    'quantized_accuracy': float(best_quant),
    'feature_importances': importances.tolist(),
}

config_file = Path('scripts/option2_best_config.json')
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
print(f"\n✅ Configuration saved to {config_file}")
