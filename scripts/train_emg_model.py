import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# make src import work when running as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.emg_model import extract_window_features, EMGLinearModel  # noqa: E402

CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]


def make_windows(df: pd.DataFrame, window: int, step: int, min_purity: float = 0.8):
    X, y, g = [], [], []

    for (subject, source_file), part in df.groupby(["subject", "source_file"], sort=False):
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
            g.append(subject)

    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(g)


def split_train_val_test(X, y, groups, split_mode: str = "group_subject", seed=42):
    if split_mode == "group_subject":
        gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

        X_train, y_train = X[train_idx], y[train_idx]
        X_temp, y_temp = X[temp_idx], y[temp_idx]
        g_temp = groups[temp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
        val_idx_rel, test_idx_rel = next(gss2.split(X_temp, y_temp, groups=g_temp))

        X_val, y_val = X_temp[val_idx_rel], y_temp[val_idx_rel]
        X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
        return X_train, y_train, X_val, y_val, X_test, y_test

    if split_mode == "stratified_random":
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        train_idx, temp_idx = next(sss1.split(X, y))

        X_train, y_train = X[train_idx], y[train_idx]
        X_temp, y_temp = X[temp_idx], y[temp_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
        val_idx_rel, test_idx_rel = next(sss2.split(X_temp, y_temp))

        X_val, y_val = X_temp[val_idx_rel], y_temp[val_idx_rel]
        X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
        return X_train, y_train, X_val, y_val, X_test, y_test

    raise ValueError(f"Unsupported split_mode: {split_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="./data/processed/emg_all_samples.csv")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--min_purity", type=float, default=0.8)
    ap.add_argument(
        "--split_mode",
        type=str,
        default="stratified_random",
        choices=["group_subject", "stratified_random"],
    )
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    X, y, groups = make_windows(df, window=args.window, step=args.step, min_purity=args.min_purity)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, groups, split_mode=args.split_mode
    )

    best_model_cfg = None
    best_val_f1 = -1.0

    model_grid = [
        {"model_type": "extra_trees", "n_estimators": 400, "max_depth": None},
        {"model_type": "extra_trees", "n_estimators": 800, "max_depth": None},
        {"model_type": "extra_trees", "n_estimators": 800, "max_depth": 24},
        {"model_type": "logreg", "C": 0.3, "max_iter": 6000},
        {"model_type": "logreg", "C": 1.0, "max_iter": 6000},
        {"model_type": "logreg", "C": 3.0, "max_iter": 6000},
    ]

    for cfg in model_grid:
        model = EMGLinearModel.create(**cfg)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_cfg = cfg

    best_model = EMGLinearModel.create(**best_model_cfg)
    best_model.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))

    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "scaler": best_model.scaler,
            "clf": best_model.clf,
            "channels": CHANNELS,
            "window": args.window,
            "step": args.step,
            "min_purity": args.min_purity,
            "best_model_cfg": best_model_cfg,
            "split_mode": args.split_mode,
        },
        outdir / "emg_linear_model.joblib",
    )

    np.save(outdir / "X_test_features.npy", X_test.astype(np.float32))
    np.save(outdir / "y_test.npy", y_test.astype(np.int32))

    metrics = {
        "best_model_cfg": best_model_cfg,
        "val_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "n_samples_after_filter": int(X.shape[0]),
        "window": int(args.window),
        "step": int(args.step),
        "min_purity": float(args.min_purity),
        "split_mode": args.split_mode,
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best model config: {best_model_cfg}")
    print(f"Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    print(f"\nSaved artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()