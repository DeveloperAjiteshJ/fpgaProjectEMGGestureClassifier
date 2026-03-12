import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.emg_model import extract_window_features

CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]


def make_windows(df: pd.DataFrame, window: int, step: int):
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
            lab = vals[np.argmax(cnts)]
            X.append(extract_window_features(w))
            y.append(lab)
            g.append(subject)
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(g)


def split_train_val_test(X, y, groups, seed=42):
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


def evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, use_scaler=False):
    if use_scaler:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)
        Xte = scaler.transform(X_test)
    else:
        Xtr, Xv, Xte = X_train, X_val, X_test

    model.fit(Xtr, y_train)
    val_pred = model.predict(Xv)
    test_pred = model.predict(Xte)

    return {
        "name": name,
        "val_f1": f1_score(y_val, val_pred, average="macro"),
        "test_acc": accuracy_score(y_test, test_pred),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/processed/emg_all_samples.csv")

    for window, step in [(150, 75), (200, 100), (250, 125), (200, 50), (250, 50)]:
        print(f"\n=== window={window}, step={step} ===")
        X, y, groups = make_windows(df, window=window, step=step)
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, groups)

        results = []
        for C in [0.1, 1.0, 3.0, 10.0]:
            results.append(
                evaluate(
                    f"LogReg C={C}",
                    LogisticRegression(C=C, max_iter=6000, solver="lbfgs"),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    use_scaler=True,
                )
            )

        for C in [0.3, 1.0, 3.0]:
            results.append(
                evaluate(
                    f"RBF-SVC C={C}",
                    SVC(C=C, kernel="rbf", gamma="scale"),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    use_scaler=True,
                )
            )

        for n in [200, 400, 800]:
            results.append(
                evaluate(
                    f"RF n={n}",
                    RandomForestClassifier(
                        n_estimators=n,
                        max_depth=None,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    use_scaler=False,
                )
            )
            results.append(
                evaluate(
                    f"ExtraTrees n={n}",
                    ExtraTreesClassifier(
                        n_estimators=n,
                        max_depth=None,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    use_scaler=False,
                )
            )

        results.append(
            evaluate(
                "LDA",
                LinearDiscriminantAnalysis(),
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                use_scaler=True,
            )
        )

        results = sorted(results, key=lambda r: (r["val_f1"], r["test_acc"]), reverse=True)
        for r in results[:8]:
            print(f"{r['name']:<18} val_f1={r['val_f1']:.4f} test_acc={r['test_acc']:.4f}")
