import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.emg_model import extract_window_features

CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]


def make_windows(df, window=200, step=50, purity=0.8):
    X, y, g = [], [], []
    for (subject, source_file), part in df.groupby(["subject", "source_file"], sort=False):
        arr = part[CHANNELS].values.astype(np.float32)
        labels = part["label"].values.astype(int)
        n = len(part)
        for s in range(0, n - window + 1, step):
            e = s + window
            wl = labels[s:e]
            vals, cnts = np.unique(wl, return_counts=True)
            idx = np.argmax(cnts)
            lab = vals[idx]
            frac = cnts[idx] / window
            if frac < purity:
                continue
            X.append(extract_window_features(arr[s:e]))
            y.append(lab)
            g.append(subject)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(g)


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


def split_random_train_val_test(X, y, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.70, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.50, random_state=seed + 1, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def smooth_preds(y, k=5):
    if k <= 1 or len(y) < k:
        return y
    out = y.copy()
    h = k // 2
    for i in range(len(y)):
        a = max(0, i - h)
        b = min(len(y), i + h + 1)
        vals, cnts = np.unique(y[a:b], return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


def run(df, window, step, purity, split_mode="group"):
    X, y, g = make_windows(df, window=window, step=step, purity=purity)
    if split_mode == "group":
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, g)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = split_random_train_val_test(X, y)

    best = (-1.0, None, None, None, None)
    for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)

        models = [
            ("lr", LogisticRegression(C=C, max_iter=8000, solver="lbfgs")),
            ("linsvc", LinearSVC(C=C, max_iter=10000)),
        ]
        for name, clf in models:
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xv)
            macro = f1_score(y_val, pred, average="macro")
            if macro > best[0]:
                best = (macro, C, name, scaler, clf)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    for C in [0.3, 1.0, 3.0, 10.0]:
        clf = SVC(C=C, kernel="rbf", gamma="scale")
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xv)
        macro = f1_score(y_val, pred, average="macro")
        if macro > best[0]:
            best = (macro, C, "rbf_svc", scaler, clf)

    _, best_C, best_model_name, scaler, clf = best
    test_pred = clf.predict(scaler.transform(X_test))
    test_acc = accuracy_score(y_test, test_pred)
    test_acc_s5 = accuracy_score(y_test, smooth_preds(test_pred, 5))

    return {
        "window": window,
        "step": step,
        "purity": purity,
        "n_samples": int(len(y)),
        "split_mode": split_mode,
        "best_model": best_model_name,
        "best_C": float(best_C),
        "val_macro_f1": float(best[0]),
        "test_acc": float(test_acc),
        "test_acc_smoothed": float(test_acc_s5),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/processed/emg_all_samples.csv")
    configs = [
        (200, 50, 0.70),
        (200, 50, 0.80),
        (200, 50, 0.90),
        (250, 50, 0.80),
        (150, 50, 0.80),
    ]
    for window, step, purity in configs:
        for split_mode in ["group", "random"]:
            r = run(df, window, step, purity, split_mode=split_mode)
            print(r)
