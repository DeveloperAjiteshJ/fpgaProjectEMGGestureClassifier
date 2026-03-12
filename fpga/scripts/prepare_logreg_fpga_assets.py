import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Keep channel order exactly aligned with training code.
CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]


def zc_count(x: np.ndarray, thr: float = 1e-4) -> float:
    x1 = x[:-1]
    x2 = x[1:]
    return float(np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) >= thr)))


def ssc_count(x: np.ndarray, thr: float = 1e-4) -> float:
    x_prev = x[:-2]
    x_mid = x[1:-1]
    x_next = x[2:]
    return float(
        np.sum(
            (((x_mid - x_prev) * (x_mid - x_next)) > 0)
            & ((np.abs(x_mid - x_prev) >= thr) | (np.abs(x_mid - x_next) >= thr))
        )
    )


def extract_window_features(window_2d: np.ndarray) -> np.ndarray:
    feats = []
    for ch in range(window_2d.shape[1]):
        s = window_2d[:, ch]
        mav = np.mean(np.abs(s))
        rms = np.sqrt(np.mean(s ** 2))
        wl = np.sum(np.abs(np.diff(s)))
        zc = zc_count(s)
        ssc = ssc_count(s)
        var = np.var(s)
        iemg = np.sum(np.abs(s))
        mean = np.mean(s)
        std = np.std(s)
        min_v = np.min(s)
        max_v = np.max(s)
        ptp = max_v - min_v
        # Use quantile with linear interpolation instead of percentile to avoid hanging
        s_sorted = np.sort(s)
        n = len(s_sorted)
        p25_idx = int(np.ceil(0.25 * n)) - 1
        p50_idx = int(np.ceil(0.50 * n)) - 1
        p75_idx = int(np.ceil(0.75 * n)) - 1
        p25 = s_sorted[max(0, p25_idx)]
        p50 = s_sorted[max(0, p50_idx)]
        p75 = s_sorted[max(0, p75_idx)]
        feats.extend([mav, rms, wl, zc, ssc, var, iemg, mean, std, min_v, max_v, ptp, p25, p50, p75])
    return np.asarray(feats, dtype=np.float32)


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


def int_to_twos_hex(v: int, bit_width: int) -> str:
    mask = (1 << bit_width) - 1
    vv = v & mask
    hex_width = (bit_width + 3) // 4
    return format(vv, f"0{hex_width}x")


def write_mem_1d(arr: np.ndarray, file_path: Path, bit_width: int):
    with file_path.open("w", encoding="ascii") as f:
        for v in arr.reshape(-1):
            f.write(int_to_twos_hex(int(v), bit_width) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="data/processed/emg_all_samples.csv")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--min_purity", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logreg_C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=20000)
    ap.add_argument("--q_frac", type=int, default=12)
    ap.add_argument("--xw_bits", type=int, default=16)
    ap.add_argument("--acc_bits", type=int, default=48)
    ap.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="If > 0, keep only the first N test samples when exporting FPGA assets.",
    )
    ap.add_argument("--outdir", type=str, default="fpga/mem")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    X, y = make_windows(df, window=args.window, step=args.step, min_purity=args.min_purity)

    # Convert labels {1..7} into {0..6} for clean hardware indexing.
    uniq = np.unique(y)
    label_to_idx = {int(l): i for i, l in enumerate(uniq.tolist())}
    y_idx = np.asarray([label_to_idx[int(v)] for v in y], dtype=np.int32)

    X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_test(X, y_idx, seed=args.seed)

    if args.max_test_samples > 0 and X_test.shape[0] > args.max_test_samples:
        X_test = X_test[: args.max_test_samples]
        y_test = y_test[: args.max_test_samples]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    try:
        clf = LogisticRegression(
            C=args.logreg_C,
            max_iter=args.max_iter,
            solver="lbfgs",
            multi_class="multinomial",
        )
    except TypeError:
        clf = LogisticRegression(
            C=args.logreg_C,
            max_iter=args.max_iter,
            solver="lbfgs",
        )
    clf.fit(Xtr, y_train)

    val_pred = clf.predict(Xv)
    test_pred = clf.predict(Xte)

    # Keep inference in standardized domain to avoid large dynamic-range clipping.
    # FPGA BRAM stores precomputed standardized test features.
    W_std = clf.coef_.astype(np.float64)
    b_std = clf.intercept_.astype(np.float64)

    Xq = quantize_signed(Xte.astype(np.float64), args.q_frac, args.xw_bits)
    Wq = quantize_signed(W_std, args.q_frac, args.xw_bits)

    # Bias is aligned to the product scale (2*q_frac) because x_q * w_q uses that scale.
    bq = quantize_signed(b_std, 2 * args.q_frac, args.acc_bits)

    # Integer-domain golden prediction for bit-true RTL comparison.
    scores_q = Xq.astype(np.int64) @ Wq.T.astype(np.int64)
    scores_q = scores_q + bq[None, :]
    y_pred_q = np.argmax(scores_q, axis=1).astype(np.int32)

    acc_float = accuracy_score(y_test, test_pred)
    f1_float = f1_score(y_test, test_pred, average="macro")
    acc_q = accuracy_score(y_test, y_pred_q)
    f1_q = f1_score(y_test, y_pred_q, average="macro")

    np.save(outdir / "X_test_float.npy", X_test.astype(np.float32))
    np.save(outdir / "X_test_std_float.npy", Xte.astype(np.float32))
    np.save(outdir / "y_test_idx.npy", y_test.astype(np.int32))
    np.save(outdir / "y_pred_float.npy", test_pred.astype(np.int32))
    np.save(outdir / "y_pred_q.npy", y_pred_q.astype(np.int32))
    np.save(outdir / "W_std_float.npy", W_std.astype(np.float32))
    np.save(outdir / "b_std_float.npy", b_std.astype(np.float32))

    write_mem_1d(Xq, outdir / "x_test_q.mem", args.xw_bits)
    write_mem_1d(y_test.astype(np.int64), outdir / "y_test_idx.mem", 8)
    write_mem_1d(y_pred_q.astype(np.int64), outdir / "golden_pred_idx.mem", 8)
    write_mem_1d(Wq, outdir / "w_q.mem", args.xw_bits)
    write_mem_1d(bq, outdir / "b_q.mem", args.acc_bits)

    meta = {
        "part": "xc7a100tcsg324-1",
        "window": int(args.window),
        "step": int(args.step),
        "min_purity": float(args.min_purity),
        "seed": int(args.seed),
        "n_samples_total": int(X.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "n_classes": int(len(uniq)),
        "labels_original": [int(x) for x in uniq.tolist()],
        "label_to_idx": {str(k): int(v) for k, v in label_to_idx.items()},
        "q_frac": int(args.q_frac),
        "xw_bits": int(args.xw_bits),
        "acc_bits": int(args.acc_bits),
        "float_model": {
            "type": "logreg_multinomial",
            "C": float(args.logreg_C),
            "max_iter": int(args.max_iter),
            "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
            "test_accuracy": float(acc_float),
            "test_macro_f1": float(f1_float),
        },
        "quantized_model": {
            "test_accuracy": float(acc_q),
            "test_macro_f1": float(f1_q),
        },
        "memory_layout": {
            "x_test_q.mem": "sample-major flattened standardized features: sample0_feat0..featN-1, sample1_feat0..",
            "w_q.mem": "class-major flattened: class0_feat0..featN-1, class1_feat0..",
            "b_q.mem": "class-major biases, same class order as w_q.mem",
            "y_test_idx.mem": "one 0-based label index per sample",
            "golden_pred_idx.mem": "bit-true expected prediction index per sample",
        },
    }

    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Prepared FPGA assets:")
    print(f"  outdir={outdir.resolve()}")
    print(f"  n_samples_test={X_test.shape[0]}, n_features={X_test.shape[1]}, n_classes={len(uniq)}")
    print(f"  float test acc={acc_float:.6f}, float test macro_f1={f1_float:.6f}")
    print(f"  quant test acc={acc_q:.6f}, quant test macro_f1={f1_q:.6f}")


if __name__ == "__main__":
    main()
