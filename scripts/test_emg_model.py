import argparse
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./artifacts/emg_linear_model.joblib")
    ap.add_argument("--x_test", type=str, default="./artifacts/X_test_features.npy")
    ap.add_argument("--y_test", type=str, default="./artifacts/y_test.npy")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    model_pkg = joblib.load(args.model)
    scaler = model_pkg["scaler"]
    clf = model_pkg["clf"]
    best_model_cfg = model_pkg.get("best_model_cfg", None)

    X_test = np.load(args.x_test).astype(np.float32)
    y_test = np.load(args.y_test).astype(np.int32)

    if scaler is None:
        y_pred = clf.predict(X_test)
    else:
        y_pred = clf.predict(scaler.transform(X_test))

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    if best_model_cfg is not None:
        print(f"Model Config: {best_model_cfg}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outdir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "best_model_cfg": best_model_cfg,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {outdir / 'confusion_matrix.csv'}")
    print(f"Saved: {outdir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()

