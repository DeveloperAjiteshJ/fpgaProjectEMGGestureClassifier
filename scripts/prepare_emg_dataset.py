import argparse
from pathlib import Path
import pandas as pd

COLS_10 = ["time", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "label"]
COLS_9 = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "label"]


def read_raw_file(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] == 10:
        df.columns = COLS_10
    elif df.shape[1] == 9:
        df.columns = COLS_9
        df.insert(0, "time", range(len(df)))
    else:
        raise ValueError(f"{fp}: unexpected column count={df.shape[1]}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="./EMG_data_for_gestures-master")
    ap.add_argument("--out_csv", type=str, default="./data/processed/emg_all_samples.csv")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    txt_files = sorted(root.glob("*/*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {root}")

    parts = []
    for fp in txt_files:
        subject = fp.parent.name
        session_token = fp.stem.split("_")[0]  # usually "1" or "2"

        df = read_raw_file(fp)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)

        df["subject"] = int(subject) if subject.isdigit() else subject
        df["session"] = int(session_token) if session_token.isdigit() else session_token
        df["source_file"] = fp.name
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)

    # Remove rest/unlabeled class 0 for gesture-only training
    all_df = all_df[all_df["label"] != 0].reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(all_df)}")
    print(f"Subjects: {all_df['subject'].nunique()}")
    print("Label counts:")
    print(all_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()