import argparse
import json
from pathlib import Path


def read_hex_lines(path: Path) -> list[str]:
    with path.open("r", encoding="ascii") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_hex_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="ascii") as f:
        for ln in lines:
            f.write(ln + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split fpga/mem test set into BRAM-fit chunks for multi-pass FPGA evaluation."
    )
    ap.add_argument("--src", type=str, default="fpga/mem", help="Source mem directory")
    ap.add_argument("--out", type=str, default="fpga/mem_chunks", help="Output chunk root")
    ap.add_argument("--chunk_size", type=int, default=1024, help="Samples per chunk")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    n_samples = int(meta["n_samples_test"])
    n_features = int(meta["n_features"])
    n_classes = int(meta["n_classes"])

    x_all = read_hex_lines(src / "x_test_q.mem")
    y_all = read_hex_lines(src / "y_test_idx.mem")
    g_all = read_hex_lines(src / "golden_pred_idx.mem")
    w_all = read_hex_lines(src / "w_q.mem")
    b_all = read_hex_lines(src / "b_q.mem")

    if len(x_all) != n_samples * n_features:
        raise ValueError("x_test_q.mem length does not match n_samples_test * n_features")
    if len(y_all) != n_samples:
        raise ValueError("y_test_idx.mem length does not match n_samples_test")
    if len(g_all) != n_samples:
        raise ValueError("golden_pred_idx.mem length does not match n_samples_test")
    if len(w_all) != n_classes * n_features:
        raise ValueError("w_q.mem length does not match n_classes * n_features")
    if len(b_all) != n_classes:
        raise ValueError("b_q.mem length does not match n_classes")

    chunk_count = 0
    golden_correct_total = 0

    for start in range(0, n_samples, args.chunk_size):
        end = min(start + args.chunk_size, n_samples)
        chunk_n = end - start

        chunk_dir = out / f"chunk_{chunk_count:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        x_chunk = x_all[start * n_features : end * n_features]
        y_chunk = y_all[start:end]
        g_chunk = g_all[start:end]

        write_hex_lines(chunk_dir / "x_test_q.mem", x_chunk)
        write_hex_lines(chunk_dir / "y_test_idx.mem", y_chunk)
        write_hex_lines(chunk_dir / "golden_pred_idx.mem", g_chunk)
        write_hex_lines(chunk_dir / "w_q.mem", w_all)
        write_hex_lines(chunk_dir / "b_q.mem", b_all)

        golden_corr = sum(int(a == b) for a, b in zip(y_chunk, g_chunk))
        golden_correct_total += golden_corr

        chunk_meta = dict(meta)
        chunk_meta["n_samples_test"] = chunk_n
        chunk_meta["chunk_index"] = chunk_count
        chunk_meta["chunk_start"] = start
        chunk_meta["chunk_end_exclusive"] = end
        chunk_meta["chunk_size"] = int(args.chunk_size)
        chunk_meta["chunk_golden_correct"] = golden_corr
        chunk_meta["chunk_golden_accuracy"] = float(golden_corr / chunk_n) if chunk_n else 0.0

        (chunk_dir / "meta.json").write_text(json.dumps(chunk_meta, indent=2), encoding="utf-8")
        chunk_count += 1

    summary = {
        "chunk_size": int(args.chunk_size),
        "n_chunks": chunk_count,
        "n_samples_total": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "golden_correct_total": golden_correct_total,
        "golden_accuracy_total": float(golden_correct_total / n_samples) if n_samples else 0.0,
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Created {chunk_count} chunks at: {out.resolve()}")
    print(
        f"Golden reference over all chunks -> correct={golden_correct_total}, "
        f"total={n_samples}, accuracy={100.0 * summary['golden_accuracy_total']:.4f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
