import argparse
import json
import sys
from pathlib import Path


def expected_hex_width(bit_width: int) -> int:
    return (bit_width + 3) // 4


def read_nonempty_lines(path: Path):
    with path.open("r", encoding="ascii") as f:
        return [ln.strip() for ln in f if ln.strip()]


def validate_hex_lines(lines, bit_width: int, name: str):
    width = expected_hex_width(bit_width)
    for i, ln in enumerate(lines, start=1):
        if len(ln) != width:
            raise ValueError(
                f"{name}: line {i} has width {len(ln)}, expected {width} for {bit_width}-bit values"
            )
        try:
            int(ln, 16)
        except ValueError as exc:
            raise ValueError(f"{name}: line {i} is not valid hex: '{ln}'") from exc


def validate_labels(lines, n_classes: int, name: str):
    for i, ln in enumerate(lines, start=1):
        val = int(ln, 16)
        if not (0 <= val < n_classes):
            raise ValueError(
                f"{name}: line {i} has class index {val}, expected [0, {n_classes - 1}]"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate fpga/mem assets against meta.json and print expected UART totals."
    )
    ap.add_argument("--memdir", type=str, default="fpga/mem", help="Directory containing mem files")
    args = ap.parse_args()

    memdir = Path(args.memdir)
    meta_path = memdir / "meta.json"

    if not meta_path.exists():
        print(f"ERROR: missing {meta_path}")
        return 1

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    n_samples = int(meta["n_samples_test"])
    n_features = int(meta["n_features"])
    n_classes = int(meta["n_classes"])
    xw_bits = int(meta["xw_bits"])
    acc_bits = int(meta["acc_bits"])

    required = {
        "x_test_q.mem": (n_samples * n_features, xw_bits),
        "w_q.mem": (n_classes * n_features, xw_bits),
        "b_q.mem": (n_classes, acc_bits),
        "y_test_idx.mem": (n_samples, 8),
        "golden_pred_idx.mem": (n_samples, 8),
    }

    print("Checking FPGA assets...")
    print(f"  memdir={memdir.resolve()}")

    for name, (expected_count, bit_width) in required.items():
        path = memdir / name
        if not path.exists():
            print(f"ERROR: missing {path}")
            return 1

        lines = read_nonempty_lines(path)
        if len(lines) != expected_count:
            print(
                f"ERROR: {name} has {len(lines)} lines, expected {expected_count} "
                f"(from meta: samples={n_samples}, features={n_features}, classes={n_classes})"
            )
            return 1

        try:
            validate_hex_lines(lines, bit_width, name)
            if name in {"y_test_idx.mem", "golden_pred_idx.mem"}:
                validate_labels(lines, n_classes=n_classes, name=name)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1

        print(f"  OK {name}: {len(lines)} lines, {bit_width}-bit hex")

    total_hex = f"{n_samples:08X}"
    print("\nPASS: assets are self-consistent with meta.json")
    print(f"Expected UART TOTAL field: TOTAL={total_hex} (dec {n_samples})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
