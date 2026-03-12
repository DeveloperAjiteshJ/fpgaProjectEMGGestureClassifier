import argparse
import json
import shutil
from pathlib import Path


FILES = [
    "x_test_q.mem",
    "w_q.mem",
    "b_q.mem",
    "y_test_idx.mem",
    "golden_pred_idx.mem",
    "meta.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage one chunk into fpga/mem for a single FPGA pass."
    )
    ap.add_argument("--chunks", type=str, default="fpga/mem_chunks", help="Chunk root")
    ap.add_argument("--chunk", type=int, required=True, help="Chunk index, e.g. 0")
    ap.add_argument("--dst", type=str, default="fpga/mem", help="Destination mem directory")
    args = ap.parse_args()

    cdir = Path(args.chunks) / f"chunk_{args.chunk:03d}"
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if not cdir.exists():
        print(f"ERROR: missing chunk directory {cdir}")
        return 1

    for name in FILES:
        src_file = cdir / name
        if not src_file.exists():
            print(f"ERROR: missing required file {src_file}")
            return 1
        shutil.copy2(src_file, dst / name)

    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    n_samples = int(meta["n_samples_test"])

    print(f"Staged chunk {args.chunk} into {dst.resolve()}")
    print(f"Chunk samples (N_SAMPLES): {n_samples}")
    print("Set this Vivado generic for this pass:")
    print(f"  N_SAMPLES={n_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
