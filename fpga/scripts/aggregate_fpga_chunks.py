import argparse
import json
import re
from pathlib import Path


def parse_uart_text(text: str) -> tuple[int, int] | None:
    m = re.search(r"CORR=([0-9A-Fa-f]{8})\s+TOTAL=([0-9A-Fa-f]{8})", text)
    if not m:
        return None
    return int(m.group(1), 16), int(m.group(2), 16)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate multi-pass FPGA UART results into one final accuracy."
    )
    ap.add_argument("--chunks", type=str, default="fpga/mem_chunks", help="Chunk root directory")
    ap.add_argument(
        "--results",
        type=str,
        default="fpga/mem_chunks/results.json",
        help="JSON file containing per-chunk corr/total results",
    )
    args = ap.parse_args()

    chunks_root = Path(args.chunks)
    results_path = Path(args.results)

    summary = json.loads((chunks_root / "summary.json").read_text(encoding="utf-8"))
    expected_chunks = int(summary["n_chunks"])
    expected_total = int(summary["n_samples_total"])

    if not results_path.exists():
        print(f"ERROR: missing {results_path}")
        print(
            "Create it with entries like:"
            " [{\"chunk\":0,\"corr\":888,\"total\":1024}, ...]"
            " or [{\"chunk\":0,\"uart\":\"CORR=00000378 TOTAL=00000400\"}, ...]"
        )
        return 1

    entries = json.loads(results_path.read_text(encoding="utf-8"))

    seen = set()
    agg_corr = 0
    agg_total = 0

    for item in entries:
        idx = int(item["chunk"])
        if idx in seen:
            raise ValueError(f"Duplicate chunk index in results: {idx}")
        seen.add(idx)

        if "corr" in item and "total" in item:
            corr = int(item["corr"])
            total = int(item["total"])
        elif "uart" in item:
            parsed = parse_uart_text(str(item["uart"]))
            if parsed is None:
                raise ValueError(f"Could not parse UART string for chunk {idx}")
            corr, total = parsed
        else:
            raise ValueError(f"Entry for chunk {idx} missing corr/total or uart")

        agg_corr += corr
        agg_total += total

    missing = [i for i in range(expected_chunks) if i not in seen]

    print(f"Chunks reported: {len(seen)}/{expected_chunks}")
    if missing:
        print(f"Missing chunks: {missing}")

    print(f"Aggregated FPGA: correct={agg_corr}, total={agg_total}")
    if agg_total:
        print(f"Aggregated FPGA accuracy: {100.0 * agg_corr / agg_total:.4f}%")

    print(f"Expected total samples from summary: {expected_total}")
    if agg_total != expected_total:
        print("WARNING: Aggregated total does not match expected total")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
