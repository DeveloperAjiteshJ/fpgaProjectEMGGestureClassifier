import argparse
import json
import re
import shutil
from pathlib import Path

import serial


FILES = ["x_test_q.mem", "w_q.mem", "b_q.mem", "y_test_idx.mem", "golden_pred_idx.mem", "meta.json"]
UART_PAT = re.compile(r"CORR=([0-9A-Fa-f]{8})\s+TOTAL=([0-9A-Fa-f]{8})")


def stage_chunk(chunks_root: Path, chunk_idx: int, dst_mem: Path) -> int:
    cdir = chunks_root / f"chunk_{chunk_idx:03d}"
    if not cdir.exists():
        raise FileNotFoundError(f"Missing chunk directory: {cdir}")

    dst_mem.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = cdir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing file in chunk: {src}")
        shutil.copy2(src, dst_mem / name)

    meta = json.loads((dst_mem / "meta.json").read_text(encoding="utf-8"))
    return int(meta["n_samples_test"])


def wait_uart_once(port: str, baud: int, timeout_s: float) -> tuple[int, int, str]:
    with serial.Serial(port, baud, timeout=0.2) as ser:
        waited = 0.0
        step = 0.2
        while timeout_s <= 0 or waited < timeout_s:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line:
                m = UART_PAT.search(line)
                if m:
                    corr = int(m.group(1), 16)
                    total = int(m.group(2), 16)
                    return corr, total, line
            waited += step
    if timeout_s > 0:
        raise TimeoutError(f"No UART result received within {timeout_s:.1f}s")
    raise TimeoutError("No UART result received")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assist multi-pass FPGA runs: stage chunk, capture UART, and aggregate final accuracy."
    )
    ap.add_argument("--chunks", type=str, default="fpga/mem_chunks", help="Chunk root")
    ap.add_argument("--dst_mem", type=str, default="fpga/mem", help="Active mem directory used by Vivado")
    ap.add_argument("--port", type=str, required=True, help="Serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="UART wait timeout per chunk in seconds; use 0 for infinite wait",
    )
    ap.add_argument("--results", type=str, default="fpga/mem_chunks/results.json")
    args = ap.parse_args()

    chunks_root = Path(args.chunks)
    dst_mem = Path(args.dst_mem)
    results_path = Path(args.results)

    summary_path = chunks_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run make_fpga_chunks.py first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_chunks = int(summary["n_chunks"])
    expected_total = int(summary["n_samples_total"])

    results = []
    agg_corr = 0
    agg_total = 0

    print(f"Running multi-pass capture for {n_chunks} chunks on {args.port} @ {args.baud}...")
    print("For each chunk: script stages files, you rebuild+program in Vivado, then press reset/start.")

    for idx in range(n_chunks):
        n_samples = stage_chunk(chunks_root, idx, dst_mem)

        print("\n" + "=" * 72)
        print(f"Chunk {idx}/{n_chunks - 1} staged. N_SAMPLES={n_samples}")
        print("Set Vivado generic (sources_1) for this pass:")
        print(f"  USE_X_BRAM=1 N_SAMPLES={n_samples}")
        print("Then run synth/impl/bitstream, program FPGA, and come back here.")
        input("Press Enter when bitstream is programmed and you are ready to press reset+start... ")

        print("Listening for one UART result line... press reset then start on board now.")
        corr, total, uart_line = wait_uart_once(args.port, args.baud, args.timeout)

        print(f"Captured: {uart_line}")
        results.append({"chunk": idx, "corr": corr, "total": total, "uart": uart_line})

        agg_corr += corr
        agg_total += total
        run_acc = 100.0 * agg_corr / agg_total if agg_total else 0.0
        print(f"Running aggregate: correct={agg_corr}, total={agg_total}, acc={run_acc:.4f}%")

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "#" * 72)
    print("Final aggregated FPGA result:")
    print(f"  correct={agg_corr}")
    print(f"  total={agg_total}")
    if agg_total:
        print(f"  accuracy={100.0 * agg_corr / agg_total:.4f}%")
    print(f"Expected total from chunk summary: {expected_total}")
    if agg_total != expected_total:
        print("WARNING: total mismatch; check per-chunk N_SAMPLES/generic settings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
