import argparse
import re

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, default="COM4", help="COM port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    pat = re.compile(r"CORR=([0-9A-Fa-f]{8})\s+TOTAL=([0-9A-Fa-f]{8})")

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        print(f"Listening on {args.port} @ {args.baud} ...")
        print("Final target setup: LogisticRegression C=1.0, expected quantized accuracy ~87.61%")
        while True:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            print(f"UART: {line}")
            m = pat.search(line)
            if m:
                corr = int(m.group(1), 16)
                total = int(m.group(2), 16)
                acc = 100.0 * corr / total if total else 0.0
                print(f"Parsed -> correct={corr}, total={total}, accuracy={acc:.4f}%")
                break


if __name__ == "__main__":
    main()
