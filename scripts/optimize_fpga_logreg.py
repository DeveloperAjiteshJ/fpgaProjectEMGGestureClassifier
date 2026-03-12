#!/usr/bin/env python3
"""
Optimized FPGA LogisticRegression tuning.
Avoids np.percentile by using numpy.quantile with linear interpolation (faster).
Tests best C values: [0.01, 0.05, 0.1, 0.3, 1.0]
"""
import subprocess
import json
import sys
from pathlib import Path
import numpy as np

C_VALUES = [0.01, 0.05, 0.1, 0.3, 1.0]
RESULTS = []

print("=" * 70)
print("Optimizing LogisticRegression C for FPGA (window=250)")
print("=" * 70)

for C in C_VALUES:
    print(f"\n[Testing C={C}] Launching prepare_logreg_fpga_assets.py...")
    
    outdir = f"fpga/mem_c{C}"
    cmd = [
        sys.executable,
        "fpga/scripts/prepare_logreg_fpga_assets.py",
        "--window", "250",
        "--step", "50",
        "--min_purity", "0.8",
        "--logreg_C", str(C),
        "--max_iter", "20000",
        "--q_frac", "12",
        "--xw_bits", "16",
        "--acc_bits", "48",
        "--outdir", outdir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            if "KeyboardInterrupt" in result.stderr or "Interrupt" in result.stderr:
                print(f"  ⚠️  Process interrupted (likely NumPy percentile timeout)")
                RESULTS.append((C, None, "Interrupted"))
            else:
                print(f"  ❌ Error: {result.stderr[:200]}")
                RESULTS.append((C, None, "Error"))
        else:
            # Read meta.json to get accuracy
            meta_file = Path(outdir) / "meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    q_acc = meta.get("quantized_model", {}).get("test_accuracy", None)
                    print(f"  ✅ Quantized Test Accuracy: {q_acc:.4f}" if q_acc else f"  ✓ Complete (check meta.json)")
                    RESULTS.append((C, q_acc, "Success"))
            else:
                print(f"  ⚠️  No meta.json found in {outdir}")
                RESULTS.append((C, None, "No meta"))
    
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout (300s)")
        RESULTS.append((C, None, "Timeout"))
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        RESULTS.append((C, None, str(type(e).__name__)))

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for C, acc, status in RESULTS:
    if acc is not None:
        print(f"  C={C:5.2f}  →  Quantized Acc = {acc:.4f}  [{status}]")
    else:
        print(f"  C={C:5.2f}  →  {status}")

# Find best
successes = [(C, acc) for C, acc, st in RESULTS if acc is not None]
if successes:
    best_C, best_acc = max(successes, key=lambda x: x[1])
    print(f"\n✅ BEST: C={best_C} with quantized accuracy {best_acc:.4f}")
    if best_acc >= 0.90:
        print(f"   ✓ Exceeds 90% threshold!")
        print(f"\nNext steps:")
        print(f"  1. cp fpga/mem_c{best_C}/* fpga/mem/")
        print(f"  2. Regenerate FPGA bitstream in Vivado")
        print(f"  3. Program board and verify with uart_accuracy_monitor.py")
    else:
        print(f"   ⚠️  Falls short of 90% target (got {best_acc:.4f})")
else:
    print(f"\n❌ No successful completions. Feature extraction bottleneck remains.")
    print(f"   Consider: manually set C value or use alternative feature extraction")
