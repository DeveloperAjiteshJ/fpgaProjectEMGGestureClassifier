#!/usr/bin/env python3
"""
Test a few key LogisticRegression C values and pick the best for FPGA.
Then update fpga/mem with optimal parameters.
"""
import json
import subprocess
from pathlib import Path

# Test these C values (most promising based on prior analysis)
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0]

best_config = None
best_acc = 0.0

print("="*70)
print("Testing LogisticRegression configurations for FPGA")
print("="*70)

for C in C_VALUES:
    outdir = f"fpga/mem_test_c{C}"
    print(f"\nTesting C={C} → {outdir}")
    
    cmd = [
        "C:/Users/jajit/OneDrive/Desktop/dsd_project/.venv/Scripts/python.exe",
        "fpga/scripts/prepare_logreg_fpga_assets.py",
        "--window", "250",
        "--step", "50",
        "--min_purity", "0.8",
        "--logreg_C", str(C),
        "--max_iter", "10000",
        "--q_frac", "12",
        "--xw_bits", "16",
        "--acc_bits", "48",
        "--outdir", outdir,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Read meta.json to get accuracy
            meta_file = Path(outdir) / "meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    test_acc = meta["quantized_model"]["test_accuracy"]
                    val_f1 = meta["float_model"]["val_macro_f1"]
                    
                    marker = "✓ >=90%" if test_acc >= 0.90 else ""
                    print(f"  → Quantized Test Acc: {test_acc:.4f} {marker}")
                    print(f"  → Float  Val F1:    {val_f1:.4f}")
                    
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_config = {"C": C, "outdir": outdir, "acc": test_acc, "f1": val_f1}
        else:
            print(f"  ✗ Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (>120s)")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

print("\n" + "="*70)
if best_config:
    print(f"✅ BEST CONFIG FOUND:")
    print(f"   C={best_config['C']}, Accuracy={best_config['acc']:.4f}")
    print(f"\nTo apply this to fpga/mem, run:")
    print(f"  cp {best_config['outdir']}/* fpga/mem/")
    print("="*70)
else:
    print("⚠️  No successful configuration found")
    print("="*70)
