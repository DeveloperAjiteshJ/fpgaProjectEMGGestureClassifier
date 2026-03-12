# FPGA Implementation Guide (Nexys A7-100T, Verilog, No HLS)

This flow runs a quantized Logistic Regression classifier on FPGA.
Data and model parameters are precomputed in Python and loaded into BRAM-backed ROMs.
For this version, BRAM stores standardized feature vectors (not raw EMG), which preserves quantized accuracy.

## 1) Generate FPGA Assets
Run from repo root:

```bash
C:/Users/jajit/OneDrive/Desktop/dsd_project/.venv/Scripts/python.exe fpga/scripts/prepare_logreg_fpga_assets.py --window 250 --step 50 --min_purity 0.8 --logreg_C 1.0 --max_iter 20000 --q_frac 12 --xw_bits 16 --acc_bits 48 --outdir fpga/mem
```

Final locked configuration for board deployment:
- model: multinomial Logistic Regression
- window/step: 250/50
- C: 1.0
- max_iter: 20000
- q_frac/xw_bits/acc_bits: 12/16/48
- expected quantized test accuracy: ~87.61%

Outputs in `fpga/mem`:
- `x_test_q.mem`
- `w_q.mem`
- `b_q.mem`
- `y_test_idx.mem`
- `golden_pred_idx.mem`
- `meta.json`

## 2) Create Vivado Project

```bash
vivado -mode batch -source fpga/vivado/create_project.tcl
```

Open project in GUI after this command.

## 3) Constraints (Critical)
1. Add official Digilent Nexys A7-100T master XDC.
2. Map these ports using exact pins from master XDC:
- `clk_100mhz`
- `rst_btn`
- `start_btn`
- `uart_tx_o`
3. Keep `LVCMOS33` and clock constraint (`10.000 ns`) consistent.

Template: `fpga/constraints/nexys_a7_100t_top_template.xdc`.

## 4) Build and Program
In Vivado GUI:
1. Run Behavioral Simulation (`tb_logreg_fpga_top`).
2. Run Synthesis.
3. Run Implementation.
4. Generate Bitstream.
5. Program Device.

## 5) Read Accuracy over UART
When `start_btn` is pressed once, FPGA processes all samples and transmits:

`CORR=XXXXXXXX TOTAL=XXXXXXXX`

(HEX values)

Use monitor script on PC:

```bash
C:/Users/jajit/OneDrive/Desktop/dsd_project/.venv/Scripts/python.exe fpga/scripts/uart_accuracy_monitor.py --port COM4 --baud 115200
```

The script decodes and prints accuracy percentage.

## 6) Data/Model Layout
See `fpga/mem/meta.json`.

Flatten order is fixed:
- `x_test_q.mem`: sample-major, then feature index
- `w_q.mem`: class-major, then feature index
- `b_q.mem`: class-major
- `y_test_idx.mem`: sample-major labels (0-based)

## 7) Notes on Accuracy
- Float model and quantized model scores are logged in `meta.json`.
- If quantized accuracy drop is high, increase `xw_bits` or adjust `q_frac`.

## 8) Why Vitis Is Not Required
This design is pure RTL and runs fully in programmable logic. Vitis is optional and not required for this flow.
