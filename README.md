# EMG Gesture Classifier with FPGA Deployment

This repository contains the code required for:
- EMG dataset preparation
- Model training and testing
- FPGA asset generation for quantized logistic regression
- FPGA RTL, testbench, constraints, and Vivado build scripts

This repo is intentionally code-only.
It does not include the raw dataset, processed dataset, trained model binaries, generated memory files, checkpoints, or Vivado generated project outputs.

## Included Files

Main Python code:
- `scripts/prepare_emg_dataset.py`
- `scripts/train_emg_model.py`
- `scripts/test_emg_model.py`
- `src/emg_model.py`

FPGA preparation scripts:
- `fpga/scripts/prepare_logreg_fpga_assets.py`
- `fpga/scripts/check_fpga_assets.py`
- `fpga/scripts/uart_accuracy_monitor.py`
- `fpga/scripts/make_fpga_chunks.py`
- `fpga/scripts/stage_fpga_chunk.py`
- `fpga/scripts/aggregate_fpga_chunks.py`
- `fpga/scripts/run_fpga_multipass.py`

RTL / FPGA implementation:
- `fpga/rtl/bram_rom_sync.v`
- `fpga/rtl/logreg_fpga_top.v`
- `fpga/rtl/top_nexys_a7.v`
- `fpga/rtl/sevenseg_accuracy.v`
- `fpga/rtl/uart_tx.v`
- `fpga/tb/tb_logreg_fpga_top.v`
- `fpga/constraints/nexys_a7_100t_top_template.xdc`
- `fpga/vivado/create_project.tcl`
- `fpga/vivado/rebuild_8bit.tcl`

## How To Download

Clone the repository:

```bash
git clone https://github.com/DeveloperAjiteshJ/fpgaProjectEMGGestureClassifier.git
cd fpgaProjectEMGGestureClassifier
```

## Python Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

The dataset is not included in this repo.
Download the EMG gesture dataset separately and place it in:

```text
EMG_data_for_gestures-master/
```

After placing the raw dataset, prepare the processed dataset:

```bash
python scripts/prepare_emg_dataset.py
```

## Train And Test The Model

Train:

```bash
python scripts/train_emg_model.py
```

Test:

```bash
python scripts/test_emg_model.py
```

## FPGA Flow

Generate FPGA assets from the trained model:

```bash
python fpga/scripts/prepare_logreg_fpga_assets.py --outdir fpga/mem
```

Check generated assets:

```bash
python fpga/scripts/check_fpga_assets.py --memdir fpga/mem
```

## Build FPGA Bitstream

Open Vivado 2024.1 and run:

```tcl
source fpga/vivado/create_project.tcl
source fpga/vivado/rebuild_8bit.tcl
```

Or from command line:

```bash
vivado -mode batch -source fpga/vivado/rebuild_8bit.tcl
```

## UART Accuracy Monitor

After programming the FPGA, read the result over UART:

```bash
python fpga/scripts/uart_accuracy_monitor.py --port COM4 --baud 115200
```

## Notes

- Stable verified FPGA configuration used in this project:
  - `X_W=8`
  - `W_W=8`
  - `ACC_W=32`
  - `q_frac=4`
- Seven-segment display is configured to show actual accuracy only.
- Raw/sample/generated files are excluded from GitHub to keep the repository lightweight.
