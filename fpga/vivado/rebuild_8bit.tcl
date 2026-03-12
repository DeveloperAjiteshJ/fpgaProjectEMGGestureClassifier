# ============================================================
# rebuild_8bit.tcl  — Clean rebuild with 8-bit quantization
# Run from project root:
#   vivado -mode batch -source fpga/vivado/rebuild_8bit.tcl
#
# Assets required (already generated in fpga/mem/):
#   xw_bits=8, q_frac=4, acc_bits=32, n_samples=4133
# Expected UART result: ~87.20%
# ============================================================

set proj_dir  "fpga/vivado/emg_logreg_fpga"
set proj_file "${proj_dir}/emg_logreg_fpga.xpr"
set mem_dir   [file normalize "fpga/mem"]
set bit_dir   "${proj_dir}/emg_logreg_fpga.runs/impl_1"

# ---- Open or recreate project ----------------------------------
if {[file exists $proj_file]} {
    open_project $proj_file
} else {
    puts "ERROR: Project not found at $proj_file"
    puts "Run create_project.tcl first."
    exit 1
}

# ---- Force-update all RTL source files -------------------------
# Remove old, re-add so Vivado picks up default-param changes
remove_files [get_files -filter {FILE_TYPE == "Verilog"}] 2>/dev/null
add_files [glob fpga/rtl/*.v]

# Remove old mem files and re-add the freshly generated ones
remove_files [get_files -filter {FILE_TYPE == "Memory File"}] 2>/dev/null
set mem_files [glob -nocomplain fpga/mem/*.mem]
if {[llength $mem_files] > 0} {
    add_files -fileset sources_1 $mem_files
}

set_property top top_nexys_a7 [current_fileset]

# ---- Set generics EXACTLY matching the 8-bit mem files ---------
set gen_str [format \
    {USE_X_BRAM=1 N_SAMPLES=4133 N_FEATURES=120 N_CLASSES=7 X_W=8 W_W=8 ACC_W=32 LABEL_W=8 X_MEM_FILE="%s/x_test_q.mem" W_MEM_FILE="%s/w_q.mem" B_MEM_FILE="%s/b_q.mem" Y_MEM_FILE="%s/y_test_idx.mem" G_MEM_FILE="%s/golden_pred_idx.mem"} \
    $mem_dir $mem_dir $mem_dir $mem_dir $mem_dir]

set_property generic $gen_str [get_filesets sources_1]
puts "Generics set: $gen_str"

update_compile_order -fileset sources_1

# ---- Clean old runs and rebuild --------------------------------
reset_run synth_1
reset_run impl_1

puts "Launching synthesis (8-bit, 4133 samples) ..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed. Check Messages tab."
    exit 1
}
puts "Synthesis done."

puts "Launching implementation + write_bitstream ..."
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed. Check Messages tab."
    exit 1
}

set bit_file "${bit_dir}/top_nexys_a7.bit"
if {[file exists $bit_file]} {
    puts "SUCCESS: Bitfile ready at $bit_file"
} else {
    puts "ERROR: Bitfile not found after implementation."
    exit 1
}
