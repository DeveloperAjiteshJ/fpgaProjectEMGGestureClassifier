# Usage from project root:
# vivado -mode batch -source fpga/vivado/create_project.tcl

set proj_name "emg_logreg_fpga"
set proj_dir  "fpga/vivado/${proj_name}"
set part_name "xc7a100tcsg324-1"

file mkdir $proj_dir
create_project $proj_name $proj_dir -part $part_name -force

add_files [glob fpga/rtl/*.v]
add_files -fileset sim_1 [glob fpga/tb/*.v]

set mem_files [glob -nocomplain fpga/mem/*.mem]
if {[llength $mem_files] > 0} {
	add_files -fileset sources_1 $mem_files
}

# Add template constraints; replace placeholders with master XDC pin mappings before bitstream.
add_files -fileset constrs_1 fpga/constraints/nexys_a7_100t_top_template.xdc

set_property top top_nexys_a7 [current_fileset]
set_property top tb_logreg_fpga_top [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "Project created: $proj_dir"
puts "Next: open Vivado GUI, import Digilent master XDC, then map pins for clk/reset/start/uart_tx_o."
