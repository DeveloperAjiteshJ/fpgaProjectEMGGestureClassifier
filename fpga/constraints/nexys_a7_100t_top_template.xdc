# IMPORTANT:
# 1) Copy the official Digilent Nexys A7-100T master XDC into your Vivado project.
# 2) Uncomment and map only the ports listed below to avoid pin mismatch.
# 3) Keep IOSTANDARD exactly as in the board master XDC.
#
# Top-level ports expected by logreg_fpga_top wrapper:
#   clk_100mhz  : board 100 MHz oscillator
#   rst_btn     : reset button (active high in wrapper below)
#   start_btn   : start inference button (active high in wrapper below)
#   uart_tx_o   : UART TX to USB-UART bridge
#
# Use the exact PACKAGE_PIN values from the Digilent master XDC for your board revision.

## Clock: CLK100MHZ (master XDC pin E3)
set_property PACKAGE_PIN E3 [get_ports clk_100mhz]
set_property IOSTANDARD LVCMOS33 [get_ports clk_100mhz]
create_clock -add -name sys_clk_pin -period 10.000 -waveform {0 5} [get_ports clk_100mhz]

## Reset button: BTNU (master XDC pin M18), active high in RTL
set_property PACKAGE_PIN M18 [get_ports rst_btn]
set_property IOSTANDARD LVCMOS33 [get_ports rst_btn]

## Start button: BTNC (master XDC pin N17), active high pulse in RTL
set_property PACKAGE_PIN N17 [get_ports start_btn]
set_property IOSTANDARD LVCMOS33 [get_ports start_btn]

## UART TX from FPGA to PC: UART_RXD_OUT (master XDC pin D4)
set_property PACKAGE_PIN D4 [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx_o]

## Seven-segment display control (active-low on Nexys A7)
set_property PACKAGE_PIN J17 [get_ports {an_o[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[0]}]
set_property PACKAGE_PIN J18 [get_ports {an_o[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[1]}]
set_property PACKAGE_PIN T9 [get_ports {an_o[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[2]}]
set_property PACKAGE_PIN J14 [get_ports {an_o[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[3]}]
set_property PACKAGE_PIN P14 [get_ports {an_o[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[4]}]
set_property PACKAGE_PIN T14 [get_ports {an_o[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[5]}]
set_property PACKAGE_PIN K2 [get_ports {an_o[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[6]}]
set_property PACKAGE_PIN U13 [get_ports {an_o[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an_o[7]}]

set_property PACKAGE_PIN T10 [get_ports {seg_o[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[0]}]
set_property PACKAGE_PIN R10 [get_ports {seg_o[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[1]}]
set_property PACKAGE_PIN K16 [get_ports {seg_o[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[2]}]
set_property PACKAGE_PIN K13 [get_ports {seg_o[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[3]}]
set_property PACKAGE_PIN P15 [get_ports {seg_o[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[4]}]
set_property PACKAGE_PIN T11 [get_ports {seg_o[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[5]}]
set_property PACKAGE_PIN L18 [get_ports {seg_o[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg_o[6]}]

set_property PACKAGE_PIN H15 [get_ports dp_o]
set_property IOSTANDARD LVCMOS33 [get_ports dp_o]

# Optional: reduce noise if start button is mechanical.
# set_property PULLDOWN true [get_ports start_btn]
