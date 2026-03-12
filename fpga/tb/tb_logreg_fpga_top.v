`timescale 1ns/1ps

module tb_logreg_fpga_top;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire uart_tx_o;
    wire done;
    wire busy;

    always #5 clk = ~clk; // 100 MHz

    logreg_fpga_top #(
        .CLK_FREQ_HZ(100_000_000),
        .BAUD_RATE(115200),
        .N_FEATURES(4),
        .N_CLASSES(2),
        .N_SAMPLES(2),
        .X_W(16),
        .W_W(16),
        .ACC_W(48),
        .LABEL_W(8),
        .X_MEM_FILE("fpga/tb/tb_x.mem"),
        .W_MEM_FILE("fpga/tb/tb_w.mem"),
        .B_MEM_FILE("fpga/tb/tb_b.mem"),
        .Y_MEM_FILE("fpga/tb/tb_y.mem")
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .uart_tx_o(uart_tx_o),
        .done(done),
        .busy(busy)
    );

    initial begin
        #100;
        rst = 1'b0;
        #50;
        start = 1'b1;
        #10;
        start = 1'b0;

        wait(done);
        #100;
        $display("TB PASS: done asserted");
        $finish;
    end
endmodule
