module top_nexys_a7 #(
    parameter integer CLK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE = 115200,
    parameter integer N_FEATURES = 120,
    parameter integer N_CLASSES = 7,
    parameter integer N_SAMPLES = 4133,
    parameter integer X_W = 8,
    parameter integer W_W = 8,
    parameter integer ACC_W = 32,
    parameter integer LABEL_W = 8,
    parameter X_MEM_FILE = "fpga/mem/x_test_q.mem",
    parameter W_MEM_FILE = "fpga/mem/w_q.mem",
    parameter B_MEM_FILE = "fpga/mem/b_q.mem",
    parameter Y_MEM_FILE = "fpga/mem/y_test_idx.mem",
    parameter G_MEM_FILE = "fpga/mem/golden_pred_idx.mem"
) (
    input  wire clk_100mhz,
    input  wire rst_btn,
    input  wire start_btn,
    output wire uart_tx_o,
    output wire [7:0] an_o,
    output wire [6:0] seg_o,
    output wire dp_o
);
    reg start_ff0, start_ff1;
    always @(posedge clk_100mhz) begin
        start_ff0 <= start_btn;
        start_ff1 <= start_ff0;
    end

    wire start_pulse = start_ff0 & ~start_ff1;
    wire [15:0] expected_pct_hundredths;
    wire [15:0] actual_pct_hundredths;

    logreg_fpga_top #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE),
        .N_FEATURES(N_FEATURES),
        .N_CLASSES(N_CLASSES),
        .N_SAMPLES(N_SAMPLES),
        .X_W(X_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .LABEL_W(LABEL_W),
        .USE_X_BRAM(1),
        .X_MEM_FILE(X_MEM_FILE),
        .W_MEM_FILE(W_MEM_FILE),
        .B_MEM_FILE(B_MEM_FILE),
        .Y_MEM_FILE(Y_MEM_FILE),
        .G_MEM_FILE(G_MEM_FILE)
    ) u_core (
        .clk(clk_100mhz),
        .rst(rst_btn),
        .start(start_pulse),
        .x_req(),
        .x_req_addr(),
        .x_ext_dout({X_W{1'b0}}),
        .x_ext_valid(1'b0),
        .uart_tx_o(uart_tx_o),
        .expected_pct_hundredths(expected_pct_hundredths),
        .actual_pct_hundredths(actual_pct_hundredths),
        .done(),
        .busy()
    );

    sevenseg_accuracy u_disp (
        .clk(clk_100mhz),
        .rst(rst_btn),
        .actual_pct_hundredths(actual_pct_hundredths),
        .an_o(an_o),
        .seg_o(seg_o),
        .dp_o(dp_o)
    );
endmodule
