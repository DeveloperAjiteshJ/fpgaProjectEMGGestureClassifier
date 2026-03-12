module uart_tx #(
    parameter integer CLK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE = 115200
) (
    input  wire clk,
    input  wire rst,
    input  wire tx_start,
    input  wire [7:0] tx_data,
    output reg  tx,
    output reg  tx_busy,
    output reg  tx_done_pulse
);
    localparam integer CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE;

    reg [31:0] clk_cnt;
    reg [3:0] bit_idx;
    reg [9:0] shifter;

    always @(posedge clk) begin
        if (rst) begin
            tx <= 1'b1;
            tx_busy <= 1'b0;
            tx_done_pulse <= 1'b0;
            clk_cnt <= 32'd0;
            bit_idx <= 4'd0;
            shifter <= 10'h3ff;
        end else begin
            tx_done_pulse <= 1'b0;

            if (!tx_busy) begin
                tx <= 1'b1;
                if (tx_start) begin
                    shifter <= {1'b1, tx_data, 1'b0};
                    tx_busy <= 1'b1;
                    clk_cnt <= 32'd0;
                    bit_idx <= 4'd0;
                    tx <= 1'b0;
                end
            end else begin
                if (clk_cnt == CLKS_PER_BIT - 1) begin
                    clk_cnt <= 32'd0;
                    bit_idx <= bit_idx + 1'b1;
                    shifter <= {1'b1, shifter[9:1]};
                    tx <= shifter[1];

                    if (bit_idx == 4'd9) begin
                        tx_busy <= 1'b0;
                        tx <= 1'b1;
                        tx_done_pulse <= 1'b1;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1'b1;
                end
            end
        end
    end
endmodule
