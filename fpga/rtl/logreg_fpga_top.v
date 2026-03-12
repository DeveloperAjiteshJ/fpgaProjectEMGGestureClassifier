module logreg_fpga_top #(
    parameter integer CLK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE = 115200,

    parameter integer N_FEATURES = 120,
    parameter integer N_CLASSES = 7,
    parameter integer N_SAMPLES = 4096,

    parameter integer X_W = 8,
    parameter integer W_W = 8,
    parameter integer ACC_W = 32,
    parameter integer LABEL_W = 8,
    parameter integer USE_X_BRAM = 1,

    parameter X_MEM_FILE = "fpga/mem/x_test_q.mem",
    parameter W_MEM_FILE = "fpga/mem/w_q.mem",
    parameter B_MEM_FILE = "fpga/mem/b_q.mem",
    parameter Y_MEM_FILE = "fpga/mem/y_test_idx.mem",
    parameter G_MEM_FILE = "fpga/mem/golden_pred_idx.mem"
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    output reg  x_req,
    output reg [31:0] x_req_addr,
    input  wire signed [X_W-1:0] x_ext_dout,
    input  wire x_ext_valid,
    output wire uart_tx_o,
    output reg [15:0] expected_pct_hundredths,
    output reg [15:0] actual_pct_hundredths,
    output reg  done,
    output reg  busy
);
    function integer ceil_pow2;
        input integer v;
        integer p;
        begin
            p = 1;
            while (p < v)
                p = p << 1;
            ceil_pow2 = p;
        end
    endfunction

    localparam integer X_DEPTH = N_SAMPLES * N_FEATURES;
    localparam integer W_DEPTH = N_CLASSES * N_FEATURES;
    localparam integer B_DEPTH = N_CLASSES;
    localparam integer Y_DEPTH = N_SAMPLES;

    // Power-of-two physical depths avoid BRAM cascade address tie-off DRC issues.
    localparam integer X_DEPTH_MEM = ceil_pow2(X_DEPTH);
    localparam integer W_DEPTH_MEM = ceil_pow2(W_DEPTH);
    localparam integer B_DEPTH_MEM = ceil_pow2(B_DEPTH);
    localparam integer Y_DEPTH_MEM = ceil_pow2(Y_DEPTH);

    localparam integer X_ADDR_W = $clog2(X_DEPTH_MEM);
    localparam integer W_ADDR_W = $clog2(W_DEPTH_MEM);
    localparam integer B_ADDR_W = $clog2(B_DEPTH_MEM);
    localparam integer Y_ADDR_W = $clog2(Y_DEPTH_MEM);

    localparam [4:0]
        S_IDLE          = 5'd0,
        S_CLASS_SETUP   = 5'd1,
        S_BIAS_REQ      = 5'd2,
        S_BIAS_CAP      = 5'd3,
        S_MAC_REQ       = 5'd4,
        S_MAC_CAP       = 5'd5,
        S_CLASS_DONE    = 5'd6,
        S_LABEL_REQ     = 5'd7,
        S_LABEL_CAP     = 5'd8,
        S_NEXT_SAMPLE   = 5'd9,
        S_REPORT_INIT   = 5'd10,
        S_REPORT_SEND   = 5'd11,
        S_REPORT_WAIT   = 5'd12,
        S_REPORT_DONE   = 5'd13,
        S_X_WAIT        = 5'd14;

    reg [4:0] state;

    reg [31:0] sample_idx;
    reg [31:0] class_idx;
    reg [31:0] feat_idx;

    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] best_score;
    reg [LABEL_W-1:0] best_class;

    reg [31:0] correct_count;
    reg [31:0] expected_correct_count;

    reg [X_ADDR_W-1:0] x_addr;
    reg [W_ADDR_W-1:0] w_addr;
    reg [B_ADDR_W-1:0] b_addr;
    reg [Y_ADDR_W-1:0] y_addr;
    reg [Y_ADDR_W-1:0] g_addr;

    wire signed [X_W-1:0] x_dout;
    wire signed [X_W-1:0] x_data = (USE_X_BRAM != 0) ? x_dout : x_ext_dout;
    wire x_data_valid = (USE_X_BRAM != 0) ? 1'b1 : x_ext_valid;
    wire signed [W_W-1:0] w_dout;
    wire signed [ACC_W-1:0] b_dout;
    wire signed [LABEL_W-1:0] y_dout;
    wire signed [LABEL_W-1:0] g_dout;

    wire signed [X_W+W_W-1:0] mul_full = x_data * w_dout;
    wire signed [ACC_W-1:0] mul_ext = {{(ACC_W-(X_W+W_W)){mul_full[X_W+W_W-1]}}, mul_full};

    generate
        if (USE_X_BRAM != 0) begin : g_x_bram
            bram_rom_sync #(
                .DATA_W(X_W),
                .DEPTH(X_DEPTH_MEM),
                .INIT_FILE(X_MEM_FILE)
            ) u_x_mem (
                .clk(clk),
                .addr(x_addr),
                .dout(x_dout)
            );
        end else begin : g_x_ext
            assign x_dout = {X_W{1'b0}};
        end
    endgenerate

    bram_rom_sync #(
        .DATA_W(W_W),
        .DEPTH(W_DEPTH_MEM),
        .INIT_FILE(W_MEM_FILE)
    ) u_w_mem (
        .clk(clk),
        .addr(w_addr),
        .dout(w_dout)
    );

    bram_rom_sync #(
        .DATA_W(ACC_W),
        .DEPTH(B_DEPTH_MEM),
        .INIT_FILE(B_MEM_FILE)
    ) u_b_mem (
        .clk(clk),
        .addr(b_addr),
        .dout(b_dout)
    );

    bram_rom_sync #(
        .DATA_W(LABEL_W),
        .DEPTH(Y_DEPTH_MEM),
        .INIT_FILE(Y_MEM_FILE)
    ) u_y_mem (
        .clk(clk),
        .addr(y_addr),
        .dout(y_dout)
    );

    bram_rom_sync #(
        .DATA_W(LABEL_W),
        .DEPTH(Y_DEPTH_MEM),
        .INIT_FILE(G_MEM_FILE)
    ) u_g_mem (
        .clk(clk),
        .addr(g_addr),
        .dout(g_dout)
    );

    reg uart_start;
    reg [7:0] uart_data;
    wire uart_busy;
    wire uart_done_pulse;

    uart_tx #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart (
        .clk(clk),
        .rst(rst),
        .tx_start(uart_start),
        .tx_data(uart_data),
        .tx(uart_tx_o),
        .tx_busy(uart_busy),
        .tx_done_pulse(uart_done_pulse)
    );

    reg [7:0] report_idx;
    reg [31:0] report_value;

    function [7:0] hex_char;
        input [3:0] nib;
        begin
            if (nib < 10)
                hex_char = 8'h30 + nib;
            else
                hex_char = 8'h41 + (nib - 10);
        end
    endfunction

    // Message format: "CORR=XXXXXXXX TOTAL=XXXXXXXX\r\n"
    function [7:0] report_byte;
        input [7:0] idx;
        input [31:0] corr;
        input [31:0] total;
        begin
            case (idx)
                8'd0: report_byte = "C";
                8'd1: report_byte = "O";
                8'd2: report_byte = "R";
                8'd3: report_byte = "R";
                8'd4: report_byte = "=";
                8'd5: report_byte = hex_char(corr[31:28]);
                8'd6: report_byte = hex_char(corr[27:24]);
                8'd7: report_byte = hex_char(corr[23:20]);
                8'd8: report_byte = hex_char(corr[19:16]);
                8'd9: report_byte = hex_char(corr[15:12]);
                8'd10: report_byte = hex_char(corr[11:8]);
                8'd11: report_byte = hex_char(corr[7:4]);
                8'd12: report_byte = hex_char(corr[3:0]);
                8'd13: report_byte = " ";
                8'd14: report_byte = "T";
                8'd15: report_byte = "O";
                8'd16: report_byte = "T";
                8'd17: report_byte = "A";
                8'd18: report_byte = "L";
                8'd19: report_byte = "=";
                8'd20: report_byte = hex_char(total[31:28]);
                8'd21: report_byte = hex_char(total[27:24]);
                8'd22: report_byte = hex_char(total[23:20]);
                8'd23: report_byte = hex_char(total[19:16]);
                8'd24: report_byte = hex_char(total[15:12]);
                8'd25: report_byte = hex_char(total[11:8]);
                8'd26: report_byte = hex_char(total[7:4]);
                8'd27: report_byte = hex_char(total[3:0]);
                8'd28: report_byte = 8'h0D;
                8'd29: report_byte = 8'h0A;
                default: report_byte = 8'h00;
            endcase
        end
    endfunction

    function [15:0] pct_hundredths;
        input [31:0] correct;
        input [31:0] total;
        reg [31:0] tmp;
        begin
            if (total == 0)
                pct_hundredths = 16'd0;
            else begin
                tmp = ((correct * 32'd10000) + (total >> 1)) / total;
                if (tmp > 32'd9999)
                    pct_hundredths = 16'd9999;
                else
                    pct_hundredths = tmp[15:0];
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            sample_idx <= 32'd0;
            class_idx <= 32'd0;
            feat_idx <= 32'd0;
            acc <= {ACC_W{1'b0}};
            best_score <= {1'b1, {(ACC_W-1){1'b0}}};
            best_class <= {LABEL_W{1'b0}};
            correct_count <= 32'd0;
            expected_correct_count <= 32'd0;
            x_addr <= {X_ADDR_W{1'b0}};
            w_addr <= {W_ADDR_W{1'b0}};
            b_addr <= {B_ADDR_W{1'b0}};
            y_addr <= {Y_ADDR_W{1'b0}};
            g_addr <= {Y_ADDR_W{1'b0}};
            uart_start <= 1'b0;
            uart_data <= 8'h00;
            report_idx <= 8'd0;
            report_value <= 32'd0;
            x_req <= 1'b0;
            x_req_addr <= 32'd0;
            expected_pct_hundredths <= 16'd0;
            actual_pct_hundredths <= 16'd0;
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            uart_start <= 1'b0;
            x_req <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        sample_idx <= 32'd0;
                        class_idx <= 32'd0;
                        feat_idx <= 32'd0;
                        correct_count <= 32'd0;
                        expected_correct_count <= 32'd0;
                        best_score <= {1'b1, {(ACC_W-1){1'b0}}};
                        best_class <= {LABEL_W{1'b0}};
                        expected_pct_hundredths <= 16'd0;
                        actual_pct_hundredths <= 16'd0;
                        state <= S_CLASS_SETUP;
                    end
                end

                S_CLASS_SETUP: begin
                    feat_idx <= 32'd0;
                    b_addr <= class_idx[B_ADDR_W-1:0];
                    state <= S_BIAS_REQ;
                end

                S_BIAS_REQ: begin
                    state <= S_BIAS_CAP;
                end

                S_BIAS_CAP: begin
                    acc <= b_dout;
                    if (USE_X_BRAM != 0) begin
                        x_addr <= sample_idx * N_FEATURES;
                    end else begin
                        x_req_addr <= sample_idx * N_FEATURES;
                        x_req <= 1'b1;
                    end
                    w_addr <= class_idx * N_FEATURES;
                    if (USE_X_BRAM != 0)
                        state <= S_MAC_REQ;
                    else
                        state <= S_X_WAIT;
                end

                S_MAC_REQ: begin
                    state <= S_MAC_CAP;
                end

                S_X_WAIT: begin
                    if (x_data_valid)
                        state <= S_MAC_CAP;
                end

                S_MAC_CAP: begin
                    acc <= acc + mul_ext;

                    if (feat_idx == N_FEATURES - 1) begin
                        state <= S_CLASS_DONE;
                    end else begin
                        feat_idx <= feat_idx + 1;
                        if (USE_X_BRAM != 0) begin
                            x_addr <= sample_idx * N_FEATURES + (feat_idx + 1);
                        end else begin
                            x_req_addr <= sample_idx * N_FEATURES + (feat_idx + 1);
                            x_req <= 1'b1;
                        end
                        w_addr <= class_idx * N_FEATURES + (feat_idx + 1);
                        if (USE_X_BRAM != 0)
                            state <= S_MAC_REQ;
                        else
                            state <= S_X_WAIT;
                    end
                end

                S_CLASS_DONE: begin
                    if (class_idx == 0 || acc > best_score) begin
                        best_score <= acc;
                        best_class <= class_idx[LABEL_W-1:0];
                    end

                    if (class_idx == N_CLASSES - 1) begin
                        y_addr <= sample_idx;
                        g_addr <= sample_idx;
                        state <= S_LABEL_REQ;
                    end else begin
                        class_idx <= class_idx + 1;
                        state <= S_CLASS_SETUP;
                    end
                end

                S_LABEL_REQ: begin
                    state <= S_LABEL_CAP;
                end

                S_LABEL_CAP: begin
                    if (best_class == y_dout[LABEL_W-1:0]) begin
                        correct_count <= correct_count + 1;
                    end
                    if (g_dout[LABEL_W-1:0] == y_dout[LABEL_W-1:0]) begin
                        expected_correct_count <= expected_correct_count + 1;
                    end
                    state <= S_NEXT_SAMPLE;
                end

                S_NEXT_SAMPLE: begin
                    if (sample_idx == N_SAMPLES - 1) begin
                        state <= S_REPORT_INIT;
                    end else begin
                        sample_idx <= sample_idx + 1;
                        class_idx <= 32'd0;
                        best_score <= {1'b1, {(ACC_W-1){1'b0}}};
                        best_class <= {LABEL_W{1'b0}};
                        state <= S_CLASS_SETUP;
                    end
                end

                S_REPORT_INIT: begin
                    actual_pct_hundredths <= pct_hundredths(correct_count, N_SAMPLES);
                    expected_pct_hundredths <= pct_hundredths(expected_correct_count, N_SAMPLES);
                    report_idx <= 8'd0;
                    state <= S_REPORT_SEND;
                end

                S_REPORT_SEND: begin
                    if (!uart_busy) begin
                        uart_data <= report_byte(report_idx, correct_count, N_SAMPLES);
                        uart_start <= 1'b1;
                        state <= S_REPORT_WAIT;
                    end
                end

                S_REPORT_WAIT: begin
                    if (uart_done_pulse) begin
                        if (report_idx == 8'd29)
                            state <= S_REPORT_DONE;
                        else begin
                            report_idx <= report_idx + 1;
                            state <= S_REPORT_SEND;
                        end
                    end
                end

                S_REPORT_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
