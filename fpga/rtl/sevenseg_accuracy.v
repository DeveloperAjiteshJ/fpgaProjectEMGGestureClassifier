module sevenseg_accuracy (
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] actual_pct_hundredths,
    output reg  [7:0]  an_o,
    output reg  [6:0]  seg_o,
    output reg         dp_o
);
    reg [19:0] refresh_cnt;
    wire [2:0] sel = refresh_cnt[19:17];

    wire [15:0] act_val = (actual_pct_hundredths > 16'd9999) ? 16'd9999 : actual_pct_hundredths;

    wire [3:0] act_tens      = (act_val / 16'd1000) % 10;
    wire [3:0] act_ones      = (act_val / 16'd100) % 10;
    wire [3:0] act_tenths    = (act_val / 16'd10) % 10;
    wire [3:0] act_hundredth = act_val % 10;

    reg [3:0] digit;

    function [6:0] seg_lut;
        input [3:0] d;
        begin
            case (d)
                4'd0: seg_lut = 7'b1000000;
                4'd1: seg_lut = 7'b1111001;
                4'd2: seg_lut = 7'b0100100;
                4'd3: seg_lut = 7'b0110000;
                4'd4: seg_lut = 7'b0011001;
                4'd5: seg_lut = 7'b0010010;
                4'd6: seg_lut = 7'b0000010;
                4'd7: seg_lut = 7'b1111000;
                4'd8: seg_lut = 7'b0000000;
                4'd9: seg_lut = 7'b0010000;
                default: seg_lut = 7'b1111111;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (rst)
            refresh_cnt <= 20'd0;
        else
            refresh_cnt <= refresh_cnt + 1'b1;
    end

    always @(*) begin
        an_o = 8'b11111111;
        dp_o = 1'b1;
        digit = 4'd15;

        case (sel)
            3'd0: begin // blank
                an_o = 8'b01111111;
            end
            3'd1: begin // blank
                an_o = 8'b10111111;
            end
            3'd2: begin // blank
                an_o = 8'b11011111;
            end
            3'd3: begin // blank
                an_o = 8'b11101111;
            end
            3'd4: begin // actual tens
                an_o = 8'b11110111;
                digit = act_tens;
            end
            3'd5: begin // actual ones
                an_o = 8'b11111011;
                digit = act_ones;
                dp_o = 1'b0;
            end
            3'd6: begin // actual tenths
                an_o = 8'b11111101;
                digit = act_tenths;
            end
            default: begin // actual hundredths
                an_o = 8'b11111110;
                digit = act_hundredth;
            end
        endcase

        seg_o = seg_lut(digit);
    end
endmodule
