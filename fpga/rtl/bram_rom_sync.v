module bram_rom_sync #(
    parameter integer DATA_W = 16,
    parameter integer DEPTH = 1024,
    parameter INIT_FILE = ""
) (
    input  wire                         clk,
    input  wire [$clog2(DEPTH)-1:0]     addr,
    output wire signed [DATA_W-1:0]     dout
);
    localparam integer ADDR_W = $clog2(DEPTH);
    localparam integer MEM_BITS = DATA_W * DEPTH;

    // Use XPM ROM primitive for robust BRAM mapping across deep memories.
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(MEM_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(DATA_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_xpm_sprom (
        .clka(clk),
        .ena(1'b1),
        .addra(addr),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .regcea(1'b1),
        .rsta(1'b0),
        .sleep(1'b0),
        .douta(dout),
        .sbiterra(),
        .dbiterra()
    );
endmodule
