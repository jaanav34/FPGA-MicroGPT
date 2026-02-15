// Parameter Memory - TESTED VERSION
// Simple dual-port RAM for model parameters
module param_memory
    import microgpt_pkg::*;
(
    input  logic                        clk,
    input  logic                        rst_n,
    
    // Write interface
    input  logic                        wr_en,
    input  logic [PARAM_ADDR_WIDTH-1:0] wr_addr,
    input  fixed_t                      wr_data,
    
    // Read interface
    input  logic                        rd_en,
    input  logic [PARAM_ADDR_WIDTH-1:0] rd_addr,
    output fixed_t                      rd_data,
    output logic                        rd_valid
);

    // Parameter storage - will be implemented as Block RAM
    fixed_t param_ram [0:TOTAL_PARAMS-1];
    
    // Read pipeline registers
    fixed_t rd_data_reg;
    logic rd_valid_reg;
    
    // Initialize RAM to zero (critical!)
    initial begin
        for (int i = 0; i < TOTAL_PARAMS; i++) begin
            param_ram[i] = '0;
        end
        // Do NOT initialize rd_data_reg or rd_valid_reg here
        // They are handled by reset in always_ff
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data_reg <= '0;
            rd_valid_reg <= 1'b0;
        end else begin
            // Write operation
            if (wr_en && wr_addr < TOTAL_PARAMS) begin
                param_ram[wr_addr] <= wr_data;
            end
            
            // Read operation
            if (rd_en && rd_addr < TOTAL_PARAMS) begin
                rd_data_reg <= param_ram[rd_addr];
                rd_valid_reg <= 1'b1;
            end else begin
                rd_valid_reg <= 1'b0;
            end
        end
    end
    
    assign rd_data = rd_data_reg;
    assign rd_valid = rd_valid_reg;

endmodule : param_memory