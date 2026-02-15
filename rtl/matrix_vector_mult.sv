// Matrix-Vector Multiplication - PRODUCTION READY
// Computes y = M * x where M is (ROWS x COLS) and x is (COLS x 1)
module matrix_vector_mult 
    import microgpt_pkg::*;
#(
    parameter int ROWS = 16,
    parameter int COLS = 16
)
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  fixed_t      vec_in [COLS-1:0],
    input  fixed_t      matrix [ROWS-1:0][COLS-1:0],  // Direct matrix input for testing
    output fixed_t      vec_out [ROWS-1:0],
    output logic        valid
);

    typedef enum logic [1:0] {
        MV_IDLE,
        MV_COMPUTE,
        MV_DONE
    } mv_state_t;
    
    mv_state_t state;
    int row_idx;
    
    // Dot product instances - one per row for parallel computation
    fixed_t dot_results [ROWS-1:0];
    logic dot_start [ROWS-1:0];
    logic dot_valid [ROWS-1:0];
    fixed_t matrix_rows [ROWS-1:0][COLS-1:0];
    
    genvar r;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_dot_products
            vector_dot_product #(.VEC_LEN(COLS)) dot_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(dot_start[r]),
                .vec_a(matrix_rows[r]),
                .vec_b(vec_in),
                .result(dot_results[r]),
                .valid(dot_valid[r])
            );
        end
    endgenerate
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= MV_IDLE;
            row_idx <= 0;
            for (int i = 0; i < ROWS; i++) begin
                vec_out[i] <= '0;
                for (int j = 0; j < COLS; j++) begin
                    matrix_rows[i][j] <= '0;
                end
            end
        end else begin
            case (state)
                MV_IDLE: begin
                    if (start) begin
                        // Load matrix rows
                        for (int i = 0; i < ROWS; i++) begin
                            for (int j = 0; j < COLS; j++) begin
                                matrix_rows[i][j] <= matrix[i][j];
                            end
                        end
                        row_idx <= 0;
                        state <= MV_COMPUTE;
                    end
                end
                
                MV_COMPUTE: begin
                    // Wait for all dot products to complete
                    if (&dot_valid) begin  // All valid signals high
                        // Capture results
                        for (int i = 0; i < ROWS; i++) begin
                            vec_out[i] <= dot_results[i];
                        end
                        state <= MV_DONE;
                    end
                end
                
                MV_DONE: begin
                    state <= MV_IDLE;
                end
            endcase
        end
    end
    
    // Start all dot products simultaneously
    always_comb begin
        for (int i = 0; i < ROWS; i++) begin
            dot_start[i] = (state == MV_COMPUTE) && !dot_valid[i];
        end
    end
    
    assign valid = (state == MV_DONE);

endmodule : matrix_vector_mult
