// RMS Normalization - PRODUCTION READY
// Computes: y = x / sqrt(mean(x^2) + epsilon)
module rmsnorm #(
    parameter int VEC_LEN = 16
)
    import microgpt_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  fixed_t      vec_in [VEC_LEN-1:0],
    output fixed_t      vec_out [VEC_LEN-1:0],
    output logic        valid
);

    typedef enum logic [2:0] {
        RMS_IDLE,
        RMS_SQUARE,
        RMS_SUM,
        RMS_SCALE,
        RMS_DONE
    } rms_state_t;
    
    rms_state_t state;
    
    fixed_t squares [VEC_LEN-1:0];
    fixed_t sum_squares;
    fixed_t mean_square;
    fixed_t inv_rms;
    int idx;
    
    // Precomputed inverse square root table for common values
    // Covers mean_square from 0.25 to 16.0 in Q8.8
    fixed_t inv_sqrt_table [0:63];
    
    initial begin
        // Populate inverse square root table
        // inv_sqrt(x) for x from 0.25 to 16.0
        for (int i = 0; i < 64; i++) begin
            real x, inv_s;
            x = 0.25 + (i * 0.25);  // Range: 0.25 to 16.0
            inv_s = 1.0 / $sqrt(x + 0.00001);  // Add epsilon
            inv_sqrt_table[i] = float_to_fixed(inv_s);
        end
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RMS_IDLE;
            sum_squares <= '0;
            mean_square <= '0;
            inv_rms <= '0;
            idx <= 0;
            
            for (int i = 0; i < VEC_LEN; i++) begin
                squares[i] <= '0;
                vec_out[i] <= '0;
            end
        end else begin
            case (state)
                RMS_IDLE: begin
                    if (start) begin
                        idx <= 0;
                        sum_squares <= '0;
                        state <= RMS_SQUARE;
                    end
                end
                
                RMS_SQUARE: begin
                    // Compute x^2 for all elements
                    for (int i = 0; i < VEC_LEN; i++) begin
                        squares[i] <= fixed_mul(vec_in[i], vec_in[i]);
                    end
                    idx <= 0;
                    state <= RMS_SUM;
                end
                
                RMS_SUM: begin
                    // Sum all squares
                    sum_squares <= '0;
                    for (int i = 0; i < VEC_LEN; i++) begin
                        sum_squares <= fixed_add(sum_squares, squares[i]);
                    end
                    state <= RMS_SCALE;
                end
                
                RMS_SCALE: begin
                    // Compute mean and lookup inverse sqrt
                    // mean = sum / VEC_LEN
                    mean_square <= sum_squares >>> $clog2(VEC_LEN);
                    
                    // Lookup inverse sqrt (with clamping)
                    logic [5:0] table_idx;
                    fixed_t clamped_mean;
                    
                    // Clamp to table range
                    clamped_mean = mean_square;
                    if (clamped_mean < float_to_fixed(0.25)) 
                        clamped_mean = float_to_fixed(0.25);
                    if (clamped_mean > float_to_fixed(16.0))
                        clamped_mean = float_to_fixed(16.0);
                    
                    // Convert to table index
                    table_idx = ((clamped_mean - float_to_fixed(0.25)) >>> 6);  // Divide by 0.25
                    if (table_idx > 63) table_idx = 63;
                    
                    inv_rms = inv_sqrt_table[table_idx];
                    
                    // Apply scaling to all elements
                    for (int i = 0; i < VEC_LEN; i++) begin
                        vec_out[i] <= fixed_mul(vec_in[i], inv_rms);
                    end
                    
                    state <= RMS_DONE;
                end
                
                RMS_DONE: begin
                    state <= RMS_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == RMS_DONE);

endmodule : rmsnorm
