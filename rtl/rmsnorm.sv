// RMS Normalization - PRODUCTION READY (FINAL)
// Computes: y = x / sqrt(mean(x^2) + epsilon)
module rmsnorm 
    import microgpt_pkg::*;
#(
    parameter int VEC_LEN = 16
)
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  fixed_t      vec_in [VEC_LEN-1:0],
    output fixed_t      vec_out [VEC_LEN-1:0],
    output logic        valid
);

    // ALL DECLARATIONS AT TOP!
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
    fixed_t temp_mean;
    real mean_val;
    fixed_t temp_sum;
    logic [7:0] idx;
    
    // Inverse square root lookup table
    // 256 entries covering 0.001 to 16.0 for better small value support
    fixed_t inv_sqrt_table [0:255];
    
    initial begin
        // Populate inverse square root table with wider range
        for (int i = 0; i < 256; i++) begin
            real x, inv_s;
            // Map index to value: 0.001 to 16.0
            // Step size = (16 - 0.001) / 256 ≈ 0.0624
            x = 0.001 + (i * 0.0624);
            inv_s = 1.0 / $sqrt(x + 0.00001);
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
                        sum_squares <= '0;
                        state <= RMS_SQUARE;
                    end
                end
                
                RMS_SQUARE: begin
                    // Compute x^2 for all elements
                    for (int i = 0; i < VEC_LEN; i++) begin
                        squares[i] <= fixed_mul(vec_in[i], vec_in[i]);
                    end
                    state <= RMS_SUM;
                end
                
                RMS_SUM: begin
                    // Sum all squares using blocking assignment
                    temp_sum = '0;
                    for (int i = 0; i < VEC_LEN; i++) begin
                        temp_sum = fixed_add(temp_sum, squares[i]);
                    end
                    sum_squares <= temp_sum;
                    state <= RMS_SCALE;
                end
                
                RMS_SCALE: begin
                   
                    
                    // Compute mean: sum / VEC_LEN
                    temp_mean = sum_squares >>> $clog2(VEC_LEN);
                    mean_square <= temp_mean;
                    
                    // Convert to real for table lookup calculation
                    mean_val = fixed_to_float(temp_mean);
                    
                    // Clamp to table range
                    if (mean_val < 0.001) mean_val = 0.001;
                    if (mean_val > 16.0) mean_val = 16.0;
                    
                    // Map to table index: (value - 0.001) / 0.0624
                    idx = $rtoi((mean_val - 0.001) / 0.0624);
                    if (idx > 255) idx = 255;
                    
                    inv_rms = inv_sqrt_table[idx];
                    
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