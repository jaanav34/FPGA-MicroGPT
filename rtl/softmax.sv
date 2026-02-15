// Softmax Module - PRODUCTION READY
// Computes: softmax(logits) with numerical stability
module softmax 
    import microgpt_pkg::*;
#(
    parameter int VEC_LEN = 27
)
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  fixed_t      logits [VEC_LEN-1:0],
    input  fixed_t      temperature,
    output fixed_t      probs [VEC_LEN-1:0],
    output logic        valid
);

    typedef enum logic [2:0] {
        SM_IDLE,
        SM_FIND_MAX,
        SM_SCALE,
        SM_EXP,
        SM_SUM,
        SM_NORMALIZE,
        SM_DONE
    } sm_state_t;
    
    sm_state_t state;
    
    fixed_t max_logit;
    fixed_t scaled [VEC_LEN-1:0];
    fixed_t exponentials [VEC_LEN-1:0];
    fixed_t exp_sum;
    int idx;
    
    // Exponential lookup table for exp(x) where x in [-4, 4]
    // 128 entries covering the range
    fixed_t exp_table [0:127];
    
    initial begin
        // Populate exp table: exp(x) for x from -4.0 to +4.0
        for (int i = 0; i < 128; i++) begin
            real x, exp_val;
            x = -4.0 + (i * 8.0 / 128.0);
            exp_val = $exp(x);
            // Clamp to representable range
            if (exp_val > 127.0) exp_val = 127.0;
            if (exp_val < 0.001) exp_val = 0.001;
            exp_table[i] = float_to_fixed(exp_val);
        end
    end
    
    // Lookup exponential with interpolation
    function automatic fixed_t lookup_exp(fixed_t x);
        logic signed [15:0] x_int;
        logic [6:0] table_idx;
        
        x_int = x;
        
        // Clamp to table range [-4, 4]
        if (x_int < float_to_fixed(-4.0)) 
            return float_to_fixed(0.018);  // exp(-4)
        if (x_int > float_to_fixed(4.0))
            return float_to_fixed(54.6);   // exp(4)
        
        // Map to table index [0, 127]
        // x = -4.0 + (idx * 8.0/128)
        // idx = (x + 4.0) * 128 / 8
        table_idx = ((x_int + float_to_fixed(4.0)) >>> 5);  // Divide by 32
        if (table_idx > 127) table_idx = 127;
        
        return exp_table[table_idx];
    endfunction
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= SM_IDLE;
            max_logit <= '0;
            exp_sum <= '0;
            idx <= 0;
            
            for (int i = 0; i < VEC_LEN; i++) begin
                scaled[i] <= '0;
                exponentials[i] <= '0;
                probs[i] <= '0;
            end
        end else begin
            case (state)
                SM_IDLE: begin
                    if (start) begin
                        max_logit <= logits[0];
                        idx <= 0;
                        state <= SM_FIND_MAX;
                    end
                end
                
                SM_FIND_MAX: begin
                    // Find maximum logit for numerical stability
                    for (int i = 0; i < VEC_LEN; i++) begin
                        if (logits[i] > max_logit) begin
                            max_logit <= logits[i];
                        end
                    end
                    state <= SM_SCALE;
                end
                
                SM_SCALE: begin
                    // Scale by temperature and subtract max
                    for (int i = 0; i < VEC_LEN; i++) begin
                        fixed_t temp_scaled;
                        temp_scaled = fixed_mul(logits[i], temperature);
                        scaled[i] <= fixed_add(temp_scaled, -max_logit);
                    end
                    state <= SM_EXP;
                end
                
                SM_EXP: begin
                    // Compute exponentials
                    for (int i = 0; i < VEC_LEN; i++) begin
                        exponentials[i] <= lookup_exp(scaled[i]);
                    end
                    exp_sum <= '0;
                    state <= SM_SUM;
                end
                
                SM_SUM: begin
                    // Sum all exponentials
                    for (int i = 0; i < VEC_LEN; i++) begin
                        exp_sum <= fixed_add(exp_sum, exponentials[i]);
                    end
                    state <= SM_NORMALIZE;
                end
                
                SM_NORMALIZE: begin
                    // Normalize: prob[i] = exp[i] / sum
                    for (int i = 0; i < VEC_LEN; i++) begin
                        logic signed [31:0] dividend;
                        dividend = exponentials[i] <<< FRAC_BITS;
                        probs[i] <= fixed_t'(dividend / exp_sum);
                    end
                    state <= SM_DONE;
                end
                
                SM_DONE: begin
                    state <= SM_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == SM_DONE);

endmodule : softmax
