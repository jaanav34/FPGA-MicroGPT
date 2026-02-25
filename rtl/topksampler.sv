// Top-K sampler with temperature scaling and LFSR-based multinomial sampling
module topk_sampler
    import microgpt_pkg::*;
#(
    parameter int K = TOP_K              // Number of candidates
)
(
    input  logic        clk,
    input  logic        rst_n,
    
    // Control
    input  logic        start,           
    input  logic [4:0]  seed,            // Random entropy source
    
    // Logits input
    input  fixed_t      logits [VOCAB_SIZE-1:0],
    
    // Output
    output logic [4:0]  token_out,       
    output logic        valid            
);

    // State and internal arrays
    typedef enum logic [2:0] {
        TK_IDLE,
        TK_FIND_TOPK,      
        TK_TEMP_SCALE,     
        TK_SOFTMAX,        
        TK_SAMPLE,         
        TK_DONE
    } topk_state_t;

    topk_state_t state;

    // Internal arrays
    fixed_t      topk_logits [K-1:0];   
    logic [4:0]  topk_ids    [K-1:0];   
    fixed_t      topk_scaled [K-1:0];   
    fixed_t      topk_probs  [K-1:0];   
    fixed_t      exp_vals    [K-1:0];   
    fixed_t      exp_table   [0:255];   
    
    // Loop variables as int (avoids zero-time simulation hangs with automatic variables)
    int          i, j, m;                
    logic [4:0]  scan_idx;               
    fixed_t      min_topk;               
    int          min_topk_pos;           
    fixed_t      max_scaled;             
    fixed_t      exp_sum;                
    logic signed [31:0] dividend;        
    logic [15:0] lfsr;                   
    logic [15:0] rand_val;               
    fixed_t      cumsum;                 
    fixed_t      threshold;              

    // Populate exponential table (Initial blocks are safe for synthesis/BRAM)
    initial begin
        for (int idx = 0; idx < 256; idx++) begin
            real x_val, e_val;
            x_val = -8.0 + (idx * 16.0 / 256.0);
            e_val = $exp(x_val);
            if (e_val > 127.0) e_val = 127.0;
            if (e_val < 0.0001) e_val = 0.0001;
            exp_table[idx] = float_to_fixed(e_val);
        end
    end
    
    // Exponential LUT lookup
    function automatic fixed_t lookup_exp(fixed_t x);
        logic signed [15:0] x_int;
        logic [7:0] t_idx;
        x_int = x;
        if (x_int < float_to_fixed(-8.0)) return float_to_fixed(0.0001);
        if (x_int > float_to_fixed(8.0))  return float_to_fixed(127.0);
        t_idx = ((x_int + float_to_fixed(8.0)) >>> (FRAC_BITS == 8 ? 4 : 0)); 
        return exp_table[t_idx];
    endfunction
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= TK_IDLE;
            token_out <= 0;
            valid     <= 0;
            scan_idx  <= 0;
            lfsr      <= 16'hACE1;
            for (m = 0; m < K; m++) begin
                topk_logits[m] <= 16'sh8000;
                topk_ids[m]    <= 0;
                topk_scaled[m] <= '0;
                topk_probs[m]  <= '0;
                exp_vals[m]    <= '0;
            end
        end else begin
            valid <= 0;
            case (state)
                TK_IDLE: begin
                    if (start) begin
                        lfsr <= {11'b0, seed} ^ 16'hACE1; // Mix seed for diversity
                        for (i = 0; i < K; i++) begin
                            topk_logits[i] <= 16'sh8000;
                            topk_ids[i]    <= 0;
                        end
                        scan_idx <= 0;
                        state <= TK_FIND_TOPK;
                    end
                end
                
                TK_FIND_TOPK: begin
                    if (scan_idx < VOCAB_SIZE) begin
                        min_topk = topk_logits[0];
                        min_topk_pos = 0;
                        for (j = 1; j < K; j++) begin
                            if (topk_logits[j] < min_topk) begin
                                min_topk = topk_logits[j];
                                min_topk_pos = j;
                            end
                        end
                        if (logits[scan_idx] > min_topk) begin
                            topk_logits[min_topk_pos] <= logits[scan_idx];
                            topk_ids[min_topk_pos]    <= scan_idx;
                        end
                        scan_idx <= scan_idx + 1;
                    end else begin
                        state <= TK_TEMP_SCALE;
                    end
                end
                
                TK_TEMP_SCALE: begin
                    for (i = 0; i < K; i++) topk_scaled[i] <= topk_logits[i] >>> TEMP_SHIFT;
                    state <= TK_SOFTMAX;
                end
                
                TK_SOFTMAX: begin
                    max_scaled = topk_scaled[0];
                    for (i = 1; i < K; i++) if (topk_scaled[i] > max_scaled) max_scaled = topk_scaled[i];
                    for (i = 0; i < K; i++) exp_vals[i] <= lookup_exp(topk_scaled[i] - max_scaled);
                    exp_sum = '0;
                    for (i = 0; i < K; i++) exp_sum = fixed_add(exp_sum, exp_vals[i]);
                    for (i = 0; i < K; i++) begin
                        dividend = exp_vals[i] <<< FRAC_BITS;
                        topk_probs[i] <= fixed_t'(dividend / exp_sum);
                    end
                    state <= TK_SAMPLE;
                end
                
                TK_SAMPLE: begin
                    lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
                    rand_val = lfsr;
                    
                    // Scale cumsum to 16-bit range to match rand_val resolution
                    cumsum = '0;
                    token_out = topk_ids[0]; 
                    
                    for (i = 0; i < K; i++) begin
                        cumsum = fixed_add(cumsum, topk_probs[i]);
                        
                        // cumsum is Q12.4; multiply by 4096 to map [0,1] → [0,65536]
                        if (rand_val <= (int'(cumsum) << 12)) begin
                            token_out = topk_ids[i];
                            break; 
                        end
                    end
                    state <= TK_DONE;
                end
                
                TK_DONE: begin
                    valid <= 1;
                    state <= TK_IDLE;
                end
                default: state <= TK_IDLE;
            endcase
        end
    end
endmodule