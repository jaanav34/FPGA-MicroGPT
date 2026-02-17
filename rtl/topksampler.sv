// ===========================================================================
// Top-K Sampler with Temperature Scaling
// ===========================================================================
// Replaces pure argmax with probabilistic sampling from top-k candidates.
//
// Algorithm:
//   1. Find top-k highest logits (parallel comparators)
//   2. Apply temperature scaling: logit' = logit >> TEMP_SHIFT
//   3. Compute softmax over top-k only
//   4. Sample using LFSR-generated random number
//
// Benefits:
//   - Non-deterministic output (diversity in generation)
//   - Temperature control (creativity vs coherence tradeoff)
//   - Efficient: only k comparisons, not full VOCAB_SIZE softmax
//
// Resource usage on VU19P:
//   - LUTs: ~500 (comparator tree + softmax)
//   - FFs:  ~300 (state + top-k storage)
//   - BRAM: 0 (all registers)
//   - DSP:  0 (fixed-point multiply done in fabric)
// ===========================================================================

module topk_sampler
    import microgpt_pkg::*;
#(
    parameter int K = TOP_K              // Number of candidates (default 5)
)
(
    input  logic        clk,
    input  logic        rst_n,
    
    // Control
    input  logic        start,           // Begin sampling
    input  logic [4:0]  seed,            // LFSR seed (use cur_pos or timer)
    
    // Logits input
    input  fixed_t      logits [VOCAB_SIZE-1:0],
    
    // Output
    output logic [4:0]  token_out,       // Sampled token ID
    output logic        valid            // High for 1 cycle when done
);

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        TK_IDLE,
        TK_FIND_TOPK,      // Find k highest logits
        TK_TEMP_SCALE,     // Apply temperature: logit >> TEMP_SHIFT
        TK_SOFTMAX,        // Compute exp and normalize over k candidates
        TK_SAMPLE,         // Use LFSR to pick from distribution
        TK_DONE
    } topk_state_t;

    topk_state_t state;

    // -----------------------------------------------------------------------
    // Top-K storage
    // -----------------------------------------------------------------------
    fixed_t      topk_logits [K-1:0];   // k highest logit values
    logic [4:0]  topk_ids    [K-1:0];   // corresponding token IDs
    fixed_t      topk_scaled [K-1:0];   // after temperature scaling
    fixed_t      topk_probs  [K-1:0];   // after softmax
    
    // -----------------------------------------------------------------------
    // Working variables (all declared at top)
    // -----------------------------------------------------------------------
    logic [4:0]  scan_idx;               // Loop counter for finding top-k
    fixed_t      min_topk;               // Smallest value in current top-k
    logic [2:0]  min_topk_pos;           // Its position in topk_logits[]
    logic [2:0]  i;
    logic [2:0]  j;
    
    // Temperature scaling
    fixed_t      max_scaled;             // Max of scaled logits (for softmax stability)
    
    // Softmax
    fixed_t      exp_vals [K-1:0];       // Exponentials
    fixed_t      exp_sum;                // Sum of exponentials
    logic signed [31:0] dividend;        // For normalization
    
    // LFSR for sampling
    logic [15:0] lfsr;                   // 16-bit LFSR
    logic [15:0] rand_val;               // Random value in [0, 65535]
    fixed_t      cumsum;                 // Cumulative probability
    logic [15:0] threshold;              // rand_val scaled to [0, 1.0] in Q12.4
    
    // -----------------------------------------------------------------------
    // Exponential lookup table (reuse from softmax module)
    // -----------------------------------------------------------------------
    fixed_t exp_table [0:255];
    
    initial begin
        // exp(x) for x in [-8, 8] in Q12.4
        for (int idx = 0; idx < 256; idx++) begin
            real x, exp_val;
            x = -8.0 + (idx * 16.0 / 256.0);
            exp_val = $exp(x);
            if (exp_val > 2047.0) exp_val = 2047.0;
            if (exp_val < 0.0001) exp_val = 0.0001;
            exp_table[idx] = float_to_fixed(exp_val);
        end
    end
    
    function automatic fixed_t lookup_exp(fixed_t x);
        logic signed [15:0] x_int;
        logic [7:0] table_idx;
        
        x_int = x;
        
        // Clamp to [-8, 8] in Q12.4
        if (x_int < float_to_fixed(-8.0))
            return float_to_fixed(0.0001);
        if (x_int > float_to_fixed(8.0))
            return float_to_fixed(2047.0);
        
        // Map to [0, 255]
        table_idx = ((x_int + float_to_fixed(8.0)) >>> 4);
        if (table_idx > 255) table_idx = 255;
        
        return exp_table[table_idx];
    endfunction
    
    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= TK_IDLE;
            token_out <= 0;
            valid     <= 0;
            scan_idx  <= 0;
            lfsr      <= 16'hACE1;  // Non-zero seed
            
            for (i = 0; i < K; i++) begin
                topk_logits[i] <= 16'sh8000;  // -2048 (most negative)
                topk_ids[i]    <= 0;
                topk_scaled[i] <= '0;
                topk_probs[i]  <= '0;
                exp_vals[i]    <= '0;
            end
            
        end else begin
            valid <= 0;  // Pulse signal
            
            case (state)
                
                // ===========================================================
                TK_IDLE: begin
                    if (start) begin
                        // Initialize LFSR with seed
                        lfsr <= {11'b0, seed};
                        if (seed == 0) lfsr <= 16'hACE1;  // Avoid all-zero
                        
                        // Reset top-k array
                        for (i = 0; i < K; i++) begin
                            topk_logits[i] <= 16'sh8000;
                            topk_ids[i]    <= 0;
                        end
                        
                        scan_idx <= 0;
                        state <= TK_FIND_TOPK;
                    end
                end
                
                // ===========================================================
                // Find top-k logits using insertion sort approach
                // ===========================================================
                TK_FIND_TOPK: begin
                    if (scan_idx < VOCAB_SIZE) begin
                        // Find minimum in current top-k
                        min_topk     = topk_logits[0];
                        min_topk_pos = 0;
                        for (i = 1; i < K; i++) begin
                            if (topk_logits[i] < min_topk) begin
                                min_topk     = topk_logits[i];
                                min_topk_pos = i;
                            end
                        end
                        
                        // If current logit > min, replace
                        if (logits[scan_idx] > min_topk) begin
                            topk_logits[min_topk_pos] <= logits[scan_idx];
                            topk_ids[min_topk_pos]    <= scan_idx;
                        end
                        
                        scan_idx <= scan_idx + 1;
                        
                    end else begin
                        state <= TK_TEMP_SCALE;
                    end
                end
                
                // ===========================================================
                // Temperature scaling: logit' = logit >> TEMP_SHIFT
                // ===========================================================
                TK_TEMP_SCALE: begin
                    for (i = 0; i < K; i++) begin
                        topk_scaled[i] <= topk_logits[i] >>> TEMP_SHIFT;
                    end
                    
                    // Find max for softmax stability
                    max_scaled = topk_scaled[0];
                    for (i = 1; i < K; i++) begin
                        if (topk_scaled[i] > max_scaled)
                            max_scaled = topk_scaled[i];
                    end
                    
                    state <= TK_SOFTMAX;
                end
                
                // ===========================================================
                // Softmax over k candidates
                // ===========================================================
                TK_SOFTMAX: begin
                    // Compute exp(scaled - max)
                    for (i = 0; i < K; i++) begin
                        exp_vals[i] <= lookup_exp(topk_scaled[i] - max_scaled);
                    end
                    
                    // Sum exponentials
                    exp_sum = '0;
                    for (i = 0; i < K; i++) begin
                        exp_sum = fixed_add(exp_sum, exp_vals[i]);
                    end
                    
                    // Normalize
                    for (i = 0; i < K; i++) begin
                        if (exp_vals[i] == '0) begin
                            topk_probs[i] <= '0;
                        end else begin
                            dividend = exp_vals[i] <<< FRAC_BITS;
                            topk_probs[i] <= fixed_t'(dividend / exp_sum);
                        end
                    end
                    
                    state <= TK_SAMPLE;
                end
                
                // ===========================================================
                // Sample using LFSR random number
                // ===========================================================
                TK_SAMPLE: begin
                    // Advance LFSR (16-bit Fibonacci LFSR, taps at 16,14,13,11)
                    lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
                    rand_val = lfsr;
                    
                    // Convert to [0, 1.0] in Q12.4: rand_val / 65536 * 16 = rand_val >> 12
                    threshold = rand_val >>> 12;  // Now in [0, 15] (Q12.4 range [0, 1.0))
                    
                    // Cumulative sampling
                    cumsum = '0;
                    token_out = topk_ids[0];  // Default to first candidate
                    
                    for (i = 0; i < K; i++) begin
                        cumsum = fixed_add(cumsum, topk_probs[i]);
                        if (threshold < cumsum) begin
                            token_out = topk_ids[i];
                            disable for;  // Break on first match
                        end
                    end
                    
                    state <= TK_DONE;
                end
                
                // ===========================================================
                TK_DONE: begin
                    valid <= 1;
                    state <= TK_IDLE;
                end
                
                default: state <= TK_IDLE;
                
            endcase
        end
    end

endmodule : topk_sampler