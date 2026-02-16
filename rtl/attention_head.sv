// ===========================================================================
// Attention Head Module
// ===========================================================================
// Implements a single attention head with scaled dot-product attention
// Uses KV cache for efficient autoregressive generation
// 
// Operation:
//   1. Compute Q, K, V projections from input
//   2. Store K, V in cache for all previous positions
//   3. Compute attention scores: score[t] = Q · K[t] / sqrt(head_dim)
//   4. Apply softmax to get attention weights
//   5. Compute output: sum(weight[t] * V[t])
//
// Features:
//   - Pipelined computation for throughput
//   - Efficient KV cache management
//   - Scaled dot-product attention
//   - Fixed-point Q8.8 arithmetic
// ===========================================================================

module attention_head
    import microgpt_pkg::*;
#(
    parameter int HEAD_DIM = 4,      // Dimension of this attention head
    parameter int N_EMBD = 16,       // Total embedding dimension
    parameter int BLOCK_SIZE = 16    // Maximum sequence length
)
(
    input  logic        clk,
    input  logic        rst_n,
    
    // Control signals
    input  logic        start,       // Start attention computation
    input  logic        clear_cache, // Clear KV cache (start of new sequence)
    input  logic [4:0]  pos,         // Current position in sequence [0, BLOCK_SIZE-1]
    
    // Input query, key, value vectors (full n_embd dimension)
    input  fixed_t      q_in [HEAD_DIM-1:0],  // Query vector for this head
    input  fixed_t      k_in [HEAD_DIM-1:0],  // Key vector for this head  
    input  fixed_t      v_in [HEAD_DIM-1:0],  // Value vector for this head
    
    // Output
    output fixed_t      head_out [HEAD_DIM-1:0], // Attention output for this head
    output logic        valid                     // Output valid signal
);

    // -----------------------------------------------------------------------
    // State Machine
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        AH_IDLE,
        AH_COMPUTE_SCORES,
        AH_SOFTMAX,
        AH_WEIGHTED_SUM,
        AH_DONE
    } attn_head_state_t;
    
    attn_head_state_t state;
    
    // -----------------------------------------------------------------------
    // KV Cache Storage
    // -----------------------------------------------------------------------
    // Store keys and values for all previous positions
    fixed_t k_cache [0:BLOCK_SIZE-1][HEAD_DIM-1:0];
    fixed_t v_cache [0:BLOCK_SIZE-1][HEAD_DIM-1:0];
    logic [4:0] cache_len;  // Number of valid entries in cache
    
    // -----------------------------------------------------------------------
    // Attention Score Computation
    // -----------------------------------------------------------------------
    fixed_t attn_scores [0:BLOCK_SIZE-1];  // Raw attention scores
    fixed_t attn_weights [0:BLOCK_SIZE-1]; // After softmax
    
    // Dot product computation
    logic        dot_start;
    logic        dot_valid;
    fixed_t      dot_result;
    fixed_t      dot_a [HEAD_DIM-1:0];
    fixed_t      dot_b [HEAD_DIM-1:0];
    
    vector_dot_product #(
        .VEC_LEN(HEAD_DIM)
    ) u_dot_product (
        .clk(clk),
        .rst_n(rst_n),
        .start(dot_start),
        .vec_a(dot_a),
        .vec_b(dot_b),
        .result(dot_result),
        .valid(dot_valid)
    );
    
    // Softmax computation
    logic   softmax_start;
    logic   softmax_valid;
    fixed_t softmax_in [0:BLOCK_SIZE-1];
    fixed_t softmax_out [0:BLOCK_SIZE-1];
    
    softmax #(
        .VEC_LEN(BLOCK_SIZE)
    ) u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .start(softmax_start),
        .logits(softmax_in),
        .probs(softmax_out),
        .valid(softmax_valid)
    );
    
    // -----------------------------------------------------------------------
    // Working Variables
    // -----------------------------------------------------------------------
    logic [4:0] score_idx;      // Index for computing scores
    logic [4:0] weight_idx;     // Index for weighted sum
    fixed_t     scale_factor;   // 1/sqrt(HEAD_DIM) for scaling
    fixed_t     accum [HEAD_DIM-1:0]; // Accumulator for weighted sum
    integer i, j;
    
    // Precompute scale factor: 1/sqrt(HEAD_DIM)
    // For HEAD_DIM=4, sqrt(4)=2, so scale = 0.5 = 128 in Q8.8
    initial begin
        case (HEAD_DIM)
            4:  scale_factor = 16'h0080;  // 0.5
            8:  scale_factor = 16'h005A;  // ~0.353
            16: scale_factor = 16'h0040;  // 0.25
            default: scale_factor = 16'h0080;
        endcase
    end
    
    // -----------------------------------------------------------------------
    // Main State Machine
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= AH_IDLE;
            cache_len <= 0;
            score_idx <= 0;
            weight_idx <= 0;
            dot_start <= 0;
            softmax_start <= 0;
            
            // Clear cache
            for (i = 0; i < BLOCK_SIZE; i++) begin
                for (j = 0; j < HEAD_DIM; j++) begin
                    k_cache[i][j] <= '0;
                    v_cache[i][j] <= '0;
                end
                attn_scores[i] <= '0;
                attn_weights[i] <= '0;
            end
            
            for (i = 0; i < HEAD_DIM; i++) begin
                head_out[i] <= '0;
                accum[i] <= '0;
            end
            
        end else begin
            // Default: clear one-cycle signals
            dot_start <= 0;
            softmax_start <= 0;
            
            case (state)
                // ---------------------------------------------------------------
                AH_IDLE: begin
                    if (clear_cache) begin
                        // Clear the KV cache for new sequence
                        cache_len <= 0;
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            for (j = 0; j < HEAD_DIM; j++) begin
                                k_cache[i][j] <= '0;
                                v_cache[i][j] <= '0;
                            end
                        end
                    end else if (start) begin
                        // Store current K, V in cache at position 'pos'
                        for (j = 0; j < HEAD_DIM; j++) begin
                            k_cache[pos][j] <= k_in[j];
                            v_cache[pos][j] <= v_in[j];
                        end
                        cache_len <= pos + 1;  // Update cache length
                        
                        // Start computing attention scores
                        score_idx <= 0;
                        state <= AH_COMPUTE_SCORES;
                    end
                end
                
                // ---------------------------------------------------------------
                AH_COMPUTE_SCORES: begin
                    // Compute Q · K[t] for all t <= pos
                    if (!dot_start && !dot_valid) begin
                        // Set up dot product inputs
                        for (j = 0; j < HEAD_DIM; j++) begin
                            dot_a[j] <= q_in[j];
                            dot_b[j] <= k_cache[score_idx][j];
                        end
                        dot_start <= 1;
                        
                    end else if (dot_valid) begin
                        // Scale and store result: score = (Q·K) / sqrt(HEAD_DIM)
                        attn_scores[score_idx] <= fixed_mul(dot_result, scale_factor);
                        
                        if (score_idx >= pos) begin
                            // Done with all scores
                            state <= AH_SOFTMAX;
                        end else begin
                            score_idx <= score_idx + 1;
                        end
                    end
                end
                
                // ---------------------------------------------------------------
                AH_SOFTMAX: begin
                    // Apply softmax to attention scores
                    if (!softmax_start) begin
                        // Prepare softmax inputs
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            if (i <= pos) begin
                                softmax_in[i] <= attn_scores[i];
                            end else begin
                                // Mask future positions with large negative value
                                softmax_in[i] <= 16'sh8000;  // Very negative in Q8.8
                            end
                        end
                        softmax_start <= 1;
                        
                    end else if (softmax_valid) begin
                        // Store softmax outputs
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            attn_weights[i] <= softmax_out[i];
                        end
                        
                        // Initialize accumulator for weighted sum
                        for (j = 0; j < HEAD_DIM; j++) begin
                            accum[j] <= '0;
                        end
                        
                        weight_idx <= 0;
                        state <= AH_WEIGHTED_SUM;
                    end
                end
                
                // ---------------------------------------------------------------
                AH_WEIGHTED_SUM: begin
                    // Compute output = sum(weight[t] * V[t])
                    // Process one position at a time
                    if (weight_idx <= pos) begin
                        for (j = 0; j < HEAD_DIM; j++) begin
                            fixed_t weighted_val;
                            weighted_val = fixed_mul(attn_weights[weight_idx], v_cache[weight_idx][j]);
                            accum[j] <= fixed_add(accum[j], weighted_val);
                        end
                        weight_idx <= weight_idx + 1;
                        
                    end else begin
                        // Copy accumulator to output
                        for (j = 0; j < HEAD_DIM; j++) begin
                            head_out[j] <= accum[j];
                        end
                        state <= AH_DONE;
                    end
                end
                
                // ---------------------------------------------------------------
                AH_DONE: begin
                    state <= AH_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == AH_DONE);
    
endmodule : attention_head