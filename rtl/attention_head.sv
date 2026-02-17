// ===========================================================================
// Attention Head Module - CORRECTED VERSION
// ===========================================================================
// Implements a single attention head with scaled dot-product attention
// Uses KV cache for efficient autoregressive generation
//
// FIXED: Proper fixed-point arithmetic in weighted sum
// ===========================================================================

module attention_head
    import microgpt_pkg::*;
#(
    parameter int HEAD_DIM = 4,
    parameter int N_EMBD = 16,
    parameter int BLOCK_SIZE = 16
)
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        clear_cache,
    input  logic [4:0]  pos,
    input  fixed_t      q_in [HEAD_DIM-1:0],
    input  fixed_t      k_in [HEAD_DIM-1:0],
    input  fixed_t      v_in [HEAD_DIM-1:0],
    output fixed_t      head_out [HEAD_DIM-1:0],
    output logic        valid
);

    typedef enum logic [2:0] {
        AH_IDLE,
        AH_COMPUTE_SCORES,
        AH_SOFTMAX,
        AH_WEIGHTED_SUM,
        AH_DONE
    } attn_head_state_t;
    
    attn_head_state_t state;
    
    // KV Cache
    fixed_t k_cache [0:BLOCK_SIZE-1][HEAD_DIM-1:0];
    fixed_t v_cache [0:BLOCK_SIZE-1][HEAD_DIM-1:0];
    logic [4:0] cache_len;
    
    // Attention computation
    fixed_t attn_scores [0:BLOCK_SIZE-1];
    fixed_t attn_weights [0:BLOCK_SIZE-1];
    
    // Dot product
    logic   dot_start, dot_valid;
    fixed_t dot_result;
    fixed_t dot_a [HEAD_DIM-1:0];
    fixed_t dot_b [HEAD_DIM-1:0];
    
    vector_dot_product #(.VEC_LEN(HEAD_DIM)) u_dot_product (
        .clk(clk), .rst_n(rst_n), .start(dot_start),
        .vec_a(dot_a), .vec_b(dot_b),
        .result(dot_result), .valid(dot_valid)
    );
    
    // Softmax
    logic   softmax_start, softmax_valid;
    fixed_t softmax_in [0:BLOCK_SIZE-1];
    fixed_t softmax_out [0:BLOCK_SIZE-1];
    
    softmax #(.VEC_LEN(BLOCK_SIZE)) u_softmax (
        .clk(clk), .rst_n(rst_n), .start(softmax_start),
        .logits(softmax_in), .probs(softmax_out), .valid(softmax_valid)
    );
    
    // Working variables
    logic [4:0] score_idx, weight_idx;
    fixed_t scale_factor;
    logic signed [31:0] accum [HEAD_DIM-1:0];  // CHANGED: 32-bit accumulator
    integer i, j;
    
    initial begin
        case (HEAD_DIM)
            4:  scale_factor = 16'h0080;  // 0.5
            8:  scale_factor = 16'h005A;  // ~0.353
            16: scale_factor = 16'h0040;  // 0.25
            default: scale_factor = 16'h0080;
        endcase
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= AH_IDLE;
            cache_len <= 0;
            score_idx <= 0;
            weight_idx <= 0;
            dot_start <= 0;
            softmax_start <= 0;
            
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
            dot_start <= 0;
            softmax_start <= 0;
            
            case (state)
                AH_IDLE: begin
                    if (clear_cache) begin
                        cache_len <= 0;
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            for (j = 0; j < HEAD_DIM; j++) begin
                                k_cache[i][j] <= '0;
                                v_cache[i][j] <= '0;
                            end
                        end
                    end else if (start) begin
                        for (j = 0; j < HEAD_DIM; j++) begin
                            k_cache[pos][j] <= k_in[j];
                            v_cache[pos][j] <= v_in[j];
                        end
                        cache_len <= pos + 1;
                        score_idx <= 0;
                        state <= AH_COMPUTE_SCORES;
                    end
                end
                
                AH_COMPUTE_SCORES: begin
                    if (!dot_start && !dot_valid) begin
                        for (j = 0; j < HEAD_DIM; j++) begin
                            dot_a[j] <= q_in[j];
                            dot_b[j] <= k_cache[score_idx][j];
                        end
                        dot_start <= 1;
                    end else if (dot_valid) begin
                        attn_scores[score_idx] <= fixed_mul(dot_result, scale_factor);
                        if (score_idx >= pos) begin
                            state <= AH_SOFTMAX;
                        end else begin
                            score_idx <= score_idx + 1;
                        end
                    end
                end
                
                AH_SOFTMAX: begin
                    if (!softmax_start) begin
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            if (i <= pos) begin
                                softmax_in[i] <= attn_scores[i];
                            end else begin
                                softmax_in[i] <= 16'sh8000;
                            end
                        end
                        softmax_start <= 1;
                    end else if (softmax_valid) begin
                        for (i = 0; i < BLOCK_SIZE; i++) begin
                            attn_weights[i] <= softmax_out[i];
                        end
                        for (j = 0; j < HEAD_DIM; j++) begin
                            accum[j] <= '0;
                        end
                        weight_idx <= 0;
                        state <= AH_WEIGHTED_SUM;
                    end
                end
                
                AH_WEIGHTED_SUM: begin
                    if (weight_idx <= pos) begin
                        for (j = 0; j < HEAD_DIM; j++) begin
                            logic signed [31:0] product;
                            logic signed [15:0] weight_val;
                            logic signed [15:0] v_val;
                            
                            // Cast to signed for proper multiplication
                            weight_val = $signed(attn_weights[weight_idx]);
                            v_val = $signed(v_cache[weight_idx][j]);
                            
                            // Multiply: Q8.8 * Q8.8 = Q16.16
                            product = weight_val * v_val;
                            
                            // CRITICAL FIX: Add to accumulator WITHOUT shifting yet
                            // Keep full precision during accumulation
                            accum[j] <= accum[j] + product;
                        end
                        weight_idx <= weight_idx + 1;
                    end else begin
                        // NOW shift accumulated result to Q8.8
                        for (j = 0; j < HEAD_DIM; j++) begin
                            head_out[j] <= fixed_t'(accum[j] >>> FRAC_BITS);
                        end
                        state <= AH_DONE;
                    end
                end
                
                AH_DONE: begin
                    state <= AH_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == AH_DONE);
    
endmodule : attention_head