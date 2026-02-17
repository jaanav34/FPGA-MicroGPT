// ===========================================================================
// Transformer Layer Module
// ===========================================================================
// One complete transformer layer, matching microGPT Python reference:
//
//   x_res = x
//   x     = rmsnorm(x)
//   x     = multi_head_attention(x)   -- uses K/V cache
//   x     = x + x_res                 -- residual connection 1
//   x_res = x
//   x     = rmsnorm(x)
//   x     = mlp(x)
//   x     = x + x_res                 -- residual connection 2
//
// Fixed-point Q8.8 throughout.
// ===========================================================================

module transformer_layer
    import microgpt_pkg::*;
#(
    parameter int N_EMBD     = 16,
    parameter int N_HEAD     = 4,
    parameter int BLOCK_SIZE = 16
)
(
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start,
    input  logic        clear_cache,   // pass through to MHA KV cache
    input  logic [4:0]  pos,

    // Input embedding vector
    input  fixed_t      x_in  [N_EMBD-1:0],

    // Attention weight matrices (flattened row-major)
    input  fixed_t      attn_wq [N_EMBD*N_EMBD-1:0],
    input  fixed_t      attn_wk [N_EMBD*N_EMBD-1:0],
    input  fixed_t      attn_wv [N_EMBD*N_EMBD-1:0],
    input  fixed_t      attn_wo [N_EMBD*N_EMBD-1:0],

    // MLP weight matrices (flattened row-major)
    input  fixed_t      mlp_fc1 [(4*N_EMBD)*N_EMBD-1:0],   // (4*n_embd) x n_embd
    input  fixed_t      mlp_fc2 [N_EMBD*(4*N_EMBD)-1:0],   // n_embd x (4*n_embd)

    // Output
    output fixed_t      x_out [N_EMBD-1:0],
    output logic        valid
);

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    typedef enum logic [3:0] {
        TL_IDLE,
        TL_RMSNORM1,        // first rmsnorm
        TL_WAIT_NORM1,
        TL_ATTN,            // multi-head attention
        TL_WAIT_ATTN,
        TL_RESIDUAL1,       // x = attn_out + x_res1
        TL_RMSNORM2,        // second rmsnorm
        TL_WAIT_NORM2,
        TL_MLP,             // feed-forward
        TL_WAIT_MLP,
        TL_RESIDUAL2,       // x = mlp_out + x_res2
        TL_DONE
    } tl_state_t;

    tl_state_t state;

    // -----------------------------------------------------------------------
    // Internal buffers  (all declared here at module level)
    // -----------------------------------------------------------------------
    fixed_t x_res1   [N_EMBD-1:0];   // residual before attn block
    fixed_t x_res2   [N_EMBD-1:0];   // residual before mlp block
    fixed_t norm1_in [N_EMBD-1:0];
    fixed_t norm2_in [N_EMBD-1:0];
    integer i;

    // -----------------------------------------------------------------------
    // RMSNorm instance  (one, used twice sequentially)
    // -----------------------------------------------------------------------
    logic   norm_start;
    logic   norm_valid;
    fixed_t norm_vec_in  [N_EMBD-1:0];
    fixed_t norm_vec_out [N_EMBD-1:0];

    rmsnorm #(.VEC_LEN(N_EMBD)) u_rmsnorm (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (norm_start),
        .vec_in (norm_vec_in),
        .vec_out(norm_vec_out),
        .valid  (norm_valid)
    );

    // -----------------------------------------------------------------------
    // Multi-head attention instance
    // -----------------------------------------------------------------------
    logic   attn_start;
    logic   attn_valid;
    fixed_t attn_in  [N_EMBD-1:0];
    fixed_t attn_out [N_EMBD-1:0];

    multi_head_attention #(
        .N_EMBD    (N_EMBD),
        .N_HEAD    (N_HEAD),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) u_mha (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (attn_start),
        .clear_cache(clear_cache),
        .pos        (pos),
        .x_in       (attn_in),
        .wq         (attn_wq),
        .wk         (attn_wk),
        .wv         (attn_wv),
        .wo         (attn_wo),
        .x_out      (attn_out),
        .valid      (attn_valid)
    );

    // -----------------------------------------------------------------------
    // MLP instance
    // -----------------------------------------------------------------------
    logic   mlp_start;
    logic   mlp_valid;
    fixed_t mlp_in  [N_EMBD-1:0];
    fixed_t mlp_out [N_EMBD-1:0];

    mlp #(.N_EMBD(N_EMBD)) u_mlp (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (mlp_start),
        .x_in       (mlp_in),
        .fc1_weights(mlp_fc1),
        .fc2_weights(mlp_fc2),
        .x_out      (mlp_out),
        .valid      (mlp_valid)
    );

    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= TL_IDLE;
            norm_start  <= 0;
            attn_start  <= 0;
            mlp_start   <= 0;

            for (i = 0; i < N_EMBD; i++) begin
                x_res1      [i] <= '0;
                x_res2      [i] <= '0;
                norm_vec_in [i] <= '0;
                norm1_in    [i] <= '0;
                norm2_in    [i] <= '0;
                attn_in     [i] <= '0;
                mlp_in      [i] <= '0;
                x_out       [i] <= '0;
            end

        end else begin
            // Default: pulse signals cleared every cycle
            norm_start <= 0;
            attn_start <= 0;
            mlp_start  <= 0;

            case (state)

                // --- Wait for start ------------------------------------------
                TL_IDLE: begin
                    if (start) begin
                        // Save input as first residual
                        for (i = 0; i < N_EMBD; i++)
                            x_res1[i] <= x_in[i];
                        state <= TL_RMSNORM1;
                    end
                end

                // --- First RMSNorm -------------------------------------------
                TL_RMSNORM1: begin
                    for (i = 0; i < N_EMBD; i++)
                        norm_vec_in[i] <= x_in[i];
                    norm_start <= 1;
                    state <= TL_WAIT_NORM1;
                end

                TL_WAIT_NORM1: begin
                    if (norm_valid) begin
                        // Feed normalised result to attention
                        for (i = 0; i < N_EMBD; i++)
                            attn_in[i] <= norm_vec_out[i];
                        state <= TL_ATTN;
                    end
                end

                // --- Multi-head attention ------------------------------------
                TL_ATTN: begin
                    attn_start <= 1;
                    state <= TL_WAIT_ATTN;
                end

                TL_WAIT_ATTN: begin
                    if (attn_valid) begin
                        state <= TL_RESIDUAL1;
                    end
                end

                // --- Residual connection 1: x = attn_out + x_res1 -----------
                TL_RESIDUAL1: begin
                    for (i = 0; i < N_EMBD; i++) begin
                        // Save as second residual AND as input to norm2
                        x_res2   [i] <= fixed_add(attn_out[i], x_res1[i]);
                        norm_vec_in[i] <= fixed_add(attn_out[i], x_res1[i]);
                    end
                    state <= TL_RMSNORM2;
                end

                // --- Second RMSNorm ------------------------------------------
                TL_RMSNORM2: begin
                    norm_start <= 1;
                    state <= TL_WAIT_NORM2;
                end

                TL_WAIT_NORM2: begin
                    if (norm_valid) begin
                        for (i = 0; i < N_EMBD; i++)
                            mlp_in[i] <= norm_vec_out[i];
                        state <= TL_MLP;
                    end
                end

                // --- MLP block -----------------------------------------------
                TL_MLP: begin
                    mlp_start <= 1;
                    state <= TL_WAIT_MLP;
                end

                TL_WAIT_MLP: begin
                    if (mlp_valid) begin
                        state <= TL_RESIDUAL2;
                    end
                end

                // --- Residual connection 2: x = mlp_out + x_res2 ------------
                TL_RESIDUAL2: begin
                    for (i = 0; i < N_EMBD; i++)
                        x_out[i] <= fixed_add(mlp_out[i], x_res2[i]);
                    state <= TL_DONE;
                end

                // --- Done ----------------------------------------------------
                TL_DONE: begin
                    state <= TL_IDLE;
                end

                default: state <= TL_IDLE;

            endcase
        end
    end

    assign valid = (state == TL_DONE);

endmodule : transformer_layer