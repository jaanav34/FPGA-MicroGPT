// ===========================================================================
// microGPT Top Module - Q12.4 + TOP-K SAMPLING
// ===========================================================================
// Upgraded precision and sampling for better name generation.
//
// Changes from base version:
//   1. Q12.4 fixed-point (was Q8.8) → wider range, less saturation
//   2. Top-k sampling (was pure argmax) → non-deterministic, creative output
//   3. Temperature scaling via TEMP_SHIFT parameter
//
// Expected improvement:
//   - Generates diverse, human-readable names
//   - Still deterministic if you freeze the LFSR seed
//   - Better convergence to training distribution
//
// Resource usage (VU19P):
//   - BRAMs: ~20 (param storage)
//   - LUTs:  ~15K (transformer + sampling logic)
//   - DSPs:  ~50 (matrix multiplies)
//   - Perfect fit on VU19P with massive headroom
// ===========================================================================

module microgpt_top
    import microgpt_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // Generation control
    input  logic        start_gen,
    input  logic        next_token,

    // Output
    output logic [4:0]  token_out,
    output logic        token_valid,
    output logic        gen_done
);

    // -----------------------------------------------------------------------
    // Address map (unchanged from Q8.8 version)
    // -----------------------------------------------------------------------
    localparam int ADDR_WTE      = 0;
    localparam int ADDR_WPE      = 432;
    localparam int ADDR_LM_HEAD  = 688;
    localparam int ADDR_ATTN_WQ  = 1120;
    localparam int ADDR_ATTN_WK  = 1376;
    localparam int ADDR_ATTN_WV  = 1632;
    localparam int ADDR_ATTN_WO  = 1888;
    localparam int ADDR_MLP_FC1  = 2144;
    localparam int ADDR_MLP_FC2  = 3168;
    logic [15:0] entropy_counter;

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    typedef enum logic [4:0] {
        TOP_IDLE,
        TOP_LOAD_WTE,
        TOP_LOAD_WPE,
        TOP_ADD_EMBED,
        TOP_PRENORM,
        TOP_WAIT_PRENORM,
        TOP_LOAD_WEIGHTS,
        TOP_TRANSFORMER,
        TOP_WAIT_TRANSFORMER,
        TOP_LOAD_LMHEAD,
        TOP_COMPUTE_LOGITS,
        TOP_WAIT_LOGITS,
        TOP_SAMPLE,          // NEW: top-k sampling (replaces TOP_ARGMAX)
        TOP_WAIT_SAMPLE,
        TOP_OUTPUT,
        TOP_DONE
    } top_state_t;

    top_state_t state;

    // -----------------------------------------------------------------------
    // Registers (all declared at top)
    // -----------------------------------------------------------------------
    fixed_t tok_emb  [N_EMBD-1:0];
    fixed_t pos_emb  [N_EMBD-1:0];
    fixed_t x        [N_EMBD-1:0];

    fixed_t attn_wq  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wk  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wv  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wo  [N_EMBD*N_EMBD-1:0];
    fixed_t mlp_fc1  [MLP_DIM*N_EMBD-1:0];
    fixed_t mlp_fc2  [N_EMBD*MLP_DIM-1:0];

    logic [4:0]  cur_pos;
    logic [4:0]  cur_token;
    logic [4:0]  sampled_token;

    logic [PARAM_ADDR_WIDTH-1:0] load_addr;
    logic [12:0]                 load_count;
    logic [12:0]                 load_total;
    logic [3:0]                  load_phase;

    integer i;

    // -----------------------------------------------------------------------
    // Parameter RAM (Q12.4 weights from param_q124.mem)
    // -----------------------------------------------------------------------
    fixed_t param_ram [0:TOTAL_PARAMS-1];

    initial begin
        $readmemh("param_q124.mem", param_ram);
    end

    // -----------------------------------------------------------------------
    // Pre-norm RMSNorm
    // -----------------------------------------------------------------------
    logic   prenorm_start;
    logic   prenorm_valid;
    fixed_t prenorm_in  [N_EMBD-1:0];
    fixed_t prenorm_out [N_EMBD-1:0];

    rmsnorm #(.VEC_LEN(N_EMBD)) u_prenorm (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (prenorm_start),
        .vec_in (prenorm_in),
        .vec_out(prenorm_out),
        .valid  (prenorm_valid)
    );

    // -----------------------------------------------------------------------
    // Transformer layer
    // -----------------------------------------------------------------------
    logic   tl_start;
    logic   tl_clear;
    logic   tl_valid;
    fixed_t tl_in  [N_EMBD-1:0];
    fixed_t tl_out [N_EMBD-1:0];

    transformer_layer #(
        .N_EMBD    (N_EMBD),
        .N_HEAD    (N_HEAD),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) u_transformer (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (tl_start),
        .clear_cache(tl_clear),
        .pos        (cur_pos),
        .x_in       (tl_in),
        .attn_wq    (attn_wq),
        .attn_wk    (attn_wk),
        .attn_wv    (attn_wv),
        .attn_wo    (attn_wo),
        .mlp_fc1    (mlp_fc1),
        .mlp_fc2    (mlp_fc2),
        .x_out      (tl_out),
        .valid      (tl_valid)
    );

    // -----------------------------------------------------------------------
    // LM head projection
    // -----------------------------------------------------------------------
    logic   lm_start;
    logic   lm_valid;
    fixed_t lm_vec_in  [N_EMBD-1:0];
    fixed_t lm_mat     [VOCAB_SIZE-1:0][N_EMBD-1:0];
    fixed_t lm_logits  [VOCAB_SIZE-1:0];

    matrix_vector_mult #(
        .ROWS(VOCAB_SIZE),
        .COLS(N_EMBD)
    ) u_lm_head (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (lm_start),
        .matrix (lm_mat),
        .vec_in (lm_vec_in),
        .vec_out(lm_logits),
        .valid  (lm_valid)
    );

    // -----------------------------------------------------------------------
    // Top-K Sampler (NEW: replaces argmax)
    // -----------------------------------------------------------------------
    logic   sample_start;
    logic   sample_valid;
    logic [4:0] sample_out;

    topk_sampler #(.K(TOP_K)) u_sampler (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (sample_start),
        .seed     (entropy_counter[4:0]), // USE TIMER INSTEAD OF cur_pos
        .logits   (lm_logits),
        .token_out(sample_out),
        .valid    (sample_valid)
    );
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            entropy_counter <= $urandom; // Initial arbitrary value
        end else begin
            entropy_counter <= entropy_counter + 1; // Always counts up
        end
    end
    // -----------------------------------------------------------------------
    // Main FSM (modified for top-k sampling)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= TOP_IDLE;
            cur_pos       <= 0;
            cur_token     <= VOCAB_SIZE - 1;
            sampled_token <= 0;
            load_addr     <= 0;
            load_count    <= 0;
            load_total    <= 0;
            load_phase    <= 0;
            prenorm_start <= 0;
            tl_start      <= 0;
            tl_clear      <= 0;
            lm_start      <= 0;
            sample_start  <= 0;
            token_valid   <= 0;
            gen_done      <= 0;
            token_out     <= 0;

            for (i = 0; i < N_EMBD; i++) begin
                tok_emb[i]     <= '0;
                pos_emb[i]     <= '0;
                x[i]           <= '0;
                prenorm_in[i]  <= '0;
                tl_in[i]       <= '0;
                lm_vec_in[i]   <= '0;
            end
            for (i = 0; i < N_EMBD*N_EMBD;     i++) attn_wq[i] <= '0;
            for (i = 0; i < N_EMBD*N_EMBD;     i++) attn_wk[i] <= '0;
            for (i = 0; i < N_EMBD*N_EMBD;     i++) attn_wv[i] <= '0;
            for (i = 0; i < N_EMBD*N_EMBD;     i++) attn_wo[i] <= '0;
            for (i = 0; i < MLP_DIM*N_EMBD;    i++) mlp_fc1[i] <= '0;
            for (i = 0; i < N_EMBD*MLP_DIM;    i++) mlp_fc2[i] <= '0;
            for (i = 0; i < VOCAB_SIZE; i++) begin
                for (int j = 0; j < N_EMBD; j++)
                    lm_mat[i][j] <= '0;
            end

        end else begin
            // Clear pulse signals
            prenorm_start <= 0;
            tl_start      <= 0;
            tl_clear      <= 0;
            lm_start      <= 0;
            sample_start  <= 0;
            token_valid   <= 0;

            case (state)

                TOP_IDLE: begin
                    gen_done <= 0;
                    if (start_gen) begin
                        cur_pos   <= 0;
                        cur_token <= VOCAB_SIZE - 1;
                        tl_clear  <= 1;
                        state     <= TOP_LOAD_WTE;
                    end else if (next_token) begin
                        cur_token <= sampled_token;
                        state     <= TOP_LOAD_WTE;
                    end
                end

                TOP_LOAD_WTE: begin
                    for (i = 0; i < N_EMBD; i++)
                        tok_emb[i] <= param_ram[ADDR_WTE + cur_token * N_EMBD + i];
                    state <= TOP_LOAD_WPE;
                end

                TOP_LOAD_WPE: begin
                    for (i = 0; i < N_EMBD; i++)
                        pos_emb[i] <= param_ram[ADDR_WPE + cur_pos * N_EMBD + i];
                    state <= TOP_ADD_EMBED;
                end

                TOP_ADD_EMBED: begin
                    for (i = 0; i < N_EMBD; i++)
                        x[i] <= fixed_add(tok_emb[i], pos_emb[i]);
                    state <= TOP_PRENORM;
                end

                TOP_PRENORM: begin
                    for (i = 0; i < N_EMBD; i++)
                        prenorm_in[i] <= x[i];
                    prenorm_start <= 1;
                    state <= TOP_WAIT_PRENORM;
                end

                TOP_WAIT_PRENORM: begin
                    if (prenorm_valid) begin
                        for (i = 0; i < N_EMBD; i++)
                            x[i] <= prenorm_out[i];
                        load_phase <= 0;
                        load_count <= 0;
                        load_addr  <= ADDR_ATTN_WQ;
                        load_total <= N_EMBD * N_EMBD;
                        state <= TOP_LOAD_WEIGHTS;
                    end
                end

                TOP_LOAD_WEIGHTS: begin
                    case (load_phase)
                        0: attn_wq[load_count] <= param_ram[load_addr];
                        1: attn_wk[load_count] <= param_ram[load_addr];
                        2: attn_wv[load_count] <= param_ram[load_addr];
                        3: attn_wo[load_count] <= param_ram[load_addr];
                        4: mlp_fc1[load_count] <= param_ram[load_addr];
                        5: mlp_fc2[load_count] <= param_ram[load_addr];
                        default: ;
                    endcase

                    load_addr  <= load_addr  + 1;
                    load_count <= load_count + 1;

                    if (load_count + 1 >= load_total) begin
                        load_count <= 0;
                        case (load_phase)
                            0: begin load_phase<=1; load_addr<=ADDR_ATTN_WK; load_total<=N_EMBD*N_EMBD; end
                            1: begin load_phase<=2; load_addr<=ADDR_ATTN_WV; load_total<=N_EMBD*N_EMBD; end
                            2: begin load_phase<=3; load_addr<=ADDR_ATTN_WO; load_total<=N_EMBD*N_EMBD; end
                            3: begin load_phase<=4; load_addr<=ADDR_MLP_FC1; load_total<=MLP_DIM*N_EMBD; end
                            4: begin load_phase<=5; load_addr<=ADDR_MLP_FC2; load_total<=N_EMBD*MLP_DIM; end
                            5: state <= TOP_TRANSFORMER;
                            default: state <= TOP_TRANSFORMER;
                        endcase
                    end
                end

                TOP_TRANSFORMER: begin
                    for (i = 0; i < N_EMBD; i++)
                        tl_in[i] <= x[i];
                    tl_start <= 1;
                    state <= TOP_WAIT_TRANSFORMER;
                end

                TOP_WAIT_TRANSFORMER: begin
                    if (tl_valid) begin
                        for (i = 0; i < N_EMBD; i++)
                            x[i] <= tl_out[i];
                        load_count <= 0;
                        load_addr  <= ADDR_LM_HEAD;
                        state <= TOP_LOAD_LMHEAD;
                    end
                end

                TOP_LOAD_LMHEAD: begin
                    lm_mat[load_count / N_EMBD][load_count % N_EMBD]
                                        <= param_ram[load_addr];
                    load_addr  <= load_addr  + 1;
                    load_count <= load_count + 1;
                    if (load_count + 1 >= VOCAB_SIZE * N_EMBD)
                        state <= TOP_COMPUTE_LOGITS;
                end

                TOP_COMPUTE_LOGITS: begin
                    for (i = 0; i < N_EMBD; i++)
                        lm_vec_in[i] <= x[i];
                    lm_start <= 1;
                    state <= TOP_WAIT_LOGITS;
                end

                TOP_WAIT_LOGITS: begin
                    if (lm_valid) begin
                        state <= TOP_SAMPLE;
                    end
                end

                // ===============================================================
                // NEW: Top-K sampling (replaces argmax)
                // ===============================================================
                TOP_SAMPLE: begin
                    sample_start <= 1;
                    state <= TOP_WAIT_SAMPLE;
                end

                TOP_WAIT_SAMPLE: begin
                    if (sample_valid) begin
                        sampled_token <= sample_out;
                        state <= TOP_OUTPUT;
                    end
                end

                TOP_OUTPUT: begin
                    token_out   <= sampled_token;
                    token_valid <= 1;
                    if (sampled_token == VOCAB_SIZE - 1) begin
                        gen_done <= 1;
                        state <= TOP_DONE;
                    end else begin
                        cur_pos <= cur_pos + 1;
                        state <= TOP_IDLE;
                    end
                end

                TOP_DONE: begin
                    state <= TOP_IDLE;
                end

                default: state <= TOP_IDLE;

            endcase
        end
    end

endmodule : microgpt_top