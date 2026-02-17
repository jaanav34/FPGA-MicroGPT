// ===========================================================================
// microGPT Top Module
// ===========================================================================
// Complete autoregressive inference engine.
//
// Matches Python reference (gpt() function) exactly:
//
//   tok_emb = wte[token_id]             <- embedding lookup
//   pos_emb = wpe[pos_id]               <- position lookup
//   x = tok_emb + pos_emb               <- add embeddings
//   x = rmsnorm(x)                      <- pre-norm (as in Python line 112)
//   x = transformer_layer(x)            <- single layer
//   logits = lm_head * x                <- project to vocab
//
// Weight address map (matches Python state_dict iteration order):
//   [0      ] wte       27 * 16  =   432  (token embeddings)
//   [432    ] wpe       16 * 16  =   256  (position embeddings)
//   [688    ] lm_head   27 * 16  =   432  (output projection)
//   [1120   ] attn_wq   16 * 16  =   256
//   [1376   ] attn_wk   16 * 16  =   256
//   [1632   ] attn_wv   16 * 16  =   256
//   [1888   ] attn_wo   16 * 16  =   256
//   [2144   ] mlp_fc1   64 * 16  =  1024
//   [3168   ] mlp_fc2   16 * 64  =  1024
//   Total                         = 4192
//
// param.mem format: $readmemh, one Q8.8 value (16-bit hex) per line,
//                   row-major, address 0 first.
// ===========================================================================

module microgpt_top
    import microgpt_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // --- generation control ---
    input  logic        start_gen,      // pulse to begin generating a new sequence
    input  logic        next_token,     // pulse to advance one position (after reading token_out)

    // --- output ---
    output logic [4:0]  token_out,      // predicted next token id
    output logic        token_valid,    // high for one cycle when token_out is ready
    output logic        gen_done        // high when BOS predicted (end of sequence)
);

    // -----------------------------------------------------------------------
    // Address base constants (matching Python state_dict order)
    // -----------------------------------------------------------------------
    localparam int ADDR_WTE      = 0;
    localparam int ADDR_WPE      = ADDR_WTE    + VOCAB_SIZE * N_EMBD;   // 432
    localparam int ADDR_LM_HEAD  = ADDR_WPE    + BLOCK_SIZE * N_EMBD;   // 688
    localparam int ADDR_ATTN_WQ  = ADDR_LM_HEAD + VOCAB_SIZE * N_EMBD;  // 1120
    localparam int ADDR_ATTN_WK  = ADDR_ATTN_WQ + N_EMBD * N_EMBD;     // 1376
    localparam int ADDR_ATTN_WV  = ADDR_ATTN_WK + N_EMBD * N_EMBD;     // 1632
    localparam int ADDR_ATTN_WO  = ADDR_ATTN_WV + N_EMBD * N_EMBD;     // 1888
    localparam int ADDR_MLP_FC1  = ADDR_ATTN_WO + N_EMBD * N_EMBD;     // 2144
    localparam int ADDR_MLP_FC2  = ADDR_MLP_FC1 + MLP_DIM * N_EMBD;    // 3168

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    typedef enum logic [4:0] {
        TOP_IDLE,
        TOP_LOAD_WTE,       // read token embedding from param_ram
        TOP_LOAD_WPE,       // read position embedding from param_ram
        TOP_ADD_EMBED,      // x = tok_emb + pos_emb
        TOP_PRENORM,        // x = rmsnorm(x)
        TOP_WAIT_PRENORM,
        TOP_LOAD_WEIGHTS,   // burst-read all 4 attn + 2 mlp weight matrices
        TOP_WAIT_WEIGHTS,
        TOP_TRANSFORMER,    // run transformer_layer
        TOP_WAIT_TRANSFORMER,
        TOP_LOAD_LMHEAD,    // read lm_head weights row by row
        TOP_WAIT_LMHEAD,
        TOP_COMPUTE_LOGITS, // logits = lm_head * x
        TOP_WAIT_LOGITS,
        TOP_ARGMAX,         // find highest logit → token_out
        TOP_OUTPUT,         // assert token_valid for one cycle
        TOP_DONE
    } top_state_t;

    top_state_t state;

    // -----------------------------------------------------------------------
    // Internal registers  (all declared at module top)
    // -----------------------------------------------------------------------
    fixed_t tok_emb  [N_EMBD-1:0];
    fixed_t pos_emb  [N_EMBD-1:0];
    fixed_t x        [N_EMBD-1:0];   // working embedding vector

    // Weight matrices held in registers (loaded once per token)
    fixed_t attn_wq  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wk  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wv  [N_EMBD*N_EMBD-1:0];
    fixed_t attn_wo  [N_EMBD*N_EMBD-1:0];
    fixed_t mlp_fc1  [MLP_DIM*N_EMBD-1:0];
    fixed_t mlp_fc2  [N_EMBD*MLP_DIM-1:0];
    fixed_t lm_head  [VOCAB_SIZE*N_EMBD-1:0];  // output projection

    // Generation state
    logic [4:0]  cur_pos;        // current generation position
    logic [4:0]  cur_token;      // current input token
    logic [4:0]  best_token;     // argmax result
    fixed_t      best_logit;     // argmax running max
    logic [4:0]  argmax_idx;     // loop counter for argmax

    // Counters for burst memory loads
    logic [PARAM_ADDR_WIDTH-1:0] load_addr;   // next param_ram read address
    logic [12:0]                 load_count;  // how many words loaded so far
    logic [12:0]                 load_total;  // how many words to load total
    logic [3:0]                  load_phase;  // which matrix we're loading (0-5)

    integer i;

    // -----------------------------------------------------------------------
    // Parameter memory (loads from param.mem at simulation start)
    // -----------------------------------------------------------------------
    fixed_t param_ram [0:TOTAL_PARAMS-1];

    initial begin
        $readmemh("params.mem", param_ram);
    end

    // -----------------------------------------------------------------------
    // Pre-norm RMSNorm instance
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
    // Transformer layer instance
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
    // LM head matrix-vector multiply
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
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= TOP_IDLE;
            cur_pos       <= 0;
            cur_token     <= VOCAB_SIZE - 1;  // BOS token
            best_token    <= 0;
            best_logit    <= 16'sh8000;       // most negative
            argmax_idx    <= 0;
            load_addr     <= 0;
            load_count    <= 0;
            load_total    <= 0;
            load_phase    <= 0;
            prenorm_start <= 0;
            tl_start      <= 0;
            tl_clear      <= 0;
            lm_start      <= 0;
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
            for (i = 0; i < VOCAB_SIZE*N_EMBD;  i++) lm_head[i] <= '0;
            for (i = 0; i < VOCAB_SIZE; i++) begin
                for (int j = 0; j < N_EMBD; j++)
                    lm_mat[i][j] <= '0;
            end

        end else begin
            // Default: clear pulse signals
            prenorm_start <= 0;
            tl_start      <= 0;
            tl_clear      <= 0;
            lm_start      <= 0;
            token_valid   <= 0;

            case (state)

                // =============================================================
                TOP_IDLE: begin
                    gen_done <= 0;
                    if (start_gen) begin
                        cur_pos   <= 0;
                        cur_token <= VOCAB_SIZE - 1;  // start with BOS
                        tl_clear  <= 1;               // flush KV cache
                        state     <= TOP_LOAD_WTE;
                    end else if (next_token) begin
                        // advance to next position using last predicted token
                        cur_token <= best_token;
                        state     <= TOP_LOAD_WTE;
                    end
                end

                // =============================================================
                // Load token embedding: wte[cur_token][0..N_EMBD-1]
                // =============================================================
                TOP_LOAD_WTE: begin
                    // Combinatorially copy N_EMBD words from param_ram
                    for (i = 0; i < N_EMBD; i++)
                        tok_emb[i] <= param_ram[ADDR_WTE + cur_token * N_EMBD + i];
                    state <= TOP_LOAD_WPE;
                end

                // =============================================================
                // Load position embedding: wpe[cur_pos][0..N_EMBD-1]
                // =============================================================
                TOP_LOAD_WPE: begin
                    for (i = 0; i < N_EMBD; i++)
                        pos_emb[i] <= param_ram[ADDR_WPE + cur_pos * N_EMBD + i];
                    state <= TOP_ADD_EMBED;
                end

                // =============================================================
                // x = tok_emb + pos_emb
                // =============================================================
                TOP_ADD_EMBED: begin
                    for (i = 0; i < N_EMBD; i++)
                        x[i] <= fixed_add(tok_emb[i], pos_emb[i]);
                    state <= TOP_PRENORM;
                end

                // =============================================================
                // x = rmsnorm(x)   (Python line 112)
                // =============================================================
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
                        // Kick off weight loading
                        load_phase <= 0;
                        load_count <= 0;
                        load_addr  <= ADDR_ATTN_WQ;
                        load_total <= N_EMBD * N_EMBD;  // first: attn_wq
                        state <= TOP_LOAD_WEIGHTS;
                    end
                end

                // =============================================================
                // Burst-load all 6 weight matrices from param_ram
                // phase 0: attn_wq  (256)
                // phase 1: attn_wk  (256)
                // phase 2: attn_wv  (256)
                // phase 3: attn_wo  (256)
                // phase 4: mlp_fc1  (1024)
                // phase 5: mlp_fc2  (1024)
                // =============================================================
                TOP_LOAD_WEIGHTS: begin
                    // Load one word per cycle
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
                        // This matrix done — advance phase
                        load_count <= 0;
                        case (load_phase)
                            0: begin load_phase<=1; load_addr<=ADDR_ATTN_WK; load_total<=N_EMBD*N_EMBD; end
                            1: begin load_phase<=2; load_addr<=ADDR_ATTN_WV; load_total<=N_EMBD*N_EMBD; end
                            2: begin load_phase<=3; load_addr<=ADDR_ATTN_WO; load_total<=N_EMBD*N_EMBD; end
                            3: begin load_phase<=4; load_addr<=ADDR_MLP_FC1; load_total<=MLP_DIM*N_EMBD; end
                            4: begin load_phase<=5; load_addr<=ADDR_MLP_FC2; load_total<=N_EMBD*MLP_DIM; end
                            5: state <= TOP_TRANSFORMER;  // all done
                            default: state <= TOP_TRANSFORMER;
                        endcase
                    end
                end

                // =============================================================
                // Run transformer layer
                // =============================================================
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
                        // Load lm_head weights
                        load_count <= 0;
                        load_addr  <= ADDR_LM_HEAD;
                        state <= TOP_LOAD_LMHEAD;
                    end
                end

                // =============================================================
                // Load lm_head weights (VOCAB_SIZE * N_EMBD = 432 words)
                // Simultaneously reshape flat array into 2D lm_mat
                // =============================================================
                TOP_LOAD_LMHEAD: begin
                    lm_head[load_count] <= param_ram[load_addr];
                    lm_mat[load_count / N_EMBD][load_count % N_EMBD]
                                        <= param_ram[load_addr];
                    load_addr  <= load_addr  + 1;
                    load_count <= load_count + 1;
                    if (load_count + 1 >= VOCAB_SIZE * N_EMBD)
                        state <= TOP_COMPUTE_LOGITS;
                end

                // =============================================================
                // logits = lm_head * x
                // =============================================================
                TOP_COMPUTE_LOGITS: begin
                    for (i = 0; i < N_EMBD; i++)
                        lm_vec_in[i] <= x[i];
                    lm_start <= 1;
                    state <= TOP_WAIT_LOGITS;
                end

                TOP_WAIT_LOGITS: begin
                    if (lm_valid) begin
                        // Seed argmax
                        best_logit <= lm_logits[0];
                        best_token <= 0;
                        argmax_idx <= 1;
                        state <= TOP_ARGMAX;
                    end
                end

                // =============================================================
                // Argmax over logits → best_token
                // =============================================================
                TOP_ARGMAX: begin
                    if (argmax_idx < VOCAB_SIZE) begin
                        if (lm_logits[argmax_idx] > best_logit) begin
                            best_logit <= lm_logits[argmax_idx];
                            best_token <= argmax_idx;
                        end
                        argmax_idx <= argmax_idx + 1;
                    end else begin
                        state <= TOP_OUTPUT;
                    end
                end

                // =============================================================
                // Emit result
                // =============================================================
                TOP_OUTPUT: begin
                    token_out   <= best_token;
                    token_valid <= 1;
                    if (best_token == VOCAB_SIZE - 1) begin
                        // Predicted BOS = end of sequence
                        gen_done <= 1;
                        state <= TOP_DONE;
                    end else begin
                        cur_pos <= cur_pos + 1;
                        state <= TOP_IDLE;  // wait for next_token pulse
                    end
                end

                TOP_DONE: begin
                    // Stay here until start_gen resets everything
                    state <= TOP_IDLE;
                end

                default: state <= TOP_IDLE;

            endcase
        end
    end

endmodule : microgpt_top