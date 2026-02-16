// ===========================================================================
// Multi-Head Attention Module
// ===========================================================================
// Combines N_HEAD attention heads operating in parallel
// Each head processes HEAD_DIM dimensions of the embedding
//
// Architecture:
//   - Projects input through Q, K, V weight matrices
//   - Splits projections across N_HEAD heads
//   - Runs attention heads in parallel (or sequentially based on resources)
//   - Concatenates head outputs
//   - Projects through output matrix
//
// Features:
//   - Configurable number of heads
//   - Efficient resource sharing
//   - Pipelined computation
//   - KV cache management
// ===========================================================================

module multi_head_attention
    import microgpt_pkg::*;
#(
    parameter int N_EMBD = 16,
    parameter int N_HEAD = 4,
    parameter int BLOCK_SIZE = 16
)
(
    input  logic        clk,
    input  logic        rst_n,
    
    // Control
    input  logic        start,
    input  logic        clear_cache,
    input  logic [4:0]  pos,
    
    // Input vector (normalized)
    input  fixed_t      x_in [N_EMBD-1:0],
    
    // Weight matrices (flattened)
    input  fixed_t      wq [N_EMBD*N_EMBD-1:0],  // Query weights
    input  fixed_t      wk [N_EMBD*N_EMBD-1:0],  // Key weights
    input  fixed_t      wv [N_EMBD*N_EMBD-1:0],  // Value weights
    input  fixed_t      wo [N_EMBD*N_EMBD-1:0],  // Output weights
    
    // Output
    output fixed_t      x_out [N_EMBD-1:0],
    output logic        valid
);

    localparam int HEAD_DIM = N_EMBD / N_HEAD;
    
    // -----------------------------------------------------------------------
    // State Machine
    // -----------------------------------------------------------------------
    typedef enum logic [3:0] {
        MHA_IDLE,
        MHA_PROJ_QKV,      // Project input to Q, K, V
        MHA_WAIT_PROJ,     // Wait for projections
        MHA_RUN_HEADS,     // Run attention heads
        MHA_WAIT_HEADS,    // Wait for heads to complete
        MHA_CONCAT,        // Concatenate head outputs
        MHA_PROJ_OUT,      // Project concatenated output
        MHA_WAIT_OUT,      // Wait for output projection
        MHA_DONE
    } mha_state_t;
    
    mha_state_t state;
    
    // -----------------------------------------------------------------------
    // Projection Results
    // -----------------------------------------------------------------------
    fixed_t q_proj [N_EMBD-1:0];  // Full Q projection
    fixed_t k_proj [N_EMBD-1:0];  // Full K projection
    fixed_t v_proj [N_EMBD-1:0];  // Full V projection
    fixed_t concat [N_EMBD-1:0];  // Concatenated head outputs
    
    // -----------------------------------------------------------------------
    // Matrix-Vector Multiplication for Projections
    // -----------------------------------------------------------------------
    logic       mv_start;
    logic       mv_valid;
    fixed_t     mv_vec_in [N_EMBD-1:0];
    fixed_t     mv_mat_in [N_EMBD*N_EMBD-1:0];
    fixed_t     mv_result [N_EMBD-1:0];
    logic [1:0] mv_select;  // 0=Q, 1=K, 2=V, 3=Out

    integer i, idx;
    
    matrix_vector_mult #(
        .ROWS(N_EMBD),
        .COLS(N_EMBD)
    ) u_matmul (
        .clk(clk),
        .rst_n(rst_n),
        .start(mv_start),
        .matrix(mv_mat_in),
        .vec_in(mv_vec_in),
        .vec_out(mv_result),
        .valid(mv_valid)
    );
    
    // -----------------------------------------------------------------------
    // Attention Heads
    // -----------------------------------------------------------------------
    // For resource efficiency, we can either:
    // 1. Instantiate N_HEAD heads and run in parallel
    // 2. Instantiate 1 head and run sequentially
    // Here we use sequential approach for minimal resource usage
    
    logic        head_start;
    logic        head_clear;
    logic        head_valid;
    fixed_t      head_q_in [HEAD_DIM-1:0];
    fixed_t      head_k_in [HEAD_DIM-1:0];
    fixed_t      head_v_in [HEAD_DIM-1:0];
    fixed_t      head_out [HEAD_DIM-1:0];
    logic [2:0]  current_head;  // Which head we're processing
    
    attention_head #(
        .HEAD_DIM(HEAD_DIM),
        .N_EMBD(N_EMBD),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) u_attn_head (
        .clk(clk),
        .rst_n(rst_n),
        .start(head_start),
        .clear_cache(head_clear),
        .pos(pos),
        .q_in(head_q_in),
        .k_in(head_k_in),
        .v_in(head_v_in),
        .head_out(head_out),
        .valid(head_valid)
    );
    
    // -----------------------------------------------------------------------
    // Control Logic
    // -----------------------------------------------------------------------
    logic [1:0] proj_count;  // Track Q, K, V projections
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= MHA_IDLE;
            mv_start <= 0;
            head_start <= 0;
            head_clear <= 0;
            current_head <= 0;
            proj_count <= 0;
            mv_select <= 0;
            
            for (i = 0; i < N_EMBD; i++) begin
                q_proj[i] <= '0;
                k_proj[i] <= '0;
                v_proj[i] <= '0;
                concat[i] <= '0;
                x_out[i] <= '0;
                mv_vec_in[i] <= '0;
            end
            
            for (i = 0; i < N_EMBD*N_EMBD; i++) begin
                mv_mat_in[i] <= '0;
            end
            
            for (i = 0; i < HEAD_DIM; i++) begin
                head_q_in[i] <= '0;
                head_k_in[i] <= '0;
                head_v_in[i] <= '0;
            end
            
        end else begin
            // Default: clear one-cycle signals
            mv_start <= 0;
            head_start <= 0;
            head_clear <= 0;
            
            case (state)
                // ---------------------------------------------------------------
                MHA_IDLE: begin
                    if (clear_cache) begin
                        // Forward cache clear to attention heads
                        head_clear <= 1;
                    end else if (start) begin
                        proj_count <= 0;
                        mv_select <= 0;
                        state <= MHA_PROJ_QKV;
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_PROJ_QKV: begin
                    // Project input through Q, K, V matrices sequentially
                    if (!mv_start && !mv_valid) begin
                        // Set up matrix-vector multiply
                        for (i = 0; i < N_EMBD; i++) begin
                            mv_vec_in[i] <= x_in[i];
                        end
                        
                        // Select weight matrix
                        case (proj_count)
                            0: begin  // Q projection
                                for (i = 0; i < N_EMBD*N_EMBD; i++) begin
                                    mv_mat_in[i] <= wq[i];
                                end
                            end
                            1: begin  // K projection
                                for (i = 0; i < N_EMBD*N_EMBD; i++) begin
                                    mv_mat_in[i] <= wk[i];
                                end
                            end
                            2: begin  // V projection
                                for (i = 0; i < N_EMBD*N_EMBD; i++) begin
                                    mv_mat_in[i] <= wv[i];
                                end
                            end
                        endcase
                        
                        mv_start <= 1;
                        state <= MHA_WAIT_PROJ;
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_WAIT_PROJ: begin
                    if (mv_valid) begin
                        // Store projection result
                        case (proj_count)
                            0: begin
                                for (i = 0; i < N_EMBD; i++) begin
                                    q_proj[i] <= mv_result[i];
                                end
                            end
                            1: begin
                                for (i = 0; i < N_EMBD; i++) begin
                                    k_proj[i] <= mv_result[i];
                                end
                            end
                            2: begin
                                for (i = 0; i < N_EMBD; i++) begin
                                    v_proj[i] <= mv_result[i];
                                end
                            end
                        endcase
                        
                        if (proj_count < 2) begin
                            proj_count <= proj_count + 1;
                            state <= MHA_PROJ_QKV;
                        end else begin
                            // All projections done, start attention heads
                            current_head <= 0;
                            state <= MHA_RUN_HEADS;
                        end
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_RUN_HEADS: begin
                    // Run attention heads sequentially
                    if (!head_start && !head_valid) begin
                        // Extract Q, K, V for current head
                        for (i = 0; i < HEAD_DIM; i++) begin
                            idx = current_head * HEAD_DIM + i;
                            head_q_in[i] <= q_proj[idx];
                            head_k_in[i] <= k_proj[idx];
                            head_v_in[i] <= v_proj[idx];
                        end
                        
                        head_start <= 1;
                        state <= MHA_WAIT_HEADS;
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_WAIT_HEADS: begin
                    if (head_valid) begin
                        // Store head output in concatenated array
                        for (i = 0; i < HEAD_DIM; i++) begin
                            idx = current_head * HEAD_DIM + i;
                            concat[idx] <= head_out[i];
                        end
                        
                        if (current_head < N_HEAD - 1) begin
                            current_head <= current_head + 1;
                            state <= MHA_RUN_HEADS;
                        end else begin
                            // All heads done, project output
                            state <= MHA_PROJ_OUT;
                        end
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_PROJ_OUT: begin
                    // Project concatenated heads through output matrix
                    if (!mv_start && !mv_valid) begin
                        for (i = 0; i < N_EMBD; i++) begin
                            mv_vec_in[i] <= concat[i];
                        end
                        
                        for (i = 0; i < N_EMBD*N_EMBD; i++) begin
                            mv_mat_in[i] <= wo[i];
                        end
                        
                        mv_start <= 1;
                        state <= MHA_WAIT_OUT;
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_WAIT_OUT: begin
                    if (mv_valid) begin
                        for (i = 0; i < N_EMBD; i++) begin
                            x_out[i] <= mv_result[i];
                        end
                        state <= MHA_DONE;
                    end
                end
                
                // ---------------------------------------------------------------
                MHA_DONE: begin
                    state <= MHA_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == MHA_DONE);
    
endmodule : multi_head_attention