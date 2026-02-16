// ===========================================================================
// MLP (Multi-Layer Perceptron) Module
// ===========================================================================
// Two-layer feed-forward network with ReLU activation
// 
// Architecture:
//   x → FC1 (n_embd → 4*n_embd) → ReLU → FC2 (4*n_embd → n_embd) → out
//
// In microGPT: hidden dimension is 4x the embedding dimension
//
// Features:
//   - Pipelined matrix-vector multiplications
//   - ReLU activation between layers
//   - Fixed-point Q8.8 arithmetic
// ===========================================================================

module mlp
    import microgpt_pkg::*;
#(
    parameter int N_EMBD = 16
)
(
    input  logic   clk,
    input  logic   rst_n,
    
    // Control
    input  logic   start,
    
    // Input vector
    input  fixed_t x_in [N_EMBD-1:0],
    
    // Weight matrices (flattened)
    input  fixed_t fc1_weights [(4*N_EMBD)*N_EMBD-1:0],  // 4*n_embd x n_embd
    input  fixed_t fc2_weights [N_EMBD*(4*N_EMBD)-1:0],  // n_embd x 4*n_embd
    
    // Output
    output fixed_t x_out [N_EMBD-1:0],
    output logic   valid
);

    localparam int HIDDEN_DIM = 4 * N_EMBD;
    
    // -----------------------------------------------------------------------
    // State Machine
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        MLP_IDLE,
        MLP_FC1,           // First linear layer
        MLP_WAIT_FC1,
        MLP_RELU,          // ReLU activation
        MLP_FC2,           // Second linear layer
        MLP_WAIT_FC2,
        MLP_DONE
    } mlp_state_t;
    
    mlp_state_t state;
    
    // -----------------------------------------------------------------------
    // Internal Buffers
    // -----------------------------------------------------------------------
    fixed_t hidden [HIDDEN_DIM-1:0];  // After FC1 + ReLU
    
    // -----------------------------------------------------------------------
    // Matrix-Vector Multiplication Units
    // -----------------------------------------------------------------------
    // We need two different sized multipliers, or reuse one with muxing
    // For simplicity, use two separate units
    
    // FC1: maps N_EMBD → 4*N_EMBD
    logic  fc1_start;
    logic  fc1_valid;
    fixed_t fc1_in [N_EMBD-1:0];
    fixed_t fc1_mat [(4*N_EMBD)*N_EMBD-1:0];
    fixed_t fc1_out [HIDDEN_DIM-1:0];
    integer i;
    
    matrix_vector_mult #(
        .ROWS(HIDDEN_DIM),
        .COLS(N_EMBD)
    ) u_fc1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(fc1_start),
        .matrix(fc1_mat),
        .vec_in(fc1_in),
        .vec_out(fc1_out),
        .valid(fc1_valid)
    );
    
    // FC2: maps 4*N_EMBD → N_EMBD
    logic  fc2_start;
    logic  fc2_valid;
    fixed_t fc2_in [HIDDEN_DIM-1:0];
    fixed_t fc2_mat [N_EMBD*HIDDEN_DIM-1:0];
    fixed_t fc2_out [N_EMBD-1:0];
    
    matrix_vector_mult #(
        .ROWS(N_EMBD),
        .COLS(HIDDEN_DIM)
    ) u_fc2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(fc2_start),
        .matrix(fc2_mat),
        .vec_in(fc2_in),
        .vec_out(fc2_out),
        .valid(fc2_valid)
    );
    
    // -----------------------------------------------------------------------
    // Main Control Logic
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= MLP_IDLE;
            fc1_start <= 0;
            fc2_start <= 0;
            
            for (i = 0; i < N_EMBD; i++) begin
                fc1_in[i] <= '0;
                x_out[i] <= '0;
            end
            
            for (i = 0; i < HIDDEN_DIM; i++) begin
                hidden[i] <= '0;
                fc2_in[i] <= '0;
            end
            
            for (i = 0; i < (HIDDEN_DIM)*N_EMBD; i++) begin
                fc1_mat[i] <= '0;
            end
            
            for (i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
                fc2_mat[i] <= '0;
            end
            
        end else begin
            // Default: clear one-cycle signals
            fc1_start <= 0;
            fc2_start <= 0;
            
            case (state)
                // ---------------------------------------------------------------
                MLP_IDLE: begin
                    if (start) begin
                        state <= MLP_FC1;
                    end
                end
                
                // ---------------------------------------------------------------
                MLP_FC1: begin
                    // Start first linear layer
                    if (!fc1_start && !fc1_valid) begin
                        // Load input and weights
                        for (i = 0; i < N_EMBD; i++) begin
                            fc1_in[i] <= x_in[i];
                        end
                        
                        for (i = 0; i < (HIDDEN_DIM)*N_EMBD; i++) begin
                            fc1_mat[i] <= fc1_weights[i];
                        end
                        
                        fc1_start <= 1;
                        state <= MLP_WAIT_FC1;
                    end
                end
                
                // ---------------------------------------------------------------
                MLP_WAIT_FC1: begin
                    if (fc1_valid) begin
                        // Store FC1 output
                        for (i = 0; i < HIDDEN_DIM; i++) begin
                            hidden[i] <= fc1_out[i];
                        end
                        state <= MLP_RELU;
                    end
                end
                
                // ---------------------------------------------------------------
                MLP_RELU: begin
                    // Apply ReLU activation: max(0, x)
                    for (i = 0; i < HIDDEN_DIM; i++) begin
                        if (hidden[i][15] == 1'b1) begin  // Negative (sign bit)
                            hidden[i] <= '0;
                        end
                        // Positive values pass through unchanged
                    end
                    state <= MLP_FC2;
                end
                
                // ---------------------------------------------------------------
                MLP_FC2: begin
                    // Start second linear layer
                    if (!fc2_start && !fc2_valid) begin
                        // Load hidden activations and weights
                        for (i = 0; i < HIDDEN_DIM; i++) begin
                            fc2_in[i] <= hidden[i];
                        end
                        
                        for (i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
                            fc2_mat[i] <= fc2_weights[i];
                        end
                        
                        fc2_start <= 1;
                        state <= MLP_WAIT_FC2;
                    end
                end
                
                // ---------------------------------------------------------------
                MLP_WAIT_FC2: begin
                    if (fc2_valid) begin
                        // Store final output
                        for (i = 0; i < N_EMBD; i++) begin
                            x_out[i] <= fc2_out[i];
                        end
                        state <= MLP_DONE;
                    end
                end
                
                // ---------------------------------------------------------------
                MLP_DONE: begin
                    state <= MLP_IDLE;
                end
            endcase
        end
    end
    
    assign valid = (state == MLP_DONE);
    
endmodule : mlp