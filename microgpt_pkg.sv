// microGPT FPGA Package - TESTED VERSION
// Global parameters and type definitions
package microgpt_pkg;

    // Model Architecture Parameters
    parameter int VOCAB_SIZE = 27;      // 26 letters + BOS token
    parameter int N_EMBD = 16;          // Embedding dimension
    parameter int N_HEAD = 4;           // Number of attention heads
    parameter int N_LAYER = 1;          // Number of transformer layers
    parameter int BLOCK_SIZE = 16;      // Maximum sequence length
    parameter int HEAD_DIM = N_EMBD / N_HEAD;  // 4
    parameter int MLP_DIM = 4 * N_EMBD; // 64
    
    // Fixed-point representation (Q8.8 format: 8 integer bits, 8 fractional bits)
    parameter int DATA_WIDTH = 16;
    parameter int FRAC_BITS = 8;
    parameter int INT_BITS = DATA_WIDTH - FRAC_BITS;
    
    // Memory parameters
    parameter int PARAM_ADDR_WIDTH = 16;
    parameter int TOTAL_PARAMS = 
        (VOCAB_SIZE * N_EMBD) +         // wte
        (BLOCK_SIZE * N_EMBD) +         // wpe
        (VOCAB_SIZE * N_EMBD) +         // lm_head
        N_LAYER * (
            (N_EMBD * N_EMBD) +         // attn_wq
            (N_EMBD * N_EMBD) +         // attn_wk
            (N_EMBD * N_EMBD) +         // attn_wv
            (N_EMBD * N_EMBD) +         // attn_wo
            (MLP_DIM * N_EMBD) +        // mlp_fc1
            (N_EMBD * MLP_DIM)          // mlp_fc2
        );
    
    // Control signals
    typedef enum logic [2:0] {
        IDLE,
        LOAD_PARAMS,
        PROCESS_TOKEN,
        COMPUTE_ATTN,
        COMPUTE_MLP,
        GENERATE_OUTPUT,
        DONE
    } state_t;
    
    // Fixed-point arithmetic
    typedef logic signed [DATA_WIDTH-1:0] fixed_t;
    
    // Convert float to fixed-point
    function automatic fixed_t float_to_fixed(real f);
        logic signed [31:0] temp;
        temp = $rtoi(f * (2.0 ** FRAC_BITS));
        // Saturate to range
        if (temp > 32767) temp = 32767;
        if (temp < -32768) temp = -32768;
        return fixed_t'(temp);
    endfunction
    
    // Convert fixed-point to float
    function automatic real fixed_to_float(fixed_t f);
        return real'(f) / (2.0 ** FRAC_BITS);
    endfunction
    
    // Multiply two fixed-point numbers
    function automatic fixed_t fixed_mul(fixed_t a, fixed_t b);
        logic signed [2*DATA_WIDTH-1:0] product;
        product = a * b;
        return fixed_t'(product >>> FRAC_BITS);
    endfunction
    
    // Add two fixed-point numbers
    function automatic fixed_t fixed_add(fixed_t a, fixed_t b);
        return a + b;
    endfunction
    
    // ReLU activation
    function automatic fixed_t relu(fixed_t x);
        return (x > 0) ? x : '0;
    endfunction
    
    // Initialize array to zero
    function automatic void zero_array_1d(ref fixed_t arr[], input int size);
        for (int i = 0; i < size; i++) begin
            arr[i] = '0;
        end
    endfunction

endpackage : microgpt_pkg
