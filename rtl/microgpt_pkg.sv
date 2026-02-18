// ===========================================================================
// microGPT FPGA Package - Q12.4 PRECISION + TOP-K SAMPLING
// ===========================================================================
// Upgraded from Q8.8 to Q12.4 for better transformer accuracy
// Added top-k sampling parameters for non-deterministic generation
// ===========================================================================

package microgpt_pkg;

    // -----------------------------------------------------------------------
    // Model Architecture
    // -----------------------------------------------------------------------
    parameter int VOCAB_SIZE = 27;
    parameter int N_EMBD     = 16;
    parameter int N_HEAD     = 4;
    parameter int N_LAYER    = 1;
    parameter int BLOCK_SIZE = 16;
    parameter int HEAD_DIM   = N_EMBD / N_HEAD;
    parameter int MLP_DIM    = 4 * N_EMBD;

    // -----------------------------------------------------------------------
    // Fixed-Point: Q12.4 (12 integer bits, 4 fractional bits)
    // -----------------------------------------------------------------------
    // Range:      [-2048.0, +2047.9375]
    // Precision:  1/16 = 0.0625
    // Why:        Wider range prevents saturation in residual connections
    //             Better suited for Transformer's accumulating values
    // -----------------------------------------------------------------------
    parameter int DATA_WIDTH = 16;
    parameter int FRAC_BITS  = 4;
    parameter int INT_BITS   = DATA_WIDTH - FRAC_BITS;  // 12

    // -----------------------------------------------------------------------
    // Top-K Sampling Configuration
    // -----------------------------------------------------------------------
    parameter int TOP_K      = 2;       // Sample from top-5 tokens
    parameter int TEMP_SHIFT = 0.3;       // Temperature = 2^(-TEMP_SHIFT)
                                         // TEMP_SHIFT=1 → temp=0.5
                                         // TEMP_SHIFT=2 → temp=0.25

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Fixed-Point Type
    // -----------------------------------------------------------------------
    typedef logic signed [DATA_WIDTH-1:0] fixed_t;

    // -----------------------------------------------------------------------
    // Arithmetic Functions
    // -----------------------------------------------------------------------
    
    // Float → Q12.4
    function automatic fixed_t float_to_fixed(real f);
        logic signed [31:0] temp;
        temp = $rtoi(f * (2.0 ** FRAC_BITS));
        if (temp >  32767) temp =  32767;   // saturate
        if (temp < -32768) temp = -32768;
        return fixed_t'(temp);
    endfunction

    // Q12.4 → Float
    function automatic real fixed_to_float(fixed_t f);
        return real'(f) / (2.0 ** FRAC_BITS);
    endfunction

    // Q12.4 × Q12.4 → Q12.4
    function automatic fixed_t fixed_mul(fixed_t a, fixed_t b);
        logic signed [2*DATA_WIDTH-1:0] product;
        product = a * b;
        return fixed_t'(product >>> FRAC_BITS);
    endfunction

    // Q12.4 + Q12.4 → Q12.4 (with saturation)
    function automatic fixed_t fixed_add(fixed_t a, fixed_t b);
        logic signed [DATA_WIDTH:0] sum;  // 17-bit for overflow detection
        sum = a + b;
        if (sum > 32767)  return 16'h7FFF;   // saturate high
        if (sum < -32768) return 16'h8000;   // saturate low
        return fixed_t'(sum);
    endfunction

    // ReLU
    function automatic fixed_t relu(fixed_t x);
        return (x > 0) ? x : '0;
    endfunction

endpackage : microgpt_pkg