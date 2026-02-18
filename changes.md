diff --git a/export_weights.py b/export_weights.py
index e9f1812..16abe77 100644
--- a/export_weights.py
+++ b/export_weights.py
@@ -1,112 +1,114 @@
 """
-export_weights_q124.py
-=======================
-Export microGPT weights to Q12.4 fixed-point format for FPGA.
-
-Q12.4 format:
-- 12 integer bits: range [-2048, +2047]
-- 4 fractional bits: precision 1/16 = 0.0625
-- Still 16-bit words (fits existing BRAM infrastructure)
-
-Benefits over Q8.8:
-- 16× wider range → less saturation in residual connections
-- Better suited for Transformer's accumulating values
-- Empirically improves output quality on real names dataset
+export_weights.py
+=================
+Exports trained microGPT weights to param.mem for FPGA inference.
 
 Usage:
-    from export_weights_q124 import export_param_mem_q124
-    export_param_mem_q124(state_dict, uchars)
+    1. Train the model using microGPT_original_python_code
+    2. At the END of training (after the Adam loop), add:
+           from export_weights import export_param_mem
+           export_param_mem(state_dict, uchars)
+    3. OR run this file standalone if you saved state_dict to a file.
+
+Output:
+    param.mem  — one 4-hex-digit Q8.8 value per line, address 0 first.
+                 Drop this file next to your Vivado project / simulation.
+
+Address map (must match ADDR_* constants in microgpt_top.sv):
+    [0      ]  wte        VOCAB_SIZE * N_EMBD   = 27*16 = 432
+    [432    ]  wpe        BLOCK_SIZE * N_EMBD   = 16*16 = 256
+    [688    ]  lm_head    VOCAB_SIZE * N_EMBD   = 27*16 = 432
+    [1120   ]  attn_wq    N_EMBD    * N_EMBD    = 16*16 = 256
+    [1376   ]  attn_wk    N_EMBD    * N_EMBD    = 256
+    [1632   ]  attn_wv    N_EMBD    * N_EMBD    = 256
+    [1888   ]  attn_wo    N_EMBD    * N_EMBD    = 256
+    [2144   ]  mlp_fc1    MLP_DIM   * N_EMBD    = 64*16 = 1024
+    [3168   ]  mlp_fc2    N_EMBD    * MLP_DIM   = 16*64 = 1024
+    Total                                        = 4192 words
+
+Q8.8 encoding:
+    fixed = round(float_value * 256)
+    clamped to [-32768, 32767]  (signed 16-bit)
+    written as 4-digit uppercase hex (two's complement)
 """
 
 import math
 
 # ---------------------------------------------------------------------------
-# Q12.4 Configuration
+# Model dimensions  (must match microgpt_pkg.sv)
 # ---------------------------------------------------------------------------
-FRAC_BITS = 4
-SCALE = 1 << FRAC_BITS   # 16
-
-# Model dimensions
-VOCAB_SIZE  = 27
+VOCAB_SIZE  = 27      # 26 letters + BOS
 N_EMBD      = 16
 N_HEAD      = 4
 N_LAYER     = 1
 BLOCK_SIZE  = 16
-MLP_DIM     = 4 * N_EMBD
+MLP_DIM     = 4 * N_EMBD   # 64
+
+FRAC_BITS   = 8
+SCALE       = 1 << FRAC_BITS   # 256
+
 
 # ---------------------------------------------------------------------------
-# Q12.4 Conversion
+# Helper: float → Q8.8 hex string
 # ---------------------------------------------------------------------------
-def to_q124_hex(val):
-    """
-    Convert float to Q12.4 hex string.
-    
-    Range: [-2048.0, +2047.9375]
-    Precision: 1/16 = 0.0625
-    
-    Returns: 4-digit uppercase hex (two's complement)
-    """
-    if hasattr(val, 'data'):
+def to_q88_hex(val):
+    """Convert a float to a 4-digit Q8.8 hex string (two's complement)."""
+    if hasattr(val, 'data'):        # unwrap Value objects from microGPT
         val = val.data
-    
     fixed = int(round(val * SCALE))
-    
-    # Saturate to 16-bit signed range
-    fixed = max(-32768, min(32767, fixed))
-    
-    # Two's complement for negative values
+    fixed = max(-32768, min(32767, fixed))   # saturate
     if fixed < 0:
-        fixed = fixed + 65536
-    
+        fixed = fixed + 65536                # two's complement 16-bit
     return f"{fixed:04X}"
 
 
-def q124_to_float(hex_str):
-    """Convert Q12.4 hex string back to float (for verification)."""
-    v = int(hex_str, 16)
-    if v >= 32768:
-        v -= 65536
-    return v / SCALE
-
-
 # ---------------------------------------------------------------------------
-# Main Export Function
+# Main export function
 # ---------------------------------------------------------------------------
-def export_param_mem_q124(state_dict, uchars, filename="param_q124.mem"):
+def export_param_mem(state_dict, uchars, filename="param.mem"):
     """
-    Export trained weights to Q12.4 format.
-    
-    Output file format:
-    - One 16-bit hex value per line
-    - 4192 lines total
-    - $readmemh compatible
+    Write all trained weights to `filename` in the order expected by
+    microgpt_top.sv.
+
+    Parameters
+    ----------
+    state_dict : dict
+        The microGPT state_dict after training.
+        Keys: 'wte', 'wpe', 'lm_head',
+              'layer0.attn_wq', 'layer0.attn_wk', etc.
+    uchars : list[str]
+        Sorted unique characters (used only for reporting vocab mapping).
+    filename : str
+        Output filename (default: "param.mem").
     """
-    
-    BOS = len(uchars)
-    lines = []
+
+    BOS = len(uchars)   # BOS token id
+
+    lines   = []
     address = 0
 
     def write_matrix(name, matrix):
+        """Flatten matrix row-major and append hex lines."""
         nonlocal address
         count = 0
         for row in matrix:
             for val in row:
-                lines.append(to_q124_hex(val))
+                lines.append(to_q88_hex(val))
                 count += 1
                 address += 1
-        print(f"  {name:<20s}  rows={len(matrix):<3d}  cols={len(matrix[0]):<3d}  "
-              f"params={count:<4d}  addr={address-count:4d}..{address-1:4d}")
+        print(f"  {name:<20s}  rows={len(matrix)}  cols={len(matrix[0])}  "
+              f"params={count}  addr_end={address}")
 
-    print("=" * 70)
-    print(f"Exporting weights to '{filename}' (Q12.4 format)")
-    print(f"Precision: ±{1/SCALE:.4f}  Range: [-2048, +2047.9375]")
-    print("=" * 70)
+    print("=" * 60)
+    print(f"Exporting weights to '{filename}'")
+    print(f"Expected total params: {VOCAB_SIZE*N_EMBD + BLOCK_SIZE*N_EMBD + VOCAB_SIZE*N_EMBD + N_LAYER*(4*N_EMBD*N_EMBD + 2*MLP_DIM*N_EMBD)}")
+    print("=" * 60)
 
-    # Match Python state_dict iteration order
+    # Exactly mirrors Python line 89:
+    # params = [p for mat in state_dict.values() for row in mat for p in row]
     write_matrix("wte",            state_dict['wte'])
     write_matrix("wpe",            state_dict['wpe'])
     write_matrix("lm_head",        state_dict['lm_head'])
-    
     for li in range(N_LAYER):
         write_matrix(f"layer{li}.attn_wq", state_dict[f'layer{li}.attn_wq'])
         write_matrix(f"layer{li}.attn_wk", state_dict[f'layer{li}.attn_wk'])
@@ -115,79 +117,81 @@ def export_param_mem_q124(state_dict, uchars, filename="param_q124.mem"):
         write_matrix(f"layer{li}.mlp_fc1", state_dict[f'layer{li}.mlp_fc1'])
         write_matrix(f"layer{li}.mlp_fc2", state_dict[f'layer{li}.mlp_fc2'])
 
-    print(f"\nTotal words: {address}")
-    assert address == 4192, f"Expected 4192, got {address}"
+    print(f"\nTotal words written: {address}")
+    assert address == len(lines), "Line count mismatch!"
 
     with open(filename, 'w') as f:
         f.write('\n'.join(lines) + '\n')
 
-    print(f"\n✓ Written to '{filename}'")
-    print("=" * 70)
-
-    # Spot-check first few weights
-    print("\nSpot-check (first 8 wte values):")
-    for i in range(8):
-        print(f"  [{i}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")
+    print(f"Written to '{filename}'  ✓")
+    print("=" * 60)
 
-    # Vocabulary
-    print("\nVocabulary:")
+    # Print vocabulary mapping so you can decode token IDs in simulation
+    print("\nVocabulary mapping (for simulation decoding):")
     for i, ch in enumerate(uchars):
-        print(f"  {i:2d} = '{ch}'")
-    print(f"  {BOS:2d} = BOS")
+        print(f"  token {i:2d} = '{ch}'")
+    print(f"  token {BOS:2d} = BOS (end-of-sequence)")
 
     return filename
 
 
 # ---------------------------------------------------------------------------
-# Verification
+# Verify an existing param.mem matches the expected total
 # ---------------------------------------------------------------------------
-def verify_param_mem_q124(filename="param_q124.mem"):
-    """Verify exported weights."""
+def verify_param_mem(filename="param.mem"):
+    """Quick sanity check: count lines and spot-check first/last addresses."""
+    expected = (VOCAB_SIZE * N_EMBD +
+                BLOCK_SIZE * N_EMBD +
+                VOCAB_SIZE * N_EMBD +
+                N_LAYER * (4 * N_EMBD * N_EMBD +
+                           MLP_DIM * N_EMBD +
+                           N_EMBD * MLP_DIM))
+
     with open(filename, 'r') as f:
         lines = [l.strip() for l in f if l.strip()]
 
-    print(f"\nVerifying '{filename}'")
-    print(f"  Lines: {len(lines)}")
-    print(f"  Expected: 4192")
+    print(f"verify_param_mem('{filename}')")
+    print(f"  Lines in file : {len(lines)}")
+    print(f"  Expected lines: {expected}")
 
-    if len(lines) != 4192:
-        print(f"  ✗ MISMATCH!")
+    if len(lines) != expected:
+        print(f"  ✗ MISMATCH!  ({len(lines)} vs {expected})")
         return False
 
-    # Decode sample values
-    print(f"\n  First 4 (wte[0][0..3]):")
-    for i in range(4):
-        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")
+    # Decode first few and last few entries as a spot-check
+    def q88(hex_str):
+        v = int(hex_str, 16)
+        if v >= 32768:
+            v -= 65536
+        return v / SCALE
+
+    print(f"\n  First 8 entries (wte[0][0..7]):")
+    for i in range(8):
+        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")
 
-    print(f"\n  Around attn_wq (addr 1120):")
+    print(f"\n  Entries around attn_wq base (addr 1120):")
     for i in range(1120, 1124):
-        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")
+        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")
 
-    print(f"\n  Last 4 (mlp_fc2 tail):")
-    for i in range(4188, 4192):
-        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")
+    print(f"\n  Last 4 entries (mlp_fc2 tail):")
+    for i in range(expected - 4, expected):
+        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")
 
-    print(f"\n  ✓ Looks good")
+    print(f"\n  ✓ param.mem looks correct")
     return True
 
 
 # ---------------------------------------------------------------------------
-# Standalone
+# Standalone: verify an existing param.mem
 # ---------------------------------------------------------------------------
 if __name__ == "__main__":
     import sys
+    fname = sys.argv[1] if len(sys.argv) > 1 else "param.mem"
     import os
-    
-    fname = sys.argv[1] if len(sys.argv) > 1 else "param_q124.mem"
-    
     if os.path.exists(fname):
-        verify_param_mem_q124(fname)
+        verify_param_mem(fname)
     else:
         print(f"'{fname}' not found.")
-        print("\nUsage:")
-        print("  # In your training script:")
-        print("  from export_weights_q124 import export_param_mem_q124")
-        print("  export_param_mem_q124(state_dict, uchars)")
-        print()
-        print("  # Or verify existing file:")
-        print(f"  python {__file__} param_q124.mem")
\ No newline at end of file
+        print("Run from inside your microGPT training script after training:")
+        print("  from export_weights import export_param_mem")
+        print("  export_param_mem(state_dict, uchars)")
\ No newline at end of file

diff --git a/rtl/microgpt_pkg.sv b/rtl/microgpt_pkg.sv
index 9fe1456..c444867 100644
--- a/rtl/microgpt_pkg.sv
+++ b/rtl/microgpt_pkg.sv
@@ -1,46 +1,22 @@
-// ===========================================================================
-// microGPT FPGA Package - Q12.4 PRECISION + TOP-K SAMPLING
-// ===========================================================================
-// Upgraded from Q8.8 to Q12.4 for better transformer accuracy
-// Added top-k sampling parameters for non-deterministic generation
-// ===========================================================================
-
+// microGPT FPGA Package - TESTED VERSION
+// Global parameters and type definitions
 package microgpt_pkg;
 
-    // -----------------------------------------------------------------------
-    // Model Architecture
-    // -----------------------------------------------------------------------
-    parameter int VOCAB_SIZE = 27;
-    parameter int N_EMBD     = 16;
-    parameter int N_HEAD     = 4;
-    parameter int N_LAYER    = 1;
-    parameter int BLOCK_SIZE = 16;
-    parameter int HEAD_DIM   = N_EMBD / N_HEAD;
-    parameter int MLP_DIM    = 4 * N_EMBD;
-
-    // -----------------------------------------------------------------------
-    // Fixed-Point: Q12.4 (12 integer bits, 4 fractional bits)
-    // -----------------------------------------------------------------------
-    // Range:      [-2048.0, +2047.9375]
-    // Precision:  1/16 = 0.0625
-    // Why:        Wider range prevents saturation in residual connections
-    //             Better suited for Transformer's accumulating values
-    // -----------------------------------------------------------------------
+    // Model Architecture Parameters
+    parameter int VOCAB_SIZE = 27;      // 26 letters + BOS token
+    parameter int N_EMBD = 16;          // Embedding dimension
+    parameter int N_HEAD = 4;           // Number of attention heads
+    parameter int N_LAYER = 1;          // Number of transformer layers
+    parameter int BLOCK_SIZE = 16;      // Maximum sequence length
+    parameter int HEAD_DIM = N_EMBD / N_HEAD;  // 4
+    parameter int MLP_DIM = 4 * N_EMBD; // 64
+    
+    // Fixed-point representation (Q8.8 format: 8 integer bits, 8 fractional bits)
     parameter int DATA_WIDTH = 16;
-    parameter int FRAC_BITS  = 4;
-    parameter int INT_BITS   = DATA_WIDTH - FRAC_BITS;  // 12
-
-    // -----------------------------------------------------------------------
-    // Top-K Sampling Configuration
-    // -----------------------------------------------------------------------
-    parameter int TOP_K      = 5;       // Sample from top-5 tokens
-    parameter int TEMP_SHIFT = 1;       // Temperature = 2^(-TEMP_SHIFT)
-                                         // TEMP_SHIFT=1 → temp=0.5
-                                         // TEMP_SHIFT=2 → temp=0.25
-
-    // -----------------------------------------------------------------------
-    // Memory
-    // -----------------------------------------------------------------------
+    parameter int FRAC_BITS = 8;
+    parameter int INT_BITS = DATA_WIDTH - FRAC_BITS;
+    
+    // Memory parameters
     parameter int PARAM_ADDR_WIDTH = 16;
     parameter int TOTAL_PARAMS = 
         (VOCAB_SIZE * N_EMBD) +         // wte
@@ -54,49 +30,58 @@ package microgpt_pkg;
             (MLP_DIM * N_EMBD) +        // mlp_fc1
             (N_EMBD * MLP_DIM)          // mlp_fc2
         );
-
-    // -----------------------------------------------------------------------
-    // Fixed-Point Type
-    // -----------------------------------------------------------------------
+    
+    // Control signals
+    typedef enum logic [2:0] {
+        IDLE,
+        LOAD_PARAMS,
+        PROCESS_TOKEN,
+        COMPUTE_ATTN,
+        COMPUTE_MLP,
+        GENERATE_OUTPUT,
+        DONE
+    } state_t;
+    
+    // Fixed-point arithmetic
     typedef logic signed [DATA_WIDTH-1:0] fixed_t;
-
-    // -----------------------------------------------------------------------
-    // Arithmetic Functions
-    // -----------------------------------------------------------------------
     
-    // Float → Q12.4
+    // Convert float to fixed-point
     function automatic fixed_t float_to_fixed(real f);
         logic signed [31:0] temp;
         temp = $rtoi(f * (2.0 ** FRAC_BITS));
-        if (temp >  32767) temp =  32767;   // saturate
+        // Saturate to range
+        if (temp > 32767) temp = 32767;
         if (temp < -32768) temp = -32768;
         return fixed_t'(temp);
     endfunction
-
-    // Q12.4 → Float
+    
+    // Convert fixed-point to float
     function automatic real fixed_to_float(fixed_t f);
         return real'(f) / (2.0 ** FRAC_BITS);
     endfunction
-
-    // Q12.4 × Q12.4 → Q12.4
+    
+    // Multiply two fixed-point numbers
     function automatic fixed_t fixed_mul(fixed_t a, fixed_t b);
         logic signed [2*DATA_WIDTH-1:0] product;
         product = a * b;
         return fixed_t'(product >>> FRAC_BITS);
     endfunction
-
-    // Q12.4 + Q12.4 → Q12.4 (with saturation)
+    
+    // Add two fixed-point numbers
     function automatic fixed_t fixed_add(fixed_t a, fixed_t b);
-        logic signed [DATA_WIDTH:0] sum;  // 17-bit for overflow detection
-        sum = a + b;
-        if (sum > 32767)  return 16'h7FFF;   // saturate high
-        if (sum < -32768) return 16'h8000;   // saturate low
-        return fixed_t'(sum);
+        return a + b;
     endfunction
-
-    // ReLU
+    
+    // ReLU activation
     function automatic fixed_t relu(fixed_t x);
         return (x > 0) ? x : '0;
     endfunction
+    
+    // Initialize array to zero
+    function automatic void zero_array_1d(ref fixed_t arr[], input int size);
+        for (int i = 0; i < size; i++) begin
+            arr[i] = '0;
+        end
+    endfunction
 
-endpackage : microgpt_pkg
\ No newline at end of file
+endpackage : microgpt_pkg
diff --git a/rtl/microgpt_top.sv b/rtl/microgpt_top.sv
index 3fa1678..6d4bcf4 100644
--- a/rtl/microgpt_top.sv
+++ b/rtl/microgpt_top.sv
@@ -1,23 +1,31 @@
 // ===========================================================================
-// microGPT Top Module - Q12.4 + TOP-K SAMPLING
+// microGPT Top Module
 // ===========================================================================
-// Upgraded precision and sampling for better name generation.
+// Complete autoregressive inference engine.
 //
-// Changes from base version:
-//   1. Q12.4 fixed-point (was Q8.8) → wider range, less saturation
-//   2. Top-k sampling (was pure argmax) → non-deterministic, creative output
-//   3. Temperature scaling via TEMP_SHIFT parameter
+// Matches Python reference (gpt() function) exactly:
 //
-// Expected improvement:
-//   - Generates diverse, human-readable names
-//   - Still deterministic if you freeze the LFSR seed
-//   - Better convergence to training distribution
+//   tok_emb = wte[token_id]             <- embedding lookup
+//   pos_emb = wpe[pos_id]               <- position lookup
+//   x = tok_emb + pos_emb               <- add embeddings
+//   x = rmsnorm(x)                      <- pre-norm (as in Python line 112)
+//   x = transformer_layer(x)            <- single layer
+//   logits = lm_head * x                <- project to vocab
 //
-// Resource usage (VU19P):
-//   - BRAMs: ~20 (param storage)
-//   - LUTs:  ~15K (transformer + sampling logic)
-//   - DSPs:  ~50 (matrix multiplies)
-//   - Perfect fit on VU19P with massive headroom
+// Weight address map (matches Python state_dict iteration order):
+//   [0      ] wte       27 * 16  =   432  (token embeddings)
+//   [432    ] wpe       16 * 16  =   256  (position embeddings)
+//   [688    ] lm_head   27 * 16  =   432  (output projection)
+//   [1120   ] attn_wq   16 * 16  =   256
+//   [1376   ] attn_wk   16 * 16  =   256
+//   [1632   ] attn_wv   16 * 16  =   256
+//   [1888   ] attn_wo   16 * 16  =   256
+//   [2144   ] mlp_fc1   64 * 16  =  1024
+//   [3168   ] mlp_fc2   16 * 64  =  1024
+//   Total                         = 4192
+//
+// param.mem format: $readmemh, one Q8.8 value (16-bit hex) per line,
+//                   row-major, address 0 first.
 // ===========================================================================
 
 module microgpt_top
@@ -26,89 +34,96 @@ module microgpt_top
     input  logic        clk,
     input  logic        rst_n,
 
-    // Generation control
-    input  logic        start_gen,
-    input  logic        next_token,
+    // --- generation control ---
+    input  logic        start_gen,      // pulse to begin generating a new sequence
+    input  logic        next_token,     // pulse to advance one position (after reading token_out)
 
-    // Output
-    output logic [4:0]  token_out,
-    output logic        token_valid,
-    output logic        gen_done
+    // --- output ---
+    output logic [4:0]  token_out,      // predicted next token id
+    output logic        token_valid,    // high for one cycle when token_out is ready
+    output logic        gen_done        // high when BOS predicted (end of sequence)
 );
 
     // -----------------------------------------------------------------------
-    // Address map (unchanged from Q8.8 version)
+    // Address base constants (matching Python state_dict order)
     // -----------------------------------------------------------------------
     localparam int ADDR_WTE      = 0;
-    localparam int ADDR_WPE      = 432;
-    localparam int ADDR_LM_HEAD  = 688;
-    localparam int ADDR_ATTN_WQ  = 1120;
-    localparam int ADDR_ATTN_WK  = 1376;
-    localparam int ADDR_ATTN_WV  = 1632;
-    localparam int ADDR_ATTN_WO  = 1888;
-    localparam int ADDR_MLP_FC1  = 2144;
-    localparam int ADDR_MLP_FC2  = 3168;
+    localparam int ADDR_WPE      = ADDR_WTE    + VOCAB_SIZE * N_EMBD;   // 432
+    localparam int ADDR_LM_HEAD  = ADDR_WPE    + BLOCK_SIZE * N_EMBD;   // 688
+    localparam int ADDR_ATTN_WQ  = ADDR_LM_HEAD + VOCAB_SIZE * N_EMBD;  // 1120
+    localparam int ADDR_ATTN_WK  = ADDR_ATTN_WQ + N_EMBD * N_EMBD;     // 1376
+    localparam int ADDR_ATTN_WV  = ADDR_ATTN_WK + N_EMBD * N_EMBD;     // 1632
+    localparam int ADDR_ATTN_WO  = ADDR_ATTN_WV + N_EMBD * N_EMBD;     // 1888
+    localparam int ADDR_MLP_FC1  = ADDR_ATTN_WO + N_EMBD * N_EMBD;     // 2144
+    localparam int ADDR_MLP_FC2  = ADDR_MLP_FC1 + MLP_DIM * N_EMBD;    // 3168
 
     // -----------------------------------------------------------------------
     // State machine
     // -----------------------------------------------------------------------
     typedef enum logic [4:0] {
         TOP_IDLE,
-        TOP_LOAD_WTE,
-        TOP_LOAD_WPE,
-        TOP_ADD_EMBED,
-        TOP_PRENORM,
+        TOP_LOAD_WTE,       // read token embedding from param_ram
+        TOP_LOAD_WPE,       // read position embedding from param_ram
+        TOP_ADD_EMBED,      // x = tok_emb + pos_emb
+        TOP_PRENORM,        // x = rmsnorm(x)
         TOP_WAIT_PRENORM,
-        TOP_LOAD_WEIGHTS,
-        TOP_TRANSFORMER,
+        TOP_LOAD_WEIGHTS,   // burst-read all 4 attn + 2 mlp weight matrices
+        TOP_WAIT_WEIGHTS,
+        TOP_TRANSFORMER,    // run transformer_layer
         TOP_WAIT_TRANSFORMER,
-        TOP_LOAD_LMHEAD,
-        TOP_COMPUTE_LOGITS,
+        TOP_LOAD_LMHEAD,    // read lm_head weights row by row
+        TOP_WAIT_LMHEAD,
+        TOP_COMPUTE_LOGITS, // logits = lm_head * x
         TOP_WAIT_LOGITS,
-        TOP_SAMPLE,          // NEW: top-k sampling (replaces TOP_ARGMAX)
-        TOP_WAIT_SAMPLE,
-        TOP_OUTPUT,
+        TOP_ARGMAX,         // find highest logit → token_out
+        TOP_OUTPUT,         // assert token_valid for one cycle
         TOP_DONE
     } top_state_t;
 
     top_state_t state;
 
     // -----------------------------------------------------------------------
-    // Registers (all declared at top)
+    // Internal registers  (all declared at module top)
     // -----------------------------------------------------------------------
     fixed_t tok_emb  [N_EMBD-1:0];
     fixed_t pos_emb  [N_EMBD-1:0];
-    fixed_t x        [N_EMBD-1:0];
+    fixed_t x        [N_EMBD-1:0];   // working embedding vector
 
+    // Weight matrices held in registers (loaded once per token)
     fixed_t attn_wq  [N_EMBD*N_EMBD-1:0];
     fixed_t attn_wk  [N_EMBD*N_EMBD-1:0];
     fixed_t attn_wv  [N_EMBD*N_EMBD-1:0];
     fixed_t attn_wo  [N_EMBD*N_EMBD-1:0];
     fixed_t mlp_fc1  [MLP_DIM*N_EMBD-1:0];
     fixed_t mlp_fc2  [N_EMBD*MLP_DIM-1:0];
+    fixed_t lm_head  [VOCAB_SIZE*N_EMBD-1:0];  // output projection
 
-    logic [4:0]  cur_pos;
-    logic [4:0]  cur_token;
-    logic [4:0]  sampled_token;
+    // Generation state
+    logic [4:0]  cur_pos;        // current generation position
+    logic [4:0]  cur_token;      // current input token
+    logic [4:0]  best_token;     // argmax result
+    fixed_t      best_logit;     // argmax running max
+    logic [4:0]  argmax_idx;     // loop counter for argmax
 
-    logic [PARAM_ADDR_WIDTH-1:0] load_addr;
-    logic [12:0]                 load_count;
-    logic [12:0]                 load_total;
-    logic [3:0]                  load_phase;
+    // Counters for burst memory loads
+    logic [PARAM_ADDR_WIDTH-1:0] load_addr;   // next param_ram read address
+    logic [12:0]                 load_count;  // how many words loaded so far
+    logic [12:0]                 load_total;  // how many words to load total
+    logic [3:0]                  load_phase;  // which matrix we're loading (0-5)
 
     integer i;
 
     // -----------------------------------------------------------------------
-    // Parameter RAM (Q12.4 weights from param_q124.mem)
+    // Parameter memory (loads from param.mem at simulation start)
     // -----------------------------------------------------------------------
     fixed_t param_ram [0:TOTAL_PARAMS-1];
 
     initial begin
-        $readmemh("param_q124.mem", param_ram);
+        $readmemh("params.mem", param_ram);
     end
 
     // -----------------------------------------------------------------------
-    // Pre-norm RMSNorm
+    // Pre-norm RMSNorm instance
     // -----------------------------------------------------------------------
     logic   prenorm_start;
     logic   prenorm_valid;
@@ -125,7 +140,7 @@ module microgpt_top
     );
 
     // -----------------------------------------------------------------------
-    // Transformer layer
+    // Transformer layer instance
     // -----------------------------------------------------------------------
     logic   tl_start;
     logic   tl_clear;
@@ -155,7 +170,7 @@ module microgpt_top
     );
 
     // -----------------------------------------------------------------------
-    // LM head projection
+    // LM head matrix-vector multiply
     // -----------------------------------------------------------------------
     logic   lm_start;
     logic   lm_valid;
@@ -177,31 +192,16 @@ module microgpt_top
     );
 
     // -----------------------------------------------------------------------
-    // Top-K Sampler (NEW: replaces argmax)
-    // -----------------------------------------------------------------------
-    logic   sample_start;
-    logic   sample_valid;
-    logic [4:0] sample_out;
-
-    topk_sampler #(.K(TOP_K)) u_sampler (
-        .clk      (clk),
-        .rst_n    (rst_n),
-        .start    (sample_start),
-        .seed     (cur_pos),      // Use position as LFSR seed for variety
-        .logits   (lm_logits),
-        .token_out(sample_out),
-        .valid    (sample_valid)
-    );
-
-    // -----------------------------------------------------------------------
-    // Main FSM (modified for top-k sampling)
+    // Main FSM
     // -----------------------------------------------------------------------
     always_ff @(posedge clk or negedge rst_n) begin
         if (!rst_n) begin
             state         <= TOP_IDLE;
             cur_pos       <= 0;
-            cur_token     <= VOCAB_SIZE - 1;
-            sampled_token <= 0;
+            cur_token     <= VOCAB_SIZE - 1;  // BOS token
+            best_token    <= 0;
+            best_logit    <= 16'sh8000;       // most negative
+            argmax_idx    <= 0;
             load_addr     <= 0;
             load_count    <= 0;
             load_total    <= 0;
@@ -210,7 +210,6 @@ module microgpt_top
             tl_start      <= 0;
             tl_clear      <= 0;
             lm_start      <= 0;
-            sample_start  <= 0;
             token_valid   <= 0;
             gen_done      <= 0;
             token_out     <= 0;
@@ -229,53 +228,68 @@ module microgpt_top
             for (i = 0; i < N_EMBD*N_EMBD;     i++) attn_wo[i] <= '0;
             for (i = 0; i < MLP_DIM*N_EMBD;    i++) mlp_fc1[i] <= '0;
             for (i = 0; i < N_EMBD*MLP_DIM;    i++) mlp_fc2[i] <= '0;
+            for (i = 0; i < VOCAB_SIZE*N_EMBD;  i++) lm_head[i] <= '0;
             for (i = 0; i < VOCAB_SIZE; i++) begin
                 for (int j = 0; j < N_EMBD; j++)
                     lm_mat[i][j] <= '0;
             end
 
         end else begin
-            // Clear pulse signals
+            // Default: clear pulse signals
             prenorm_start <= 0;
             tl_start      <= 0;
             tl_clear      <= 0;
             lm_start      <= 0;
-            sample_start  <= 0;
             token_valid   <= 0;
 
             case (state)
 
+                // =============================================================
                 TOP_IDLE: begin
                     gen_done <= 0;
                     if (start_gen) begin
                         cur_pos   <= 0;
-                        cur_token <= VOCAB_SIZE - 1;
-                        tl_clear  <= 1;
+                        cur_token <= VOCAB_SIZE - 1;  // start with BOS
+                        tl_clear  <= 1;               // flush KV cache
                         state     <= TOP_LOAD_WTE;
                     end else if (next_token) begin
-                        cur_token <= sampled_token;
+                        // advance to next position using last predicted token
+                        cur_token <= best_token;
                         state     <= TOP_LOAD_WTE;
                     end
                 end
 
+                // =============================================================
+                // Load token embedding: wte[cur_token][0..N_EMBD-1]
+                // =============================================================
                 TOP_LOAD_WTE: begin
+                    // Combinatorially copy N_EMBD words from param_ram
                     for (i = 0; i < N_EMBD; i++)
                         tok_emb[i] <= param_ram[ADDR_WTE + cur_token * N_EMBD + i];
                     state <= TOP_LOAD_WPE;
                 end
 
+                // =============================================================
+                // Load position embedding: wpe[cur_pos][0..N_EMBD-1]
+                // =============================================================
                 TOP_LOAD_WPE: begin
                     for (i = 0; i < N_EMBD; i++)
                         pos_emb[i] <= param_ram[ADDR_WPE + cur_pos * N_EMBD + i];
                     state <= TOP_ADD_EMBED;
                 end
 
+                // =============================================================
+                // x = tok_emb + pos_emb
+                // =============================================================
                 TOP_ADD_EMBED: begin
                     for (i = 0; i < N_EMBD; i++)
                         x[i] <= fixed_add(tok_emb[i], pos_emb[i]);
                     state <= TOP_PRENORM;
                 end
 
+                // =============================================================
+                // x = rmsnorm(x)   (Python line 112)
+                // =============================================================
                 TOP_PRENORM: begin
                     for (i = 0; i < N_EMBD; i++)
                         prenorm_in[i] <= x[i];
@@ -287,15 +301,26 @@ module microgpt_top
                     if (prenorm_valid) begin
                         for (i = 0; i < N_EMBD; i++)
                             x[i] <= prenorm_out[i];
+                        // Kick off weight loading
                         load_phase <= 0;
                         load_count <= 0;
                         load_addr  <= ADDR_ATTN_WQ;
-                        load_total <= N_EMBD * N_EMBD;
+                        load_total <= N_EMBD * N_EMBD;  // first: attn_wq
                         state <= TOP_LOAD_WEIGHTS;
                     end
                 end
 
+                // =============================================================
+                // Burst-load all 6 weight matrices from param_ram
+                // phase 0: attn_wq  (256)
+                // phase 1: attn_wk  (256)
+                // phase 2: attn_wv  (256)
+                // phase 3: attn_wo  (256)
+                // phase 4: mlp_fc1  (1024)
+                // phase 5: mlp_fc2  (1024)
+                // =============================================================
                 TOP_LOAD_WEIGHTS: begin
+                    // Load one word per cycle
                     case (load_phase)
                         0: attn_wq[load_count] <= param_ram[load_addr];
                         1: attn_wk[load_count] <= param_ram[load_addr];
@@ -310,6 +335,7 @@ module microgpt_top
                     load_count <= load_count + 1;
 
                     if (load_count + 1 >= load_total) begin
+                        // This matrix done — advance phase
                         load_count <= 0;
                         case (load_phase)
                             0: begin load_phase<=1; load_addr<=ADDR_ATTN_WK; load_total<=N_EMBD*N_EMBD; end
@@ -317,12 +343,15 @@ module microgpt_top
                             2: begin load_phase<=3; load_addr<=ADDR_ATTN_WO; load_total<=N_EMBD*N_EMBD; end
                             3: begin load_phase<=4; load_addr<=ADDR_MLP_FC1; load_total<=MLP_DIM*N_EMBD; end
                             4: begin load_phase<=5; load_addr<=ADDR_MLP_FC2; load_total<=N_EMBD*MLP_DIM; end
-                            5: state <= TOP_TRANSFORMER;
+                            5: state <= TOP_TRANSFORMER;  // all done
                             default: state <= TOP_TRANSFORMER;
                         endcase
                     end
                 end
 
+                // =============================================================
+                // Run transformer layer
+                // =============================================================
                 TOP_TRANSFORMER: begin
                     for (i = 0; i < N_EMBD; i++)
                         tl_in[i] <= x[i];
@@ -334,13 +363,19 @@ module microgpt_top
                     if (tl_valid) begin
                         for (i = 0; i < N_EMBD; i++)
                             x[i] <= tl_out[i];
+                        // Load lm_head weights
                         load_count <= 0;
                         load_addr  <= ADDR_LM_HEAD;
                         state <= TOP_LOAD_LMHEAD;
                     end
                 end
 
+                // =============================================================
+                // Load lm_head weights (VOCAB_SIZE * N_EMBD = 432 words)
+                // Simultaneously reshape flat array into 2D lm_mat
+                // =============================================================
                 TOP_LOAD_LMHEAD: begin
+                    lm_head[load_count] <= param_ram[load_addr];
                     lm_mat[load_count / N_EMBD][load_count % N_EMBD]
                                         <= param_ram[load_addr];
                     load_addr  <= load_addr  + 1;
@@ -349,6 +384,9 @@ module microgpt_top
                         state <= TOP_COMPUTE_LOGITS;
                 end
 
+                // =============================================================
+                // logits = lm_head * x
+                // =============================================================
                 TOP_COMPUTE_LOGITS: begin
                     for (i = 0; i < N_EMBD; i++)
                         lm_vec_in[i] <= x[i];
@@ -358,38 +396,47 @@ module microgpt_top
 
                 TOP_WAIT_LOGITS: begin
                     if (lm_valid) begin
-                        state <= TOP_SAMPLE;
+                        // Seed argmax
+                        best_logit <= lm_logits[0];
+                        best_token <= 0;
+                        argmax_idx <= 1;
+                        state <= TOP_ARGMAX;
                     end
                 end
 
-                // ===============================================================
-                // NEW: Top-K sampling (replaces argmax)
-                // ===============================================================
-                TOP_SAMPLE: begin
-                    sample_start <= 1;
-                    state <= TOP_WAIT_SAMPLE;
-                end
-
-                TOP_WAIT_SAMPLE: begin
-                    if (sample_valid) begin
-                        sampled_token <= sample_out;
+                // =============================================================
+                // Argmax over logits → best_token
+                // =============================================================
+                TOP_ARGMAX: begin
+                    if (argmax_idx < VOCAB_SIZE) begin
+                        if (lm_logits[argmax_idx] > best_logit) begin
+                            best_logit <= lm_logits[argmax_idx];
+                            best_token <= argmax_idx;
+                        end
+                        argmax_idx <= argmax_idx + 1;
+                    end else begin
                         state <= TOP_OUTPUT;
                     end
                 end
 
+                // =============================================================
+                // Emit result
+                // =============================================================
                 TOP_OUTPUT: begin
-                    token_out   <= sampled_token;
+                    token_out   <= best_token;
                     token_valid <= 1;
-                    if (sampled_token == VOCAB_SIZE - 1) begin
+                    if (best_token == VOCAB_SIZE - 1) begin
+                        // Predicted BOS = end of sequence
                         gen_done <= 1;
                         state <= TOP_DONE;
                     end else begin
                         cur_pos <= cur_pos + 1;
-                        state <= TOP_IDLE;
+                        state <= TOP_IDLE;  // wait for next_token pulse
                     end
                 end
 
                 TOP_DONE: begin
+                    // Stay here until start_gen resets everything
                     state <= TOP_IDLE;
                 end
 
diff --git a/rtl/topksampler.sv b/rtl/topksampler.sv
deleted file mode 100644
index d669afa..0000000
--- a/rtl/topksampler.sv
+++ /dev/null
@@ -1,280 +0,0 @@
-// ===========================================================================
-// Top-K Sampler with Temperature Scaling
-// ===========================================================================
-// Replaces pure argmax with probabilistic sampling from top-k candidates.
-//
-// Algorithm:
-//   1. Find top-k highest logits (parallel comparators)
-//   2. Apply temperature scaling: logit' = logit >> TEMP_SHIFT
-//   3. Compute softmax over top-k only
-//   4. Sample using LFSR-generated random number
-//
-// Benefits:
-//   - Non-deterministic output (diversity in generation)
-//   - Temperature control (creativity vs coherence tradeoff)
-//   - Efficient: only k comparisons, not full VOCAB_SIZE softmax
-//
-// Resource usage on VU19P:
-//   - LUTs: ~500 (comparator tree + softmax)
-//   - FFs:  ~300 (state + top-k storage)
-//   - BRAM: 0 (all registers)
-//   - DSP:  0 (fixed-point multiply done in fabric)
-// ===========================================================================
-
-module topk_sampler
-    import microgpt_pkg::*;
-#(
-    parameter int K = TOP_K              // Number of candidates (default 5)
-)
-(
-    input  logic        clk,
-    input  logic        rst_n,
-    
-    // Control
-    input  logic        start,           // Begin sampling
-    input  logic [4:0]  seed,            // LFSR seed (use cur_pos or timer)
-    
-    // Logits input
-    input  fixed_t      logits [VOCAB_SIZE-1:0],
-    
-    // Output
-    output logic [4:0]  token_out,       // Sampled token ID
-    output logic        valid            // High for 1 cycle when done
-);
-
-    // -----------------------------------------------------------------------
-    // State machine
-    // -----------------------------------------------------------------------
-    typedef enum logic [2:0] {
-        TK_IDLE,
-        TK_FIND_TOPK,      // Find k highest logits
-        TK_TEMP_SCALE,     // Apply temperature: logit >> TEMP_SHIFT
-        TK_SOFTMAX,        // Compute exp and normalize over k candidates
-        TK_SAMPLE,         // Use LFSR to pick from distribution
-        TK_DONE
-    } topk_state_t;
-
-    topk_state_t state;
-
-    // -----------------------------------------------------------------------
-    // Top-K storage
-    // -----------------------------------------------------------------------
-    fixed_t      topk_logits [K-1:0];   // k highest logit values
-    logic [4:0]  topk_ids    [K-1:0];   // corresponding token IDs
-    fixed_t      topk_scaled [K-1:0];   // after temperature scaling
-    fixed_t      topk_probs  [K-1:0];   // after softmax
-    
-    // -----------------------------------------------------------------------
-    // Working variables (all declared at top)
-    // -----------------------------------------------------------------------
-    logic [4:0]  scan_idx;               // Loop counter for finding top-k
-    fixed_t      min_topk;               // Smallest value in current top-k
-    logic [2:0]  min_topk_pos;           // Its position in topk_logits[]
-    logic [2:0]  i;
-    logic [2:0]  j;
-    
-    // Temperature scaling
-    fixed_t      max_scaled;             // Max of scaled logits (for softmax stability)
-    
-    // Softmax
-    fixed_t      exp_vals [K-1:0];       // Exponentials
-    fixed_t      exp_sum;                // Sum of exponentials
-    logic signed [31:0] dividend;        // For normalization
-    
-    // LFSR for sampling
-    logic [15:0] lfsr;                   // 16-bit LFSR
-    logic [15:0] rand_val;               // Random value in [0, 65535]
-    fixed_t      cumsum;                 // Cumulative probability
-    logic [15:0] threshold;              // rand_val scaled to [0, 1.0] in Q12.4
-    
-    // -----------------------------------------------------------------------
-    // Exponential lookup table (reuse from softmax module)
-    // -----------------------------------------------------------------------
-    fixed_t exp_table [0:255];
-    
-    initial begin
-        // exp(x) for x in [-8, 8] in Q12.4
-        for (int idx = 0; idx < 256; idx++) begin
-            real x, exp_val;
-            x = -8.0 + (idx * 16.0 / 256.0);
-            exp_val = $exp(x);
-            if (exp_val > 2047.0) exp_val = 2047.0;
-            if (exp_val < 0.0001) exp_val = 0.0001;
-            exp_table[idx] = float_to_fixed(exp_val);
-        end
-    end
-    
-    function automatic fixed_t lookup_exp(fixed_t x);
-        logic signed [15:0] x_int;
-        logic [7:0] table_idx;
-        
-        x_int = x;
-        
-        // Clamp to [-8, 8] in Q12.4
-        if (x_int < float_to_fixed(-8.0))
-            return float_to_fixed(0.0001);
-        if (x_int > float_to_fixed(8.0))
-            return float_to_fixed(2047.0);
-        
-        // Map to [0, 255]
-        table_idx = ((x_int + float_to_fixed(8.0)) >>> 4);
-        if (table_idx > 255) table_idx = 255;
-        
-        return exp_table[table_idx];
-    endfunction
-    
-    // -----------------------------------------------------------------------
-    // Main FSM
-    // -----------------------------------------------------------------------
-    always_ff @(posedge clk or negedge rst_n) begin
-        if (!rst_n) begin
-            state     <= TK_IDLE;
-            token_out <= 0;
-            valid     <= 0;
-            scan_idx  <= 0;
-            lfsr      <= 16'hACE1;  // Non-zero seed
-            
-            for (i = 0; i < K; i++) begin
-                topk_logits[i] <= 16'sh8000;  // -2048 (most negative)
-                topk_ids[i]    <= 0;
-                topk_scaled[i] <= '0;
-                topk_probs[i]  <= '0;
-                exp_vals[i]    <= '0;
-            end
-            
-        end else begin
-            valid <= 0;  // Pulse signal
-            
-            case (state)
-                
-                // ===========================================================
-                TK_IDLE: begin
-                    if (start) begin
-                        // Initialize LFSR with seed
-                        lfsr <= {11'b0, seed};
-                        if (seed == 0) lfsr <= 16'hACE1;  // Avoid all-zero
-                        
-                        // Reset top-k array
-                        for (i = 0; i < K; i++) begin
-                            topk_logits[i] <= 16'sh8000;
-                            topk_ids[i]    <= 0;
-                        end
-                        
-                        scan_idx <= 0;
-                        state <= TK_FIND_TOPK;
-                    end
-                end
-                
-                // ===========================================================
-                // Find top-k logits using insertion sort approach
-                // ===========================================================
-                TK_FIND_TOPK: begin
-                    if (scan_idx < VOCAB_SIZE) begin
-                        // Find minimum in current top-k
-                        min_topk     = topk_logits[0];
-                        min_topk_pos = 0;
-                        for (i = 1; i < K; i++) begin
-                            if (topk_logits[i] < min_topk) begin
-                                min_topk     = topk_logits[i];
-                                min_topk_pos = i;
-                            end
-                        end
-                        
-                        // If current logit > min, replace
-                        if (logits[scan_idx] > min_topk) begin
-                            topk_logits[min_topk_pos] <= logits[scan_idx];
-                            topk_ids[min_topk_pos]    <= scan_idx;
-                        end
-                        
-                        scan_idx <= scan_idx + 1;
-                        
-                    end else begin
-                        state <= TK_TEMP_SCALE;
-                    end
-                end
-                
-                // ===========================================================
-                // Temperature scaling: logit' = logit >> TEMP_SHIFT
-                // ===========================================================
-                TK_TEMP_SCALE: begin
-                    for (i = 0; i < K; i++) begin
-                        topk_scaled[i] <= topk_logits[i] >>> TEMP_SHIFT;
-                    end
-                    
-                    // Find max for softmax stability
-                    max_scaled = topk_scaled[0];
-                    for (i = 1; i < K; i++) begin
-                        if (topk_scaled[i] > max_scaled)
-                            max_scaled = topk_scaled[i];
-                    end
-                    
-                    state <= TK_SOFTMAX;
-                end
-                
-                // ===========================================================
-                // Softmax over k candidates
-                // ===========================================================
-                TK_SOFTMAX: begin
-                    // Compute exp(scaled - max)
-                    for (i = 0; i < K; i++) begin
-                        exp_vals[i] <= lookup_exp(topk_scaled[i] - max_scaled);
-                    end
-                    
-                    // Sum exponentials
-                    exp_sum = '0;
-                    for (i = 0; i < K; i++) begin
-                        exp_sum = fixed_add(exp_sum, exp_vals[i]);
-                    end
-                    
-                    // Normalize
-                    for (i = 0; i < K; i++) begin
-                        if (exp_vals[i] == '0) begin
-                            topk_probs[i] <= '0;
-                        end else begin
-                            dividend = exp_vals[i] <<< FRAC_BITS;
-                            topk_probs[i] <= fixed_t'(dividend / exp_sum);
-                        end
-                    end
-                    
-                    state <= TK_SAMPLE;
-                end
-                
-                // ===========================================================
-                // Sample using LFSR random number
-                // ===========================================================
-                TK_SAMPLE: begin
-                    // Advance LFSR (16-bit Fibonacci LFSR, taps at 16,14,13,11)
-                    lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
-                    rand_val = lfsr;
-                    
-                    // Convert to [0, 1.0] in Q12.4: rand_val / 65536 * 16 = rand_val >> 12
-                    threshold = rand_val >>> 12;  // Now in [0, 15] (Q12.4 range [0, 1.0))
-                    
-                    // Cumulative sampling
-                    cumsum = '0;
-                    token_out = topk_ids[0];  // Default to first candidate
-                    
-                    for (i = 0; i < K; i++) begin
-                        cumsum = fixed_add(cumsum, topk_probs[i]);
-                        if (threshold < cumsum) begin
-                            token_out = topk_ids[i];
-                            disable for;  // Break on first match
-                        end
-                    end
-                    
-                    state <= TK_DONE;
-                end
-                
-                // ===========================================================
-                TK_DONE: begin
-                    valid <= 1;
-                    state <= TK_IDLE;
-                end
-                
-                default: state <= TK_IDLE;
-                
-            endcase
-        end
-    end
-
-endmodule : topk_sampler
\ No newline at end of file
