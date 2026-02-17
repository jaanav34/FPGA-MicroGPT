"""
export_weights.py
=================
Exports trained microGPT weights to param.mem for FPGA inference.

Usage:
    1. Train the model using microGPT_original_python_code
    2. At the END of training (after the Adam loop), add:
           from export_weights import export_param_mem
           export_param_mem(state_dict, uchars)
    3. OR run this file standalone if you saved state_dict to a file.

Output:
    param.mem  — one 4-hex-digit Q8.8 value per line, address 0 first.
                 Drop this file next to your Vivado project / simulation.

Address map (must match ADDR_* constants in microgpt_top.sv):
    [0      ]  wte        VOCAB_SIZE * N_EMBD   = 27*16 = 432
    [432    ]  wpe        BLOCK_SIZE * N_EMBD   = 16*16 = 256
    [688    ]  lm_head    VOCAB_SIZE * N_EMBD   = 27*16 = 432
    [1120   ]  attn_wq    N_EMBD    * N_EMBD    = 16*16 = 256
    [1376   ]  attn_wk    N_EMBD    * N_EMBD    = 256
    [1632   ]  attn_wv    N_EMBD    * N_EMBD    = 256
    [1888   ]  attn_wo    N_EMBD    * N_EMBD    = 256
    [2144   ]  mlp_fc1    MLP_DIM   * N_EMBD    = 64*16 = 1024
    [3168   ]  mlp_fc2    N_EMBD    * MLP_DIM   = 16*64 = 1024
    Total                                        = 4192 words

Q8.8 encoding:
    fixed = round(float_value * 256)
    clamped to [-32768, 32767]  (signed 16-bit)
    written as 4-digit uppercase hex (two's complement)
"""

import math

# ---------------------------------------------------------------------------
# Model dimensions  (must match microgpt_pkg.sv)
# ---------------------------------------------------------------------------
VOCAB_SIZE  = 27      # 26 letters + BOS
N_EMBD      = 16
N_HEAD      = 4
N_LAYER     = 1
BLOCK_SIZE  = 16
MLP_DIM     = 4 * N_EMBD   # 64

FRAC_BITS   = 8
SCALE       = 1 << FRAC_BITS   # 256


# ---------------------------------------------------------------------------
# Helper: float → Q8.8 hex string
# ---------------------------------------------------------------------------
def to_q88_hex(val):
    """Convert a float to a 4-digit Q8.8 hex string (two's complement)."""
    if hasattr(val, 'data'):        # unwrap Value objects from microGPT
        val = val.data
    fixed = int(round(val * SCALE))
    fixed = max(-32768, min(32767, fixed))   # saturate
    if fixed < 0:
        fixed = fixed + 65536                # two's complement 16-bit
    return f"{fixed:04X}"


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------
def export_param_mem(state_dict, uchars, filename="param.mem"):
    """
    Write all trained weights to `filename` in the order expected by
    microgpt_top.sv.

    Parameters
    ----------
    state_dict : dict
        The microGPT state_dict after training.
        Keys: 'wte', 'wpe', 'lm_head',
              'layer0.attn_wq', 'layer0.attn_wk', etc.
    uchars : list[str]
        Sorted unique characters (used only for reporting vocab mapping).
    filename : str
        Output filename (default: "param.mem").
    """

    BOS = len(uchars)   # BOS token id

    lines   = []
    address = 0

    def write_matrix(name, matrix):
        """Flatten matrix row-major and append hex lines."""
        nonlocal address
        count = 0
        for row in matrix:
            for val in row:
                lines.append(to_q88_hex(val))
                count += 1
                address += 1
        print(f"  {name:<20s}  rows={len(matrix)}  cols={len(matrix[0])}  "
              f"params={count}  addr_end={address}")

    print("=" * 60)
    print(f"Exporting weights to '{filename}'")
    print(f"Expected total params: {VOCAB_SIZE*N_EMBD + BLOCK_SIZE*N_EMBD + VOCAB_SIZE*N_EMBD + N_LAYER*(4*N_EMBD*N_EMBD + 2*MLP_DIM*N_EMBD)}")
    print("=" * 60)

    # Exactly mirrors Python line 89:
    # params = [p for mat in state_dict.values() for row in mat for p in row]
    write_matrix("wte",            state_dict['wte'])
    write_matrix("wpe",            state_dict['wpe'])
    write_matrix("lm_head",        state_dict['lm_head'])
    for li in range(N_LAYER):
        write_matrix(f"layer{li}.attn_wq", state_dict[f'layer{li}.attn_wq'])
        write_matrix(f"layer{li}.attn_wk", state_dict[f'layer{li}.attn_wk'])
        write_matrix(f"layer{li}.attn_wv", state_dict[f'layer{li}.attn_wv'])
        write_matrix(f"layer{li}.attn_wo", state_dict[f'layer{li}.attn_wo'])
        write_matrix(f"layer{li}.mlp_fc1", state_dict[f'layer{li}.mlp_fc1'])
        write_matrix(f"layer{li}.mlp_fc2", state_dict[f'layer{li}.mlp_fc2'])

    print(f"\nTotal words written: {address}")
    assert address == len(lines), "Line count mismatch!"

    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Written to '{filename}'  ✓")
    print("=" * 60)

    # Print vocabulary mapping so you can decode token IDs in simulation
    print("\nVocabulary mapping (for simulation decoding):")
    for i, ch in enumerate(uchars):
        print(f"  token {i:2d} = '{ch}'")
    print(f"  token {BOS:2d} = BOS (end-of-sequence)")

    return filename


# ---------------------------------------------------------------------------
# Verify an existing param.mem matches the expected total
# ---------------------------------------------------------------------------
def verify_param_mem(filename="param.mem"):
    """Quick sanity check: count lines and spot-check first/last addresses."""
    expected = (VOCAB_SIZE * N_EMBD +
                BLOCK_SIZE * N_EMBD +
                VOCAB_SIZE * N_EMBD +
                N_LAYER * (4 * N_EMBD * N_EMBD +
                           MLP_DIM * N_EMBD +
                           N_EMBD * MLP_DIM))

    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"verify_param_mem('{filename}')")
    print(f"  Lines in file : {len(lines)}")
    print(f"  Expected lines: {expected}")

    if len(lines) != expected:
        print(f"  ✗ MISMATCH!  ({len(lines)} vs {expected})")
        return False

    # Decode first few and last few entries as a spot-check
    def q88(hex_str):
        v = int(hex_str, 16)
        if v >= 32768:
            v -= 65536
        return v / SCALE

    print(f"\n  First 8 entries (wte[0][0..7]):")
    for i in range(8):
        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")

    print(f"\n  Entries around attn_wq base (addr 1120):")
    for i in range(1120, 1124):
        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")

    print(f"\n  Last 4 entries (mlp_fc2 tail):")
    for i in range(expected - 4, expected):
        print(f"    [{i:4d}] {lines[i]}  →  {q88(lines[i]):+.4f}")

    print(f"\n  ✓ param.mem looks correct")
    return True


# ---------------------------------------------------------------------------
# Standalone: verify an existing param.mem
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "param.mem"
    import os
    if os.path.exists(fname):
        verify_param_mem(fname)
    else:
        print(f"'{fname}' not found.")
        print("Run from inside your microGPT training script after training:")
        print("  from export_weights import export_param_mem")
        print("  export_param_mem(state_dict, uchars)")