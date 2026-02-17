"""
export_weights_q124.py
=======================
Export microGPT weights to Q12.4 fixed-point format for FPGA.

Q12.4 format:
- 12 integer bits: range [-2048, +2047]
- 4 fractional bits: precision 1/16 = 0.0625
- Still 16-bit words (fits existing BRAM infrastructure)

Benefits over Q8.8:
- 16× wider range → less saturation in residual connections
- Better suited for Transformer's accumulating values
- Empirically improves output quality on real names dataset

Usage:
    from export_weights_q124 import export_param_mem_q124
    export_param_mem_q124(state_dict, uchars)
"""

import math

# ---------------------------------------------------------------------------
# Q12.4 Configuration
# ---------------------------------------------------------------------------
FRAC_BITS = 4
SCALE = 1 << FRAC_BITS   # 16

# Model dimensions
VOCAB_SIZE  = 27
N_EMBD      = 16
N_HEAD      = 4
N_LAYER     = 1
BLOCK_SIZE  = 16
MLP_DIM     = 4 * N_EMBD

# ---------------------------------------------------------------------------
# Q12.4 Conversion
# ---------------------------------------------------------------------------
def to_q124_hex(val):
    """
    Convert float to Q12.4 hex string.
    
    Range: [-2048.0, +2047.9375]
    Precision: 1/16 = 0.0625
    
    Returns: 4-digit uppercase hex (two's complement)
    """
    if hasattr(val, 'data'):
        val = val.data
    
    fixed = int(round(val * SCALE))
    
    # Saturate to 16-bit signed range
    fixed = max(-32768, min(32767, fixed))
    
    # Two's complement for negative values
    if fixed < 0:
        fixed = fixed + 65536
    
    return f"{fixed:04X}"


def q124_to_float(hex_str):
    """Convert Q12.4 hex string back to float (for verification)."""
    v = int(hex_str, 16)
    if v >= 32768:
        v -= 65536
    return v / SCALE


# ---------------------------------------------------------------------------
# Main Export Function
# ---------------------------------------------------------------------------
def export_param_mem_q124(state_dict, uchars, filename="param_q124.mem"):
    """
    Export trained weights to Q12.4 format.
    
    Output file format:
    - One 16-bit hex value per line
    - 4192 lines total
    - $readmemh compatible
    """
    
    BOS = len(uchars)
    lines = []
    address = 0

    def write_matrix(name, matrix):
        nonlocal address
        count = 0
        for row in matrix:
            for val in row:
                lines.append(to_q124_hex(val))
                count += 1
                address += 1
        print(f"  {name:<20s}  rows={len(matrix):<3d}  cols={len(matrix[0]):<3d}  "
              f"params={count:<4d}  addr={address-count:4d}..{address-1:4d}")

    print("=" * 70)
    print(f"Exporting weights to '{filename}' (Q12.4 format)")
    print(f"Precision: ±{1/SCALE:.4f}  Range: [-2048, +2047.9375]")
    print("=" * 70)

    # Match Python state_dict iteration order
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

    print(f"\nTotal words: {address}")
    assert address == 4192, f"Expected 4192, got {address}"

    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\n✓ Written to '{filename}'")
    print("=" * 70)

    # Spot-check first few weights
    print("\nSpot-check (first 8 wte values):")
    for i in range(8):
        print(f"  [{i}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")

    # Vocabulary
    print("\nVocabulary:")
    for i, ch in enumerate(uchars):
        print(f"  {i:2d} = '{ch}'")
    print(f"  {BOS:2d} = BOS")

    return filename


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_param_mem_q124(filename="param_q124.mem"):
    """Verify exported weights."""
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\nVerifying '{filename}'")
    print(f"  Lines: {len(lines)}")
    print(f"  Expected: 4192")

    if len(lines) != 4192:
        print(f"  ✗ MISMATCH!")
        return False

    # Decode sample values
    print(f"\n  First 4 (wte[0][0..3]):")
    for i in range(4):
        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")

    print(f"\n  Around attn_wq (addr 1120):")
    for i in range(1120, 1124):
        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")

    print(f"\n  Last 4 (mlp_fc2 tail):")
    for i in range(4188, 4192):
        print(f"    [{i:4d}] {lines[i]}  →  {q124_to_float(lines[i]):+8.4f}")

    print(f"\n  ✓ Looks good")
    return True


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    
    fname = sys.argv[1] if len(sys.argv) > 1 else "param_q124.mem"
    
    if os.path.exists(fname):
        verify_param_mem_q124(fname)
    else:
        print(f"'{fname}' not found.")
        print("\nUsage:")
        print("  # In your training script:")
        print("  from export_weights_q124 import export_param_mem_q124")
        print("  export_param_mem_q124(state_dict, uchars)")
        print()
        print("  # Or verify existing file:")
        print(f"  python {__file__} param_q124.mem")