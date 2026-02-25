# microGPT FPGA - Testing Guide

## Testing Hierarchy

The testbench suite follows a bottom-up validation strategy. Higher-level modules depend on lower-level ones being correct, so tests should be run in order.

```
Level 0: Fundamentals
  ├─ Fixed-point arithmetic
  └─ Type conversions

Level 1: Basic Building Blocks
  ├─ Parameter Memory
  └─ Vector Dot Product

Level 2: Math Operations
  ├─ Matrix-Vector Multiply
  ├─ RMSNorm
  └─ Softmax

Level 3: Neural Network Layers
  ├─ Attention Head
  ├─ Multi-Head Attention
  └─ MLP Block

Level 4: Complete Layer
  └─ Transformer Layer

Level 5: Full System
  └─ microGPT Top
```

---

## Running Tests in Vivado

Each testbench in `tb/` is standalone and requires only its direct RTL dependencies.

1. Create a new project and add the source files listed for the test.
2. Set the testbench as the simulation top (right-click → Set as Top in Simulation Sources).
3. Run Behavioral Simulation.
4. Check the TCL Console for the PASS/FAIL summary.

### File dependencies per test

| Test | Sources |
|------|---------|
| Fixed-point | `rtl/microgpt_pkg.sv`, `tb/tb_fixed_point.sv` |
| Vector dot product | above + `rtl/vector_dot_product.sv`, `tb/tb_vector_dot_product.sv` |
| Parameter memory | `rtl/microgpt_pkg.sv`, `rtl/param_memory.sv`, `tb/tb_param_memory.sv` |
| RMSNorm | `rtl/microgpt_pkg.sv`, `rtl/rmsnorm.sv`, `tb/tb_rmsnorm.sv` |
| Softmax | `rtl/microgpt_pkg.sv`, `rtl/softmax.sv`, `tb/tb_softmax.sv` |
| Matrix-vector | above + `rtl/vector_dot_product.sv`, `rtl/matrix_vector_mult.sv`, `tb/tb_matrix_vector_mult.sv` |

### TCL Script

```tcl
cd /path/to/project
source sim/setup_tests.tcl
setup_test vector_dot
launch_simulation
```

---

## Test Reference

### Test 0: Fixed-Point Arithmetic (`tb/tb_fixed_point.sv`)

Verifies Q8.8 (or Q12.4 depending on `FRAC_BITS` in `microgpt_pkg.sv`) fixed-point conversion, addition, multiplication, and saturation behavior.

Expected output ends with:
```
All Fixed-Point Tests Complete!
```

Errors should be below the quantization step (1/256 for Q8.8, 1/16 for Q12.4).

---

### Test 1: Vector Dot Product (`tb/tb_vector_dot_product.sv`)

Verifies the 4-element dot product state machine across 7 cases: all zeros, all ones, integer sequences, mixed signs, fractional values, a standard [1,2,3,4]·[5,6,7,8] case, and a large negative case.

Expected summary:
```
Test Summary:
  Passed: 7
  Failed: 0

✓ ALL TESTS PASSED!
```

In the waveform: state should transition IDLE → MULTIPLY → ACCUMULATE → DONE, with `valid` pulsing in DONE.

---

### Test 2: Parameter Memory (`tb/tb_param_memory.sv`)

22 tests covering single write/read, sequential writes, random-address access, overwrites, uninitialized read (expect zero), and pattern verification. Total parameter space: 4192 words.

Expected summary:
```
Test Summary:
  Passed: 22
  Failed: 0

✓ ALL TESTS PASSED!
```

---

### Test 3: Matrix-Vector Multiply (`tb/tb_matrix_vector_mult.sv`)

6 tests including identity matrix, zero matrix, scaling, and mixed-value cases. All dot product instances run in parallel.

Expected summary:
```
Test Summary:
  Passed: 6
  Failed: 0

✓ ALL TESTS PASSED!
```

---

### Test 4: RMS Normalization (`tb/tb_rmsnorm.sv`)

5 tests verifying that output RMS is approximately 1.0 across different input scales. Uses an inverse sqrt lookup table (256 entries, range 0.001–16.0).

Expected output RMS error < 0.2 for all cases.

---

### Test 5: Softmax (`tb/tb_softmax.sv`)

6 tests: uniform logits, one dominant logit, negative logits, temperature variation, near-zero logits, and a probability-sum verification. Uses a numerically stable max-subtracted exponential LUT.

Key check: probabilities must sum to ~1.0 (error < 0.001).

---

## Debugging Notes

- **Uninitialized signals (X in waveform):** Check that reset properly initializes all state registers.
- **Accumulation overflow:** Fixed-point accumulation can saturate with many large values. Verify `fixed_add` uses saturating arithmetic from `microgpt_pkg.sv`.
- **LUT initialization:** `initial` blocks in `rmsnorm.sv` and `softmax.sv` populate lookup tables at simulation startup. These are synthesis-safe for Xilinx BRAM initialization.
- **Timing:** `param_memory.sv` has a one-cycle read latency — `rd_valid` goes high the cycle after `rd_en`.

---

## Component Status

See `docs/STATUS.md` for current test coverage and integration status per module.
