# microGPT FPGA - Bottom-Up Testing Guide

## Philosophy: Test Every Component Individually

We're building this design from the ground up, testing each piece thoroughly before moving to the next level.

## Testing Hierarchy

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

## How to Run Each Test

### Prerequisites

All tests are in `tb/` directory. Each test is standalone and can be run independently.

### Running in Vivado

1. **Create new project** (or use existing)

2. **Add files for the test you want:**
   ```
   For Fixed-Point Test:
     - rtl/microgpt_pkg.sv
     - tb/tb_fixed_point.sv
   
   For Vector Dot Product Test:
     - rtl/microgpt_pkg.sv
     - rtl/vector_dot_product.sv
     - tb/tb_vector_dot_product.sv
   
   For Parameter Memory Test:
     - rtl/microgpt_pkg.sv
     - rtl/param_memory.sv
     - tb/tb_param_memory.sv
   ```

3. **Set testbench as simulation top**
   - In Simulation Sources, right-click testbench
   - Set as Top

4. **Run Behavioral Simulation**

5. **Check TCL Console for results**

---

## Test 0: Fixed-Point Arithmetic

**File:** `tb/tb_fixed_point.sv`

**What it tests:**
- Float ↔ Fixed-point conversion accuracy
- Addition of fixed-point numbers
- Multiplication of fixed-point numbers
- Range limits of Q8.8 format

**Expected output:**
```
==============================================
Fixed-Point Q8.8 Arithmetic Test
==============================================

TEST 1: Float to Fixed Conversion
----------------------------------------------
Float:   0.0000 → Fixed: 0x0000 (     0) → Float:   0.0000 | Error: 0.000000
Float:   1.0000 → Fixed: 0x0100 (   256) → Float:   1.0000 | Error: 0.000000
Float:  -1.0000 → Fixed: 0xff00 (  -256) → Float:  -1.0000 | Error: 0.000000
...

TEST 2: Addition
----------------------------------------------
 1.000 +  2.000 =  3.000 | Expected:  3.000 | Error: 0.00000 ✓
 0.500 +  0.250 =  0.750 | Expected:  0.750 | Error: 0.00000 ✓
...

TEST 3: Multiplication
----------------------------------------------
 2.000 ×  3.000 =  6.000 | Expected:  6.000 | Error: 0.00000 ✓
 0.500 ×  0.500 =  0.250 | Expected:  0.250 | Error: 0.00000 ✓
...

All Fixed-Point Tests Complete!
```

**What to look for:**
- ✅ All errors should be < 0.005
- ✅ No warnings
- ✅ Conversion errors are tiny due to Q8.8 resolution (1/256)

**If it fails:**
- Check that FRAC_BITS = 8 in package
- Verify multiplication shift (>>> FRAC_BITS)

---

## Test 1: Vector Dot Product

**File:** `tb/tb_vector_dot_product.sv`

**What it tests:**
- Vector dot product: sum(a[i] * b[i])
- State machine operation
- Fixed-point accumulation accuracy
- Edge cases (zeros, negatives, fractional)

**Expected output:**
```
==============================================
Vector Dot Product Test (VEC_LEN = 4)
==============================================

TEST 1: All zeros
  Input A: [0.00, 0.00, 0.00, 0.00]
  Input B: [0.00, 0.00, 0.00, 0.00]
  Expected: 0.0000
  Actual:   0.0000
  Error:    0.000000
  Result: PASS ✓

TEST 2: All ones
  Input A: [1.00, 1.00, 1.00, 1.00]
  Input B: [1.00, 1.00, 1.00, 1.00]
  Expected: 4.0000
  Actual:   4.0000
  Error:    0.000000
  Result: PASS ✓

...

TEST 6: Standard dot product
  Input A: [1.00, 2.00, 3.00, 4.00]
  Input B: [5.00, 6.00, 7.00, 8.00]
  Expected: 70.0000
  Actual:   70.0000
  Error:    0.000000
  Result: PASS ✓

==============================================
Test Summary:
  Passed: 7
  Failed: 0
  Total:  7

✓ ALL TESTS PASSED!
```

**What to look for:**
- ✅ All 7 tests pass
- ✅ Error < 0.1 for all tests
- ✅ `valid` signal pulses correctly
- ✅ State machine: IDLE → MULTIPLY → ACCUMULATE → DONE

**In waveform, check:**
1. Start pulse triggers computation
2. State transitions cleanly
3. Products array gets filled in MULTIPLY state
4. Accumulator builds up in ACCUMULATE
5. Valid goes high in DONE state
6. Returns to IDLE

**If it fails:**
- Check fixed_mul() implementation
- Verify accumulation logic doesn't overflow
- Look for uninitialized signals

---

## Test 2: Parameter Memory

**File:** `tb/tb_param_memory.sv`

**What it tests:**
- Write operation
- Read operation
- Write-then-read (data integrity)
- Uninitialized locations (should be zero)
- Address boundaries
- Overwrite capability

**Expected output:**
```
==============================================
Parameter Memory Test
Total params: 4192
==============================================

TEST 1: Write and read single value
  Addr    0: Wrote  1.5000, Read  1.5000, Error 0.000000

TEST 2: Write and read multiple sequential values
  Addr    0: Wrote  0.0000, Read  0.0000, Error 0.000000
  Addr    1: Wrote  0.5000, Read  0.5000, Error 0.000000
  Addr    2: Wrote  1.0000, Read  1.0000, Error 0.000000
  ...

TEST 3: Write to various addresses
  Addr  100: Wrote  2.2500, Read  2.2500, Error 0.000000
  Addr  500: Wrote -1.7500, Read -1.7500, Error 0.000000
  Addr 1000: Wrote  0.1250, Read  0.1250, Error 0.000000

TEST 4: Overwrite existing value
  Addr    0: Wrote  3.5000, Read  3.5000, Error 0.000000
  Addr    0: Wrote -2.2500, Read -2.2500, Error 0.000000

TEST 5: Read uninitialized location (should be 0)
  Addr 2000: Expected  0.0000, Read  0.0000, Error 0.000000

...

Test Summary:
  Passed: 22
  Failed: 0

✓ ALL TESTS PASSED!
```

**What to look for:**
- ✅ All reads match writes
- ✅ Uninitialized memory reads as zero
- ✅ Overwrite works correctly
- ✅ No address violations

**In waveform, check:**
1. wr_en causes data to be stored
2. rd_en triggers read
3. rd_valid goes high one cycle after rd_en
4. rd_data contains correct value when rd_valid

**If it fails:**
- Verify initial block zeros the memory
- Check address bounds checking
- Ensure rd_valid timing is correct

---

---

## Test 3: Matrix-Vector Multiply

**File:** `tb/tb_matrix_vector_mult.sv`

**What it tests:**
- Matrix-vector multiplication: y = M * x
- Uses vector_dot_product internally (already tested!)
- Multiple simultaneous dot products
- Various matrix patterns

**Expected output:**
```
==============================================
Matrix-Vector Multiply Test (4x4)
==============================================

TEST 1: Identity Matrix
  Matrix:
    [ 1.00,  0.00,  0.00,  0.00]
    [ 0.00,  1.00,  0.00,  0.00]
    [ 0.00,  0.00,  1.00,  0.00]
    [ 0.00,  0.00,  0.00,  1.00]
  Vector: [1.00, 2.00, 3.00, 4.00]
  Expected: [1.00, 2.00, 3.00, 4.00]
  Actual:   [1.00, 2.00, 3.00, 4.00]
  Errors:   [0.0000, 0.0000, 0.0000, 0.0000]
  Result: PASS ✓

...

Test Summary:
  Passed: 6
  Failed: 0

✓ ALL TESTS PASSED!
```

**What to look for:**
- ✅ All 6 tests pass
- ✅ Identity matrix returns input unchanged
- ✅ Zero matrix returns zeros
- ✅ Errors < 0.1 for all elements

**In waveform, check:**
1. All dot product units start simultaneously
2. Each computes one row of output
3. Results captured when all dot_valid signals high
4. Valid asserted when complete

**If it fails:**
- Vector dot product test should pass first!
- Check matrix loading from inputs
- Verify all dot product instances instantiated correctly

---

## Test 4: RMS Normalization

**File:** `tb/tb_rmsnorm.sv`

**What it tests:**
- RMS normalization: y = x / sqrt(mean(x²) + ε)
- Output should have RMS ≈ 1.0
- Preserves relative magnitudes
- Handles various input scales

**Expected output:**
```
==============================================
RMSNorm Test (VEC_LEN = 4)
==============================================

TEST 1: All ones (should normalize to ~1.0)
  Input: [1.00, 1.00, 1.00, 1.00]
  Input RMS: 1.0000
  Output: [1.0000, 1.0000, 1.0000, 1.0000]
  Output RMS: 1.0000
  Expected RMS: 1.0000
  RMS Error: 0.0000
  Result: PASS ✓

TEST 2: Scaled values
  Input: [2.00, 2.00, 2.00, 2.00]
  Input RMS: 2.0000
  Output: [1.0000, 1.0000, 1.0000, 1.0000]
  Output RMS: 1.0000
  Expected RMS: 1.0000
  RMS Error: 0.0000
  Result: PASS ✓

...

Test Summary:
  Passed: 5
  Failed: 0

✓ ALL TESTS PASSED!
```

**What to look for:**
- ✅ Output RMS is always close to 1.0
- ✅ Relative magnitudes preserved
- ✅ Works with small and large inputs
- ✅ RMS error < 0.2

**In waveform, check:**
1. Squares computed correctly
2. Sum accumulates all squares
3. Mean calculated by division
4. Inverse sqrt lookup happens
5. All elements scaled by same factor

**If it fails:**
- Check square computation (x * x)
- Verify sum doesn't overflow
- Check inverse sqrt table initialization
- Ensure division by vector length correct

---

## Test 5: Softmax

**File:** `tb/tb_softmax.sv`

**What it tests:**
- Softmax: prob[i] = exp(logit[i]) / Σ exp(logit[j])
- Numerical stability (subtract max)
- Temperature scaling
- Probability sum = 1.0

**Expected output:**
```
==============================================
Softmax Test (VEC_LEN = 5)
==============================================

TEST 1: Uniform logits
  Logits: [0.00, 0.00, 0.00, 0.00, 0.00]
  Temperature: 1.00
  Expected: [0.200, 0.200, 0.200, 0.200, 0.200]
  Actual:   [0.200, 0.200, 0.200, 0.200, 0.200]
  Max Error: 0.0000
  Result: PASS ✓

TEST 2: One dominant logit
  Logits: [10.00, 0.00, 0.00, 0.00, 0.00]
  Temperature: 1.00
  Expected: [0.990, 0.003, 0.003, 0.003, 0.003]
  Actual:   [0.988, 0.003, 0.003, 0.003, 0.003]
  Max Error: 0.0020
  Result: PASS ✓

...

TEST 6: Probability sum verification
  Logits: [2.00, -1.00, 3.00, 0.00, 1.50]
  Probs: [0.241, 0.012, 0.655, 0.033, 0.146]
  Sum: 1.000000 (Expected: 1.000000)
  Sum Error: 0.000000
  Result: PASS ✓

Test Summary:
  Passed: 6
  Failed: 0

✓ ALL TESTS PASSED!
```

**What to look for:**
- ✅ Probabilities always sum to ~1.0
- ✅ Uniform input → uniform output
- ✅ Dominant logit → high probability
- ✅ Temperature affects distribution
- ✅ No NaN or overflow

**In waveform, check:**
1. Max finding works correctly
2. Scaling subtracts max from all
3. Exponentials computed via lookup
4. Sum accumulates all exponentials
5. Division normalizes correctly

**If it fails:**
- Check exp lookup table initialization
- Verify max subtraction for stability
- Ensure sum doesn't overflow
- Check division implementation

---

## Next Steps

Once all Level 0, 1, and 2 tests pass:

### You Have Built:
✅ Fixed-point arithmetic  
✅ Vector operations  
✅ Parameter storage  
✅ Matrix multiplication  
✅ Normalization  
✅ Probability distributions  

### Ready to Build:
- Attention mechanisms (uses matrix mult + softmax)
- MLP layers (uses matrix mult + relu)
- Complete transformer layer
- Full GPT model

Each component above is **proven and verified**. No more guessing!

---

## Debugging Strategy

When a test fails:

### 1. Read the Error Message
The testbenches print detailed information about what went wrong.

### 2. Check the Waveform
- Are signals initialized?
- Do state machines transition?
- Are there any 'X' values?

### 3. Add More $display
In the RTL, add debug prints:
```systemverilog
$display("State: %s, idx=%0d, acc=%0d", state.name(), idx, accumulator);
```

### 4. Reduce Complexity
If Test 6 fails but Test 1 passes, the issue is with larger values or accumulation.

### 5. Test in Isolation
Create a minimal testbench that only tests the failing case.

---

## Success Criteria

Before moving to the next level:

✅ All tests at current level pass
✅ No warnings in synthesis (if you've run it)
✅ Waveforms look clean (no X's during operation)
✅ You understand WHY each test passes

---

## File Organization

```
microgpt_fpga_tested/
├── rtl/
│   ├── microgpt_pkg.sv           # Always needed
│   ├── param_memory.sv           # Level 1
│   ├── vector_dot_product.sv     # Level 1
│   ├── matrix_vector_mult.sv     # Level 2 (TODO)
│   ├── rmsnorm.sv                # Level 2 (TODO)
│   └── softmax.sv                # Level 2 (TODO)
├── tb/
│   ├── tb_fixed_point.sv         # Level 0 ✓
│   ├── tb_param_memory.sv        # Level 1 ✓
│   ├── tb_vector_dot_product.sv  # Level 1 ✓
│   ├── tb_matrix_vector.sv       # Level 2 (TODO)
│   └── ...
└── docs/
    └── TESTING_GUIDE.md          # This file
```

---

Start with Test 0 (fixed-point) and work your way up. Don't skip levels!

Each passing test gives you confidence that the foundation is solid.

**Let's build this right! 🚀**
