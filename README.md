# microGPT FPGA - Tested & Verified Version

## What's Different in This Version?

This is a **bottom-up tested** implementation. Instead of hoping the full system works, we test each component individually and build confidence layer by layer.

### Key Improvements:

1. **Exhaustive Component Tests** - Every module has its own testbench
2. **Clear Expected Outputs** - Each test prints PASS/FAIL
3. **Incremental Complexity** - Start simple, add complexity gradually
4. **Debug-Friendly** - Easy to see exactly where things break
5. **Proven Foundation** - Don't move up until lower levels pass

---

## Quick Start

### Step 1: Run the Fundamental Tests

These test the math that everything else depends on:

```
1. Test Fixed-Point Arithmetic
   Files needed:
     - rtl/microgpt_pkg.sv
     - tb/tb_fixed_point.sv
   
   This MUST pass before anything else!
   
2. Test Vector Dot Product
   Files needed:
     - rtl/microgpt_pkg.sv
     - rtl/vector_dot_product.sv
     - tb/tb_vector_dot_product.sv
   
   This is used everywhere in matrix operations.
   
3. Test Parameter Memory
   Files needed:
     - rtl/microgpt_pkg.sv
     - rtl/param_memory.sv
     - tb/tb_param_memory.sv
   
   This stores all your trained weights.
```

### Step 2: Check All Tests Pass

Each test should print:
```
✓ ALL TESTS PASSED!
```

If you see:
```
✗ SOME TESTS FAILED!
```

**STOP. FIX IT. Don't proceed.**

### Step 3: Only Then Build Up

Once Level 0 and Level 1 tests all pass, we can build:
- Matrix operations
- Normalization layers  
- Attention mechanisms
- Complete transformer

---

## How to Use in Vivado

### Method 1: Manual Setup

1. Create new Vivado project
2. Add the files for one test (see Quick Start above)
3. Set testbench as simulation top
4. Run Behavioral Simulation
5. Check TCL Console for results

### Method 2: Using TCL Script

1. Create Vivado project
2. In TCL Console, navigate to project directory:
   ```tcl
   cd /path/to/microgpt_fpga_tested
   ```

3. Source the setup script:
   ```tcl
   source sim/setup_tests.tcl
   ```

4. Run a test:
   ```tcl
   setup_test fixed_point
   launch_simulation
   ```

5. After checking results, try next test:
   ```tcl
   setup_test vector_dot
   launch_simulation
   ```

---

## Directory Structure

```
microgpt_fpga_tested/
│
├── rtl/                    # RTL modules
│   ├── microgpt_pkg.sv    # Package (types, functions)
│   ├── param_memory.sv    # Parameter storage
│   └── vector_dot_product.sv
│
├── tb/                     # Testbenches
│   ├── tb_fixed_point.sv      # Tests Q8.8 arithmetic
│   ├── tb_param_memory.sv     # Tests memory read/write
│   └── tb_vector_dot_product.sv  # Tests dot product
│
├── sim/                    # Simulation support
│   └── setup_tests.tcl    # Quick test setup script
│
└── docs/
    ├── TESTING_GUIDE.md   # Comprehensive testing guide
    └── README.md          # This file
```

---

## What Each Test Does

### Fixed-Point Arithmetic (`tb_fixed_point.sv`)

**Purpose:** Verify the math works correctly

**Tests:**
- Float to fixed-point conversion
- Fixed-point to float conversion  
- Addition of fixed-point numbers
- Multiplication of fixed-point numbers
- Range limits

**Why it matters:**
All neural network computations use this math. If this is wrong, everything is wrong.

**Expected result:**
```
All Fixed-Point Tests Complete!
```

---

### Vector Dot Product (`tb_vector_dot_product.sv`)

**Purpose:** Verify dot product: result = Σ(a[i] × b[i])

**Tests:**
- All zeros (edge case)
- All ones (simple accumulation)
- Integer sequences
- Mixed positive/negative
- Fractional values
- Standard dot product
- Large negative values

**Why it matters:**
Used in every matrix multiply, every attention computation. This is the workhorse operation.

**Expected result:**
```
Test Summary:
  Passed: 7
  Failed: 0

✓ ALL TESTS PASSED!
```

---

### Parameter Memory (`tb_param_memory.sv`)

**Purpose:** Verify weight storage works correctly

**Tests:**
- Single write/read
- Sequential writes
- Random address access
- Overwrite existing data
- Read uninitialized (should be zero)
- Pattern write/read

**Why it matters:**
All your trained model weights live here. If this fails, the model gets garbage weights.

**Expected result:**
```
Test Summary:
  Passed: 22
  Failed: 0

✓ ALL TESTS PASSED!
```

---

## Debugging Tips

### If Fixed-Point Test Fails:

**Symptom:** Conversion errors > 0.01

**Check:**
1. FRAC_BITS = 8 in package?
2. Multiply function doing `>>> FRAC_BITS`?
3. Are you saturating on overflow?

**Fix:** Look at the specific test that failed. If multiplication fails but addition passes, issue is in `fixed_mul()`.

---

### If Vector Dot Product Fails:

**Symptom:** Some tests pass, some fail

**Check in waveform:**
1. Are all products computed correctly in MULTIPLY state?
2. Is accumulator building up in ACCUMULATE state?
3. Are any values showing as 'X'?

**Common issues:**
- Accumulator not initialized to zero
- Products array has uninitialized values
- Multiply happening but results not stored

---

### If Parameter Memory Fails:

**Symptom:** Read-back doesn't match write

**Check in waveform:**
1. Is wr_en actually high when writing?
2. Is the address in bounds?
3. Does rd_valid come high after rd_en?
4. Look at param_ram[addr] - does it update?

**Common issues:**
- Address out of bounds (silently ignored)
- Timing: reading before write completes
- Memory not initialized (should be all zeros)

---

## Next Steps (After All Tests Pass)

### Level 2: Build Matrix Operations

1. **Matrix-Vector Multiply**
   - Uses vector_dot_product internally
   - Test with known matrices first
   - e.g., Identity matrix should return input vector

2. **RMSNorm**
   - Test with simple known values
   - Check that output has correct magnitude

3. **Softmax**
   - Critical for attention
   - Test with simple logits
   - Verify probabilities sum to 1.0

### Level 3: Neural Network Layers

Only after Level 2 all passes!

### Level 4: Complete System

Only after Level 3 all passes!

---

## Testing Philosophy

**Traditional approach:**
```
Build everything → Test → Debug everything → Cry
                                    ↑
                              You are here
```

**Our approach:**
```
Test A → ✓ → Build B → Test B → ✓ → Build C → Test C → ✓
                                                          ↓
                                                    Ship it! 🚀
```

Each component is **proven** before you build the next level.

---

## Success Criteria

Before moving to next component:

- ✅ All tests pass
- ✅ No X's in waveform (during operation)
- ✅ You understand what the test is doing
- ✅ You can explain why it passes

Don't proceed with failures. Fix them first!

---

## Why This Approach Works

1. **Isolates Problems**
   - If vec_dot test fails, problem is in vec_dot only
   - Not scattered across 10 files

2. **Builds Confidence**
   - Each passing test = solid foundation
   - No "hope it works" guessing

3. **Saves Time**
   - 10 minutes per test now
   - Saves hours of debugging later

4. **Easy to Verify**
   - Clear PASS/FAIL output
   - Not interpreting waveforms for hours

5. **Incremental Progress**
   - Each test is a milestone
   - Visible progress

---

## Common Questions

### Q: Can I skip the tests and go straight to the full system?
**A:** You can, but you'll waste time debugging. These tests take 30 minutes total. Debugging the full system takes days.

### Q: My test is failing. Can I just proceed anyway?
**A:** No! The next level depends on this one working. Fix it now.

### Q: Do I need to test in simulation AND hardware?
**A:** Simulation first! Fix all issues there. Hardware testing comes later.

### Q: Can I modify the tests?
**A:** Yes! Add more test cases if you want extra confidence.

### Q: The fixed-point test passed. Can I trust the math now?
**A:** Yes! That's the point. Once it passes, all higher levels can rely on it.

---

## Timeline Estimate

- Fixed-Point Test: 5 minutes
- Vector Dot Test: 10 minutes  
- Param Memory Test: 10 minutes
- **Total Level 0-1: 25 minutes**

Then you have a **proven foundation** to build on!

---

Start with `tb_fixed_point.sv` and work your way up.

**Good luck! 🎯**
