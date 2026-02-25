# microGPT FPGA

A fixed-point transformer inference engine implemented in SystemVerilog, targeting Xilinx FPGAs. The model is a miniature GPT-style character-level language model trained on a names dataset, with weights exported to Q12.4 fixed-point format and loaded into on-chip BRAM at simulation startup.

The design is validated through a bottom-up testbench hierarchy — each RTL module has its own standalone testbench before being integrated into higher-level components.

---

## Architecture

```
microGPT Top
└── Transformer Layer
    ├── Multi-Head Attention
    │   └── Attention Head (× N_HEAD)
    │       └── Matrix-Vector Multiply
    │           └── Vector Dot Product
    ├── MLP Block
    │   └── Matrix-Vector Multiply
    ├── RMS Normalization
    └── Softmax + Top-K Sampler

Supporting modules:
  param_memory.sv   – BRAM-backed weight storage ($readmemh from .mem file)
  microgpt_pkg.sv   – Shared types, fixed-point functions, global parameters
```

Fixed-point format: **Q12.4** (16-bit signed, 4 fractional bits; range ±2048, precision 0.0625). Changed from Q8.8 to reduce saturation in residual accumulation paths.

---

## Directory Structure

```
rtl/          SystemVerilog source modules
tb/           Standalone testbenches (one per module)
sim/          TCL simulation setup script
docs/         This file and component status
export_weights.py   Converts PyTorch weights to Q12.4 .mem format
verify_complete.py  Python reference model for output comparison
name_reference.py   Baseline character-level inference in Python
```

---

## Simulation

### Vivado (manual)

1. Create a new project and add the source files for the module under test.
2. Set the testbench as the simulation top.
3. Run Behavioral Simulation.
4. Check the TCL Console for PASS/FAIL output.

Each testbench prints a summary in the format:
```
Test Summary:
  Passed: N
  Failed: 0
```

### TCL Script

From the Vivado TCL Console:
```tcl
cd /path/to/project
source sim/setup_tests.tcl
setup_test vector_dot
launch_simulation
```

Supported test names: `fixed_point`, `vector_dot`, `param_memory`, `rmsnorm`, `softmax`, `matmul`.

---

## Weight Export

Weights are trained in Python (standard PyTorch character-level GPT) and exported with `export_weights.py`:

```python
from export_weights import export_param_mem_q124
export_param_mem_q124(state_dict, uchars)
```

This writes `param_q124.mem` — 4192 16-bit hex values in `$readmemh` format. The address layout matches the iteration order in `param_memory.sv`.

To verify an existing `.mem` file:
```bash
python export_weights.py param_q124.mem
```

---

## Module Test Status

See `docs/PRODUCTION_STATUS.md` for current test coverage per component.

---

## Parameters (microgpt_pkg.sv)

| Parameter  | Value | Notes                          |
|------------|-------|--------------------------------|
| VOCAB_SIZE | 27    | 26 letters + BOS token         |
| N_EMBD     | 16    | Embedding dimension            |
| N_HEAD     | 4     | Attention heads                |
| N_LAYER    | 1     | Transformer layers             |
| BLOCK_SIZE | 16    | Context window                 |
| FRAC_BITS  | 4     | Q12.4 fractional bits          |
| TOP_K      | 5     | Sampling candidates            |
