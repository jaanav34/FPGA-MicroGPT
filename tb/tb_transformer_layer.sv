// ===========================================================================
// Transformer Layer Testbench
// ===========================================================================
// ALL declarations at the TOP of every module / task / function.
// All test values pre-verified within Q8.8 range [-128.0, +127.996].
//
// Q8.8 budget per test (worst case):
//   input 0.5 → RMSNorm ≈ 1.0 → MHA (identity) ≈ 1.0
//   residual1 = 1.0 + 0.5 = 1.5  ✓
//   RMSNorm2 ≈ 1.0 → MLP (diag 0.03125):
//     fc1: 16×0.03125×1.0 = 0.5, relu = 0.5
//     fc2: 64×0.03125×0.5 = 1.0
//   residual2 = 1.0 + 1.5 = 2.5  ✓  (well within range)
// ===========================================================================

`timescale 1ns/1ps

module tb_transformer_layer;
    import microgpt_pkg::*;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int N_EMBD     = 16;
    localparam int N_HEAD     = 4;
    localparam int BLOCK_SIZE = 16;
    localparam int CLK_PERIOD = 10;
    localparam int W_ATTN     = N_EMBD * N_EMBD;         // 256
    localparam int W_FC1      = (4*N_EMBD) * N_EMBD;     // 1024
    localparam int W_FC2      = N_EMBD * (4*N_EMBD);     // 1024

    // -----------------------------------------------------------------------
    // DUT ports
    // -----------------------------------------------------------------------
    logic        clk;
    logic        rst_n;
    logic        start;
    logic        clear_cache;
    logic [4:0]  pos;

    fixed_t x_in     [N_EMBD-1:0];
    fixed_t attn_wq  [W_ATTN-1:0];
    fixed_t attn_wk  [W_ATTN-1:0];
    fixed_t attn_wv  [W_ATTN-1:0];
    fixed_t attn_wo  [W_ATTN-1:0];
    fixed_t mlp_fc1  [W_FC1-1:0];
    fixed_t mlp_fc2  [W_FC2-1:0];
    fixed_t x_out    [N_EMBD-1:0];
    logic   valid;

    // -----------------------------------------------------------------------
    // Book-keeping (module-level)
    // -----------------------------------------------------------------------
    int pass_count;
    int fail_count;
    int test_num;

    // -----------------------------------------------------------------------
    // DUT
    // -----------------------------------------------------------------------
    transformer_layer #(
        .N_EMBD    (N_EMBD),
        .N_HEAD    (N_HEAD),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .clear_cache(clear_cache),
        .pos        (pos),
        .x_in       (x_in),
        .attn_wq    (attn_wq),
        .attn_wk    (attn_wk),
        .attn_wv    (attn_wv),
        .attn_wo    (attn_wo),
        .mlp_fc1    (mlp_fc1),
        .mlp_fc2    (mlp_fc2),
        .x_out      (x_out),
        .valid      (valid)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // Task: reset_dut
    // -----------------------------------------------------------------------
    task reset_dut();
        // --- declarations ---
        int i;
        // --- body ---
        rst_n       = 0;
        start       = 0;
        clear_cache = 0;
        pos         = 0;
        for (i = 0; i < N_EMBD;  i++) x_in[i]    = '0;
        for (i = 0; i < W_ATTN;  i++) begin
            attn_wq[i] = '0;
            attn_wk[i] = '0;
            attn_wv[i] = '0;
            attn_wo[i] = '0;
        end
        for (i = 0; i < W_FC1;  i++) mlp_fc1[i] = '0;
        for (i = 0; i < W_FC2;  i++) mlp_fc2[i] = '0;
        repeat(6) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Task: wait_for_valid
    // -----------------------------------------------------------------------
    task wait_for_valid();
        // --- declarations ---
        int timeout_cnt;
        // --- body ---
        timeout_cnt = 0;
        while (!valid) begin
            @(posedge clk);
            timeout_cnt = timeout_cnt + 1;
            if (timeout_cnt > 10000) begin
                $display("  ERROR: Timeout in test %0d", test_num);
                fail_count = fail_count + 1;
                disable wait_for_valid;
            end
        end
        @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Task: do_clear_cache
    // -----------------------------------------------------------------------
    task do_clear_cache();
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Task: do_start
    // -----------------------------------------------------------------------
    task do_start();
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
    endtask

    // -----------------------------------------------------------------------
    // Task: set_uniform_input
    // -----------------------------------------------------------------------
    task set_uniform_input(input real val);
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < N_EMBD; i++)
            x_in[i] = float_to_fixed(val);
    endtask

    // -----------------------------------------------------------------------
    // Task: set_identity_attn_weights
    //   wq = wk = wv = wo = I (identity)
    //   Max projection value = max(x_in) = 1.0  ✓
    // -----------------------------------------------------------------------
    task set_identity_attn_weights();
        // --- declarations ---
        int r;
        int c;
        // --- body ---
        for (r = 0; r < N_EMBD; r++) begin
            for (c = 0; c < N_EMBD; c++) begin
                attn_wq[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
                attn_wk[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
                attn_wv[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
                attn_wo[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: set_zero_attn_weights
    // -----------------------------------------------------------------------
    task set_zero_attn_weights();
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < W_ATTN; i++) begin
            attn_wq[i] = '0;
            attn_wk[i] = '0;
            attn_wv[i] = '0;
            attn_wo[i] = '0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: set_safe_mlp_weights
    //   Diagonal fc1 and fc2 weights = 0.25
    //
    //   Q8.8 budget (diagonal path only):
    //     fc1 out[r] = 0.25 * norm2_out[r] ≈ 0.25 * 1.0 = 0.25  ✓
    //     after ReLU = 0.25
    //     fc2 out[r] = 0.25 * hidden[r] = 0.25 * 0.25 = 0.0625
    //     In Q8.8: 0.0625 * 256 = 16  → survives fixed-point truncation ✓
    //
    //   NOTE: 0.03125 was too small — 0.03125²=0.000977 → Q8.8 truncates to 0.
    // -----------------------------------------------------------------------
    task set_safe_mlp_weights();
        // --- declarations ---
        int r;
        int c;
        int hd;
        // --- body ---
        hd = 4 * N_EMBD;   // HIDDEN_DIM = 64
        for (r = 0; r < hd; r++) begin
            for (c = 0; c < N_EMBD; c++) begin
                mlp_fc1[r*N_EMBD + c] = (r == c && r < N_EMBD)
                                         ? float_to_fixed(0.25) : '0;
            end
        end
        for (r = 0; r < N_EMBD; r++) begin
            for (c = 0; c < hd; c++) begin
                mlp_fc2[r*hd + c] = (r == c && c < N_EMBD)
                                     ? float_to_fixed(0.25) : '0;
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Task: set_zero_mlp_weights
    // -----------------------------------------------------------------------
    task set_zero_mlp_weights();
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < W_FC1; i++) mlp_fc1[i] = '0;
        for (i = 0; i < W_FC2; i++) mlp_fc2[i] = '0;
    endtask

    // -----------------------------------------------------------------------
    // Function: avg_output  — returns mean of all x_out elements
    // -----------------------------------------------------------------------
    function automatic real avg_output();
        // --- declarations ---
        int  i;
        real s;
        // --- body ---
        s = 0.0;
        for (i = 0; i < N_EMBD; i++)
            s = s + fixed_to_float(x_out[i]);
        return s / N_EMBD;
    endfunction

    // -----------------------------------------------------------------------
    // Function: check_all_close
    // -----------------------------------------------------------------------
    function automatic logic check_all_close(input real expected, input real tol);
        // --- declarations ---
        int   i;
        real  val;
        real  err;
        logic ok;
        // --- body ---
        ok = 1;
        for (i = 0; i < N_EMBD; i++) begin
            val = fixed_to_float(x_out[i]);
            err = val - expected;
            if (err < 0.0) err = -err;
            if (err > tol) begin
                $display("    MISMATCH x_out[%0d]=%.4f expected %.4f (err %.4f > tol %.4f)",
                         i, val, expected, err, tol);
                ok = 0;
            end
        end
        return ok;
    endfunction

    // -----------------------------------------------------------------------
    // Function: check_output_range
    // -----------------------------------------------------------------------
    function automatic logic check_output_range(input real lo, input real hi);
        // --- declarations ---
        int   i;
        real  val;
        logic ok;
        // --- body ---
        ok = 1;
        for (i = 0; i < N_EMBD; i++) begin
            val = fixed_to_float(x_out[i]);
            if (val < lo || val > hi) begin
                $display("    RANGE FAIL x_out[%0d]=%.4f not in [%.4f, %.4f]",
                         i, val, lo, hi);
                ok = 0;
            end
        end
        return ok;
    endfunction

    // -----------------------------------------------------------------------
    // Task: display_output
    // -----------------------------------------------------------------------
    task display_output(input string label);
        // --- declarations ---
        int i;
        // --- body ---
        $display("  %s", label);
        for (i = 0; i < N_EMBD; i++)
            $display("    x_out[%02d] = %6d  (%7.4f)", i, x_out[i], fixed_to_float(x_out[i]));
    endtask

    // =========================================================================
    // TEST 1 — Zero Input → Zero Output
    // =========================================================================
    // x=0 → rmsnorm(0)=0 → attn(0)=0 → res1=0+0=0 → norm(0)=0 → mlp(0)=0
    // → res2 = 0+0 = 0.  Output must be exactly zero.
    // =========================================================================
    task test1_zero_input();
        // --- declarations ---
        logic ok;
        // --- body ---
        test_num = 1;
        $display("\n=== Test 1: Zero Input → Zero Output ===");

        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(0.0);
        pos = 0;

        do_start();
        wait_for_valid();
        display_output("Outputs:");

        ok = check_all_close(0.0, 0.05);
        if (ok) begin
            $display("  ✓ Test 1 PASSED");
            pass_count++;
        end else begin
            $display("  ✗ Test 1 FAILED");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 2 — Zero Weights → Output Equals Input  (only residuals survive)
    // =========================================================================
    // With zero attn AND zero mlp weights:
    //   attn_out = 0  →  res1 = 0 + x_in = x_in
    //   mlp_out  = 0  →  res2 = 0 + x_in = x_in
    // Output should equal x_in through both residual paths.
    // =========================================================================
    task test2_zero_weights_residual_passthrough();
        // --- declarations ---
        logic ok;
        real  input_val;
        // --- body ---
        test_num   = 2;
        input_val  = 0.5;
        $display("\n=== Test 2: Zero Weights → Residuals Pass Through Input ===");

        do_clear_cache();
        set_zero_attn_weights();
        set_zero_mlp_weights();
        set_uniform_input(input_val);
        pos = 0;

        do_start();
        wait_for_valid();
        display_output("Outputs:");

        // Both residuals add x_in to zero → output = x_in = 0.5
        ok = check_all_close(input_val, 0.1);
        if (ok) begin
            $display("  ✓ Test 2 PASSED  (residuals preserved input %.3f)", input_val);
            pass_count++;
        end else begin
            $display("  ✗ Test 2 FAILED  (expected %.3f at every element)", input_val);
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 3 — Identity Weights, Single Position
    // =========================================================================
    // Verify the full data path with identity weights produces a valid output.
    //
    // Budget:
    //   x=0.5 → rmsnorm ≈ 1.0 → attn(I) ≈ 1.0 → res1 = 1.0+0.5 = 1.5
    //   rmsnorm(1.5) ≈ 1.0 → mlp(diag 0.03125):
    //     fc1 = 16×0.03125×1.0 = 0.5, relu=0.5
    //     fc2 = 64×0.03125×0.5 = 1.0
    //   res2 = 1.0 + 1.5 = 2.5  ← safe ✓
    //
    //   We just verify the output is in the safe range [0.5, 4.0].
    // =========================================================================
    task test3_identity_single_pos();
        // --- declarations ---
        logic ok;
        real  avg;
        // --- body ---
        test_num = 3;
        $display("\n=== Test 3: Identity Weights, Single Position ===");

        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(0.5);
        pos = 0;

        do_start();
        wait_for_valid();
        display_output("Outputs:");

        avg = avg_output();
        $display("  avg output = %.4f  (expected in [0.5, 4.0])", avg);

        ok = check_output_range(0.5, 4.0);
        if (ok) begin
            $display("  ✓ Test 3 PASSED");
            pass_count++;
        end else begin
            $display("  ✗ Test 3 FAILED");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 4 — Residuals Are Active (output > input)
    // =========================================================================
    // With identity weights, the attention output ≈ rmsnorm(x_in) ≈ 1.0.
    // Residual1 adds x_in=0.5, so output after attn block ≈ 1.5 > 0.5.
    // Therefore the transformer output MUST be > the raw input.
    // =========================================================================
    task test4_residuals_increase_output();
        // --- declarations ---
        real  avg;
        real  input_val;
        logic ok;
        // --- body ---
        test_num  = 4;
        input_val = 0.5;
        $display("\n=== Test 4: Residual Connections Increase Output ===");

        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(input_val);
        pos = 0;

        do_start();
        wait_for_valid();

        avg = avg_output();
        $display("  input = %.4f,  avg output = %.4f", input_val, avg);

        ok = (avg > input_val);
        if (ok) begin
            $display("  ✓ Test 4 PASSED  (residuals lifted output above input)");
            pass_count++;
        end else begin
            $display("  ✗ Test 4 FAILED  (output %.4f not greater than input %.4f)",
                     avg, input_val);
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 5 — Cache Clear Reproduces Same Output
    // =========================================================================
    // Run pos=0 twice: once fresh, once after polluting the cache with extra
    // positions and then clearing.  Both runs must give the same output.
    // =========================================================================
    task test5_cache_clear_reproducibility();
        // --- declarations ---
        real  avg_first;
        real  avg_second;
        real  diff;
        int   p;
        logic ok;
        // --- body ---
        test_num = 5;
        $display("\n=== Test 5: Cache Clear → Reproducible Output ===");

        // --- first run ---
        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();

        avg_first = avg_output();
        $display("  First run  (pos=0, fresh cache) avg = %.4f", avg_first);

        // --- pollute cache with extra positions ---
        for (p = 1; p <= 3; p++) begin
            set_uniform_input(0.125 * real'(p));
            pos = p;
            do_start();
            wait_for_valid();
        end

        // --- clear and re-run pos=0 with same input ---
        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();

        avg_second = avg_output();
        $display("  Second run (pos=0, after clear) avg = %.4f", avg_second);

        diff = avg_first - avg_second;
        if (diff < 0.0) diff = -diff;

        ok = (diff < 0.05);
        if (ok) begin
            $display("  ✓ Test 5 PASSED  (diff = %.4f)", diff);
            pass_count++;
        end else begin
            $display("  ✗ Test 5 FAILED  (diff %.4f too large)", diff);
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 6 — Two Sequential Positions Have Different Outputs
    // =========================================================================
    // Feed pos=0 with x=0.25, then pos=1 with x=0.5.
    // The second position attends over two entries in the KV cache,
    // so its output MUST differ from the first.
    // =========================================================================
    task test6_sequential_outputs_differ();
        // --- declarations ---
        real  avg0;
        real  avg1;
        logic ok;
        // --- body ---
        test_num = 6;
        $display("\n=== Test 6: Two Positions → Outputs Differ ===");

        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();

        // position 0
        set_uniform_input(0.25);
        pos = 0;
        do_start();
        wait_for_valid();
        avg0 = avg_output();
        $display("  pos=0  input=0.25  avg_output=%.4f", avg0);

        // position 1
        set_uniform_input(0.5);
        pos = 1;
        do_start();
        wait_for_valid();
        avg1 = avg_output();
        $display("  pos=1  input=0.50  avg_output=%.4f", avg1);

        ok = (avg1 != avg0);
        if (ok) begin
            $display("  ✓ Test 6 PASSED  (pos-1 output differs from pos-0)");
            pass_count++;
        end else begin
            $display("  ✗ Test 6 FAILED  (outputs identical — KV cache not updating)");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 7 — Four Sequential Positions, No Overflow
    // =========================================================================
    // Feed 4 positions with inputs 0.125, 0.25, 0.375, 0.5.
    // Verify every output stays within [-4.0, 8.0] (well within Q8.8).
    // =========================================================================
    task test7_four_positions_no_overflow();
        // --- declarations ---
        int   p;
        real  avg;
        logic ok;
        // --- body ---
        test_num = 7;
        $display("\n=== Test 7: Four Sequential Positions, No Overflow ===");

        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        ok = 1;

        for (p = 0; p < 4; p++) begin
            set_uniform_input(0.125 * real'(p + 1));
            pos = p;
            do_start();
            wait_for_valid();

            avg = avg_output();
            $display("  pos=%0d  input=%.3f  avg_output=%.4f",
                     p, 0.125 * real'(p+1), avg);

            if (avg < -4.0 || avg > 8.0) begin
                $display("    OVERFLOW: %.4f out of safe range", avg);
                ok = 0;
            end
        end

        if (ok) begin
            $display("  ✓ Test 7 PASSED  (no overflow across 4 positions)");
            pass_count++;
        end else begin
            $display("  ✗ Test 7 FAILED  (overflow detected)");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 8 — MLP Path Contributes to Output
    // =========================================================================
    // Compare: zero MLP weights vs safe MLP weights (same attn weights).
    // When MLP weights are non-zero, the output should differ.
    // =========================================================================
    task test8_mlp_contributes();
        // --- declarations ---
        real  avg_no_mlp;
        real  avg_with_mlp;
        real  diff;
        logic ok;
        // --- body ---
        test_num = 8;
        $display("\n=== Test 8: MLP Contribution to Output ===");

        // --- run with zero MLP ---
        do_clear_cache();
        set_identity_attn_weights();
        set_zero_mlp_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();
        avg_no_mlp = avg_output();
        $display("  MLP=zero   avg_output = %.4f", avg_no_mlp);

        // --- run with safe MLP ---
        do_clear_cache();
        set_identity_attn_weights();
        set_safe_mlp_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();
        avg_with_mlp = avg_output();
        $display("  MLP=diag   avg_output = %.4f", avg_with_mlp);

        diff = avg_with_mlp - avg_no_mlp;
        if (diff < 0.0) diff = -diff;

        ok = (diff > 0.01);
        if (ok) begin
            $display("  ✓ Test 8 PASSED  (MLP changed output by %.4f)", diff);
            pass_count++;
        end else begin
            $display("  ✗ Test 8 FAILED  (MLP made no difference, diff=%.4f)", diff);
            fail_count++;
        end
    endtask

    // =========================================================================
    // Main
    // =========================================================================
    initial begin
        $display("============================================================");
        $display("  Transformer Layer Testbench  (N_EMBD=%0d, N_HEAD=%0d)",
                 N_EMBD, N_HEAD);
        $display("============================================================");

        pass_count = 0;
        fail_count = 0;
        test_num   = 0;

        reset_dut();

        test1_zero_input();
        test2_zero_weights_residual_passthrough();
        test3_identity_single_pos();
        test4_residuals_increase_output();
        test5_cache_clear_reproducibility();
        test6_sequential_outputs_differ();
        test7_four_positions_no_overflow();
        test8_mlp_contributes();

        $display("\n============================================================");
        $display("  Results: %0d passed, %0d failed  (total %0d)",
                 pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ✓ ALL TESTS PASSED");
        else
            $display("  ✗ SOME TESTS FAILED");
        $display("============================================================\n");
        $finish;
    end

    // Watchdog
    initial begin
        #5000000;
        $display("ERROR: global watchdog fired — simulation stuck");
        $finish;
    end

endmodule : tb_transformer_layer