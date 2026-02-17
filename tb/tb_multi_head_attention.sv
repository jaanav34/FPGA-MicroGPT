// ===========================================================================
// Multi-Head Attention Testbench
// ===========================================================================
// ALL variable declarations are at the TOP of every module/task/function.
// All test values pre-verified to stay within Q8.8 range [-128.0, +127.996].
//
// Q8.8 Safety Rules used throughout:
//   - Matrix weights: max 0.125 (with 16 inputs, max output = 16*0.125*1.0 = 2.0)
//   - Input values:   max 1.0
//   - Dot products:   max HEAD_DIM * proj_max^2 = 4 * 2.0 * 2.0 = 16.0 (safe)
//   - No value ever exceeds +/- 10.0 in any intermediate computation
// ===========================================================================

`timescale 1ns/1ps

module tb_multi_head_attention;
    import microgpt_pkg::*;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int N_EMBD     = 16;
    localparam int N_HEAD     = 4;
    localparam int HEAD_DIM   = N_EMBD / N_HEAD;   // 4
    localparam int BLOCK_SIZE = 16;
    localparam int CLK_PERIOD = 10;
    localparam int W_SZ       = N_EMBD * N_EMBD;   // 256 weights per matrix

    // -----------------------------------------------------------------------
    // DUT Ports
    // -----------------------------------------------------------------------
    logic   clk;
    logic   rst_n;
    logic   start;
    logic   clear_cache;
    logic [4:0] pos;

    fixed_t x_in  [N_EMBD-1:0];
    fixed_t wq    [W_SZ-1:0];
    fixed_t wk    [W_SZ-1:0];
    fixed_t wv    [W_SZ-1:0];
    fixed_t wo    [W_SZ-1:0];
    fixed_t x_out [N_EMBD-1:0];
    logic   valid;

    // -----------------------------------------------------------------------
    // Test book-keeping (declared once, used everywhere)
    // -----------------------------------------------------------------------
    int pass_count;
    int fail_count;
    int test_num;

    // -----------------------------------------------------------------------
    // DUT Instantiation
    // -----------------------------------------------------------------------
    multi_head_attention #(
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
        .wq         (wq),
        .wk         (wk),
        .wv         (wv),
        .wo         (wo),
        .x_out      (x_out),
        .valid      (valid)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // Helper: reset_dut
    // -----------------------------------------------------------------------
    task reset_dut();
        // --- declarations ---
        int i;
        // --- body ---
        rst_n       = 0;
        start       = 0;
        clear_cache = 0;
        pos         = 0;
        for (i = 0; i < N_EMBD; i++) x_in[i] = '0;
        for (i = 0; i < W_SZ;   i++) begin
            wq[i] = '0;
            wk[i] = '0;
            wv[i] = '0;
            wo[i] = '0;
        end
        repeat(6) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Helper: wait_for_valid  (with timeout guard)
    // -----------------------------------------------------------------------
    task wait_for_valid();
        // --- declarations ---
        int timeout_cnt;
        // --- body ---
        timeout_cnt = 0;
        while (!valid) begin
            @(posedge clk);
            timeout_cnt = timeout_cnt + 1;
            if (timeout_cnt > 5000) begin
                $display("  ERROR: Timeout in test %0d", test_num);
                fail_count = fail_count + 1;
                disable wait_for_valid;
            end
        end
        @(posedge clk);   // consume the valid cycle
    endtask

    // -----------------------------------------------------------------------
    // Helper: do_clear_cache
    // -----------------------------------------------------------------------
    task do_clear_cache();
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Helper: do_start
    // -----------------------------------------------------------------------
    task do_start();
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
    endtask

    // -----------------------------------------------------------------------
    // Helper: set_identity_weights
    //   Sets wq = wk = wv = wo = identity matrix (1.0 on diagonal)
    //   After projection: Q = K = V = x_in  (pass-through)
    //   Max projection value = max(x_in) = 1.0  (safe)
    // -----------------------------------------------------------------------
    task set_identity_weights();
        // --- declarations ---
        int r, c;
        // --- body ---
        for (r = 0; r < N_EMBD; r++) begin
            for (c = 0; c < N_EMBD; c++) begin
                if (r == c) begin
                    wq[r*N_EMBD + c] = float_to_fixed(1.0);
                    wk[r*N_EMBD + c] = float_to_fixed(1.0);
                    wv[r*N_EMBD + c] = float_to_fixed(1.0);
                    wo[r*N_EMBD + c] = float_to_fixed(1.0);
                end else begin
                    wq[r*N_EMBD + c] = '0;
                    wk[r*N_EMBD + c] = '0;
                    wv[r*N_EMBD + c] = '0;
                    wo[r*N_EMBD + c] = '0;
                end
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Helper: set_uniform_weights
    //   All weights = w_val
    //   With N_EMBD=16 and w_val=0.0625:
    //     max projection = 16 * 0.0625 * 1.0 = 1.0  (safe)
    // -----------------------------------------------------------------------
    task set_uniform_weights(input real w_val);
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < W_SZ; i++) begin
            wq[i] = float_to_fixed(w_val);
            wk[i] = float_to_fixed(w_val);
            wv[i] = float_to_fixed(w_val);
            wo[i] = float_to_fixed(w_val);
        end
    endtask

    // -----------------------------------------------------------------------
    // Helper: set_zero_weights
    // -----------------------------------------------------------------------
    task set_zero_weights();
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < W_SZ; i++) begin
            wq[i] = '0;
            wk[i] = '0;
            wv[i] = '0;
            wo[i] = '0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Helper: set_uniform_input
    // -----------------------------------------------------------------------
    task set_uniform_input(input real val);
        // --- declarations ---
        int i;
        // --- body ---
        for (i = 0; i < N_EMBD; i++)
            x_in[i] = float_to_fixed(val);
    endtask

    // -----------------------------------------------------------------------
    // Helper: display_output
    // -----------------------------------------------------------------------
    task display_output(input string label);
        // --- declarations ---
        int i;
        // --- body ---
        $display("  %s", label);
        for (i = 0; i < N_EMBD; i++)
            $display("    x_out[%02d] = %6d  (%7.4f)", i, x_out[i], fixed_to_float(x_out[i]));
    endtask

    // -----------------------------------------------------------------------
    // Helper: check_all_close
    //   Returns 1 if every output element is within tol of expected
    // -----------------------------------------------------------------------
    function automatic logic check_all_close(
        input real expected,
        input real tol
    );
        // --- declarations ---
        int    i;
        real   val;
        real   err;
        logic  ok;
        // --- body ---
        ok = 1;
        for (i = 0; i < N_EMBD; i++) begin
            val = fixed_to_float(x_out[i]);
            err = val - expected;
            if (err < 0.0) err = -err;
            if (err > tol) begin
                $display("    MISMATCH x_out[%0d] = %.4f  expected %.4f  (err %.4f > tol %.4f)",
                         i, val, expected, err, tol);
                ok = 0;
            end
        end
        return ok;
    endfunction

    // -----------------------------------------------------------------------
    // Helper: check_output_range
    //   Returns 1 if every output element is within [lo, hi]
    // -----------------------------------------------------------------------
    function automatic logic check_output_range(
        input real lo,
        input real hi
    );
        // --- declarations ---
        int  i;
        real val;
        logic ok;
        // --- body ---
        ok = 1;
        for (i = 0; i < N_EMBD; i++) begin
            val = fixed_to_float(x_out[i]);
            if (val < lo || val > hi) begin
                $display("    RANGE FAIL x_out[%0d] = %.4f  not in [%.4f, %.4f]",
                         i, val, lo, hi);
                ok = 0;
            end
        end
        return ok;
    endfunction

    // =========================================================================
    // TEST 1 — Zero Input → Zero Output
    // =========================================================================
    // Rationale: With any weights, zero input must produce zero output at every
    //            stage (projection, attention, output projection).
    // Q8.8 check: trivial — all zeros.
    // =========================================================================
    task test1_zero_input();
        // --- declarations ---
        logic ok;
        // --- body ---
        test_num = 1;
        $display("\n=== Test 1: Zero Input → Zero Output ===");

        do_clear_cache();
        set_uniform_weights(0.0625);   // any non-zero weights
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
    // TEST 2 — Zero Weights → Zero Output (non-zero input)
    // =========================================================================
    // Rationale: Zero weight matrices must silence any input.
    // Q8.8 check: all zeros — trivial.
    // =========================================================================
    task test2_zero_weights();
        // --- declarations ---
        logic ok;
        // --- body ---
        test_num = 2;
        $display("\n=== Test 2: Zero Weights → Zero Output ===");

        do_clear_cache();
        set_zero_weights();
        set_uniform_input(0.5);     // non-zero input

        pos = 0;
        do_start();
        wait_for_valid();
        display_output("Outputs:");

        ok = check_all_close(0.0, 0.05);
        if (ok) begin
            $display("  ✓ Test 2 PASSED");
            pass_count++;
        end else begin
            $display("  ✗ Test 2 FAILED");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 3 — Identity Weights, Single Position
    // =========================================================================
    // Rationale: With identity wq/wk/wv/wo the module is a pass-through.
    //   Input:  x = [0.5, 0.5, ..., 0.5]
    //   Q = K = V = x  (identity projection)
    //   Single position → softmax weight = 1.0
    //   Head output = V = x (per head)
    //   Concat = x
    //   Final = wo * x = identity * x = x
    //
    //   Expected output ≈ 0.5 at every element.
    //
    // Q8.8 check:
    //   Projection: 1.0 * 0.5 = 0.5  ✓
    //   Dot(Q,K) per head: 4*(0.5*0.5) = 1.0, scaled *0.5 = 0.5  ✓
    //   Output: 1.0 * 0.5 = 0.5  ✓
    // =========================================================================
    task test3_identity_single_pos();
        // --- declarations ---
        logic ok;
        // --- body ---
        test_num = 3;
        $display("\n=== Test 3: Identity Weights, Single Position ===");

        do_clear_cache();
        set_identity_weights();
        set_uniform_input(0.5);

        pos = 0;
        do_start();
        wait_for_valid();
        display_output("Outputs:");

        // After a full identity round-trip the value can drift slightly
        // from fixed-point rounding; allow 0.15 tolerance.
        ok = check_all_close(0.5, 0.15);
        if (ok) begin
            $display("  ✓ Test 3 PASSED  (expected ≈ 0.5 per element)");
            pass_count++;
        end else begin
            $display("  ✗ Test 3 FAILED");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 4 — Output Scales With Input (Single Position)
    // =========================================================================
    // Rationale: Double the input → double the output (linearity check).
    //   Run twice: x=0.25 then x=0.5.  Output[1] should be ~2× Output[0].
    //
    // Q8.8 check:
    //   With identity weights and input 0.25 → output ≈ 0.25  (max 0.5, safe)
    //   With identity weights and input 0.50 → output ≈ 0.50  (max 1.0, safe)
    //   Note: linearity is approximate due to softmax non-linearity.
    //         We test that output increases monotonically, not exact 2×.
    // =========================================================================
    task test4_input_scaling();
        // --- declarations ---
        real  out_low;
        real  out_high;
        int   i;
        logic ok;
        // --- body ---
        test_num = 4;
        $display("\n=== Test 4: Output Scales With Input ===");

        // --- run with low input ---
        do_clear_cache();
        set_identity_weights();
        set_uniform_input(0.25);
        pos = 0;
        do_start();
        wait_for_valid();

        out_low = 0.0;
        for (i = 0; i < N_EMBD; i++)
            out_low = out_low + fixed_to_float(x_out[i]);
        out_low = out_low / N_EMBD;
        $display("  Input = 0.25 → avg output = %.4f", out_low);

        // --- run with high input ---
        do_clear_cache();
        set_identity_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();

        out_high = 0.0;
        for (i = 0; i < N_EMBD; i++)
            out_high = out_high + fixed_to_float(x_out[i]);
        out_high = out_high / N_EMBD;
        $display("  Input = 0.50 → avg output = %.4f", out_high);

        ok = (out_high > out_low);
        if (ok) begin
            $display("  ✓ Test 4 PASSED  (larger input → larger output)");
            pass_count++;
        end else begin
            $display("  ✗ Test 4 FAILED  (output did not increase)");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 5 — Two Positions: Output Changes Between Positions
    // =========================================================================
    // Rationale: With the KV cache active, the second position attends over
    //            both positions.  Its output must differ from the first.
    //
    //   Position 0: x = [0.25, 0.25, ..., 0.25]
    //   Position 1: x = [0.50, 0.50, ..., 0.50]  (different input)
    //
    // Q8.8 check:
    //   Identity weights, inputs 0.25 and 0.50 → all intermediate
    //   values stay in [0, 1.0]  ✓
    // =========================================================================
    task test5_two_positions();
        // --- declarations ---
        real  avg0;
        real  avg1;
        int   i;
        logic ok;
        // --- body ---
        test_num = 5;
        $display("\n=== Test 5: Two Positions, Output Changes ===");

        do_clear_cache();
        set_identity_weights();

        // --- position 0 ---
        set_uniform_input(0.25);
        pos = 0;
        do_start();
        wait_for_valid();

        avg0 = 0.0;
        for (i = 0; i < N_EMBD; i++)
            avg0 = avg0 + fixed_to_float(x_out[i]);
        avg0 = avg0 / N_EMBD;
        $display("  Position 0 avg output = %.4f", avg0);

        // --- position 1 ---
        set_uniform_input(0.5);
        pos = 1;
        do_start();
        wait_for_valid();

        avg1 = 0.0;
        for (i = 0; i < N_EMBD; i++)
            avg1 = avg1 + fixed_to_float(x_out[i]);
        avg1 = avg1 / N_EMBD;
        $display("  Position 1 avg output = %.4f", avg1);

        ok = (avg1 != avg0);
        if (ok) begin
            $display("  ✓ Test 5 PASSED  (outputs differ between positions)");
            pass_count++;
        end else begin
            $display("  ✗ Test 5 FAILED  (outputs identical — KV cache not updating)");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 6 — Cache Clear Resets Attention
    // =========================================================================
    // Rationale: After processing 2 positions, clearing the cache and
    //            re-running position 0 with the same input must reproduce
    //            the same output as the very first run.
    //
    // Q8.8 check: identical to Test 3 — all values ≤ 1.0  ✓
    // =========================================================================
    task test6_cache_clear();
        // --- declarations ---
        real  first_avg;
        real  second_avg;
        real  diff;
        int   i;
        logic ok;
        // --- body ---
        test_num = 6;
        $display("\n=== Test 6: Cache Clear Resets Attention ===");

        // --- first run: just position 0 ---
        do_clear_cache();
        set_identity_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();

        first_avg = 0.0;
        for (i = 0; i < N_EMBD; i++)
            first_avg = first_avg + fixed_to_float(x_out[i]);
        first_avg = first_avg / N_EMBD;
        $display("  First  run (pos=0 after fresh clear) avg = %.4f", first_avg);

        // --- fill cache with extra positions ---
        set_uniform_input(0.125);
        pos = 1;
        do_start();
        wait_for_valid();

        set_uniform_input(0.25);
        pos = 2;
        do_start();
        wait_for_valid();

        // --- clear and repeat position 0 ---
        do_clear_cache();
        set_identity_weights();
        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();

        second_avg = 0.0;
        for (i = 0; i < N_EMBD; i++)
            second_avg = second_avg + fixed_to_float(x_out[i]);
        second_avg = second_avg / N_EMBD;
        $display("  Second run (pos=0 after cache clear) avg = %.4f", second_avg);

        diff = first_avg - second_avg;
        if (diff < 0.0) diff = -diff;

        ok = (diff < 0.05);
        if (ok) begin
            $display("  ✓ Test 6 PASSED  (cache clear restored original output)");
            pass_count++;
        end else begin
            $display("  ✗ Test 6 FAILED  (cache clear did not work, diff = %.4f)", diff);
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 7 — Output Projection Scales Result
    // =========================================================================
    // Rationale: Keep wq/wk/wv as identity.  Change wo from identity to a
    //            uniform-0.0625 matrix.  The output projection now sums all
    //            16 elements of the concatenated head output and multiplies
    //            by 0.0625:
    //              out[i] = sum_j(concat[j]) * 0.0625
    //                     = 16 * 0.5 * 0.0625  = 0.5   (safe Q8.8)
    //
    // Q8.8 check:  16 * 0.5 * 0.0625 = 0.5  ✓
    // =========================================================================
    task test7_output_projection();
        // --- declarations ---
        int   r;
        int   c;
        int   i;
        logic ok;
        // --- body ---
        test_num = 7;
        $display("\n=== Test 7: Output Projection Scales Result ===");

        do_clear_cache();

        // wq = wk = wv = identity
        for (r = 0; r < N_EMBD; r++) begin
            for (c = 0; c < N_EMBD; c++) begin
                wq[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
                wk[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
                wv[r*N_EMBD + c] = (r == c) ? float_to_fixed(1.0) : '0;
            end
        end

        // wo = uniform 0.0625  →  out[i] = 16 * concat_avg * 0.0625
        for (i = 0; i < W_SZ; i++)
            wo[i] = float_to_fixed(0.0625);

        set_uniform_input(0.5);
        pos = 0;
        do_start();
        wait_for_valid();
        display_output("Outputs:");

        // concat ≈ [0.5, ..., 0.5], so out[i] ≈ 16*0.5*0.0625 = 0.5
        ok = check_output_range(0.2, 0.8);
        if (ok) begin
            $display("  ✓ Test 7 PASSED  (output projection working)");
            pass_count++;
        end else begin
            $display("  ✗ Test 7 FAILED");
            fail_count++;
        end
    endtask

    // =========================================================================
    // TEST 8 — Sequential Run: 4 Positions, Outputs Monotonically Change
    // =========================================================================
    // Rationale: Feed 4 positions with linearly increasing inputs.
    //   As more context is added, the attended output should shift.
    //   We just verify no output overflows or goes NaN (all in valid range).
    //
    //   Inputs: 0.125, 0.25, 0.375, 0.5  (all safe)
    //   With identity weights, all intermediate values ≤ 0.5  ✓
    // =========================================================================
    task test8_sequential_4pos();
        // --- declarations ---
        real  vals [3:0];    // one real per position
        int   p;
        int   i;
        logic ok;
        real  avg;
        // --- body ---
        test_num = 8;
        $display("\n=== Test 8: Sequential 4 Positions, All Outputs Valid ===");

        do_clear_cache();
        set_identity_weights();
        ok = 1;

        for (p = 0; p < 4; p++) begin
            set_uniform_input(0.125 * real'(p + 1));  // 0.125, 0.25, 0.375, 0.5
            pos = p;
            do_start();
            wait_for_valid();

            avg = 0.0;
            for (i = 0; i < N_EMBD; i++)
                avg = avg + fixed_to_float(x_out[i]);
            avg = avg / N_EMBD;
            vals[p] = avg;

            $display("  Position %0d  input=%.3f  avg_output=%.4f",
                     p, 0.125 * real'(p+1), avg);

            if (avg < -2.0 || avg > 2.0) begin
                $display("    ERROR: output %.4f out of safe range [-2, 2]", avg);
                ok = 0;
            end
        end

        if (ok) begin
            $display("  ✓ Test 8 PASSED  (all 4 positions produced valid outputs)");
            pass_count++;
        end else begin
            $display("  ✗ Test 8 FAILED  (overflow/underflow detected)");
            fail_count++;
        end
    endtask

    // =========================================================================
    // Main
    // =========================================================================
    initial begin
        $display("============================================================");
        $display("  Multi-Head Attention Testbench  (N_EMBD=%0d, N_HEAD=%0d)",
                 N_EMBD, N_HEAD);
        $display("============================================================");

        pass_count = 0;
        fail_count = 0;
        test_num   = 0;

        reset_dut();

        test1_zero_input();
        test2_zero_weights();
        test3_identity_single_pos();
        test4_input_scaling();
        test5_two_positions();
        test6_cache_clear();
        test7_output_projection();
        test8_sequential_4pos();

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
        #2000000;
        $display("ERROR: global watchdog fired — simulation stuck");
        $finish;
    end

endmodule : tb_multi_head_attention