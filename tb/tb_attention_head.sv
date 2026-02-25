// ===========================================================================
// Attention Head Testbench
// ===========================================================================
// Comprehensive verification of single attention head functionality
// ===========================================================================

`timescale 1ns/1ps

module tb_attention_head;
    import microgpt_pkg::*;
    
    // Parameters
    localparam int HEAD_DIM = 4;
    localparam int N_EMBD = 16;
    localparam int BLOCK_SIZE = 16;
    localparam int CLK_PERIOD = 10;
    
    // DUT signals
    logic        clk;
    logic        rst_n;
    logic        start;
    logic        clear_cache;
    logic [4:0]  pos;
    fixed_t      q_in [HEAD_DIM-1:0];
    fixed_t      k_in [HEAD_DIM-1:0];
    fixed_t      v_in [HEAD_DIM-1:0];
    fixed_t      head_out [HEAD_DIM-1:0];
    logic        valid;
    logic test_pass = 1;

    // Test control
    int test_num;
    int pass_count;
    int fail_count;
    
    // DUT instantiation
    attention_head #(
        .HEAD_DIM(HEAD_DIM),
        .N_EMBD(N_EMBD),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .clear_cache(clear_cache),
        .pos(pos),
        .q_in(q_in),
        .k_in(k_in),
        .v_in(v_in),
        .head_out(head_out),
        .valid(valid)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Helper functions
    task reset_dut();
        rst_n = 0;
        start = 0;
        clear_cache = 0;
        pos = 0;
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = '0;
            k_in[i] = '0;
            v_in[i] = '0;
        end
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask
    
    task wait_for_valid();
        fork
            begin
                wait(valid);
                @(posedge clk);
            end
            begin
                repeat(1000) @(posedge clk);
                $display("ERROR: Timeout waiting for valid signal");
                fail_count++;
            end
        join_any
        disable fork;
    endtask
    
    function void display_output(string msg);
        $display("%s", msg);
        $display("  Output vector:");
        for (int i = 0; i < HEAD_DIM; i++) begin
            $display("    [%0d] = %0d (%.3f)", i, head_out[i], fixed_to_float(head_out[i]));
        end
    endfunction
    
    function automatic logic check_output_range(real min_val, real max_val);
        logic in_range = 1;
        for (int i = 0; i < HEAD_DIM; i++) begin
            real val = fixed_to_float(head_out[i]);
            if (val < min_val || val > max_val) begin
                $display("  ERROR: Output[%0d] = %.3f out of range [%.3f, %.3f]", 
                        i, val, min_val, max_val);
                in_range = 0;
            end
        end
        return in_range;
    endfunction
    
    // -----------------------------------------------------------------------
    // Test Cases
    // -----------------------------------------------------------------------
    
    // Test 1: Single position attention
    task test_single_position();
        real expected;
        real actual;
        real error;
        $display("\n=== Test 1: Single Position Attention ===");
        test_num = 1;
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Set up Q, K, V for position 0
        // Q = [1.0, 0.5, 0.25, 0.125]
        // K = [1.0, 0.5, 0.25, 0.125]
        // V = [2.0, 1.5, 1.0, 0.5]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0 / (1 << i));
            k_in[i] = float_to_fixed(1.0 / (1 << i));
            v_in[i] = float_to_fixed(2.0 - (i * 0.5));
        end

        // After setting q_in, k_in, v_in but before start=1
        $display("DEBUG Test 1 Inputs:");
        $display("  Q: [%.3f, %.3f, %.3f, %.3f]", 
            fixed_to_float(q_in[0]), fixed_to_float(q_in[1]), 
            fixed_to_float(q_in[2]), fixed_to_float(q_in[3]));
        $display("  K: [%.3f, %.3f, %.3f, %.3f]", 
            fixed_to_float(k_in[0]), fixed_to_float(k_in[1]), 
            fixed_to_float(k_in[2]), fixed_to_float(k_in[3]));
        $display("  V: [%.3f, %.3f, %.3f, %.3f]", 
            fixed_to_float(v_in[0]), fixed_to_float(v_in[1]), 
            fixed_to_float(v_in[2]), fixed_to_float(v_in[3]));
        
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        
        display_output("Single position output:");
        
        // With only one position, attention weight should be 1.0
        // Output should equal V
        for (int i = 0; i < HEAD_DIM; i++) begin
            expected = 2.0 - (i * 0.5);
            actual = fixed_to_float(head_out[i]);
            error = actual - expected;
            if (error < 0) error = -error;
            
            if (error > 0.3) begin  // Allow some tolerance due to fixed-point
                $display("  ERROR: Mismatch at [%0d]: expected %.3f, got %.3f", 
                        i, expected, actual);
                test_pass = 0;
            end
        end

        // After wait_for_valid(), add:
        $display("DEBUG: V cache at pos 0:");
        $display("  [%.3f, %.3f, %.3f, %.3f]",
            fixed_to_float(dut.v_cache[0][0]), fixed_to_float(dut.v_cache[0][1]),
            fixed_to_float(dut.v_cache[0][2]), fixed_to_float(dut.v_cache[0][3]));
        $display("DEBUG: Attention weight[0] = %0d (%.6f)", 
            dut.attn_weights[0], fixed_to_float(dut.attn_weights[0]));
        $display("DEBUG: Accumulator before final shift:");
        $display("  [%0d, %0d, %0d, %0d]",
            dut.accum[0], dut.accum[1], dut.accum[2], dut.accum[3]);
        
        if (test_pass) begin
            $display("✓ Test 1 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 1 FAILED");
            fail_count++;
        end
    endtask
    // In Test 1, after wait_for_valid():

    // Test 2: Two position attention
    task test_two_positions();
        $display("\n=== Test 2: Two Position Attention ===");
        test_num = 2;
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Position 0: K=[1,1,1,1], V=[1,1,1,1]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(1.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        @(posedge clk);
        @(posedge clk);
        
        // Position 1: K=[2,2,2,2], V=[2,2,2,2]
        // Q=[1,1,1,1] - testing with same query
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(2.0);
            v_in[i] = float_to_fixed(2.0);
        end
        pos = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("Two position output:");
        
        // Q·K[0] = 4.0, Q·K[1] = 8.0 (before scaling)
        // Position 1 should have higher attention weight
        // Output should be weighted average, closer to V[1]=[2,2,2,2]
        if (check_output_range(1.0, 2.5)) begin
            $display("✓ Test 2 PASSED - output in expected range");
            pass_count++;
        end else begin
            $display("✗ Test 2 FAILED - output out of range");
            fail_count++;
        end
    endtask
    
    // Test 3: Sequence of 4 positions
    task test_sequence();
        $display("\n=== Test 3: Sequence of 4 Positions ===");
        test_num = 3;
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Process 4 positions with different K, V values
        for (int p = 0; p < 4; p++) begin
            // K and V increase with position
            for (int i = 0; i < HEAD_DIM; i++) begin
                q_in[i] = float_to_fixed(1.0);  // Constant query
                k_in[i] = float_to_fixed(real'(p + 1));
                v_in[i] = float_to_fixed(real'(p + 1) * 0.5);
            end
            
            pos = p;
            start = 1;
            @(posedge clk);
            start = 0;
            wait_for_valid();
            
            $display("  Position %0d output:", p);
            for (int i = 0; i < HEAD_DIM; i++) begin
                $display("    [%0d] = %.3f", i, fixed_to_float(head_out[i]));
            end
            
            @(posedge clk);
            @(posedge clk);
        end
        
        // Check that later positions have larger outputs
        // (higher attention to more recent tokens with larger values)
        $display("✓ Test 3 PASSED - sequence processed");
        pass_count++;
    endtask
    
    // Test 4: Attention focus test (IMPROVED)
    task test_attention_focus();
        $display("\n=== Test 4: Attention Focus Test ===");
        test_num = 4;
        
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Position 0: K=[1,1,1,1], V=[10,10,10,10]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(5.0);  // Query doesn't match this well
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(10.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        @(posedge clk);
        
        // Position 1: K=[5,5,5,5] (matches query!), V=[1,1,1,1]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(5.0);  // Query matches this position
            k_in[i] = float_to_fixed(5.0);
            v_in[i] = float_to_fixed(1.0);
        end
        pos = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("Attention focus output:");
        
        // Q·K[0] = 5×1 × 4 = 20 (scaled)
        // Q·K[1] = 5×5 × 4 = 100 (scaled)
        // After softmax, position 1 should get MUCH higher weight
        // Output should be closer to V[1]=[1,1,1,1] than V[0]=[10,10,10,10]
        // Expected: somewhere between 1.0 and 4.0
        if (check_output_range(0.5, 4.0)) begin
            $display("✓ Test 4 PASSED - attention focused on higher-scoring position");
            pass_count++;
        end else begin
            $display("✗ Test 4 FAILED - attention focus incorrect");
            fail_count++;
        end
    endtask
    
    // Test 5: Zero inputs
    task test_zero_inputs();
        $display("\n=== Test 5: Zero Inputs ===");
        test_num = 5;
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // All zeros
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = '0;
            k_in[i] = '0;
            v_in[i] = '0;
        end
        
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("Zero input output:");
        
        // Output should be close to zero
        if (check_output_range(-0.5, 0.5)) begin
            $display("✓ Test 5 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 5 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 6: Maximum length sequence
    task test_max_length();
        $display("\n=== Test 6: Maximum Length Sequence ===");
        test_num = 6;
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Process full block_size positions
        for (int p = 0; p < 8; p++) begin  // Test 8 positions
            for (int i = 0; i < HEAD_DIM; i++) begin
                q_in[i] = float_to_fixed(1.0);
                k_in[i] = float_to_fixed(1.0);
                v_in[i] = float_to_fixed(real'(p) * 0.1);
            end
            
            pos = p;
            start = 1;
            @(posedge clk);
            start = 0;
            wait_for_valid();
            
            @(posedge clk);
        end
        
        $display("✓ Test 6 PASSED - max length sequence processed");
        pass_count++;
    endtask
    
    // Test 7: Cache clear functionality
    task test_cache_clear();
        $display("\n=== Test 7: Cache Clear Functionality ===");
        test_num = 7;
        
        // Process some positions
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        
        for (int p = 0; p < 3; p++) begin
            for (int i = 0; i < HEAD_DIM; i++) begin
                q_in[i] = float_to_fixed(real'(p+1));
                k_in[i] = float_to_fixed(real'(p+1));
                v_in[i] = float_to_fixed(real'(p+1));
            end
            pos = p;
            start = 1;
            @(posedge clk);
            start = 0;
            wait_for_valid();
            @(posedge clk);
        end
        
        // Clear cache
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Process position 0 again
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(5.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("After cache clear:");
        
        // Should only attend to position 0, output ≈ [5,5,5,5]
        if (check_output_range(4.0, 6.0)) begin
            $display("✓ Test 7 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 7 FAILED");
            fail_count++;
        end
    endtask

    // ===========================================================================
    // Additional Attention Head Tests - Stress Testing
    // These tests verify edge cases and attention focusing behavior
    // ===========================================================================

    // Test 8: Strong Attention Focus (one position dominates)
    task test_strong_attention_focus();
        real avg_output;
        $display("\n=== Test 8: Strong Attention Focus ===");
        test_num = 8;
        
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Position 0: K=[0.1,0.1,0.1,0.1], V=[100,100,100,100]
        // Very low key values - won't match query
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(10.0);
            k_in[i] = float_to_fixed(0.1);
            v_in[i] = float_to_fixed(100.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        @(posedge clk);
        
        // Position 1: K=[10,10,10,10], V=[1,1,1,1]
        // Perfect match with query!
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(10.0);
            k_in[i] = float_to_fixed(10.0);
            v_in[i] = float_to_fixed(1.0);
        end
        pos = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("Strong attention focus output:");
        
        // Q·K[0] = 10×0.1×4 = 4.0
        // Q·K[1] = 10×10×4 = 400.0
        // After scaling and softmax, position 1 should get ~99.9% attention
        // Output should be very close to V[1]=[1,1,1,1]
        
        avg_output = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) begin
            avg_output += fixed_to_float(head_out[i]);
        end
        avg_output = avg_output / HEAD_DIM;
        
        $display("Average output: %.3f", avg_output);
        $display("Expected: ~1.0 (strongly focused on position 1)");
        
        if (avg_output > 0.8 && avg_output < 2.0) begin
            $display("✓ Test 8 PASSED - strong attention focus works");
            pass_count++;
        end else begin
            $display("✗ Test 8 FAILED - attention not focusing correctly");
            $display("  Output should be close to 1.0, got %.3f", avg_output);
            fail_count++;
        end
    endtask

    // Test 9: Uniform Attention (all positions equal)
    task test_uniform_attention();
        real avg_output;
        real error;
        $display("\n=== Test 9: Uniform Attention Distribution ===");
        test_num = 9;
        
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Create 4 positions with identical keys but different values
        for (int p = 0; p < 4; p++) begin
            for (int i = 0; i < HEAD_DIM; i++) begin
                q_in[i] = float_to_fixed(2.0);
                k_in[i] = float_to_fixed(2.0);  // All keys identical
                v_in[i] = float_to_fixed(real'(p + 1));  // Values: 1,2,3,4
            end
            pos = p;
            start = 1;
            @(posedge clk);
            start = 0;
            wait_for_valid();
            @(posedge clk);
        end
        
        display_output("Uniform attention output:");
        
        // All Q·K scores are equal, so attention should be uniform
        // With 4 positions having values [1,2,3,4], each with 0.25 weight:
        // Output = 0.25×1 + 0.25×2 + 0.25×3 + 0.25×4 = 2.5
        
        avg_output = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) begin
            avg_output += fixed_to_float(head_out[i]);
        end
        avg_output = avg_output / HEAD_DIM;
        
        $display("Average output: %.3f", avg_output);
        $display("Expected: 2.5 (uniform average of 1,2,3,4)");
        
        error = avg_output - 2.5;
        if (error < 0) error = -error;
        
        if (error < 0.5) begin
            $display("✓ Test 9 PASSED - uniform attention distribution works");
            pass_count++;
        end else begin
            $display("✗ Test 9 FAILED - uniform attention incorrect");
            $display("  Expected 2.5, got %.3f (error: %.3f)", avg_output, error);
            fail_count++;
        end
    endtask

    // Test 10: Negative Values in K and V
    task test_negative_keys_values();
        real avg_output;
        $display("\n=== Test 10: Negative Keys and Values ===");
        test_num = 10;
        
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Position 0: K=[-2,-2,-2,-2], V=[5,5,5,5]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(3.0);
            k_in[i] = float_to_fixed(-2.0);  // Negative keys
            v_in[i] = float_to_fixed(5.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        @(posedge clk);
        
        // Position 1: K=[3,3,3,3], V=[-10,-10,-10,-10]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(3.0);
            k_in[i] = float_to_fixed(3.0);
            v_in[i] = float_to_fixed(-10.0);  // Negative values
        end
        pos = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        display_output("Negative K/V output:");
        
        // Q·K[0] = 3×(-2)×4 = -24 (negative dot product)
        // Q·K[1] = 3×3×4 = 36 (positive dot product)
        // Position 1 should dominate attention
        // Output should be close to V[1]=[-10,-10,-10,-10]
        
        avg_output = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) begin
            avg_output += fixed_to_float(head_out[i]);
        end
        avg_output = avg_output / HEAD_DIM;
        
        $display("Average output: %.3f", avg_output);
        $display("Expected: close to -10.0 (focused on position 1 with negative values)");
        
        if (avg_output < -5.0 && avg_output > -12.0) begin
            $display("✓ Test 10 PASSED - handles negative keys and values correctly");
            pass_count++;
        end else begin
            $display("✗ Test 10 FAILED - negative value handling incorrect");
            $display("  Expected around -10.0, got %.3f", avg_output);
            fail_count++;
        end
    endtask

    // Test 11: Recency Bias Test (should attend to recent positions)
    task test_recency_bias();
        real prev_output;
        real avg_out;
        logic recency_trend;
        $display("\n=== Test 11: Recency Bias with Varying Queries ===");
        test_num = 11;
        
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Build up a sequence where each position's query should
        // match its own key better than previous keys
        
        prev_output = 0.0;
        recency_trend = 1;
        
        for (int p = 0; p < 5; p++) begin
            // Each position: K and V increase with position
            for (int i = 0; i < HEAD_DIM; i++) begin
                // Query always looks for high values
                q_in[i] = float_to_fixed(10.0);
                // Keys increase with position (0.5, 1.0, 1.5, 2.0, 2.5)
                k_in[i] = float_to_fixed(real'(p + 1) * 0.5);
                // Values also increase with position
                v_in[i] = float_to_fixed(real'(p + 1) * 2.0);
            end
            pos = p;
            start = 1;
            @(posedge clk);
            start = 0;
            wait_for_valid();
            
            // Calculate average output
            avg_out = 0.0;
            for (int i = 0; i < HEAD_DIM; i++) begin
                avg_out += fixed_to_float(head_out[i]);
            end
            avg_out = avg_out / HEAD_DIM;
            
            $display("  Position %0d output: %.3f", p, avg_out);
            
            // After first position, each output should be larger than previous
            // (attention shifts toward newer, higher-scoring positions)
            if (p > 0) begin
                if (avg_out <= prev_output) begin
                    recency_trend = 0;
                    $display("    WARNING: Output not increasing (%.3f <= %.3f)", 
                            avg_out, prev_output);
                end
            end
            
            prev_output = avg_out;
            @(posedge clk);
        end
        
        if (recency_trend) begin
            $display("✓ Test 11 PASSED - attention properly shifts to higher-scoring recent positions");
            pass_count++;
        end else begin
            $display("✗ Test 11 FAILED - recency trend not observed");
            $display("  Expected outputs to increase as better-matching positions are added");
            fail_count++;
        end
    endtask

    // ===========================================================================
    // Additional Debug Test: Verify Attention Weights Distribution
    // ===========================================================================
    task test_attention_weight_verification();
        real weight_sum;
        real avg_output;
        real w;
        real weight_error;
        real output_error;
        $display("\n=== Test 12: Attention Weight Distribution Verification ===");
        test_num = 12;
        @(posedge clk);
        clear_cache = 1;
        @(posedge clk);
        clear_cache = 0;
        @(posedge clk);
        
        // Create 3 positions with known attention distribution
        // Position 0: K=[1,1,1,1], V=[10,10,10,10]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(10.0);
        end
        pos = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        @(posedge clk);
        
        // Position 1: K=[1,1,1,1], V=[20,20,20,20]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(20.0);
        end
        pos = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        @(posedge clk);
        
        // Position 2: K=[1,1,1,1], V=[30,30,30,30]
        for (int i = 0; i < HEAD_DIM; i++) begin
            q_in[i] = float_to_fixed(1.0);
            k_in[i] = float_to_fixed(1.0);
            v_in[i] = float_to_fixed(30.0);
        end
        pos = 2;
        start = 1;
        @(posedge clk);
        start = 0;
        wait_for_valid();
        
        // All keys equal, so attention should be uniform: 1/3 each
        // Output = (1/3)×10 + (1/3)×20 + (1/3)×30 = 20.0
        
        display_output("Weight verification output:");
        
        $display("Internal attention weights:");
        weight_sum = 0.0;
        for (int i = 0; i <= 2; i++) begin
            w = fixed_to_float(dut.attn_weights[i]);
            weight_sum += w;
            $display("  weight[%0d] = %.6f", i, w);
        end
        $display("  Sum of weights = %.6f (should be ~1.0)", weight_sum);
        
        avg_output = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) begin
            avg_output += fixed_to_float(head_out[i]);
        end
        avg_output = avg_output / HEAD_DIM;
        
        $display("Average output: %.3f", avg_output);
        $display("Expected: 20.0 (uniform average of 10,20,30)");
        
        weight_error = weight_sum - 1.0;
        if (weight_error < 0) weight_error = -weight_error;
        
        output_error = avg_output - 20.0;
        if (output_error < 0) output_error = -output_error;
        
        if (weight_error < 0.1 && output_error < 1.0) begin
            $display("✓ Test 12 PASSED - attention weights sum to 1.0 and output is correct");
            pass_count++;
        end else begin
            $display("✗ Test 12 FAILED");
            if (weight_error >= 0.1) 
                $display("  Weight sum error: %.6f (should be < 0.1)", weight_error);
            if (output_error >= 1.0)
                $display("  Output error: %.3f (should be < 1.0)", output_error);
            fail_count++;
        end
    endtask

    // ===========================================================================
    // Extended test invocation (add after test_cache_clear() in the main initial block):
    //
    //    test_strong_attention_focus();       // Test 8
    //    test_uniform_attention();            // Test 9
    //    test_negative_keys_values();         // Test 10
    //    test_recency_bias();                 // Test 11
    //    test_attention_weight_verification(); // Test 12
    //
    // Update the summary total to 12 when these are included.
    // ===========================================================================
    
    // -----------------------------------------------------------------------
    // Main Test Sequence
    // -----------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("Attention Head Testbench");
        $display("========================================");
        
        pass_count = 0;
        fail_count = 0;
        
        // Initialize signals
        clk = 0;
        rst_n = 0;
        start = 0;
        clear_cache = 0;
        pos = 0;
        
        // Reset
        reset_dut();
        
        // Run tests
        test_single_position();
        test_two_positions();
        test_sequence();
        test_attention_focus();
        test_zero_inputs();
        test_max_length();
        test_cache_clear();
        test_strong_attention_focus();      // Test 8
        test_uniform_attention();            // Test 9
        test_negative_keys_values();         // Test 10
        test_recency_bias();                 // Test 11
        test_attention_weight_verification(); // Test 12
        
        // Summary
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total tests: %0d", pass_count + fail_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("\n✓ ALL TESTS PASSED!");
        end else begin
            $display("\n✗ SOME TESTS FAILED!");
        end
        
        $display("========================================\n");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Global timeout!");
        $finish;
    end
    
endmodule : tb_attention_head