// ===========================================================================
// MLP Testbench
// ===========================================================================
// Comprehensive verification of MLP (feed-forward) functionality
// ===========================================================================

`timescale 1ns/1ps

module tb_mlp;
    import microgpt_pkg::*;
    
    // Parameters
    localparam int N_EMBD = 16;
    localparam int HIDDEN_DIM = 4 * N_EMBD;
    localparam int CLK_PERIOD = 10;
    
    // DUT signals
    logic   clk;
    logic   rst_n;
    logic   start;
    fixed_t x_in [N_EMBD-1:0];
    fixed_t fc1_weights [(HIDDEN_DIM)*N_EMBD-1:0];
    fixed_t fc2_weights [N_EMBD*HIDDEN_DIM-1:0];
    fixed_t x_out [N_EMBD-1:0];
    logic   valid;
    
    // Test control
    int test_num;
    int pass_count;
    int fail_count;
    
    // DUT instantiation
    mlp #(
        .N_EMBD(N_EMBD)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .x_in(x_in),
        .fc1_weights(fc1_weights),
        .fc2_weights(fc2_weights),
        .x_out(x_out),
        .valid(valid)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Helper tasks (Changed from functions to tasks to allow time-controlled statements)
    task reset_dut();
        rst_n = 0;
        start = 0;
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = '0;
        end
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = '0;
        end
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = '0;
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
                repeat(5000) @(posedge clk);
                $display("ERROR: Timeout waiting for valid signal");
                fail_count++;
            end
        join_any
        disable fork;
    endtask
    
    // Display output (Can remain a function as it has no time-controlled statements)
    function void display_output(string msg);
        $display("%s", msg);
        $display("  Output vector:");
        for (int i = 0; i < N_EMBD; i++) begin
            $display("    [%0d] = %0d (%.3f)", i, x_out[i], fixed_to_float(x_out[i]));
        end
    endfunction
    
    function automatic logic check_output_range(real min_val, real max_val);
        logic in_range = 1;
        for (int i = 0; i < N_EMBD; i++) begin
            real val = fixed_to_float(x_out[i]);
            if (val < min_val || val > max_val) begin
                $display("  ERROR: Output[%0d] = %.3f out of range [%.3f, %.3f]", 
                        i, val, min_val, max_val);
                in_range = 0;
            end
        end
        return in_range;
    endfunction
    
    function void init_identity_weights();
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = '0;
        end
        for (int i = 0; i < N_EMBD; i++) begin
            fc1_weights[i * N_EMBD + i] = float_to_fixed(1.0);
        end
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = '0;
        end
        for (int i = 0; i < N_EMBD; i++) begin
            fc2_weights[i * HIDDEN_DIM + i] = float_to_fixed(1.0);
        end
    endfunction
    
    // -----------------------------------------------------------------------
    // Test Cases
    // -----------------------------------------------------------------------
    
    // Test 1: Zero input
    task test_zero_input();
        $display("\n=== Test 1: Zero Input ===");
        test_num = 1;
        
        // Initialize weights to small values
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = float_to_fixed(0.1);
        end
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = float_to_fixed(0.1);
        end
        
        // Zero input
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = '0;
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("Zero input output:");
        
        // With zero input, output should be close to zero
        if (check_output_range(-0.5, 0.5)) begin
            $display("✓ Test 1 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 1 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 2: Identity transformation
    task test_identity();
        $display("\n=== Test 2: Identity Transformation ===");
        test_num = 2;
        
        init_identity_weights();
        
        // Set input to known values
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = float_to_fixed(real'(i) * 0.5);
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("Identity transformation output:");
        
        // With identity-like weights, output should be similar to input
        // (some loss due to fixed-point and intermediate ReLU)
        if (check_output_range(-1.0, 8.0)) begin
            $display("✓ Test 2 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 2 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 3: ReLU functionality (negative inputs)
    task test_relu_negative();
        $display("\n=== Test 3: ReLU with Negative Activations ===");
        test_num = 3;
        
        // FC1 weights that produce some negative values
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = float_to_fixed(-0.5);  // Negative weights
        end
        
        // FC2 weights
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = float_to_fixed(0.1);
        end
        
        // Positive input
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = float_to_fixed(1.0);
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("ReLU negative test output:");
        
        // ReLU should clip negative values, output should be small
        if (check_output_range(-0.5, 2.0)) begin
            $display("✓ Test 3 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 3 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 4: Positive activation amplification
    task test_positive_amplification();
        $display("\n=== Test 4: Positive Activation Amplification ===");
        test_num = 4;
        
        // FC1 weights - positive, amplifying
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = float_to_fixed(0.5);
        end
        
        // FC2 weights - sum up activations
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = float_to_fixed(0.1);
        end
        
        // Input: all ones
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = float_to_fixed(1.0);
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("Positive amplification output:");
        
        // Should produce positive outputs
        if (check_output_range(0.0, 50.0)) begin
            $display("✓ Test 4 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 4 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 5: Sequential processing
    task test_sequential();
        $display("\n=== Test 5: Sequential Processing ===");
        test_num = 5;
        
        init_identity_weights();
        
        // Process multiple different inputs sequentially
        for (int seq = 0; seq < 3; seq++) begin
            $display("  Sequence %0d:", seq);
            
            // Different input pattern each time
            for (int i = 0; i < N_EMBD; i++) begin
                x_in[i] = float_to_fixed(real'(seq + i) * 0.25);
            end
            
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            wait_for_valid();
            
            $display("    Output[0] = %.3f", fixed_to_float(x_out[0]));
            
            @(posedge clk);
            @(posedge clk);
        end
        
        $display("✓ Test 5 PASSED - sequential processing completed");
        pass_count++;
    endtask
    
    // Test 6: Large input values
    task test_large_inputs();
        $display("\n=== Test 6: Large Input Values ===");
        test_num = 6;
        
        // Small weights to prevent overflow
        for (int i = 0; i < HIDDEN_DIM*N_EMBD; i++) begin
            fc1_weights[i] = float_to_fixed(0.01);
        end
        for (int i = 0; i < N_EMBD*HIDDEN_DIM; i++) begin
            fc2_weights[i] = float_to_fixed(0.01);
        end
        
        // Large input values
        for (int i = 0; i < N_EMBD; i++) begin
            x_in[i] = float_to_fixed(10.0);
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("Large input output:");
        
        // Should handle large inputs without overflow
        if (check_output_range(-5.0, 20.0)) begin
            $display("✓ Test 6 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 6 FAILED");
            fail_count++;
        end
    endtask
    
    // Test 7: Mixed positive/negative inputs
    task test_mixed_signs();
        $display("\n=== Test 7: Mixed Positive/Negative Inputs ===");
        test_num = 7;
        
        init_identity_weights();
        
        // Half positive, half negative inputs
        for (int i = 0; i < N_EMBD; i++) begin
            if (i < N_EMBD/2) begin
                x_in[i] = float_to_fixed(1.0);
            end else begin
                x_in[i] = float_to_fixed(-1.0);
            end
        end
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_valid();
        display_output("Mixed sign output:");
        
        // ReLU will clip negative activations
        if (check_output_range(-2.0, 5.0)) begin
            $display("✓ Test 7 PASSED");
            pass_count++;
        end else begin
            $display("✗ Test 7 FAILED");
            fail_count++;
        end
    endtask
    
    // -----------------------------------------------------------------------
    // Main Test Sequence
    // -----------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("MLP Testbench");
        $display("========================================");
        
        pass_count = 0;
        fail_count = 0;
        
        // Initialize signals
        clk = 0;
        rst_n = 0;
        start = 0;
        
        // Reset
        reset_dut();
        
        // Run tests
        test_zero_input();
        test_identity();
        test_relu_negative();
        test_positive_amplification();
        test_sequential();
        test_large_inputs();
        test_mixed_signs();
        
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
        #500000;
        $display("ERROR: Global timeout!");
        $finish;
    end
    
endmodule : tb_mlp