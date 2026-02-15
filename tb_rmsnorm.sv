`timescale 1ns/1ps

module tb_rmsnorm;
    import microgpt_pkg::*;
    
    localparam VEC_LEN = 4;
    
    logic clk, rst_n, start, valid;
    fixed_t vec_in [VEC_LEN-1:0];
    fixed_t vec_out [VEC_LEN-1:0];
    
    int pass_count, fail_count;
    
    rmsnorm #(.VEC_LEN(VEC_LEN)) dut (.*);
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("==============================================");
        $display("RMSNorm Test (VEC_LEN = %0d)", VEC_LEN);
        $display("==============================================\n");
        
        rst_n = 0;
        start = 0;
        pass_count = 0;
        fail_count = 0;
        
        #20 rst_n = 1;
        #20;
        
        // Test 1: All ones
        $display("TEST 1: All ones (should normalize to ~1.0)");
        test_rmsnorm(
            '{1.0, 1.0, 1.0, 1.0},
            1.0  // Expected RMS magnitude
        );
        
        // Test 2: Simple scaling
        $display("\nTEST 2: Scaled values");
        test_rmsnorm(
            '{2.0, 2.0, 2.0, 2.0},
            1.0  // Should normalize to 1.0
        );
        
        // Test 3: Mixed values
        $display("\nTEST 3: Mixed values");
        test_rmsnorm(
            '{1.0, 2.0, 3.0, 4.0},
            1.0  // Should normalize
        );
        
        // Test 4: Zero mean (but non-zero RMS)
        $display("\nTEST 4: Zero mean");
        test_rmsnorm(
            '{-1.0, 1.0, -1.0, 1.0},
            1.0
        );
        
        // Test 5: Small values
        $display("\nTEST 5: Small values");
        test_rmsnorm(
            '{0.1, 0.2, 0.3, 0.4},
            1.0
        );
        
        // Summary
        $display("\n==============================================");
        $display("Test Summary:");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        if (fail_count == 0) $display("\n✓ ALL TESTS PASSED!");
        else $display("\n✗ SOME TESTS FAILED!");
        $display("==============================================");
        
        #100 $finish;
    end
    
    task test_rmsnorm(
        real input_vals [VEC_LEN-1:0],
        real expected_rms
    );
        real output_vals [VEC_LEN-1:0];
        real input_rms, output_rms;
        real rms_error;
        
        // Load input
        for (int i = 0; i < VEC_LEN; i++) begin
            vec_in[i] = float_to_fixed(input_vals[i]);
        end
        
        // Compute expected RMS of input
        input_rms = 0.0;
        for (int i = 0; i < VEC_LEN; i++) begin
            input_rms += input_vals[i] * input_vals[i];
        end
        input_rms = $sqrt(input_rms / VEC_LEN);
        
        $display("  Input: [%0.2f, %0.2f, %0.2f, %0.2f]",
                 input_vals[0], input_vals[1], input_vals[2], input_vals[3]);
        $display("  Input RMS: %0.4f", input_rms);
        
        // Run normalization
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait(valid == 1);
        @(posedge clk);
        
        // Check output
        for (int i = 0; i < VEC_LEN; i++) begin
            output_vals[i] = fixed_to_float(vec_out[i]);
        end
        
        // Compute RMS of output
        output_rms = 0.0;
        for (int i = 0; i < VEC_LEN; i++) begin
            output_rms += output_vals[i] * output_vals[i];
        end
        output_rms = $sqrt(output_rms / VEC_LEN);
        
        $display("  Output: [%0.4f, %0.4f, %0.4f, %0.4f]",
                 output_vals[0], output_vals[1], output_vals[2], output_vals[3]);
        $display("  Output RMS: %0.4f", output_rms);
        
        rms_error = expected_rms - output_rms;
        $display("  Expected RMS: %0.4f", expected_rms);
        $display("  RMS Error: %0.4f", rms_error);
        
        if (rms_error < 0.2 && rms_error > -0.2) begin
            $display("  Result: PASS ✓");
            pass_count++;
        end else begin
            $display("  Result: FAIL ✗ (RMS error too large)");
            fail_count++;
        end
        
        repeat(10) @(posedge clk);
    endtask
    
    initial #500000 $finish;

endmodule
