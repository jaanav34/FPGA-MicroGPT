`timescale 1ns/1ps

module tb_softmax;
    import microgpt_pkg::*;
    
    localparam VEC_LEN = 5;  // Small for easy verification
    
    logic clk, rst_n, start, valid;
    fixed_t logits [VEC_LEN-1:0];
    fixed_t temperature;
    fixed_t probs [VEC_LEN-1:0];
    
    int pass_count, fail_count;
    
    softmax #(.VEC_LEN(VEC_LEN)) dut (.*);
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("==============================================");
        $display("Softmax Test (VEC_LEN = %0d)", VEC_LEN);
        $display("==============================================\n");
        
        rst_n = 0;
        start = 0;
        temperature = float_to_fixed(1.0);
        pass_count = 0;
        fail_count = 0;
        
        #20 rst_n = 1;
        #20;
        
        // Test 1: Uniform logits (should give uniform probabilities)
        $display("TEST 1: Uniform logits");
        test_softmax(
            '{0.0, 0.0, 0.0, 0.0, 0.0},
            1.0,
            '{0.2, 0.2, 0.2, 0.2, 0.2}  // Equal probabilities
        );
        
        // Test 2: One dominant logit
        $display("\nTEST 2: One dominant logit");
        test_softmax(
            '{10.0, 0.0, 0.0, 0.0, 0.0},
            1.0,
            '{0.99, 0.0025, 0.0025, 0.0025, 0.0025}  // First dominates
        );
        
        // Test 3: Simple case
        $display("\nTEST 3: Simple ascending");
        test_softmax(
            '{1.0, 2.0, 3.0, 4.0, 5.0},
            1.0,
            '{0.012, 0.032, 0.087, 0.236, 0.643}  // Approximate
        );
        
        // Test 4: Negative logits
        $display("\nTEST 4: Negative logits");
        test_softmax(
            '{-1.0, -2.0, -3.0, -4.0, -5.0},
            1.0,
            '{0.643, 0.236, 0.087, 0.032, 0.012}  // Reversed
        );
        
        // Test 5: Temperature scaling (high temp = more uniform)
        $display("\nTEST 5: High temperature");
        test_softmax(
            '{1.0, 2.0, 3.0, 4.0, 5.0},
            2.0,  // Higher temperature
            '{0.11, 0.15, 0.20, 0.25, 0.29}  // More uniform than Test 3
        );
        
        // Test 6: Sum should always be ~1.0
        $display("\nTEST 6: Probability sum verification");
        test_softmax_sum(
            '{2.0, -1.0, 3.0, 0.0, 1.5},
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
    
    task test_softmax(
        real input_logits [VEC_LEN-1:0],
        real temp,
        real expected_probs [VEC_LEN-1:0]
    );
        real actual_probs [VEC_LEN-1:0];
        real errors [VEC_LEN-1:0];
        real max_error;
        logic test_pass;
        
        // Load inputs
        for (int i = 0; i < VEC_LEN; i++) begin
            logits[i] = float_to_fixed(input_logits[i]);
        end
        temperature = float_to_fixed(temp);
        
        $display("  Logits: [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]",
                 input_logits[0], input_logits[1], input_logits[2], 
                 input_logits[3], input_logits[4]);
        $display("  Temperature: %0.2f", temp);
        
        // Run softmax
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait(valid == 1);
        @(posedge clk);
        
        // Check outputs
        test_pass = 1;
        max_error = 0.0;
        for (int i = 0; i < VEC_LEN; i++) begin
            actual_probs[i] = fixed_to_float(probs[i]);
            errors[i] = expected_probs[i] - actual_probs[i];
            if (errors[i] > max_error || errors[i] < -max_error) begin
                max_error = (errors[i] > 0) ? errors[i] : -errors[i];
            end
            
            if (errors[i] > 0.1 || errors[i] < -0.1) begin
                test_pass = 0;
            end
        end
        
        $display("  Expected: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]",
                 expected_probs[0], expected_probs[1], expected_probs[2],
                 expected_probs[3], expected_probs[4]);
        
        $display("  Actual:   [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]",
                 actual_probs[0], actual_probs[1], actual_probs[2],
                 actual_probs[3], actual_probs[4]);
        
        $display("  Max Error: %0.4f", max_error);
        
        if (test_pass) begin
            $display("  Result: PASS ✓");
            pass_count++;
        end else begin
            $display("  Result: FAIL ✗ (Error too large)");
            fail_count++;
        end
        
        repeat(10) @(posedge clk);
    endtask
    
    task test_softmax_sum(
        real input_logits [VEC_LEN-1:0],
        real temp
    );
        real actual_probs [VEC_LEN-1:0];
        real prob_sum;
        real sum_error;
        
        // Load inputs
        for (int i = 0; i < VEC_LEN; i++) begin
            logits[i] = float_to_fixed(input_logits[i]);
        end
        temperature = float_to_fixed(temp);
        
        $display("  Logits: [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]",
                 input_logits[0], input_logits[1], input_logits[2],
                 input_logits[3], input_logits[4]);
        
        // Run softmax
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait(valid == 1);
        @(posedge clk);
        
        // Sum probabilities
        prob_sum = 0.0;
        for (int i = 0; i < VEC_LEN; i++) begin
            actual_probs[i] = fixed_to_float(probs[i]);
            prob_sum += actual_probs[i];
        end
        
        sum_error = 1.0 - prob_sum;
        
        $display("  Probs: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]",
                 actual_probs[0], actual_probs[1], actual_probs[2],
                 actual_probs[3], actual_probs[4]);
        
        $display("  Sum: %0.6f (Expected: 1.000000)", prob_sum);
        $display("  Sum Error: %0.6f", sum_error);
        
        if (sum_error < 0.05 && sum_error > -0.05) begin
            $display("  Result: PASS ✓");
            pass_count++;
        end else begin
            $display("  Result: FAIL ✗ (Sum not close to 1.0)");
            fail_count++;
        end
        
        repeat(10) @(posedge clk);
    endtask
    
    initial #500000 $finish;

endmodule
