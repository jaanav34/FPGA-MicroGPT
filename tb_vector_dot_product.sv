`timescale 1ns/1ps

module tb_vector_dot_product;
    import microgpt_pkg::*;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // DUT signals
    localparam VEC_LEN = 4;  // Small for easy verification
    logic start;
    fixed_t vec_a [VEC_LEN-1:0];
    fixed_t vec_b [VEC_LEN-1:0];
    fixed_t result;
    logic valid;
    
    // Test tracking
    int test_num;
    int pass_count;
    int fail_count;
    
    // Instantiate DUT
    vector_dot_product #(
        .VEC_LEN(VEC_LEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .vec_a(vec_a),
        .vec_b(vec_b),
        .result(result),
        .valid(valid)
    );
    
    // Clock generation (100 MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Main test
    initial begin
        $display("==============================================");
        $display("Vector Dot Product Test (VEC_LEN = %0d)", VEC_LEN);
        $display("==============================================\n");
        
        // Initialize
        rst_n = 0;
        start = 0;
        test_num = 0;
        pass_count = 0;
        fail_count = 0;
        
        for (int i = 0; i < VEC_LEN; i++) begin
            vec_a[i] = '0;
            vec_b[i] = '0;
        end
        
        // Reset
        #20;
        rst_n = 1;
        #20;
        
        // Test 1: All zeros
        $display("TEST 1: All zeros");
        test_dot_product(
            '{0.0, 0.0, 0.0, 0.0},
            '{0.0, 0.0, 0.0, 0.0},
            0.0
        );
        
        // Test 2: All ones
        $display("\nTEST 2: All ones");
        test_dot_product(
            '{1.0, 1.0, 1.0, 1.0},
            '{1.0, 1.0, 1.0, 1.0},
            4.0
        );
        
        // Test 3: Simple integers
        $display("\nTEST 3: Simple integers");
        test_dot_product(
            '{1.0, 2.0, 3.0, 4.0},
            '{1.0, 1.0, 1.0, 1.0},
            10.0  // 1+2+3+4
        );
        
        // Test 4: Mixed positive/negative
        $display("\nTEST 4: Mixed signs");
        test_dot_product(
            '{1.0, -1.0, 1.0, -1.0},
            '{1.0, 1.0, 1.0, 1.0},
            0.0  // 1-1+1-1
        );
        
        // Test 5: Fractional values
        $display("\nTEST 5: Fractional values");
        test_dot_product(
            '{0.5, 0.5, 0.5, 0.5},
            '{2.0, 2.0, 2.0, 2.0},
            4.0  // 0.5*2*4
        );
        
        // Test 6: Actual dot product
        $display("\nTEST 6: Standard dot product");
        test_dot_product(
            '{1.0, 2.0, 3.0, 4.0},
            '{5.0, 6.0, 7.0, 8.0},
            70.0  // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32
        );
        
        // Test 7: Negative values
        $display("\nTEST 7: Negative values");
        test_dot_product(
            '{-1.0, -2.0, -3.0, -4.0},
            '{1.0, 2.0, 3.0, 4.0},
            -30.0  // -1-4-9-16
        );
        
        // Summary
        $display("\n==============================================");
        $display("Test Summary:");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("  Total:  %0d", test_num);
        if (fail_count == 0) begin
            $display("\n✓ ALL TESTS PASSED!");
        end else begin
            $display("\n✗ SOME TESTS FAILED!");
        end
        $display("==============================================");
        
        #100;
        $finish;
    end
    
    // Task to test dot product
    task test_dot_product(
        real a_vals [VEC_LEN-1:0],
        real b_vals [VEC_LEN-1:0],
        real expected
    );
        real actual;
        real error;
        
        test_num++;
        
        // Load vectors
        $display("  Input A: [%0.2f, %0.2f, %0.2f, %0.2f]", 
                 a_vals[0], a_vals[1], a_vals[2], a_vals[3]);
        $display("  Input B: [%0.2f, %0.2f, %0.2f, %0.2f]", 
                 b_vals[0], b_vals[1], b_vals[2], b_vals[3]);
        
        for (int i = 0; i < VEC_LEN; i++) begin
            vec_a[i] = float_to_fixed(a_vals[i]);
            vec_b[i] = float_to_fixed(b_vals[i]);
        end
        
        // Start computation
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for result
        wait(valid == 1);
        @(posedge clk);
        
        // Check result
        actual = fixed_to_float(result);
        error = expected - actual;
        
        $display("  Expected: %0.4f", expected);
        $display("  Actual:   %0.4f", actual);
        $display("  Error:    %0.6f", error);
        
        if (error < 0.1 && error > -0.1) begin
            $display("  Result: PASS ✓");
            pass_count++;
        end else begin
            $display("  Result: FAIL ✗ (Error too large!)");
            fail_count++;
        end
        
        // Wait a bit
        repeat(5) @(posedge clk);
    endtask
    
    // Timeout watchdog
    initial begin
        #100000;  // 100 us
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule
