`timescale 1ns/1ps

module tb_matrix_vector_mult;
    import microgpt_pkg::*;
    
    // Parameters
    localparam ROWS = 4;
    localparam COLS = 4;
    
    // DUT signals
    logic clk, rst_n, start, valid;
    fixed_t vec_in [COLS-1:0];
    fixed_t matrix [ROWS-1:0][COLS-1:0];
    fixed_t vec_out [ROWS-1:0];
    
    int pass_count, fail_count;
    
    // Instantiate DUT
    matrix_vector_mult #(
        .ROWS(ROWS),
        .COLS(COLS)
    ) dut (.*);
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Main test
    initial begin
        $display("==============================================");
        $display("Matrix-Vector Multiply Test (%0dx%0d)", ROWS, COLS);
        $display("==============================================\n");
        
        rst_n = 0;
        start = 0;
        pass_count = 0;
        fail_count = 0;
        
        #20 rst_n = 1;
        #20;
        
        // Test 1: Identity matrix
        $display("TEST 1: Identity Matrix");
        test_multiply(
            // Matrix: Identity
            '{'{1.0, 0.0, 0.0, 0.0},
              '{0.0, 1.0, 0.0, 0.0},
              '{0.0, 0.0, 1.0, 0.0},
              '{0.0, 0.0, 0.0, 1.0}},
            // Vector
            '{1.0, 2.0, 3.0, 4.0},
            // Expected: same as input
            '{1.0, 2.0, 3.0, 4.0}
        );
        
        // Test 2: All ones matrix
        $display("\nTEST 2: All Ones Matrix");
        test_multiply(
            '{'{1.0, 1.0, 1.0, 1.0},
              '{1.0, 1.0, 1.0, 1.0},
              '{1.0, 1.0, 1.0, 1.0},
              '{1.0, 1.0, 1.0, 1.0}},
            '{1.0, 2.0, 3.0, 4.0},
            '{10.0, 10.0, 10.0, 10.0}  // Sum of input vector
        );
        
        // Test 3: Diagonal matrix (scaling)
        $display("\nTEST 3: Diagonal Matrix (Scaling)");
        test_multiply(
            '{'{2.0, 0.0, 0.0, 0.0},
              '{0.0, 3.0, 0.0, 0.0},
              '{0.0, 0.0, 4.0, 0.0},
              '{0.0, 0.0, 0.0, 5.0}},
            '{1.0, 1.0, 1.0, 1.0},
            '{2.0, 3.0, 4.0, 5.0}
        );
        
        // Test 4: Zero matrix
        $display("\nTEST 4: Zero Matrix");
        test_multiply(
            '{'{0.0, 0.0, 0.0, 0.0},
              '{0.0, 0.0, 0.0, 0.0},
              '{0.0, 0.0, 0.0, 0.0},
              '{0.0, 0.0, 0.0, 0.0}},
            '{1.0, 2.0, 3.0, 4.0},
            '{0.0, 0.0, 0.0, 0.0}
        );
        
        // Test 5: General matrix
        $display("\nTEST 5: General Matrix");
        test_multiply(
            '{'{1.0, 2.0, 3.0, 4.0},
              '{5.0, 6.0, 7.0, 8.0},
              '{9.0, 10.0, 11.0, 12.0},
              '{13.0, 14.0, 15.0, 16.0}},
            '{1.0, 0.0, 0.0, 0.0},
            '{1.0, 5.0, 9.0, 13.0}  // First column
        );
        
        // Test 6: Negative values
        $display("\nTEST 6: Negative Values");
        test_multiply(
            '{'{-1.0, 0.0, 0.0, 0.0},
              '{0.0, -1.0, 0.0, 0.0},
              '{0.0, 0.0, -1.0, 0.0},
              '{0.0, 0.0, 0.0, -1.0}},
            '{1.0, 2.0, 3.0, 4.0},
            '{-1.0, -2.0, -3.0, -4.0}
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
    
    task test_multiply(
        real mat [ROWS-1:0][COLS-1:0],
        real vec [COLS-1:0],
        real expected [ROWS-1:0]
    );
        real actual [ROWS-1:0];
        real errors [ROWS-1:0];
        logic test_pass;
        
        // Convert inputs to fixed-point
        for (int i = 0; i < ROWS; i++) begin
            for (int j = 0; j < COLS; j++) begin
                matrix[i][j] = float_to_fixed(mat[i][j]);
            end
        end
        
        for (int j = 0; j < COLS; j++) begin
            vec_in[j] = float_to_fixed(vec[j]);
        end
        
        // Start computation
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for result
        wait(valid == 1);
        @(posedge clk);
        
        // Check results
        test_pass = 1;
        for (int i = 0; i < ROWS; i++) begin
            actual[i] = fixed_to_float(vec_out[i]);
            errors[i] = expected[i] - actual[i];
            
            if (errors[i] > 0.1 || errors[i] < -0.1) begin
                test_pass = 0;
            end
        end
        
        // Display results
        $display("  Matrix:");
        for (int i = 0; i < ROWS; i++) begin
            $write("    [");
            for (int j = 0; j < COLS; j++) begin
                $write("%6.2f", mat[i][j]);
                if (j < COLS-1) $write(", ");
            end
            $display("]");
        end
        
        $display("  Vector: [%0.2f, %0.2f, %0.2f, %0.2f]", 
                 vec[0], vec[1], vec[2], vec[3]);
        
        $display("  Expected: [%0.2f, %0.2f, %0.2f, %0.2f]",
                 expected[0], expected[1], expected[2], expected[3]);
        
        $display("  Actual:   [%0.2f, %0.2f, %0.2f, %0.2f]",
                 actual[0], actual[1], actual[2], actual[3]);
        
        $display("  Errors:   [%0.4f, %0.4f, %0.4f, %0.4f]",
                 errors[0], errors[1], errors[2], errors[3]);
        
        if (test_pass) begin
            $display("  Result: PASS ✓");
            pass_count++;
        end else begin
            $display("  Result: FAIL ✗");
            fail_count++;
        end
        
        repeat(10) @(posedge clk);
    endtask
    
    initial #500000 $finish;  // Timeout

endmodule
