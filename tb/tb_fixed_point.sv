`timescale 1ns/1ps

// Test Fixed-Point Arithmetic
// Q8.8 format: 8 integer bits, 8 fractional bits
module tb_fixed_point;

    // Fixed-point type
    typedef logic signed [15:0] fixed_t;
    
    // Test variables
    fixed_t a, b, sum, product;
    real a_float, b_float, sum_float, prod_float;
    
    // Helper functions
    function fixed_t float_to_fixed(real f);
        return fixed_t'($rtoi(f * 256.0));
    endfunction
    
    function real fixed_to_float(fixed_t f);
        return real'(f) / 256.0;
    endfunction
    
    function fixed_t fixed_add(fixed_t x, fixed_t y);
        return x + y;
    endfunction
    
    function fixed_t fixed_mul(fixed_t x, fixed_t y);
        logic signed [31:0] temp;
        temp = x * y;
        return fixed_t'(temp >>> 8);
    endfunction
    
    initial begin
        $display("==============================================");
        $display("Fixed-Point Q8.8 Arithmetic Test");
        $display("==============================================\n");
        
        // Test 1: Basic conversions
        $display("TEST 1: Float to Fixed Conversion");
        $display("----------------------------------------------");
        test_conversion(0.0);
        test_conversion(1.0);
        test_conversion(-1.0);
        test_conversion(0.5);
        test_conversion(-0.5);
        test_conversion(2.25);
        test_conversion(-3.75);
        test_conversion(127.99);
        test_conversion(-128.0);
        
        // Test 2: Addition
        $display("\nTEST 2: Addition");
        $display("----------------------------------------------");
        test_add(1.0, 2.0);
        test_add(0.5, 0.25);
        test_add(-1.5, 2.5);
        test_add(10.75, -3.25);
        
        // Test 3: Multiplication
        $display("\nTEST 3: Multiplication");
        $display("----------------------------------------------");
        test_mul(2.0, 3.0);
        test_mul(0.5, 0.5);
        test_mul(-2.0, 1.5);
        test_mul(4.0, 0.25);
        test_mul(1.5, -2.5);
        
        // Test 4: Range limits
        $display("\nTEST 4: Range Limits (Q8.8 = -128 to +127.996)");
        $display("----------------------------------------------");
        test_conversion(127.5);
        test_conversion(-128.0);
        $display("Values outside range will saturate or overflow");
        
        $display("\n==============================================");
        $display("All Fixed-Point Tests Complete!");
        $display("==============================================");
        $finish;
    end
    
    task test_conversion(real val);
        fixed_t fx;
        real recovered;
        real error;
        
        fx = float_to_fixed(val);
        recovered = fixed_to_float(fx);
        error = val - recovered;
        
        $display("Float: %8.4f → Fixed: 0x%04h (%6d) → Float: %8.4f | Error: %8.6f", 
                 val, fx, fx, recovered, error);
        
        if (error > 0.005 || error < -0.005) begin
            $display("  WARNING: Large conversion error!");
        end
    endtask
    
    task test_add(real x, real y);
        fixed_t fx, fy, fsum;
        real expected, actual, error;
        
        fx = float_to_fixed(x);
        fy = float_to_fixed(y);
        fsum = fixed_add(fx, fy);
        
        expected = x + y;
        actual = fixed_to_float(fsum);
        error = expected - actual;
        
        $display("%6.3f + %6.3f = %6.3f | Expected: %6.3f | Error: %7.5f %s", 
                 x, y, actual, expected, error,
                 (error > 0.01 || error < -0.01) ? "⚠️" : "✓");
    endtask
    
    task test_mul(real x, real y);
        fixed_t fx, fy, fprod;
        real expected, actual, error;
        
        fx = float_to_fixed(x);
        fy = float_to_fixed(y);
        fprod = fixed_mul(fx, fy);
        
        expected = x * y;
        actual = fixed_to_float(fprod);
        error = expected - actual;
        
        $display("%6.3f × %6.3f = %6.3f | Expected: %6.3f | Error: %7.5f %s", 
                 x, y, actual, expected, error,
                 (error > 0.05 || error < -0.05) ? "⚠️" : "✓");
    endtask

endmodule
