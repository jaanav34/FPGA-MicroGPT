`timescale 1ns/1ps

module tb_param_memory;
    import microgpt_pkg::*;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // DUT signals
    logic wr_en;
    logic [PARAM_ADDR_WIDTH-1:0] wr_addr;
    fixed_t wr_data;
    logic rd_en;
    logic [PARAM_ADDR_WIDTH-1:0] rd_addr;
    fixed_t rd_data;
    logic rd_valid;
    
    // Test tracking
    int pass_count;
    int fail_count;
    
    // Instantiate DUT
    param_memory dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data),
        .rd_valid(rd_valid)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Main test
    initial begin
        $display("==============================================");
        $display("Parameter Memory Test");
        $display("Total params: %0d", TOTAL_PARAMS);
        $display("==============================================\n");
        
        // Initialize
        rst_n = 0;
        wr_en = 0;
        wr_addr = 0;
        wr_data = '0;
        rd_en = 0;
        rd_addr = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Reset
        #20;
        rst_n = 1;
        #20;
        
        // Test 1: Write and read single value
        $display("TEST 1: Write and read single value");
        test_write_read(0, 1.5);
        
        // Test 2: Write and read multiple values
        $display("\nTEST 2: Write and read multiple sequential values");
        for (int i = 0; i < 10; i++) begin
            test_write_read(i, real'(i) * 0.5);
        end
        
        // Test 3: Write to various addresses
        $display("\nTEST 3: Write to various addresses");
        test_write_read(100, 2.25);
        test_write_read(500, -1.75);
        test_write_read(1000, 0.125);
        
        // Test 4: Overwrite existing value
        $display("\nTEST 4: Overwrite existing value");
        test_write_read(0, 3.5);
        test_write_read(0, -2.25);  // Overwrite
        
        // Test 5: Read without prior write (should be zero)
        $display("\nTEST 5: Read uninitialized location (should be 0)");
        test_read_only(2000, 0.0);
        
        // Test 6: Sequential write/read
        $display("\nTEST 6: Sequential write pattern");
        $display("  Writing pattern...");
        for (int i = 0; i < 16; i++) begin
            @(posedge clk);
            wr_en = 1;
            wr_addr = i;
            wr_data = float_to_fixed(real'(i) * 0.1);
        end
        @(posedge clk);
        wr_en = 0;
        
        $display("  Reading back pattern...");
        for (int i = 0; i < 16; i++) begin
            test_read_only(i, real'(i) * 0.1);
        end
        
        // Summary
        $display("\n==============================================");
        $display("Test Summary:");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        if (fail_count == 0) begin
            $display("\n✓ ALL TESTS PASSED!");
        end else begin
            $display("\n✗ SOME TESTS FAILED!");
        end
        $display("==============================================");
        
        #100;
        $finish;
    end
    
    // Task to write and immediately read
    task test_write_read(int addr, real value);
        real read_back;
        real error;
        
        // Write
        @(posedge clk);
        wr_en = 1;
        wr_addr = addr;
        wr_data = float_to_fixed(value);
        
        @(posedge clk);
        wr_en = 0;
        
        // Read
        @(posedge clk);
        rd_en = 1;
        rd_addr = addr;
        
        @(posedge clk);
        wait(rd_valid == 1);
        
        read_back = fixed_to_float(rd_data);
        error = value - read_back;
        
        $display("  Addr %4d: Wrote %7.4f, Read %7.4f, Error %8.6f", 
                 addr, value, read_back, error);
        
        if (error < 0.01 && error > -0.01) begin
            pass_count++;
        end else begin
            $display("    FAIL: Error too large!");
            fail_count++;
        end
        
        @(posedge clk);
        rd_en = 0;
        repeat(2) @(posedge clk);
    endtask
    
    // Task to read only
    task test_read_only(int addr, real expected);
        real read_back;
        real error;
        
        @(posedge clk);
        rd_en = 1;
        rd_addr = addr;
        
        @(posedge clk);
        wait(rd_valid == 1);
        
        read_back = fixed_to_float(rd_data);
        error = expected - read_back;
        
        $display("  Addr %4d: Expected %7.4f, Read %7.4f, Error %8.6f", 
                 addr, expected, read_back, error);
        
        if (error < 0.01 && error > -0.01) begin
            pass_count++;
        end else begin
            $display("    FAIL: Error too large!");
            fail_count++;
        end
        
        @(posedge clk);
        rd_en = 0;
        repeat(2) @(posedge clk);
    endtask
    
    // Timeout watchdog
    initial begin
        #100000;
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule
