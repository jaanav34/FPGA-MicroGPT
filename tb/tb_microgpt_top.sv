// ===========================================================================
// microGPT Top-Level Testbench
// ===========================================================================
`timescale 1ns/1ps

module tb_microgpt_top;
    import microgpt_pkg::*;

    // --- Declarations at module top ---
    logic        clk;
    logic        rst_n;
    logic        start_gen;
    logic        next_token;
    logic [4:0]  token_out;
    logic        token_valid;
    logic        gen_done;
    
    int          pass_count;
    int          fail_count;
    int          token_count;
    real         start_time;
    real         end_time;
    int         random_delay;

    // --- DUT Instantiation ---
    microgpt_top #(
        .TOP_K(TOP_K),            // Correct parameter mapping
        .TEMP_SHIFT(TEMP_SHIFT)
    ) dut (
        .clk(clk),                // Correct port mapping
        .rst_n(rst_n),
        .start_gen(start_gen),
        .next_token(next_token),
        .token_out(token_out),
        .token_valid(token_valid),
        .gen_done(gen_done)
    );

    // --- Clock Generation ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // --- Helper Tasks (Declarations at top of task) ---
    task reset_system();
        begin
            rst_n = 0;
            start_gen = 0;
            next_token = 0;
            repeat(10) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);
        end
    endtask

    task run_generation();
        // --- task declarations ---
        int max_tokens;
        begin
            max_tokens = 20;
            token_count = 0;
            $display("\n--- Starting Autoregressive Generation ---");
            
          // Inside task run_generation
            @(posedge clk);
            start_gen = 1;
            @(posedge clk);
            start_gen = 0;
            
            // FORCE START WITH 'b' (Token 1) instead of BOS
            // Note: You may need to add a "force" or wait for the IDLE state to transition
            dut.cur_token = 12;
            // Loop until BOS token (gen_done) or limit reached
            while (!gen_done && token_count < max_tokens) begin
                // Wait for the engine to compute the token
                wait(token_valid || gen_done);
                
                if (token_valid) begin
                    $display("  [Pos %0d] Predicted Token ID: %0d (Probable char: %c)", 
                              token_count, token_out, token_out + 97); // Basic char mapping
                    token_count++;
                    // Inside the while loop, when a token is valid:
                    final_name = {final_name, token_out+97}; // Append the character
                    // Request next token
                    @(posedge clk);
                    next_token = 1;
                    @(posedge clk);
                    next_token = 0;
                end
                
                // Safety break to prevent infinite simulation
                if (token_count >= max_tokens) begin
                    $display("WARNING: Generation reached max token limit.");
                    break;
                end
            end
            // ... after the while loop ...
        	$display("--- Generation Complete: %0d tokens generated ---\n", token_count);

        	// Open, Write, Flush, and Close explicitly
        	f_log = $fopen("names_local.csv", "a"); 
        	if (f_log) begin
            	$fdisplay(f_log, "%0d, %0d, %s", TOP_K, TEMP_SHIFT, final_name);
            	$fflush(f_log); // Force write to disk
            	$fclose(f_log);
            	$display("DATA_SAVED: Parameters %0d/%0d, Name: %s", TOP_K, TEMP_SHIFT, final_name);
        	end else begin
            	$display("ERROR: Could not open names_local.csv for writing");
        	end
        end
    endtask

    // --- Main Simulation Sequence ---
    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("========================================");
        $display("  microGPT Full Engine Verification");
        $display("========================================");

        reset_system();
        // random delay before starting generation ---
        random_delay = $urandom_range(10, 100); 
        $display("Applying initial entropy delay: %0d cycles", random_delay);
        repeat(random_delay) @(posedge clk);
        // Start timer
        start_time = $realtime;
        
        run_generation();
        
        end_time = $realtime;
        
        // Final Status
        if (token_count > 0) begin
            $display("SUCCESS: Engine completed inference in %0.2f ns", end_time - start_time);
            pass_count++;
        end else begin
            $display("ERROR: No tokens were generated.");
            fail_count++;
        end

        $display("\nSummary: %0d Passed, %0d Failed", pass_count, fail_count);
        $finish;
    end

endmodule
