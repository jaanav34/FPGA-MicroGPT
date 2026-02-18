# run_batch.tcl
set total_runs 500
set output_file "all_generated_names.csv"

# Initialize CSV header if it doesn't exist
if {![file exists $output_file]} {
    set f [open $output_file w]
    puts $f "TopK, TempShift, Name"
    close $f
}

for {set i 1} {$i <= $total_runs} {incr i} {
    # Generate random parameters for this run
    set k_val [expr {int(rand() * 9) + 2}] ;# Top-K between 2 and 10
    set t_val [expr {int(rand() * 3)}]     ;# Temp Shift between 0 and 2
    
    puts "========================================"
    puts " RUN $i / $total_runs: TopK=$k_val, Temp=$t_val"
    puts "========================================"

    # 1. Launch simulation with parameter overrides and random seed
    # -generic overrides the parameters we added in Step 1
    launch_simulation -generic "TOP_K=$k_val" -generic "TEMP_SHIFT=$t_val" -sv_seed random
    
    # 2. Run for 50ms (or until $finish is hit in TB)
    run 50 ms
    
    # 3. Close the simulation to free memory for the next run
    close_sim
}

puts "DONE! All names collected in $output_file"