# run_batch.tcl
set total_runs 500
set master_csv "all_generated_names.csv"

# Construct simulation directory path
set project_dir [get_property DIRECTORY [current_project]]
set project_name [get_property NAME [current_project]]
set sim_dir "$project_dir/${project_name}.sim/sim_1/behav/xsim"

# Initialize Master CSV header at root
if {![file exists $master_csv]} {
    set f [open $master_csv w]
    puts $f "TopK, TempShift, Name"
    close $f
}

for {set i 1} {$i <= $total_runs} {incr i} {
    set k_val [expr {int(rand() * 9) + 2}] 
    set t_val [expr {int(rand() * 3)}]     
    
    puts "========================================"
    puts " RUN $i / $total_runs: TopK=$k_val, Temp=$t_val"
    puts "========================================"

    # 1. Setup parameters and seed
    set_property generic "TOP_K=$k_val TEMP_SHIFT=$t_val" [get_filesets sim_1]
    set_property -name {xsim.simulate.xsim.more_options} -value {-sv_seed random} -objects [get_filesets sim_1]

    # 2. Launch and Run
    launch_simulation
    run 50 ms
    
    # 3. Retrieve and Aggregate Data
    set local_file "$sim_dir/names_local.csv"
    if {[file exists $local_file]} {
        set fin [open $local_file r]
        set row_data [read $fin]
        close $fin
        
        set fout [open $master_csv a]
        puts -nonewline $fout $row_data
        close $fout
        puts "SUCCESS: Aggregated Run $i into $master_csv"
    } else {
        puts "ERROR: Could not find result file at $local_file"
    }

    # 4. Cleanup
    close_sim
}