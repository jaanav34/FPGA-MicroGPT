# Vivado TCL Script - Quick Test Setup
# Run this in Vivado TCL console to quickly set up tests

proc setup_test {test_name} {
    puts "Setting up test: $test_name"
    
    # Remove all simulation sources
    remove_files -fileset sim_1 [get_files -of_objects [get_filesets sim_1]]
    
    # Add package (always needed)
    add_files -fileset sim_1 -norecurse rtl/microgpt_pkg.sv
    
    switch $test_name {
        "fixed_point" {
            add_files -fileset sim_1 -norecurse tb/tb_fixed_point.sv
            set_property top tb_fixed_point [get_filesets sim_1]
        }
        "vector_dot" {
            add_files -fileset sim_1 -norecurse rtl/vector_dot_product.sv
            add_files -fileset sim_1 -norecurse tb/tb_vector_dot_product.sv
            set_property top tb_vector_dot_product [get_filesets sim_1]
        }
        "param_mem" {
            add_files -fileset sim_1 -norecurse rtl/param_memory.sv
            add_files -fileset sim_1 -norecurse tb/tb_param_memory.sv
            set_property top tb_param_memory [get_filesets sim_1]
        }
        "matrix_vector" {
            add_files -fileset sim_1 -norecurse rtl/vector_dot_product.sv
            add_files -fileset sim_1 -norecurse rtl/matrix_vector_mult.sv
            add_files -fileset sim_1 -norecurse tb/tb_matrix_vector_mult.sv
            set_property top tb_matrix_vector_mult [get_filesets sim_1]
        }
        "rmsnorm" {
            add_files -fileset sim_1 -norecurse rtl/rmsnorm.sv
            add_files -fileset sim_1 -norecurse tb/tb_rmsnorm.sv
            set_property top tb_rmsnorm [get_filesets sim_1]
        }
        "softmax" {
            add_files -fileset sim_1 -norecurse rtl/softmax.sv
            add_files -fileset sim_1 -norecurse tb/tb_softmax.sv
            set_property top tb_softmax [get_filesets sim_1]
        }
        default {
            puts "ERROR: Unknown test '$test_name'"
            puts "Available tests:"
            puts "  LEVEL 0-1: fixed_point, vector_dot, param_mem"
            puts "  LEVEL 2:   matrix_vector, rmsnorm, softmax"
            return
        }
    }
    
    # Update compile order
    update_compile_order -fileset sim_1
    
    puts "Test setup complete. Run with: launch_simulation"
}

# Example usage:
# setup_test fixed_point
# launch_simulation

puts "Test setup commands loaded!"
puts "Usage: setup_test <test_name>"
puts ""
puts "Available Tests:"
puts "  LEVEL 0-1 (Foundation):"
puts "    fixed_point   - Q8.8 arithmetic verification"
puts "    vector_dot    - Dot product computation"
puts "    param_mem     - Parameter memory read/write"
puts ""
puts "  LEVEL 2 (Math Operations):"
puts "    matrix_vector - Matrix-vector multiplication"
puts "    rmsnorm       - RMS normalization"
puts "    softmax       - Softmax with temperature"
puts ""
puts "Example: setup_test matrix_vector"
puts "         launch_simulation"