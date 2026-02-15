#!/bin/bash
# Complete Test Runner for microGPT FPGA
# Run all tests in order and report results

set -e

echo "=========================================="
echo "microGPT FPGA - Complete Test Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a single test
run_test() {
    local test_name=$1
    local test_file=$2
    local deps=$3
    
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Create temporary simulation directory
    mkdir -p sim_temp
    cd sim_temp
    
    # Copy required files
    cp ../rtl/microgpt_pkg.sv .
    for dep in $deps; do
        cp ../rtl/$dep .
    done
    cp ../tb/$test_file .
    
    # Run with iverilog (if available)
    if command -v iverilog &> /dev/null; then
        echo "Compiling with iverilog..."
        iverilog -g2012 -o test.vvp microgpt_pkg.sv $deps $test_file
        
        echo "Running simulation..."
        vvp test.vvp > test_output.log 2>&1
        
        # Check results
        if grep -q "ALL TESTS PASSED" test_output.log; then
            echo -e "${GREEN}✓ PASS${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            cat test_output.log
        else
            echo -e "${RED}✗ FAIL${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            cat test_output.log
        fi
    else
        echo -e "${YELLOW}⚠ Skipped (iverilog not found)${NC}"
        echo "  Run manually in Vivado"
    fi
    
    cd ..
    rm -rf sim_temp
    echo ""
}

# Level 0 Tests
echo "=========================================="
echo "LEVEL 0: Fundamental Tests"
echo "=========================================="
echo ""

run_test "Fixed-Point Arithmetic" "tb_fixed_point.sv" ""

# Level 1 Tests  
echo "=========================================="
echo "LEVEL 1: Basic Building Blocks"
echo "=========================================="
echo ""

run_test "Vector Dot Product" "tb_vector_dot_product.sv" "vector_dot_product.sv"
run_test "Parameter Memory" "tb_param_memory.sv" "param_memory.sv"

# Level 2 Tests
echo "=========================================="
echo "LEVEL 2: Math Operations"
echo "=========================================="
echo ""

run_test "Matrix-Vector Multiply" "tb_matrix_vector_mult.sv" "vector_dot_product.sv matrix_vector_mult.sv"
run_test "RMS Normalization" "tb_rmsnorm.sv" "rmsnorm.sv"
run_test "Softmax" "tb_softmax.sv" "softmax.sv"

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo ""
    echo "You have a solid foundation. Ready to build higher-level components!"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED!${NC}"
    echo ""
    echo "Fix failing tests before proceeding to next level."
    exit 1
fi
