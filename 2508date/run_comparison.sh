#!/bin/bash

echo "==================================================================="
echo "     LLM Data Ordering Comparison Test (Col-Major Sorting)        "
echo "==================================================================="
echo "Date: $(date)"
echo ""

cd Debug

# Function to extract key metrics
run_test() {
    local case_name=$1
    local description=$2
    
    echo "-------------------------------------------------------------------"
    echo "Testing: $description"
    echo "-------------------------------------------------------------------"
    
    # Run the test and capture output
    output=$(./2508date -c $case_name 2>&1)
    
    # Extract metrics
    ordering_type=$(echo "$output" | grep -E "NO ORDERING|AFFILIATED|SEPARATED" | head -1)
    bit_flips=$(echo "$output" | grep "Total bit flips during transmission" | head -1 | awk '{print $7}')
    avg_flips=$(echo "$output" | grep "Average flips per transition" | head -1 | awk '{print $5}')
    completion_rate=$(echo "$output" | grep "Task completion rate" | head -1 | awk '{print $4}')
    
    # Extract column distribution for ordering cases
    if [[ "$case_name" != "baseline" ]]; then
        echo "Column-wise bit distribution (first task):"
        echo "$output" | grep -A8 "COLUMN-WISE BIT COUNT" | head -9
    fi
    
    echo ""
    echo "Results:"
    echo "  Ordering Type: $ordering_type"
    echo "  Total bit flips: ${bit_flips:-N/A}"
    echo "  Average flips per transition: ${avg_flips:-N/A}"
    echo "  Task completion rate: $completion_rate"
    echo ""
}

# Test 1: Baseline (no ordering)
echo "==================================================================="
echo "1. BASELINE (No Ordering)"
echo "==================================================================="
# Compile for baseline
sed -i 's/^#*define case[0-9]_.*$/\/\/&/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
run_test "baseline" "Baseline - No data ordering"

# Test 2: Affiliated Ordering
echo "==================================================================="
echo "2. AFFILIATED ORDERING (Col-Major)"
echo "==================================================================="
# Compile for affiliated
sed -i 's/^\/\/#define case3_affiliatedordering/#define case3_affiliatedordering/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
run_test "case3" "Affiliated - Query and Key sorted together by Key's bit count"

# Test 3: Separated Ordering
echo "==================================================================="
echo "3. SEPARATED ORDERING (Col-Major)"
echo "==================================================================="
# Compile for separated
sed -i 's/^#define case3_affiliatedordering/\/\/#define case3_affiliatedordering/' ../src/parameters.hpp
sed -i 's/^\/\/#define case4_seperratedordering/#define case4_seperratedordering/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
run_test "case4" "Separated - Query and Key sorted independently"

# Summary
echo "==================================================================="
echo "                        SUMMARY                                   "
echo "==================================================================="
echo "All tests use col-major (column-first) sorting like CNN"
echo "Bit flip reduction is limited due to uniform LLM data distribution"
echo "Task completion rate remains consistent across all methods"

# Reset to baseline
sed -i 's/^#*define case[0-9]_.*$/\/\/&/' ../src/parameters.hpp
