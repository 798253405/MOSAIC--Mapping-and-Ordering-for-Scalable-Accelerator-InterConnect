#!/bin/bash

echo "=== LLM Bit Flip Reduction Test Suite ==="
echo "Testing all ordering configurations..."
echo ""

# Function to compile with specific case
compile_case() {
    local case_name=$1
    local define_flag=$2
    echo "Compiling $case_name..."
    
    # Clean build
    make clean > /dev/null 2>&1
    
    # Compile with the specific case flag
    if [ -z "$define_flag" ]; then
        # Baseline - no special flags
        g++ -O0 -g3 -Wall -c ../src/llmmacnet.cpp -o src/llmmacnet.o 2>/dev/null
    else
        # With ordering flag
        g++ -O0 -g3 -Wall -D$define_flag -c ../src/llmmacnet.cpp -o src/llmmacnet.o 2>/dev/null
    fi
    
    # Rebuild everything
    make all -j8 > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Compiled successfully"
    else
        echo "  ✗ Compilation failed"
        return 1
    fi
    return 0
}

# Function to run test and extract results
run_test() {
    local case_name=$1
    echo "Running $case_name test..."
    
    # Run and extract key metrics
    local output=$(./2508date -c $case_name 2>&1)
    
    # Extract ordering type
    local ordering=$(echo "$output" | grep -E "(NO ORDERING|AFFILIATED|SEPARATED)" | head -1)
    
    # Extract bit flip data
    local bit_flips=$(echo "$output" | grep "Total bit flips during transmission" | head -1 | awk '{print $7}')
    
    # Extract performance metrics
    local avg_cycles=$(echo "$output" | grep "Total: Avg=" | awk -F'Avg=' '{print $2}' | awk '{print $1}')
    
    echo "  Ordering: $ordering"
    echo "  Bit flips per task: $bit_flips"
    echo "  Avg cycles: $avg_cycles"
    echo ""
}

# Test 1: Baseline (no ordering)
echo "=== TEST 1: BASELINE (No Ordering) ==="
compile_case "Baseline" ""
if [ $? -eq 0 ]; then
    run_test "baseline"
fi

# Test 2: Affiliated Ordering
echo "=== TEST 2: AFFILIATED ORDERING ==="
compile_case "Affiliated" "case3_affiliatedordering"
if [ $? -eq 0 ]; then
    run_test "case3"
fi

# Test 3: Separated Ordering
echo "=== TEST 3: SEPARATED ORDERING ==="
compile_case "Separated" "case4_seperratedordering"
if [ $? -eq 0 ]; then
    run_test "case4"
fi

echo "=== Test Suite Complete ==="