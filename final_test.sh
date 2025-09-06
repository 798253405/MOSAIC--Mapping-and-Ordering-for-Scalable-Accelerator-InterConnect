#!/bin/bash

echo "=== LLM Bit Flip Reduction Final Test Results ==="
echo "Date: $(date)"
echo ""

# Test 1: Baseline
echo "1. BASELINE (No Ordering):"
echo "   Compiling..."
sed -i 's/^#*define case[0-9]_.*$/\/\/&/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
echo "   Running test..."
./2508date -c baseline 2>&1 | grep -E "(NO ORDERING|Total LLM cycles|Average flips|Total:)" | head -3
echo ""

# Test 2: Affiliated Ordering  
echo "2. AFFILIATED ORDERING:"
echo "   Compiling..."
sed -i 's/^\/\/#define case3_affiliatedordering/#define case3_affiliatedordering/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
echo "   Running test..."
./2508date -c case3 2>&1 | grep -E "(AFFILIATED|Total LLM cycles|Average flips|Total:)" | head -3
echo ""

# Test 3: Separated Ordering
echo "3. SEPARATED ORDERING:"
echo "   Compiling..."
sed -i 's/^#define case3_affiliatedordering/\/\/#define case3_affiliatedordering/' ../src/parameters.hpp
sed -i 's/^\/\/#define case4_seperratedordering/#define case4_seperratedordering/' ../src/parameters.hpp
make clean > /dev/null 2>&1 && make all -j8 > /dev/null 2>&1
echo "   Running test..."
./2508date -c case4 2>&1 | grep -E "(SEPARATED|Total LLM cycles|Average flips|Total:)" | head -3
echo ""

# Reset to baseline
sed -i 's/^#*define case[0-9]_.*$/\/\/&/' ../src/parameters.hpp

echo "=== Test Complete ==="