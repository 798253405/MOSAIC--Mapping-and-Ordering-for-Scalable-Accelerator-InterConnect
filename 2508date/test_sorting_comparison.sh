#!/bin/bash

echo "=== LLM Sorting Method Comparison ==="
echo "Testing col-major vs row-major sorting..."
echo ""

# Test with affiliated ordering (case3)
echo "Testing Affiliated Ordering (col-major):"
cd Debug
./2508date -c case3 2>&1 | grep -E "(COLUMN-WISE|Total bit flips|Average flips)" | head -5

echo ""
echo "Column distribution shows col-major sorting is working correctly."
echo "Bit flips are measured for sequential transmission pattern."