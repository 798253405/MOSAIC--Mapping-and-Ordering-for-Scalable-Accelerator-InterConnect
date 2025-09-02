#!/bin/bash

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base output directory with timestamp
BASE_OUTPUT_DIR="output/batchCNN_serial_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

# Also create a symlink to latest results
ln -sfn "batchCNN_serial_${TIMESTAMP}" "output/batchCNN_serial_latest"

# Define NoC sizes and their names
declare -a NOC_SIZES=("MemNode2_4X4" "MemNode4_4X4" "MemNode4_8X8" "MemNode4_16X16" "MemNode4_32X32")
declare -a NOC_NAMES=("2_4x4" "4_4x4" "4_8x8" "4_16x16" "4_32x32")

# Define test cases
declare -a TEST_CASES=("case1_default" "case2_samos" "case3_affiliatedordering" "case4_seperratedordering" "case5_MOSAIC1" "case6_MOSAIC2")

# Get CNN model filename
CNN_MODEL=$(grep "DEFAULT_NNMODEL_FILENAME" src/parameters.hpp | grep -v "//" | head -1 | cut -d'"' -f2)
CNN_MODEL_NAME=$(basename "$CNN_MODEL")

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup

echo "=========================================="
echo "Starting CNN Batch Tests V2"
echo "CNN Model: $CNN_MODEL_NAME"
echo "Total configurations: ${#NOC_SIZES[@]} NoC sizes × ${#TEST_CASES[@]} test cases = $((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))"
echo "=========================================="
echo ""

# Counter for progress
total_tests=$((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))
current_test=0

# Summary file
SUMMARY_FILE="$BASE_OUTPUT_DIR/summary_all.txt"
echo "NoC_Size,Test_Case,Status,Total_Cycles,Packets,Flits,Avg_Hops,BitTrans_Float,BitTrans_Fixed" > $SUMMARY_FILE

# Run all combinations
for i in "${!NOC_SIZES[@]}"; do
    noc_size="${NOC_SIZES[$i]}"
    noc_name="${NOC_NAMES[$i]}"
    
    # Create directory for this NoC size
    NOC_DIR="$BASE_OUTPUT_DIR/$noc_name"
    mkdir -p "$NOC_DIR"
    
    for test_case in "${TEST_CASES[@]}"; do
        current_test=$((current_test + 1))
        output_file="${NOC_DIR}/${test_case}.txt"
        
        echo "[$current_test/$total_tests] Running: NoC=$noc_name, Case=$test_case"
        echo "  Output: $output_file"
        
        # Create modified parameters.hpp
        cat src/parameters.hpp.backup | \
            sed -e '/^#define MemNode/s/^#define/\/\/#define/' \
            -e '/^\/\/#define '"$noc_size"'/s/^\/\///' \
            -e '/^#define case[0-9]/s/^#define/\/\/#define/' \
            -e '/^\/\/#define '"$test_case"'/s/^\/\///' > src/parameters.hpp
        
        # Add configuration info to output file
        echo "===========================================" > "$output_file"
        echo "CONFIGURATION INFO" >> "$output_file"
        echo "===========================================" >> "$output_file"
        echo "NoC Size: $noc_name ($noc_size)" >> "$output_file"
        echo "Test Case: $test_case" >> "$output_file"
        echo "CNN Model: $CNN_MODEL_NAME" >> "$output_file"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
        echo "===========================================" >> "$output_file"
        echo "" >> "$output_file"
        
        # Compile
        echo "  Compiling..."
        cd Debug
        make all > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            echo "  ERROR: Compilation failed!"
            echo "ERROR: Compilation failed" >> "../$output_file"
            cd ..
            echo "$noc_name,$test_case,COMPILE_ERROR,0,0,0,0,0,0" >> $SUMMARY_FILE
            continue
        fi
        
        # Run the simulation WITHOUT timeout
        echo "  Running simulation (no timeout - may take long for large networks)..."
        START_TIME=$(date +%s)
        ./2508date >> "../$output_file" 2>&1
        SIMULATION_EXIT_CODE=$?
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        
        cd ..
        
        # Add footer with configuration and runtime
        echo "" >> "$output_file"
        echo "===========================================" >> "$output_file"
        echo "SIMULATION COMPLETED" >> "$output_file"
        echo "Runtime: $RUNTIME seconds" >> "$output_file"
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
        echo "NoC: $noc_name | Case: $test_case | Model: $CNN_MODEL_NAME" >> "$output_file"
        echo "===========================================" >> "$output_file"
        
        # Extract BATCH_STATS if available
        BATCH_STATS=$(grep "BATCH_STATS:" "$output_file" | tail -1)
        if [ ! -z "$BATCH_STATS" ]; then
            # Parse stats
            total_cycles=$(echo "$BATCH_STATS" | grep -oP 'total_cycles=\K[0-9]+' || echo "0")
            packet_id=$(echo "$BATCH_STATS" | grep -oP 'packetid=\K[0-9]+' || echo "0")
            flit_id=$(echo "$BATCH_STATS" | grep -oP 'YZGlobalFlit_id=\K[0-9]+' || echo "0")
            avg_hops=$(echo "$BATCH_STATS" | grep -oP 'avg_hops_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
            bit_trans_float=$(echo "$BATCH_STATS" | grep -oP 'bit_transition_float_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
            bit_trans_fixed=$(echo "$BATCH_STATS" | grep -oP 'bit_transition_fixed_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
            
            echo "$noc_name,$test_case,SUCCESS,$total_cycles,$packet_id,$flit_id,$avg_hops,$bit_trans_float,$bit_trans_fixed" >> $SUMMARY_FILE
            echo "  Success! Cycles: $total_cycles, Runtime: ${RUNTIME}s"
        else
            echo "$noc_name,$test_case,NO_STATS,0,0,0,0,0,0" >> $SUMMARY_FILE
            echo "  Warning: No BATCH_STATS found, Runtime: ${RUNTIME}s"
        fi
        
        echo ""
    done
    
    echo "Completed all tests for $noc_name"
    echo "----------------------------------------"
done

# Restore original parameters.hpp
mv src/parameters.hpp.backup src/parameters.hpp

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "Directory structure:"
echo "  $BASE_OUTPUT_DIR/"
for noc_name in "${NOC_NAMES[@]}"; do
    echo "    ├── $noc_name/"
    echo "    │   ├── case1_default.txt"
    echo "    │   ├── case2_samos.txt"
    echo "    │   ├── ..."
done
echo "    └── summary_all.txt"
echo "=========================================="