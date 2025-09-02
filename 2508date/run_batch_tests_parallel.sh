#!/bin/bash

# Pipeline parallel batch testing script
# Strategy: Sequential compilation, parallel execution

# ========== CONFIGURATION ==========
# Maximum number of parallel execution slots
MAX_PARALLEL_SLOTS=6  # Change this value to adjust parallel execution limit

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base output directory with timestamp
BASE_OUTPUT_DIR="output/batchCNN_parallel_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

# Create symlink to latest results
ln -sfn "batchCNN_parallel_${TIMESTAMP}" "output/batchCNN_parallel_latest"

# Define NoC sizes and their names
declare -a NOC_SIZES=("MemNode2_4X4" "MemNode4_4X4" "MemNode4_8X8" "MemNode4_16X16" "MemNode4_32X32")
declare -a NOC_NAMES=("2_4x4" "4_4x4" "4_8x8" "4_16x16" "4_32x32")

# Define test cases
declare -a TEST_CASES=("case1_default" "case2_samos" "case3_affiliatedordering" "case4_seperratedordering" "case5_MOSAIC1" "case6_MOSAIC2")

# Get CNN model filename
CNN_MODEL=$(grep "DEFAULT_NNMODEL_FILENAME" src/parameters.hpp | grep -v "//" | head -1 | cut -d'"' -f2)
CNN_MODEL_NAME=$(basename "$CNN_MODEL")

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup_pipeline

# Create compilation directory
COMPILE_DIR="${BASE_OUTPUT_DIR}/compile_workspace"
mkdir -p "$COMPILE_DIR"
cp -r src "$COMPILE_DIR/src"
cp -r Debug "$COMPILE_DIR/Debug"

# Create execution directories for parallel runs
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${i}"
    mkdir -p "$EXEC_DIR/Debug"
done

echo "==========================================="
echo "Starting Pipeline Parallel CNN Batch Tests"
echo "CNN Model: $CNN_MODEL_NAME"
echo "Total configurations: ${#NOC_SIZES[@]} NoC sizes × ${#TEST_CASES[@]} test cases = $((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))"
echo "Strategy: Sequential compilation, parallel execution (max $MAX_PARALLEL_SLOTS)"
echo "==========================================="
echo ""

# Initialize summary file
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "NoC_Size,Test_Case,Status,Total_Cycles,Packets,Flits,Avg_Hops,BitTrans_Float,BitTrans_Fixed,Runtime" > $SUMMARY_FILE

# Job counter
TOTAL_JOBS=$((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))
COMPILED_COUNT=0
RUNNING_COUNT=0
COMPLETED_COUNT=0

# Function to compile a configuration
compile_config() {
    local noc_size=$1
    local noc_name=$2
    local test_case=$3
    local job_id=$4
    
    # Create output directory for this NoC configuration
    local NOC_DIR="${BASE_OUTPUT_DIR}/${noc_name}"
    mkdir -p "$NOC_DIR"
    
    echo "[Compile][$job_id/$TOTAL_JOBS] Compiling: NoC=$noc_name, Case=$test_case"
    
    # Create modified parameters.hpp
    cat src/parameters.hpp.backup_pipeline | \
        sed -e '/^#define MemNode/s/^#define/\/\/#define/' \
        -e '/^\/\/#define '"$noc_size"'/s/^\/\///' \
        -e '/^#define case[0-9]/s/^#define/\/\/#define/' \
        -e '/^\/\/#define '"$test_case"'/s/^\/\///' > "$COMPILE_DIR/src/parameters.hpp"
    
    # Compile
    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > /dev/null 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null
    
    if [ $COMPILE_RESULT -eq 0 ]; then
        # Save compiled binary with unique name
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success: NoC=$noc_name, Case=$test_case"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED: NoC=$noc_name, Case=$test_case"
        echo "$noc_name,$test_case,COMPILE_ERROR,0,0,0,0,0,0,0" >> $SUMMARY_FILE
        return 1
    fi
}

# Function to run a test (executed in background)
run_test() {
    local noc_name=$1
    local test_case=$2
    local job_id=$3
    local exec_slot=$4
    
    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local NOC_DIR="${BASE_OUTPUT_DIR}/${noc_name}"
    mkdir -p "$NOC_DIR"  # Ensure directory exists
    local output_file="$(realpath ${NOC_DIR}/${test_case}.txt)"  # Use absolute path
    
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Starting: NoC=$noc_name, Case=$test_case"
    
    # Copy binary to execution directory
    cp "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}" "$EXEC_DIR/Debug/2508date"
    
    # Add header to output file
    echo "===========================================" > "$output_file"
    echo "CONFIGURATION INFO" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "NoC Size: $noc_name" >> "$output_file"
    echo "Test Case: $test_case" >> "$output_file"
    echo "CNN Model: $CNN_MODEL_NAME" >> "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "" >> "$output_file"
    
    # Run simulation (no timeout limit)
    START_TIME=$(date +%s)
    cd "$EXEC_DIR/Debug"
    ./2508date >> "$output_file" 2>&1
    cd - > /dev/null
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))
    
    # Extract BATCH_STATS
    BATCH_STATS=$(grep "BATCH_STATS:" "$output_file" | tail -1)
    if [ ! -z "$BATCH_STATS" ]; then
        total_cycles=$(echo "$BATCH_STATS" | grep -oP 'total_cycles=\K[0-9]+' || echo "0")
        packet_id=$(echo "$BATCH_STATS" | grep -oP 'packetid=\K[0-9]+' || echo "0")
        flit_id=$(echo "$BATCH_STATS" | grep -oP 'YZGlobalFlit_id=\K[0-9]+' || echo "0")
        avg_hops=$(echo "$BATCH_STATS" | grep -oP 'avg_hops_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
        bit_trans_float=$(echo "$BATCH_STATS" | grep -oP 'bit_transition_float_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
        bit_trans_fixed=$(echo "$BATCH_STATS" | grep -oP 'bit_transition_fixed_per_flit=\K[0-9]+\.?[0-9]*' || echo "0")
        
        echo "$noc_name,$test_case,SUCCESS,$total_cycles,$packet_id,$flit_id,$avg_hops,$bit_trans_float,$bit_trans_fixed,$RUNTIME" >> $SUMMARY_FILE
        echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Completed: Cycles=$total_cycles, Runtime=${RUNTIME}s"
    else
        echo "$noc_name,$test_case,NO_STATS,0,0,0,0,0,0,$RUNTIME" >> $SUMMARY_FILE
        echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Completed: No BATCH_STATS, Runtime=${RUNTIME}s"
    fi
    
    # Clean up binary
    rm -f "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}"
    
    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
}

# Main pipeline loop
echo "Starting compilation and execution pipeline..."
echo ""

# Arrays to track running jobs
declare -a RUNNING_PIDS
declare -a RUNNING_SLOTS
declare -a RUNNING_NAMES

# Initialize slots as available
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

# Job ID counter
JOB_ID=0

# Compile and launch jobs
for i in "${!NOC_SIZES[@]}"; do
    noc_size="${NOC_SIZES[$i]}"
    noc_name="${NOC_NAMES[$i]}"
    
    for test_case in "${TEST_CASES[@]}"; do
        JOB_ID=$((JOB_ID + 1))
        
        # Compile the configuration
        compile_config "$noc_size" "$noc_name" "$test_case" "$JOB_ID"
        
        if [ $? -eq 0 ]; then
            # Wait for an available execution slot
            while true; do
                # Check for completed jobs
                for ((slot=1; slot<=MAX_PARALLEL_SLOTS; slot++)); do
                    if [ "${RUNNING_SLOTS[$slot]}" -ne 0 ]; then
                        # Check if this job is still running
                        if ! kill -0 "${RUNNING_SLOTS[$slot]}" 2>/dev/null; then
                            # Job completed, slot is free
                            wait "${RUNNING_SLOTS[$slot]}" 2>/dev/null
                            echo "[Pipeline] Slot $slot is now free"
                            RUNNING_SLOTS[$slot]=0
                        fi
                    fi
                done
                
                # Find a free slot
                FREE_SLOT=0
                for ((slot=1; slot<=MAX_PARALLEL_SLOTS; slot++)); do
                    if [ "${RUNNING_SLOTS[$slot]}" -eq 0 ]; then
                        FREE_SLOT=$slot
                        break
                    fi
                done
                
                if [ $FREE_SLOT -ne 0 ]; then
                    # Launch the test in background
                    run_test "$noc_name" "$test_case" "$JOB_ID" "$FREE_SLOT" &
                    RUNNING_SLOTS[$FREE_SLOT]=$!
                    RUNNING_NAMES[$FREE_SLOT]="${noc_name}_${test_case}"
                    echo "[Pipeline] Launched test in slot $FREE_SLOT (PID: ${RUNNING_SLOTS[$FREE_SLOT]})"
                    break
                else
                    # All slots busy, wait a bit
                    sleep 2
                fi
            done
        fi
        
        # Progress report
        COMPLETED=$(grep -c "SUCCESS\|NO_STATS" "$SUMMARY_FILE" 2>/dev/null || echo "0")
        RUNNING=$(jobs -r | wc -l)
        echo "[Progress] Compiled: $JOB_ID/$TOTAL_JOBS | Running: $RUNNING | Completed: $COMPLETED/$TOTAL_JOBS"
        echo ""
    done
done

# Wait for all remaining jobs to complete
echo "Waiting for all running tests to complete..."
for ((slot=1; slot<=MAX_PARALLEL_SLOTS; slot++)); do
    if [ "${RUNNING_SLOTS[$slot]}" -ne 0 ]; then
        wait "${RUNNING_SLOTS[$slot]}" 2>/dev/null
    fi
done

# Cleanup
rm -rf "$COMPILE_DIR"
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    rm -rf "${BASE_OUTPUT_DIR}/exec_${i}"
done

# Restore original parameters.hpp
mv src/parameters.hpp.backup_pipeline src/parameters.hpp

echo ""
echo "==========================================="
echo "All tests completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""

# Final statistics
echo "Final Statistics:"
SUCCESS_COUNT=$(grep -c "SUCCESS" "$SUMMARY_FILE" 2>/dev/null || echo "0")
FAILED_COUNT=$(grep -c "ERROR\|NO_STATS" "$SUMMARY_FILE" 2>/dev/null || echo "0")
echo "  Successful tests: $SUCCESS_COUNT"
echo "  Failed tests: $FAILED_COUNT"
echo ""

echo "Performance Summary:"
echo "  Top 5 best power reduction (lowest BitTrans_Float):"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -n | head -5 | while IFS=',' read -r noc test status cycles packets flits hops float fixed runtime; do
    printf "    %s/%s: %.2f bit flips/flit\n" "$noc" "$test" "$float"
done
echo ""
echo "  Top 5 worst power (highest BitTrans_Float):"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -rn | head -5 | while IFS=',' read -r noc test status cycles packets flits hops float fixed runtime; do
    printf "    %s/%s: %.2f bit flips/flit\n" "$noc" "$test" "$float"
done
echo "==========================================="