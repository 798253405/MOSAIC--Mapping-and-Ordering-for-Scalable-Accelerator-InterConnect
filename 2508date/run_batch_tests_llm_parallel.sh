#!/bin/bash

# LLM Pipeline parallel batch testing script
# Based on CNN batch script, adapted for LLM testing

# ========== CONFIGURATION ==========
# Maximum number of parallel execution slots
MAX_PARALLEL_SLOTS=6  # Adjust based on your system capabilities

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base output directory with timestamp
BASE_OUTPUT_DIR="output/batchLLM_parallel_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

# Create symlink to latest results
ln -sfn "batchLLM_parallel_${TIMESTAMP}" "output/batchLLM_latest"

# Define NoC sizes and their names (same as CNN)
declare -a NOC_SIZES=("DATEMC2_4X4" "DATEMC8_8X8" "DATEMC32_16X16" "DATEMC128_32X32")
declare -a NOC_NAMES=("2mc_4x4" "8mc_8x8" "32mc_16x16" "128mc_32x32")

# Define LLM test cases
declare -a TEST_CASES=(
    "baseline"           # No optimization
    "samos_only"         # SAMOS mapping only
    "separated_only"     # Separated ordering only
    "affiliated_only"    # Affiliated ordering only
    "samos_separated"    # SAMOS + Separated ordering
    "samos_affiliated"   # SAMOS + Affiliated ordering
)

# Get LLM model info
echo "LLM Mode: Testing with real LLaMA matrices (8x128 output)"

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup_llm

# Create compilation directory
COMPILE_DIR="${BASE_OUTPUT_DIR}/compile_workspace"
mkdir -p "$COMPILE_DIR"
cp -r src "$COMPILE_DIR/src"
cp -r Debug "$COMPILE_DIR/Debug"

# Create execution directories for parallel runs
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${i}"
    mkdir -p "$EXEC_DIR/Debug"
    mkdir -p "$EXEC_DIR/src/Input"
    ln -s "$(realpath src/Input/llminput)" "$EXEC_DIR/src/Input/llminput"
done

echo "==========================================="
echo "Starting Pipeline Parallel LLM Batch Tests"
echo "LLM Configuration: 8x128 attention output matrix"
echo "Total configurations: ${#NOC_SIZES[@]} NoC sizes × ${#TEST_CASES[@]} test cases = $((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))"
echo "Strategy: Sequential compilation, parallel execution (max $MAX_PARALLEL_SLOTS)"
echo "==========================================="
echo ""

# Initialize summary file (simplified - no stats collection)
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "NoC_Size,Test_Case,Runtime" > $SUMMARY_FILE

# Job counter
TOTAL_JOBS=$((${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))

# Function to configure test case in parameters.hpp
configure_test_case() {
    local test_case=$1
    local params_file=$2
    
    # First, disable all case definitions
    sed -i 's|^#define case[0-9]_[a-zA-Z]*|//&|g' "$params_file"
    
    # Enable specific case based on test case
    case "$test_case" in
        "baseline")
            # Enable case1_default for baseline
            sed -i 's|^//#define case1_default|#define case1_default|' "$params_file"
            ;;
        "samos_only")
            # Enable case2_samos
            sed -i 's|^//#define case2_samos|#define case2_samos|' "$params_file"
            ;;
        "separated_only")
            # Enable case4_seperratedordering
            sed -i 's|^//#define case4_seperratedordering|#define case4_seperratedordering|' "$params_file"
            ;;
        "affiliated_only")
            # Enable case3_affiliatedordering
            sed -i 's|^//#define case3_affiliatedordering|#define case3_affiliatedordering|' "$params_file"
            ;;
        "samos_separated")
            # Enable case6_MOSAIC2 (SAMOS + Separated)
            sed -i 's|^//#define case6_MOSAIC2|#define case6_MOSAIC2|' "$params_file"
            ;;
        "samos_affiliated")
            # Enable case5_MOSAIC1 (SAMOS + Affiliated)
            sed -i 's|^//#define case5_MOSAIC1|#define case5_MOSAIC1|' "$params_file"
            ;;
    esac
    
    # Ensure LLM mode is enabled
    sed -i 's|^//#define YZLLMSwitchON|#define YZLLMSwitchON|' "$params_file"
    sed -i 's|^#define mode 0|#define mode 1|' "$params_file"
}

# Function to compile a configuration
compile_config() {
    local noc_size=$1
    local noc_name=$2
    local test_case=$3
    local job_id=$4
    
    echo "[Compile][$job_id/$TOTAL_JOBS] Compiling: NoC=$noc_name, Case=$test_case"
    
    # Create modified parameters.hpp
    cp src/parameters.hpp.backup_llm "$COMPILE_DIR/src/parameters.hpp"
    
    # Configure NoC size
    sed -i 's|^#define DATEMC[0-9]*_[0-9X]*|//&|g' "$COMPILE_DIR/src/parameters.hpp"
    sed -i "s|^//#define $noc_size|#define $noc_size|" "$COMPILE_DIR/src/parameters.hpp"
    
    # Configure test case
    configure_test_case "$test_case" "$COMPILE_DIR/src/parameters.hpp"
    
    # Compile
    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > compile_log.txt 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null
    
    if [ $COMPILE_RESULT -eq 0 ]; then
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success: NoC=$noc_name, Case=$test_case"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED: NoC=$noc_name, Case=$test_case"
        echo "Compile error log:"
        tail -20 "$COMPILE_DIR/Debug/compile_log.txt"
        echo "$noc_name,$test_case,COMPILE_ERROR" >> $SUMMARY_FILE
        return 1
    fi
}

# Function to run a test
run_test() {
    local noc_name=$1
    local test_case=$2
    local job_id=$3
    local exec_slot=$4
    
    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local NOC_DIR="${BASE_OUTPUT_DIR}/${noc_name}"
    mkdir -p "$NOC_DIR"
    local output_file="$(realpath ${NOC_DIR}/${test_case}.txt)"
    
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Starting: NoC=$noc_name, Case=$test_case"
    
    # Copy binary to execution directory
    cp "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}" "$EXEC_DIR/Debug/2508date"
    
    # Add header to output file
    echo "===========================================" > "$output_file"
    echo "LLM CONFIGURATION INFO" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "NoC Size: $noc_name" >> "$output_file"
    echo "Test Case: $test_case" >> "$output_file"
    echo "LLM Mode: Real LLaMA matrices (8x128)" >> "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "" >> "$output_file"
    
    # Run simulation (no timeout)
    START_TIME=$(date +%s)
    cd "$EXEC_DIR/Debug"
    ./2508date >> "$output_file" 2>&1
    cd - > /dev/null
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))
    
    echo "$noc_name,$test_case,$RUNTIME" >> $SUMMARY_FILE
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Completed: Runtime=${RUNTIME}s"
    
    # Clean up binary
    rm -f "${BASE_OUTPUT_DIR}/binary_${noc_name}_${test_case}"
}

# Main pipeline loop
echo "Starting LLM compilation and execution pipeline..."
echo ""

# Arrays to track running jobs
declare -a RUNNING_SLOTS
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

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
                        if ! kill -0 "${RUNNING_SLOTS[$slot]}" 2>/dev/null; then
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
                    echo "[Pipeline] Launched test in slot $FREE_SLOT (PID: ${RUNNING_SLOTS[$FREE_SLOT]})"
                    break
                else
                    sleep 2
                fi
            done
        fi
        
        # Progress report
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" | wc -l)
        RUNNING=$(jobs -r | wc -l)
        echo "[Progress] Compiled: $JOB_ID/$TOTAL_JOBS | Running: $RUNNING | Completed: $COMPLETED/$TOTAL_JOBS"
        echo ""
    done
done

# Wait for all remaining jobs
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
mv src/parameters.hpp.backup_llm src/parameters.hpp

echo ""
echo "==========================================="
echo "All LLM tests completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo "==========================================="
