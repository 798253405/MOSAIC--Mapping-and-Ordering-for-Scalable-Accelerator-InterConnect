#!/bin/bash

# DarkNet-19 Batch Testing Script
# Iterates over: eval modes (fulleval/randomeval) × NoC sizes × test cases (1-9)
# Strategy: Sequential compilation, parallel execution (max 8 slots)

# ========== CONFIGURATION ==========
MAX_PARALLEL_SLOTS=8

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base output directory
BASE_OUTPUT_DIR="output/batch_darknet_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"

# Create symlink to latest results
ln -sfn "batch_darknet_${TIMESTAMP}" "output/batch_darknet_latest"

# Define evaluation modes
declare -a EVAL_MODES=("fulleval" "randomeval")

# Define NoC sizes and their names
declare -a NOC_SIZES=("TACOMC2_4X4" "TACOMC8_8X8" "TACOMC4_4X4" "TACOMC4_8X8")
declare -a NOC_NAMES=("2mc_4x4" "8mc_8x8" "4mc_4x4" "4mc_8x8")

# Define test cases (1-9)
declare -a TEST_CASES=(
    "case1_default"
    "case2_TACOall128BitInvert"
    "case3_PartialBusInvert"
    "case4_affiliatedordering"
    "case5_seperratedordering"
    "case6_affiliatedordering_TACOall128BitInvert"
    "case7_affiliatedordering_PartialBusInvert"
    "case8_seperratedordering_TACOall128BitInvert"
    "case9_seperratedordering_PartialBusInvert"
)

# Short names for cases
declare -a CASE_NAMES=("case1" "case2" "case3" "case4" "case5" "case6" "case7" "case8" "case9")

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup_darknet_batch

# Enable DarkNet model in backup (disable VGG and Lenet)
sed -i 's|^#define modelVGG|//#define modelVGG|g' src/parameters.hpp.backup_darknet_batch
sed -i 's|^#define modelLenet|//#define modelLenet|g' src/parameters.hpp.backup_darknet_batch
sed -i 's|^//#define modelDarknet|#define modelDarknet|g' src/parameters.hpp.backup_darknet_batch

# Create compilation workspace
COMPILE_DIR="${BASE_OUTPUT_DIR}/compile_workspace"
mkdir -p "$COMPILE_DIR"
cp -r src "$COMPILE_DIR/src"
cp -r Debug "$COMPILE_DIR/Debug"

# Create execution directories
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${i}"
    mkdir -p "$EXEC_DIR/Debug"
done

# Calculate total jobs
TOTAL_JOBS=$((${#EVAL_MODES[@]} * ${#NOC_SIZES[@]} * ${#TEST_CASES[@]}))

echo "==========================================="
echo "Starting DarkNet-19 Batch Tests"
echo "Model: DarkNet-19"
echo "Eval Modes: ${#EVAL_MODES[@]} (${EVAL_MODES[*]})"
echo "NoC Sizes: ${#NOC_SIZES[@]} (${NOC_NAMES[*]})"
echo "Test Cases: ${#TEST_CASES[@]}"
echo "Total configurations: $TOTAL_JOBS"
echo "Parallel execution slots: $MAX_PARALLEL_SLOTS"
echo "Output directory: $BASE_OUTPUT_DIR"
echo "==========================================="
echo ""

# Initialize summary file
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "Eval_Mode,NoC_Size,Test_Case,Runtime_Seconds" > "$SUMMARY_FILE"

# Function to compile a configuration
compile_config() {
    local eval_mode=$1
    local noc_size=$2
    local noc_name=$3
    local test_case=$4
    local case_name=$5
    local job_id=$6

    echo "[Compile][$job_id/$TOTAL_JOBS] Compiling: Eval=$eval_mode, NoC=$noc_name, Case=$case_name"

    # Start from backup (with DarkNet enabled)
    cp src/parameters.hpp.backup_darknet_batch "$COMPILE_DIR/src/parameters.hpp"

    # 1. Set eval mode
    if [ "$eval_mode" == "fulleval" ]; then
        sed -i 's|^#define randomeval|//#define randomeval|g' "$COMPILE_DIR/src/parameters.hpp"
        sed -i 's|^//#define fulleval|#define fulleval|g' "$COMPILE_DIR/src/parameters.hpp"
    else
        sed -i 's|^#define fulleval|//#define fulleval|g' "$COMPILE_DIR/src/parameters.hpp"
        sed -i 's|^//#define randomeval|#define randomeval|g' "$COMPILE_DIR/src/parameters.hpp"
    fi

    # 2. Disable all TACOMC configurations, enable the selected one
    sed -i 's|^#define TACOMC[0-9]*_[0-9X]*|//&|g' "$COMPILE_DIR/src/parameters.hpp"
    sed -i "s|^//#define $noc_size|#define $noc_size|" "$COMPILE_DIR/src/parameters.hpp"

    # 3. Disable all case configurations, enable the selected one
    sed -i 's|^#define case[0-9]_[a-zA-Z_]*|//&|g' "$COMPILE_DIR/src/parameters.hpp"
    sed -i "s|^//#define $test_case|#define $test_case|" "$COMPILE_DIR/src/parameters.hpp"

    # Compile
    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > /dev/null 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null

    if [ $COMPILE_RESULT -eq 0 ]; then
        # Save compiled binary with unique name
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/binary_${eval_mode}_${noc_name}_${case_name}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success: Eval=$eval_mode, NoC=$noc_name, Case=$case_name"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED: Eval=$eval_mode, NoC=$noc_name, Case=$case_name"
        echo "$eval_mode,$noc_name,$case_name,COMPILE_ERROR" >> "$SUMMARY_FILE"
        return 1
    fi
}

# Function to run a test (executed in background)
run_test() {
    local eval_mode=$1
    local noc_name=$2
    local case_name=$3
    local job_id=$4
    local exec_slot=$5

    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"

    # Create output directory: eval_mode/noc_name/
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${eval_mode}/${noc_name}"
    mkdir -p "$OUTPUT_DIR"

    local output_file="$(realpath ${OUTPUT_DIR}/${case_name}.txt)"

    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Starting: Eval=$eval_mode, NoC=$noc_name, Case=$case_name"

    # Copy binary to execution directory
    cp "${BASE_OUTPUT_DIR}/binary_${eval_mode}_${noc_name}_${case_name}" "$EXEC_DIR/Debug/2508date"

    # Add header to output file
    {
        echo "==========================================="
        echo "CONFIGURATION INFO"
        echo "==========================================="
        echo "Model: DarkNet-19"
        echo "Eval Mode: $eval_mode"
        echo "NoC Size: $noc_name"
        echo "Test Case: $case_name"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "==========================================="
        echo ""
    } > "$output_file"

    # Run simulation
    START_TIME=$(date +%s)
    cd "$EXEC_DIR/Debug"
    ./2508date >> "$output_file" 2>&1
    cd - > /dev/null
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    # Record to summary
    echo "$eval_mode,$noc_name,$case_name,$RUNTIME" >> "$SUMMARY_FILE"
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Completed: Runtime=${RUNTIME}s"

    # Clean up binary
    rm -f "${BASE_OUTPUT_DIR}/binary_${eval_mode}_${noc_name}_${case_name}"
}

# Initialize slots as available
declare -a RUNNING_SLOTS
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

# Job ID counter
JOB_ID=0

echo "Starting compilation and execution pipeline..."
echo ""

# Main loop: eval_mode -> noc_size -> test_case
for eval_mode in "${EVAL_MODES[@]}"; do
    for i in "${!NOC_SIZES[@]}"; do
        noc_size="${NOC_SIZES[$i]}"
        noc_name="${NOC_NAMES[$i]}"

        for j in "${!TEST_CASES[@]}"; do
            test_case="${TEST_CASES[$j]}"
            case_name="${CASE_NAMES[$j]}"

            JOB_ID=$((JOB_ID + 1))

            # Compile the configuration
            compile_config "$eval_mode" "$noc_size" "$noc_name" "$test_case" "$case_name" "$JOB_ID"

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
                        run_test "$eval_mode" "$noc_name" "$case_name" "$JOB_ID" "$FREE_SLOT" &
                        RUNNING_SLOTS[$FREE_SLOT]=$!
                        echo "[Pipeline] Launched test in slot $FREE_SLOT (PID: ${RUNNING_SLOTS[$FREE_SLOT]})"
                        break
                    else
                        sleep 2
                    fi
                done
            fi

            # Progress report
            COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l || echo "0")
            RUNNING=$(jobs -r | wc -l)
            echo "[Progress] Compiled: $JOB_ID/$TOTAL_JOBS | Running: $RUNNING | Completed: $COMPLETED/$TOTAL_JOBS"
            echo ""
        done
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
mv src/parameters.hpp.backup_darknet_batch src/parameters.hpp

echo ""
echo "==========================================="
echo "All DarkNet-19 tests completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""

# Final statistics
echo "Test Results Summary:"
echo "---------------------"
TOTAL_COMPLETED=$(tail -n +2 "$SUMMARY_FILE" | wc -l)
COMPILE_ERRORS=$(grep -c "COMPILE_ERROR" "$SUMMARY_FILE" 2>/dev/null || echo "0")
echo "  Total completed: $TOTAL_COMPLETED"
echo "  Compile errors: $COMPILE_ERRORS"
echo ""

echo "Results by Eval Mode:"
for eval_mode in "${EVAL_MODES[@]}"; do
    COUNT=$(grep "^$eval_mode," "$SUMMARY_FILE" | wc -l)
    echo "  $eval_mode: $COUNT tests"
done
echo ""

echo "==========================================="
