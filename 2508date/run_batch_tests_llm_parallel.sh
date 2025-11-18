#!/bin/bash

# LLM Pipeline parallel batch testing script
# Tests multiple NoC sizes, test cases, and token sizes

# ========== CONFIGURATION ==========
# Maximum number of parallel execution slots
MAX_PARALLEL_SLOTS=6  # Adjust based on your system capabilities

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base output directory with timestamp
BASE_OUTPUT_DIR="output/llm_batch_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

# Create symlink to latest results
ln -sfn "llm_batch_${TIMESTAMP}" "output/llm_batch_latest"

# Define NoC sizes and their names
declare -a NOC_SIZES=("NOCSIZEMC2_4X4" "NOCSIZEMC8_8X8" "NOCSIZEMC32_16X16" "NOCSIZEMC128_32X32")
declare -a NOC_NAMES=("2mc_4x4" "8mc_8x8" "32mc_16x16" "128mc_32x32")

# Define token sizes
declare -a TOKEN_SIZES=(1 2)
declare -a TOKEN_NAMES=("token8" "token128")

# Define LLM test cases (10 cases)
declare -a TEST_CASES=(
    "case1_baseline"          # case1_default: Row mapping
    "case2_samos"             # case2_samos: SAMOS only
    "case3_affiliated"        # case3_affiliatedordering
    "case4_separated"         # case4_seperratedordering
    "case5_combo1"            # case5_COMBO1: SAMOS + Affiliated
    "case6_combo2"            # case6_COMBO2: SAMOS + Affiliated + Separated
    "case7_fireadvance"       # case7_FireAdvance
    "case8_binaryswitch"      # case8_BinarySwitch
    "case9_mosaic1"           # case9_MOSAIC1: Full optimization w/o separated
    "case10_mosaic2"          # case10_MOSAIC2: Full optimization
)

echo "==========================================="
echo "Starting Pipeline Parallel LLM Batch Tests"
echo "Configuration:"
echo "  NoC Sizes: ${#NOC_SIZES[@]} (4x4, 8x8, 16x16, 32x32)"
echo "  Test Cases: ${#TEST_CASES[@]} (case1-case10)"
echo "  Token Sizes: ${#TOKEN_SIZES[@]} (8 tokens, 128 tokens)"
echo "  Total Tests: $((${#NOC_SIZES[@]} * ${#TEST_CASES[@]} * ${#TOKEN_SIZES[@]}))"
echo "  Parallel Slots: $MAX_PARALLEL_SLOTS"
echo "==========================================="
echo ""

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

# Initialize summary file
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "Token_Size,NoC_Size,Test_Case,Runtime" > $SUMMARY_FILE

# Job counter
TOTAL_JOBS=$((${#NOC_SIZES[@]} * ${#TEST_CASES[@]} * ${#TOKEN_SIZES[@]}))

# Function to configure test case in parameters.hpp
configure_test_case() {
    local test_case=$1
    local params_file=$2

    # First, disable all case definitions
    sed -i 's|^#define case[0-9]*_[a-zA-Z_0-9]*|//&|g' "$params_file"

    # Enable specific case based on test case
    case "$test_case" in
        "case1_baseline")
            sed -i 's|^//#define case1_default|#define case1_default|' "$params_file"
            ;;
        "case2_samos")
            sed -i 's|^//#define case2_samos|#define case2_samos|' "$params_file"
            ;;
        "case3_affiliated")
            sed -i 's|^//#define case3_affiliatedordering|#define case3_affiliatedordering|' "$params_file"
            ;;
        "case4_separated")
            sed -i 's|^//#define case4_seperratedordering|#define case4_seperratedordering|' "$params_file"
            ;;
        "case5_combo1")
            sed -i 's|^//#define case5_COMBO1|#define case5_COMBO1|' "$params_file"
            ;;
        "case6_combo2")
            sed -i 's|^//#define case6_COMBO2|#define case6_COMBO2|' "$params_file"
            ;;
        "case7_fireadvance")
            sed -i 's|^//#define case7_FireAdvance|#define case7_FireAdvance|' "$params_file"
            ;;
        "case8_binaryswitch")
            sed -i 's|^//#define case8_BinarySwitch|#define case8_BinarySwitch|' "$params_file"
            ;;
        "case9_mosaic1")
            sed -i 's|^//#define case9_MOSAIC1|#define case9_MOSAIC1|' "$params_file"
            ;;
        "case10_mosaic2")
            sed -i 's|^//#define case10_MOSAIC2|#define case10_MOSAIC2|' "$params_file"
            ;;
    esac

    # Ensure LLM mode is enabled
    sed -i 's|^//#define YZLLMSwitchON|#define YZLLMSwitchON|' "$params_file"
}

# Function to configure token size in parameters.hpp
configure_token_size() {
    local token_size=$1
    local params_file=$2

    # Disable all LLM_TOKEN_SIZE definitions
    sed -i 's|^#define LLM_TOKEN_SIZE [0-9]*|//&|g' "$params_file"

    # Enable specific token size
    if [ "$token_size" -eq 1 ]; then
        sed -i 's|^//#define LLM_TOKEN_SIZE 1|#define LLM_TOKEN_SIZE 1|' "$params_file"
    elif [ "$token_size" -eq 2 ]; then
        sed -i 's|^//#define LLM_TOKEN_SIZE 2|#define LLM_TOKEN_SIZE 2|' "$params_file"
    fi
}

# Function to compile a configuration
compile_config() {
    local noc_size=$1
    local noc_name=$2
    local test_case=$3
    local token_size=$4
    local token_name=$5
    local job_id=$6

    echo "[Compile][$job_id/$TOTAL_JOBS] Compiling: Token=$token_name, NoC=$noc_name, Case=$test_case"

    # Create modified parameters.hpp
    cp src/parameters.hpp.backup_llm "$COMPILE_DIR/src/parameters.hpp"

    # Configure NoC size
    sed -i 's|^#define NOCSIZEMC[0-9]*_[0-9X]*|//&|g' "$COMPILE_DIR/src/parameters.hpp"
    sed -i "s|^//#define $noc_size|#define $noc_size|" "$COMPILE_DIR/src/parameters.hpp"

    # Configure test case
    configure_test_case "$test_case" "$COMPILE_DIR/src/parameters.hpp"

    # Configure token size
    configure_token_size "$token_size" "$COMPILE_DIR/src/parameters.hpp"

    # Compile
    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > compile_log.txt 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null

    if [ $COMPILE_RESULT -eq 0 ]; then
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/binary_${token_name}_${noc_name}_${test_case}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success: Token=$token_name, NoC=$noc_name, Case=$test_case"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED: Token=$token_name, NoC=$noc_name, Case=$test_case"
        echo "Compile error log:"
        tail -20 "$COMPILE_DIR/Debug/compile_log.txt"
        echo "$token_name,$noc_name,$test_case,COMPILE_ERROR" >> $SUMMARY_FILE
        return 1
    fi
}

# Function to run a test
run_test() {
    local noc_name=$1
    local test_case=$2
    local token_size=$3
    local token_name=$4
    local job_id=$5
    local exec_slot=$6

    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local TOKEN_DIR="${BASE_OUTPUT_DIR}/${token_name}"
    local NOC_DIR="${TOKEN_DIR}/${noc_name}"
    mkdir -p "$NOC_DIR"
    local output_file="$(realpath ${NOC_DIR}/${test_case}.txt)"

    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Starting: Token=$token_name, NoC=$noc_name, Case=$test_case"

    # Copy binary to execution directory
    cp "${BASE_OUTPUT_DIR}/binary_${token_name}_${noc_name}_${test_case}" "$EXEC_DIR/Debug/2508date"

    # Add header to output file
    echo "===========================================" > "$output_file"
    echo "LLM CONFIGURATION INFO" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "Token Size: $token_name (LLM_TOKEN_SIZE=$token_size)" >> "$output_file"
    echo "NoC Size: $noc_name" >> "$output_file"
    echo "Test Case: $test_case" >> "$output_file"
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

    echo "$token_name,$noc_name,$test_case,$RUNTIME" >> $SUMMARY_FILE
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Completed: Runtime=${RUNTIME}s"

    # Clean up binary
    rm -f "${BASE_OUTPUT_DIR}/binary_${token_name}_${noc_name}_${test_case}"
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

# Outer loop: Token sizes (token8 first, then token128)
for t in "${!TOKEN_SIZES[@]}"; do
    token_size="${TOKEN_SIZES[$t]}"
    token_name="${TOKEN_NAMES[$t]}"

    echo ""
    echo "=========================================="
    echo "Starting Token Size: $token_name (LLM_TOKEN_SIZE=$token_size)"
    echo "=========================================="
    echo ""

    # Middle loop: NoC sizes
    for i in "${!NOC_SIZES[@]}"; do
        noc_size="${NOC_SIZES[$i]}"
        noc_name="${NOC_NAMES[$i]}"

        # Inner loop: Test cases
        for test_case in "${TEST_CASES[@]}"; do
            JOB_ID=$((JOB_ID + 1))

            # Compile the configuration
            compile_config "$noc_size" "$noc_name" "$test_case" "$token_size" "$token_name" "$JOB_ID"

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
                        run_test "$noc_name" "$test_case" "$token_size" "$token_name" "$JOB_ID" "$FREE_SLOT" &
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
            echo "[Progress] Total: $JOB_ID/$TOTAL_JOBS | Running: $RUNNING | Completed: $COMPLETED/$TOTAL_JOBS"
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
mv src/parameters.hpp.backup_llm src/parameters.hpp

echo ""
echo "==========================================="
echo "All LLM tests completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "  ├── token8/ (40 tests with LLM_TOKEN_SIZE=1)"
echo "  └── token128/ (40 tests with LLM_TOKEN_SIZE=2)"
echo "Summary file: $SUMMARY_FILE"
echo "==========================================="
