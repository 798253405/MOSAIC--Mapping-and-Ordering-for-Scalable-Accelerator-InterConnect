#!/bin/bash

# VC Channel Sweep: baseline vs mosaic1 across VC_PER_VN values
# Fixed: token1, NOCSIZEMC2_4X4

# ========== CONFIGURATION ==========
MAX_PARALLEL_SLOTS=6

declare -a VC_VALUES=(2 4 8)
declare -a TEST_CASES=("baseline" "mosaic1")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="output/batchVC_parallel_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR
ln -sfn "batchVC_parallel_${TIMESTAMP}" "output/batchVC_latest"

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup_llm

# Create compilation directory
COMPILE_DIR="${BASE_OUTPUT_DIR}/compile_workspace"
mkdir -p "$COMPILE_DIR"
cp -r src "$COMPILE_DIR/src"
cp -r Debug "$COMPILE_DIR/Debug"

# Create execution directories
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${i}"
    mkdir -p "$EXEC_DIR/Debug/src/Input"
    ln -s "$(realpath src/Input/llminput)" "$EXEC_DIR/Debug/src/Input/llminput"
done

TOTAL_JOBS=$((${#VC_VALUES[@]} * ${#TEST_CASES[@]}))

echo "==========================================="
echo "Starting VC Channel Sweep"
echo "Matrix: ${#VC_VALUES[@]} VC values × ${#TEST_CASES[@]} test cases = $TOTAL_JOBS total"
echo "VC_PER_VN values: ${VC_VALUES[*]}"
echo "Test cases: ${TEST_CASES[*]}"
echo "Fixed: token1, NOCSIZEMC2_4X4"
echo "Parallel execution slots: $MAX_PARALLEL_SLOTS"
echo "==========================================="
echo ""

SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "VC_Value,Test_Case,Runtime" > $SUMMARY_FILE

# Function to configure test case
configure_test_case() {
    local test_case=$1
    local params_file=$2

    # Disable all case definitions
    sed -i 's|^#define case[0-9][0-9]*_[a-zA-Z]*|//&|g' "$params_file"

    case "$test_case" in
        "baseline")
            sed -i 's|^//#define case1_default|#define case1_default|' "$params_file"
            ;;
        "mosaic1")
            sed -i 's|^//#define case9_MOSAIC1|#define case9_MOSAIC1|' "$params_file"
            ;;
    esac

    # Ensure LLM mode is enabled
    sed -i 's|^//#define AUTHORLLMSwitchON|#define AUTHORLLMSwitchON|' "$params_file"
}

# Function to compile
compile_config() {
    local vc_val=$1
    local test_case=$2
    local job_id=$3

    echo "[Compile][$job_id/$TOTAL_JOBS] VC=$vc_val, Case=$test_case"

    # Start from clean backup
    cp src/parameters.hpp.backup_llm "$COMPILE_DIR/src/parameters.hpp"
    local params_file="$COMPILE_DIR/src/parameters.hpp"

    # (a) Disable all NoC sizes
    sed -i 's|^#define NOCSIZEMC[0-9]*_[0-9X]*|//&|g' "$params_file"
    # (b) Enable 4x4
    sed -i 's|^//#define NOCSIZEMC2_4X4|#define NOCSIZEMC2_4X4|' "$params_file"

    # (c,d,e) Configure test case + enable LLM mode
    configure_test_case "$test_case" "$params_file"

    # (f) Fix token size to 1
    sed -i 's|^#define LLM_TOKEN_SIZE|//#define LLM_TOKEN_SIZE|g' "$params_file"
    sed -i 's|^//#define LLM_TOKEN_SIZE 1|#define LLM_TOKEN_SIZE 1|' "$params_file"

    # (g) Set VC_PER_VN
    sed -i "s|^#define VC_PER_VN .*|#define VC_PER_VN $vc_val|" "$params_file"

    # Compile
    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > compile_log.txt 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null

    local binary_name="binary_vc${vc_val}_${test_case}"
    if [ $COMPILE_RESULT -eq 0 ]; then
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/${binary_name}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED"
        tail -10 "$COMPILE_DIR/Debug/compile_log.txt"
        echo "vc$vc_val,$test_case,COMPILE_ERROR" >> $SUMMARY_FILE
        return 1
    fi
}

# Function to run a test
run_test() {
    local vc_val=$1
    local test_case=$2
    local job_id=$3
    local exec_slot=$4

    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local OUT_DIR="${BASE_OUTPUT_DIR}/vc${vc_val}"
    mkdir -p "$OUT_DIR"
    local output_file="$(realpath ${OUT_DIR}/${test_case}.txt)"
    local binary_name="binary_vc${vc_val}_${test_case}"

    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] vc${vc_val}/$test_case"

    cp "${BASE_OUTPUT_DIR}/${binary_name}" "$EXEC_DIR/Debug/2508date"

    echo "===========================================" > "$output_file"
    echo "VC_PER_VN: $vc_val" >> "$output_file"
    echo "Test Case: $test_case" >> "$output_file"
    echo "NoC Size: 2mc_4x4" >> "$output_file"
    echo "Token Size: token1" >> "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "" >> "$output_file"

    START_TIME=$(date +%s)
    cd "$EXEC_DIR/Debug"
    ./2508date >> "$output_file" 2>&1
    cd - > /dev/null
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo "vc$vc_val,$test_case,$RUNTIME" >> $SUMMARY_FILE
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Done: ${RUNTIME}s"

    rm -f "${BASE_OUTPUT_DIR}/${binary_name}"
}

# Main pipeline loop
declare -a RUNNING_SLOTS
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

JOB_ID=0

for vc_val in "${VC_VALUES[@]}"; do
    for test_case in "${TEST_CASES[@]}"; do
        JOB_ID=$((JOB_ID + 1))

        compile_config "$vc_val" "$test_case" "$JOB_ID"

        if [ $? -eq 0 ]; then
            while true; do
                for ((slot=1; slot<=MAX_PARALLEL_SLOTS; slot++)); do
                    if [ "${RUNNING_SLOTS[$slot]}" -ne 0 ]; then
                        if ! kill -0 "${RUNNING_SLOTS[$slot]}" 2>/dev/null; then
                            wait "${RUNNING_SLOTS[$slot]}" 2>/dev/null
                            RUNNING_SLOTS[$slot]=0
                        fi
                    fi
                done

                FREE_SLOT=0
                for ((slot=1; slot<=MAX_PARALLEL_SLOTS; slot++)); do
                    if [ "${RUNNING_SLOTS[$slot]}" -eq 0 ]; then
                        FREE_SLOT=$slot
                        break
                    fi
                done

                if [ $FREE_SLOT -ne 0 ]; then
                    run_test "$vc_val" "$test_case" "$JOB_ID" "$FREE_SLOT" &
                    RUNNING_SLOTS[$FREE_SLOT]=$!
                    break
                else
                    sleep 2
                fi
            done
        fi

        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" | wc -l)
        echo "[Progress] $JOB_ID/$TOTAL_JOBS compiled | $COMPLETED completed"
        echo ""
    done
done

# Wait for remaining jobs
echo "Waiting for remaining jobs..."
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
echo "VC Channel Sweep completed!"
echo "Results: $BASE_OUTPUT_DIR"
echo ""
TOTAL_RUNS=$(tail -n +2 "$SUMMARY_FILE" | wc -l)
FAILED=$(grep -c "COMPILE_ERROR" "$SUMMARY_FILE" 2>/dev/null || echo "0")
echo "Total: $TOTAL_RUNS | Success: $((TOTAL_RUNS - FAILED)) | Failed: $FAILED"
echo ""
echo "Runtime summary (sorted):"
tail -n +2 "$SUMMARY_FILE" | grep -v "COMPILE_ERROR" | sort -t',' -k3 -n | while IFS=',' read -r vc test runtime; do
    printf "  %-6s / %-20s : %ss\n" "$vc" "$test" "$runtime"
done
echo "==========================================="
