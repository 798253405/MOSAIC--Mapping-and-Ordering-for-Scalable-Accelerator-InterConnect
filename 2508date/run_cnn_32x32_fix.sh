#!/bin/bash

# Re-run failed CNN 128mc_32x32 cases: FireAdvance, MOSAIC1, MOSAIC2
# Also runs baseline for verification

MAX_PARALLEL_SLOTS=4

declare -a TEST_CASES=("case1_default" "case7_FireAdvance" "case9_MOSAIC1" "case10_MOSAIC2")
NOC_SIZE="NOCSIZEMC128_32X32"
NOC_NAME="128mc_32x32"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="output/batchCNN_32x32fix_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

# Backup original parameters.hpp
cp src/parameters.hpp src/parameters.hpp.backup_pipeline

# Create compilation directory
COMPILE_DIR="${BASE_OUTPUT_DIR}/compile_workspace"
mkdir -p "$COMPILE_DIR"
cp -r src "$COMPILE_DIR/src"
cp -r Debug "$COMPILE_DIR/Debug"

# Create execution directories
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${i}"
    mkdir -p "$EXEC_DIR/Debug/src"
    ln -s "$(realpath src/Input)" "$EXEC_DIR/Debug/src/Input"
done

TOTAL_JOBS=${#TEST_CASES[@]}

echo "==========================================="
echo "Re-running CNN 128mc_32x32 failed cases"
echo "Cases: ${TEST_CASES[*]}"
echo "==========================================="
echo ""

SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "NoC_Size,Test_Case,Runtime" > $SUMMARY_FILE

compile_config() {
    local test_case=$1
    local job_id=$2

    echo "[Compile][$job_id/$TOTAL_JOBS] $NOC_NAME/$test_case"

    cp src/parameters.hpp.backup_pipeline "$COMPILE_DIR/src/parameters.hpp"
    local params_file="$COMPILE_DIR/src/parameters.hpp"

    # Disable all NoC sizes, enable 32x32
    sed -i 's|^#define NOCSIZEMC[0-9]*_[0-9X]*|//&|g' "$params_file"
    sed -i "s|^//#define $NOC_SIZE|#define $NOC_SIZE|" "$params_file"

    # Disable LLM mode
    sed -i 's|^#define AUTHORLLMSwitchON|//#define AUTHORLLMSwitchON|' "$params_file"

    # Disable all cases, enable target
    sed -i 's|^#define case[0-9][0-9]*_[a-zA-Z]*|//&|g' "$params_file"
    sed -i "s|^//#define $test_case|#define $test_case|" "$params_file"

    cd "$COMPILE_DIR/Debug"
    make clean > /dev/null 2>&1
    make all > compile_log.txt 2>&1
    COMPILE_RESULT=$?
    cd - > /dev/null

    local binary_name="binary_${test_case}"
    if [ $COMPILE_RESULT -eq 0 ]; then
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/${binary_name}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED"
        tail -10 "$COMPILE_DIR/Debug/compile_log.txt"
        echo "$NOC_NAME,$test_case,COMPILE_ERROR" >> $SUMMARY_FILE
        return 1
    fi
}

run_test() {
    local test_case=$1
    local job_id=$2
    local exec_slot=$3

    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local OUT_DIR="${BASE_OUTPUT_DIR}/${NOC_NAME}"
    mkdir -p "$OUT_DIR"
    local output_file="$(realpath ${OUT_DIR}/${test_case}.txt)"
    local binary_name="binary_${test_case}"

    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] $test_case"

    cp "${BASE_OUTPUT_DIR}/${binary_name}" "$EXEC_DIR/Debug/2508date"

    echo "===========================================" > "$output_file"
    echo "NoC Size: $NOC_NAME" >> "$output_file"
    echo "Test Case: $test_case" >> "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
    echo "===========================================" >> "$output_file"
    echo "" >> "$output_file"

    START_TIME=$(date +%s)
    cd "$EXEC_DIR/Debug"
    ./2508date >> "$output_file" 2>&1
    cd - > /dev/null
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo "$NOC_NAME,$test_case,$RUNTIME" >> $SUMMARY_FILE
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Done: ${RUNTIME}s"

    rm -f "${BASE_OUTPUT_DIR}/${binary_name}"
}

# Main loop
declare -a RUNNING_SLOTS
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

JOB_ID=0
for test_case in "${TEST_CASES[@]}"; do
    JOB_ID=$((JOB_ID + 1))

    compile_config "$test_case" "$JOB_ID"

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
                run_test "$test_case" "$JOB_ID" "$FREE_SLOT" &
                RUNNING_SLOTS[$FREE_SLOT]=$!
                break
            else
                sleep 2
            fi
        done
    fi
done

# Wait for remaining
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

# Restore
mv src/parameters.hpp.backup_pipeline src/parameters.hpp

echo ""
echo "==========================================="
echo "Done! Results: $BASE_OUTPUT_DIR/$NOC_NAME/"
echo ""
cat $SUMMARY_FILE
echo "==========================================="
