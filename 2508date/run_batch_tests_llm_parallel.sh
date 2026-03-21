#!/bin/bash

# LLM Pipeline parallel batch testing script
# Full matrix: NoC sizes × test cases × LLM token sizes

# ========== CONFIGURATION ==========
MAX_PARALLEL_SLOTS=6

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="output/batchLLM_parallel_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR
ln -sfn "batchLLM_parallel_${TIMESTAMP}" "output/batchLLM_latest"

# Define NoC sizes
declare -a NOC_SIZES=("NOCSIZEMC2_4X4" "NOCSIZEMC8_8X8" "NOCSIZEMC32_16X16" "NOCSIZEMC128_32X32")
declare -a NOC_NAMES=("2mc_4x4" "8mc_8x8" "32mc_16x16" "128mc_32x32")

# Define ALL 12 test cases
declare -a TEST_CASES=(
    "baseline"
    "samos_only"
    "affiliated_only"
    "separated_only"
    "samos_affiliated"
    "samos_separated"
    "fire_advance"
    "binary_switch"
    "mosaic1"
    "mosaic2"
    "lsbs_only"
    "lsbs_affiliated"
)

# Define LLM token sizes
declare -a TOKEN_SIZES=(1 2)
declare -a TOKEN_NAMES=("token1" "token2")

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

TOTAL_JOBS=$((${#NOC_SIZES[@]} * ${#TEST_CASES[@]} * ${#TOKEN_SIZES[@]}))

echo "==========================================="
echo "Starting Full LLM Batch Tests"
echo "Matrix: ${#TOKEN_SIZES[@]} token sizes × ${#NOC_SIZES[@]} NoC sizes × ${#TEST_CASES[@]} test cases = $TOTAL_JOBS total"
echo "Parallel execution slots: $MAX_PARALLEL_SLOTS"
echo "==========================================="
echo ""

SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_all.txt"
echo "Token_Size,NoC_Size,Test_Case,Runtime" > $SUMMARY_FILE

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
        "samos_only")
            sed -i 's|^//#define case2_samos|#define case2_samos|' "$params_file"
            ;;
        "affiliated_only")
            sed -i 's|^//#define case3_affiliatedordering|#define case3_affiliatedordering|' "$params_file"
            ;;
        "separated_only")
            sed -i 's|^//#define case4_seperratedordering|#define case4_seperratedordering|' "$params_file"
            ;;
        "samos_affiliated")
            sed -i 's|^//#define case5_COMBO1|#define case5_COMBO1|' "$params_file"
            ;;
        "samos_separated")
            sed -i 's|^//#define case6_COMBO2|#define case6_COMBO2|' "$params_file"
            ;;
        "fire_advance")
            sed -i 's|^//#define case7_FireAdvance|#define case7_FireAdvance|' "$params_file"
            ;;
        "binary_switch")
            sed -i 's|^//#define case8_BinarySwitch|#define case8_BinarySwitch|' "$params_file"
            ;;
        "mosaic1")
            sed -i 's|^//#define case9_MOSAIC1|#define case9_MOSAIC1|' "$params_file"
            ;;
        "mosaic2")
            sed -i 's|^//#define case10_MOSAIC2|#define case10_MOSAIC2|' "$params_file"
            ;;
        "lsbs_only")
            sed -i 's|^//#define case11_LSBSaturation|#define case11_LSBSaturation|' "$params_file"
            ;;
        "lsbs_affiliated")
            sed -i 's|^//#define case12_LSBSaturationAffilaited|#define case12_LSBSaturationAffilaited|' "$params_file"
            ;;
    esac

    # Ensure LLM mode is enabled
    sed -i 's|^//#define AUTHORLLMSwitchON|#define AUTHORLLMSwitchON|' "$params_file"
}

# Function to configure LLM token size
configure_token_size() {
    local token_size=$1
    local params_file=$2

    # Comment out all LLM_TOKEN_SIZE lines
    sed -i 's|^#define LLM_TOKEN_SIZE|//#define LLM_TOKEN_SIZE|g' "$params_file"
    # Enable the selected token size
    sed -i "s|^//#define LLM_TOKEN_SIZE $token_size|#define LLM_TOKEN_SIZE $token_size|" "$params_file"
}

# Function to compile
compile_config() {
    local noc_size=$1
    local noc_name=$2
    local test_case=$3
    local token_size=$4
    local token_name=$5
    local job_id=$6

    echo "[Compile][$job_id/$TOTAL_JOBS] NoC=$noc_name, Case=$test_case, Token=$token_name"

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

    local binary_name="binary_${token_name}_${noc_name}_${test_case}"
    if [ $COMPILE_RESULT -eq 0 ]; then
        cp "$COMPILE_DIR/Debug/2508date" "${BASE_OUTPUT_DIR}/${binary_name}"
        echo "[Compile][$job_id/$TOTAL_JOBS] Success"
        return 0
    else
        echo "[Compile][$job_id/$TOTAL_JOBS] FAILED"
        tail -10 "$COMPILE_DIR/Debug/compile_log.txt"
        echo "$token_name,$noc_name,$test_case,COMPILE_ERROR" >> $SUMMARY_FILE
        return 1
    fi
}

# Function to run a test
run_test() {
    local noc_name=$1
    local test_case=$2
    local token_name=$3
    local job_id=$4
    local exec_slot=$5

    local EXEC_DIR="${BASE_OUTPUT_DIR}/exec_${exec_slot}"
    local OUT_DIR="${BASE_OUTPUT_DIR}/${token_name}/${noc_name}"
    mkdir -p "$OUT_DIR"
    local output_file="$(realpath ${OUT_DIR}/${test_case}.txt)"
    local binary_name="binary_${token_name}_${noc_name}_${test_case}"

    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] $token_name/$noc_name/$test_case"

    cp "${BASE_OUTPUT_DIR}/${binary_name}" "$EXEC_DIR/Debug/2508date"

    echo "===========================================" > "$output_file"
    echo "Token Size: $token_name" >> "$output_file"
    echo "NoC Size: $noc_name" >> "$output_file"
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

    echo "$token_name,$noc_name,$test_case,$RUNTIME" >> $SUMMARY_FILE
    echo "[Execute][$job_id/$TOTAL_JOBS][Slot $exec_slot] Done: ${RUNTIME}s"

    rm -f "${BASE_OUTPUT_DIR}/${binary_name}"
}

# Main pipeline loop
declare -a RUNNING_SLOTS
for ((i=1; i<=MAX_PARALLEL_SLOTS; i++)); do
    RUNNING_SLOTS[$i]=0
done

JOB_ID=0

for t in "${!TOKEN_SIZES[@]}"; do
    token_size="${TOKEN_SIZES[$t]}"
    token_name="${TOKEN_NAMES[$t]}"

    for i in "${!NOC_SIZES[@]}"; do
        noc_size="${NOC_SIZES[$i]}"
        noc_name="${NOC_NAMES[$i]}"

        for test_case in "${TEST_CASES[@]}"; do
            JOB_ID=$((JOB_ID + 1))

            compile_config "$noc_size" "$noc_name" "$test_case" "$token_size" "$token_name" "$JOB_ID"

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
                        run_test "$noc_name" "$test_case" "$token_name" "$JOB_ID" "$FREE_SLOT" &
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
echo "All LLM tests completed!"
echo "Results: $BASE_OUTPUT_DIR"
echo ""
TOTAL_RUNS=$(tail -n +2 "$SUMMARY_FILE" | wc -l)
FAILED=$(grep -c "COMPILE_ERROR" "$SUMMARY_FILE" 2>/dev/null || echo "0")
echo "Total: $TOTAL_RUNS | Success: $((TOTAL_RUNS - FAILED)) | Failed: $FAILED"
echo ""
echo "Runtime summary (sorted):"
tail -n +2 "$SUMMARY_FILE" | grep -v "COMPILE_ERROR" | sort -t',' -k4 -n | while IFS=',' read -r tok noc test runtime; do
    printf "  %s / %s / %-20s : %ss\n" "$tok" "$noc" "$test" "$runtime"
done
echo "==========================================="
