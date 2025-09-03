# Batch Testing Guide for NoC Simulation

## Current Git Status
- Branch: `ongoing`
- Status: Clean working tree
- Latest commit: `abca20f`

## Project Structure
```
2508date/
├── Debug/              # Build directory
│   ├── makefile
│   ├── src/           # Object files
│   └── 2508date       # Executable
├── src/
│   ├── main.cpp       # Main file with statistics
│   ├── parameters.hpp # Configuration file
│   ├── Input/         # Network and weight files
│   │   ├── newnet2.txt
│   │   └── weight.txt
│   └── NoC/           # NoC implementation
└── output/            # Test results directory
```

## Key Configuration Files

### 1. Network Configuration: `src/Input/newnet2.txt`
```
Input 224 224 3
Conv2D 3 5 5 16 relu 0 1
```

### 2. Parameters: `src/parameters.hpp`
Key settings to check:
- Line 69: `#define MemNode4_8X8` (or other NoC sizes)
- Line 76-81: Test case selection (case1_default to case6_MOSAIC2)
- Line 135: `#define samplingWindowLength 100`

## Compilation Commands

### From Debug directory:
```bash
cd Debug
make clean
make all
```

### Check executable:
```bash
ls -la 2508date
```

## Batch Test Scripts

### 1. Serial Batch Test: `run_batch_tests_serial.sh`
- Runs 30 configurations sequentially (5 NoC sizes × 6 test cases)
- Output: `output/batchCNN_serial_YYYYMMDD_HHMMSS/`
- Runtime: ~3-4 hours

### 2. Parallel Batch Test: `run_batch_tests_parallel.sh`
- Runs with 6 parallel workers
- Output: `output/batchCNN_parallel_YYYYMMDD_HHMMSS/`
- Runtime: ~1 hour
- Configurable: `MAX_PARALLEL_SLOTS=6` (line 4)

## Parallel Testing Architecture

### Core Design: Compile-Execute Pipeline

**NOT compiling all configurations at once**, but using a **pipeline approach**:

```
Compilation Stage → Execution Stage (parallel)
       ↓                    ↓
Compile next one → Multiple slots run simultaneously
```

### Implementation Details

#### 1. Environment Preparation
- Creates 6 independent execution directories (`exec_1` to `exec_6`)
- Each directory is a complete project copy to avoid file conflicts
- One dedicated compilation directory (`compile_dir`)

#### 2. Compilation Pipeline
```bash
for each configuration (30 total):
    1. Modify parameters.hpp (NoC size + test case)
    2. Compile to generate binary
    3. Save binary as binary_${noc}_${case}
    4. Immediately assign to idle execution slot
```

#### 3. Parallel Execution Management
```bash
# 6 execution slots running in parallel
Slot 1: Running 4x4_case1 ─┐
Slot 2: Running 4x4_case2 ─┤
Slot 3: Running 4x4_case3 ─┼─→ Simultaneous execution
Slot 4: Running 4x4_case4 ─┤
Slot 5: Running 4x4_case5 ─┤
Slot 6: Running 4x4_case6 ─┘

# When a slot finishes, immediately assign next task
```

### Key Technical Features

#### File Locking Mechanism:
```bash
# Use flock to ensure atomic slot allocation
exec 200>$LOCKFILE
flock -x 200
# Allocate free slot
flock -u 200
```

#### Independent Execution Environment:
- Each slot has its own working directory
- Avoids file conflicts during parallel execution
- Output files use absolute paths

#### Smart Scheduling:
```bash
# Find free slot
for slot in {1..6}; do
    if [ ! -f "slot_${slot}.lock" ]; then
        # Assign this slot
        break
    fi
done
```

### Timeline Example

```
Time →
T0: Compile 4x4_case1
T1: Compile 4x4_case2, [Execute 4x4_case1 in Slot1]
T2: Compile 4x4_case3, [Execute 4x4_case2 in Slot2]
T3: Compile 4x4_case4, [Execute 4x4_case3 in Slot3]
...
T6: Compile 4x4_case7, [All 6 slots running]
T7: 4x4_case1 done → Slot1 free → Assign 4x4_case7
```

### Advantages

1. **Full CPU utilization**: Compilation and execution in parallel
2. **Memory efficiency**: No need to store 30 compiled binaries
3. **Flexible scheduling**: Fast tests complete and run next immediately
4. **Fault tolerance**: One test failure doesn't affect others

### Performance Comparison

| Method | Time | Description |
|--------|------|-------------|
| Serial | 3-4 hours | 30 tests run sequentially |
| Parallel | ~1 hour | 6 slots running concurrently |
| Speedup | 3-4x | Limited by slowest test |

This design is particularly suitable for NoC simulations where runtime varies greatly (4x4 takes minutes, 32x32 takes hours)

## Running Batch Tests

### Step 1: Ensure scripts are in place
```bash
ls -la run_batch_tests*.sh
```

### Step 2: Run parallel batch test (recommended)
```bash
./run_batch_tests_parallel.sh
```

### Step 3: Monitor progress
```bash
# Watch output folder being populated
watch -n 5 "ls -la output/batchCNN_parallel_*/4_*.txt | wc -l"

# Check a specific test progress
tail -f output/batchCNN_parallel_*/4_8x8_case1_default.txt
```

## Test Cases Explained

1. **case1_default**: Basic rowmapping
2. **case2_samos**: SAMOS adaptive mapping
3. **case3_affiliatedordering**: rowmapping + YzAffiliatedOrdering
4. **case4_seperratedordering**: rowmapping + YzAffiliatedOrdering + YZSeperatedOrdering_reArrangeInput
5. **case5_MOSAIC1**: SAMOS + YzAffiliatedOrdering
6. **case6_MOSAIC2**: SAMOS + YzAffiliatedOrdering + YZSeperatedOrdering_reArrangeInput

## NoC Configurations
- **MemNode2_4X4**: 4×4 NoC, 2 memory nodes
- **MemNode4_4X4**: 4×4 NoC, 4 memory nodes
- **MemNode4_8X8**: 8×8 NoC, 4 memory nodes
- **MemNode4_16X16**: 16×16 NoC, 4 memory nodes
- **MemNode4_32X32**: 32×32 NoC, 4 memory nodes

## Analysis Scripts

### 1. Analyze results: `analyze_batch_results_v2.sh`
```bash
./analyze_batch_results_v2.sh output/batchCNN_parallel_YYYYMMDD_HHMMSS
```

### 2. Python visualization: `professional_analysis.py`
```bash
python3 professional_analysis.py output/batchCNN_parallel_YYYYMMDD_HHMMSS
```

## Important Notes for Large Memory Machine

1. **Memory Requirements**:
   - 4×4 NoC: ~1GB RAM
   - 8×8 NoC: ~2GB RAM
   - 16×16 NoC: ~4GB RAM
   - 32×32 NoC: ~8-16GB RAM

2. **Runtime Expectations**:
   - Larger NoCs (32×32) can take 10-100x longer than smaller ones
   - SAMOS tests may take slightly less time than baseline

3. **File Output**:
   - Each test generates a .txt file with cycle progress and BATCH_STATS
   - Final summary in `summary_all.txt`

## Quick Single Test

To run a single configuration test:
```bash
cd Debug

# Configure test case in src/parameters.hpp
# Line 76-81: Choose one case (e.g., #define case2_samos)
# Line 69: Choose NoC size (e.g., #define MemNode4_8X8)

# Compile
make clean && make all

# Run
./2508date > ../output/test_output.txt 2>&1
```

## Troubleshooting

1. **If compilation fails**: Check g++ version (needs C++11 support)
2. **If tests hang**: Large NoCs (32×32) can take hours
3. **If out of memory**: Reduce NoC size or use machine with more RAM
4. **Missing output**: Ensure output/ directory exists

## Expected Output Format

Each test file contains:
```
BATCH_STATS: total_cycles=XXX packetid=XXX YZGlobalFlit_id=XXX 
YZGlobalFlitPass=XXX avg_hops_per_flit=XXX 
bit_transition_float_per_flit=XXX bit_transition_fixed_per_flit=XXX
```

## Contact
For issues or questions about the test setup, check the git log for recent changes or review src/main.cpp lines 147-156 for output configuration.