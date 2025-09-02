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
3. **case3_affiliatedordering**: rowmapping + flitLevelFlipping
4. **case4_seperratedordering**: rowmapping + flitLevelFlipping + reArrangeInput
5. **case5_MOSAIC1**: SAMOS + flitLevelFlipping
6. **case6_MOSAIC2**: SAMOS + flitLevelFlipping + reArrangeInput

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