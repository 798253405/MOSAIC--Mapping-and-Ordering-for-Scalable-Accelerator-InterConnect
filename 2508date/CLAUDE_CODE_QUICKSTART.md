# Claude Code Quick Start Guide

## 🚀 Quick Commands for Claude Code

### Check environment and run batch tests:
```bash
# 1. Check current location
pwd
# Should be: /home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date

# 2. Check git status
git status
# Should show: branch 'ongoing', clean working tree

# 3. Run parallel batch test (recommended)
cd /home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date
./run_batch_tests_parallel.sh

# 4. Monitor progress
watch -n 10 "ls -la output/batchCNN_parallel_*/4_*.txt 2>/dev/null | wc -l; echo '/30 tests completed'"
```

## 📁 Essential Files Location

```
2508date/
├── run_batch_tests_parallel.sh    # Main batch script (6 parallel workers)
├── run_batch_tests_serial.sh      # Serial version (slower)
├── professional_analysis.py       # Results visualization
├── Debug/
│   ├── makefile                   # Build configuration
│   └── 2508date                   # Executable (generated after make)
├── src/
│   ├── main.cpp                   # Line 148-156: Output configuration
│   ├── parameters.hpp             # Line 69: NoC size, Line 76-81: Test cases
│   └── Input/
│       └── newnet2.txt           # Neural network configuration
└── output/                        # All test results go here
```

## ⚙️ Configuration Changes

### To modify NoC size (src/parameters.hpp line 69):
```cpp
//#define MemNode2_4X4    // 4x4 NoC, 2 MC cores
//#define MemNode4_4X4    // 4x4 NoC, 4 MC cores  
#define MemNode4_8X8      // 8x8 NoC, 4 MC cores (current)
//#define MemNode4_16X16   // 16x16 NoC, 4 MC cores
//#define MemNode4_32X32   // 32x32 NoC, 4 MC cores
```

### To modify test case (src/parameters.hpp line 76-81):
```cpp
//#define case1_default
#define case2_samos        // Current setting
//#define case3_affiliatedordering
//#define case4_seperratedordering
//#define case5_MOSAIC1
//#define case6_MOSAIC2
```

## 🔧 Compilation

```bash
cd Debug
make clean && make all
# Verify executable exists
ls -la 2508date
```

## 📊 Check Results

### After batch test completes:
```bash
# Find the output folder (most recent)
ls -ltd output/batchCNN_parallel_* | head -1

# View summary
cat output/batchCNN_parallel_*/summary_all.txt

# Visualize results
python3 professional_analysis.py output/batchCNN_parallel_YYYYMMDD_HHMMSS
```

## ⚠️ Important Notes

1. **Memory requirements**: 32x32 NoC needs 8-16GB RAM
2. **Runtime**: Full batch test takes ~1 hour with 6 parallel workers
3. **Output location**: Always check results in `output/` directory, not `Debug/output/`
4. **Parallel slots**: Adjust `MAX_PARALLEL_SLOTS=6` in run_batch_tests_parallel.sh if needed

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| No executable | Run `cd Debug && make all` |
| Tests hang | Large NoCs take hours, check with `tail -f output/...txt` |
| Out of memory | Reduce NoC size or use larger machine |
| No output files | Ensure `output/` directory exists: `mkdir -p output` |

## 📈 Expected Performance

| NoC Size | Baseline Cycles | SAMOS Improvement | Runtime |
|----------|----------------|-------------------|---------|
| 4x4 | ~5.5M | ~5-6% | 5 min |
| 8x8 | ~2.3M | ~3-5% | 20 min |
| 16x16 | ~2.2M | ~0.5% | 45 min |
| 32x32 | ~2.4M | ~0.5% | 2+ hours |

## 🎯 Single Test Example

```bash
# Configure for 8x8 SAMOS test
cd Debug
# Edit ../src/parameters.hpp: set case2_samos and MemNode4_8X8
make clean && make all
./2508date > ../output/single_test.txt 2>&1
grep BATCH_STATS ../output/single_test.txt
```

## 📝 Key Metrics in Output

- **total_cycles**: Overall simulation time (lower is better)
- **avg_hops_per_flit**: Network traversal distance
- **bit_transition_float_per_flit**: Power metric (lower is better)
- **YZGlobalFlitPass**: Total flits transmitted

Remember: When running on a new machine, first check that the branch is 'ongoing' and working tree is clean!