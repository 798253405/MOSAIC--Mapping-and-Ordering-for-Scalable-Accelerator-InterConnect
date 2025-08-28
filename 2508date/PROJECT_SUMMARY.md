# NoC-based LLM注意力机制加速器项目

## 项目概述

基于Network-on-Chip (NoC)的LLM注意力机制硬件加速器，通过多个MAC单元并行计算和智能任务映射策略实现高效加速。

## 系统架构

### 核心组件
- **LLMMACnet** (`llmmacnet.cpp/hpp`): LLM网络控制器，管理任务分发和结果收集
- **LLMMAC** (`llmmac.cpp/hpp`): MAC计算单元，执行注意力计算
- **NoC互联**: 16×16网格，4个内存节点，支持多种映射策略
- **参数配置** (`parameters.hpp`): 系统参数和优化开关

### 关键特性
1. **SAMOS自适应映射**: 基于采样的动态任务分配（Hamilton最大余数法）
2. **Flit级别翻转优化**: 减少数据传输功耗
3. **分层调试系统**: 三级调试输出控制（Level 1-3）
4. **可配置性能监控**: 支持时间和周期双模式报告
5. **三种测试配置**:
   - Test Case 1: 4×4小矩阵（快速验证）
   - Test Case 2: 128×128矩阵（LLaMA风格）
   - Test Case 3: 256×256大矩阵（性能测试）

## 技术实现

### 任务映射策略（互斥选择）
```cpp
// parameters.hpp配置
#define rowmapping         // 基准row映射
//#define YZSAMOSSampleMapping  // SAMOS自适应映射
//#define flitLevelFlippingSwitch  // Ordering优化
```

### SAMOS工作流程
1. **采样阶段** (前10个周期): 收集MAC性能数据
2. **分配阶段**: Hamilton算法计算最优任务分配
3. **执行阶段**: 按优化后的映射执行任务

### 注意力计算流程
```cpp
// 每个任务计算 Q·K / sqrt(d_k) 并应用tanh
1. 接收Query和Key向量
2. 计算点积
3. 缩放除以sqrt(向量维度)
4. 应用tanh激活函数
5. 更新输出矩阵对应位置
```

## 性能优化成果

### 已完成优化
- ✅ 修复MAC输出全零问题（消息类型错误）
- ✅ 等待周期从10,000减少到100（**41倍加速**）
- ✅ 解决调试输出导致的超时问题
- ✅ 实现SAMOS与Row mapping的互斥控制

### 性能对比（预期）
| 配置 | 小矩阵 | 中等矩阵 | 大矩阵 |
|------|--------|----------|---------|
| Baseline | 基准 | 基准 | 基准 |
| SAMOS | 0% | 10-20% | 15-30% |
| Ordering | 0% | 5-10% | 5-15% |
| SAMOS+Ordering | 0% | 15-25% | 20-35% |

## 配置选项

### 调试级别控制 (parameters.hpp)
```cpp
// 三级调试系统
#define LLM_DEBUG_LEVEL 1  // 1-3，控制调试信息详细程度

// Level 1: 基础周期进度 + 层级信息 + 系统时间戳
// Level 2: 任务级别和节点级别详细信息  
// Level 3: 详细的数据包级别调试
```

### 性能监控配置 (parameters.hpp)
```cpp
// 性能监控配置
#define PERF_REPORT_ENABLED        // 启用/禁用性能报告
#define PERF_REPORT_INTERVAL_SEC 30 // 时间间隔报告（秒）
#define PERF_REPORT_INTERVAL_CYCLES 50000 // 周期间隔报告
#define PERF_USE_TIME_BASED true   // true=时间模式，false=周期模式
```

### 测试用例选择
```cpp
#define LLM_TEST_CASE 1 // 1=4x4小矩阵, 2=128x128中等, 3=256x256大矩阵
```

### NoC规模配置
```cpp
#define MemNode4_16X16   // 16x16 NoC，4个内存控制器（推荐Test Case 2）
// 其他选项：MemNode2_4x4, MemNode4_4X4, MemNode4_8X8, MemNode4_32X32
```

## 使用指南

### 编译运行
```bash
cd Debug
make clean && make -j8
./2508date
```

### 配置切换

**重要：所有配置都在`src/parameters.hpp`中集中管理**

```cpp
// 1. 选择测试用例（根据需求选择合适规模）
#define LLM_TEST_CASE 2  // <-- 修改这里切换测试用例
// Test Case 1: 4×4矩阵 - 快速功能验证
// Test Case 2: 128×128矩阵 - 模拟实际LLaMA注意力计算（推荐）
// Test Case 3: 256×256矩阵 - 大规模性能测试

// 2. 选择NoC规模（匹配测试用例）
#define MemNode4_16X16   // 当前：16×16 NoC（适合Test Case 2）
// 推荐配置：
// Test Case 1 → MemNode4_4X4
// Test Case 2 → MemNode4_16X16
// Test Case 3 → MemNode4_32X32

// 3. 选择映射策略（rowmapping和SAMOS二选一）
#define rowmapping           // 基准映射（默认）
//#define YZSAMOSSampleMapping // SAMOS自适应映射

// 4. Ordering优化（可选）
//#define flitLevelFlippingSwitch  // 取消注释以启用
```

**配置组合建议：**
- 初始测试：Test Case 1 + rowmapping
- 性能基准：Test Case 2 + rowmapping  
- SAMOS验证：Test Case 2 + YZSAMOSSampleMapping
- 最大优化：Test Case 2 + YZSAMOSSampleMapping + flitLevelFlippingSwitch

### 验证结果
```bash
cd src/pycharmCodes
python3 hellotest.py 2  # 运行对应测试用例验证
```

## 关键参数调优

### SAMOS参数
- `samplingWindowLength`: 采样窗口长度（当前10，建议5-20）
- Hamilton算法权重因子
- 任务批量大小

### NoC配置建议
- 4×4矩阵 → 4×4 NoC
- 128×128矩阵 → 16×16 NoC
- 256×256矩阵 → 32×32 NoC

## 项目文件说明

### 核心代码
- `src/llmmac.cpp/hpp`: MAC计算单元实现
- `src/llmmacnet.cpp/hpp`: 网络控制器和任务管理
- `src/parameters.hpp`: 系统配置参数
- `src/NoC/`: NoC路由器和互联实现

### 验证脚本
- `src/pycharmCodes/hellotest.py`: Python功能验证脚本

### 输出文件
- `src/output/llm_attention_output.txt`: 最终计算结果
- `src/output/cpp_*.txt`: 中间矩阵数据

## 下一步工作

1. **参数优化**: 调整SAMOS采样窗口获得最佳性能
2. **大规模测试**: 在256×256矩阵上完整验证
3. **负载均衡**: 改进Hamilton算法的任务分配策略
4. **性能分析**: 添加更详细的性能计数器

## 联系信息

项目维护：YZ
最后更新：2025-08-28