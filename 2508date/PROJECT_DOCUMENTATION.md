# NoC-based LLM/CNN 加速器项目文档

## 项目概述
基于Network-on-Chip (NoC)的深度学习硬件加速器，支持CNN和LLM两种工作模式，通过多个MAC单元并行计算和智能任务映射策略实现高效加速。

## 系统架构

### 核心组件
- **LLMMACnet** (`llmmacnet.cpp/hpp`): LLM网络控制器，管理任务分发和结果收集
- **LLMMAC** (`llmmac.cpp/hpp`): MAC计算单元，执行注意力计算
- **MACnet** (`MACnet.cpp/hpp`): CNN网络控制器
- **MAC** (`MAC.cpp/hpp`): CNN MAC计算单元
- **NoC互联**: 16×16网格，4个内存节点，支持多种映射策略
- **参数配置** (`parameters.hpp`): 系统参数和优化开关

### 关键特性
1. **SAMOS自适应映射**: 基于采样的动态任务分配（Hamilton最大余数法）
2. **Flit级别翻转优化**: 减少数据传输功耗
3. **分层调试系统**: 三级调试输出控制
4. **可配置性能监控**: 支持时间和周期双模式报告

## LLM模式详解

### 当前配置 (8×128矩阵)
- **输入矩阵 (X_input)**: 8×4096
- **权重矩阵 (Wq)**: 128×4096  
- **输出矩阵 (Q)**: 8×128 (Q = X @ Wq^T)
- **数据来源**: `../src/Input/llminput/`
  - `X_input.txt`: 输入数据
  - `Wq.txt`: Query权重

### 任务分解机制
每个输出像素(pixel)计算分解为多个子任务：
- 4096维dot product分成128个32元素的chunks
- 每个子任务处理部分数据，生成partial sum
- 最终聚合所有partial sums得到像素值

### LLM状态机流程
```
State 0 (IDLE) → State 1 (REQUEST) → State 2 (WAIT) → State 3 (COMPUTE) → State 4 (COMPLETE)
```

### 包类型
- **Type 0 (REQUEST)**: MAC请求数据 (MAC → Memory)
- **Type 1 (RESPONSE)**: Memory返回数据 (Memory → MAC)  
- **Type 2/3 (RESULT)**: 计算结果 (MAC → Memory)

## CNN模式详解

### 测试配置
- **Test Case 1**: 4×4小矩阵（快速验证）
- **Test Case 2**: 128×128矩阵（标准测试）
- **Test Case 3**: 256×256大矩阵（性能测试）

### 卷积计算流程
1. 接收输入特征图和权重
2. 执行卷积运算
3. 应用激活函数
4. 更新输出特征图

## 编译与运行

### 编译命令
```bash
cd Debug
g++ -std=c++11 -o 2508date ../src/*.cpp ../src/NoC/*.cpp -I../src -I../src/NoC
```

### 运行测试
```bash
./2508date
```

### 配置参数 (parameters.hpp)
```cpp
// 选择工作模式
#define LLM_TEST_MODE  // LLM模式
//#define CNN_TEST_MODE  // CNN模式

// 选择映射策略（互斥）
#define rowmapping         // 基准row映射
//#define YZSAMOSSampleMapping  // SAMOS自适应映射
//#define YzAffiliatedOrdering  // Ordering优化

// 调试级别
#define LLM_DEBUG_LEVEL 1  // 1=基础, 2=详细, 3=全部
```

## 性能优化策略

### 1. SAMOS自适应映射
- **采样阶段**: 前10个周期收集MAC性能数据
- **分配阶段**: Hamilton算法计算最优任务分配
- **执行阶段**: 按优化后的映射执行任务

### 2. 数据排序优化
- **Affiliated Ordering**: Query和Key矩阵根据Key的bit count一起排序
- **Separated Ordering**: Query和Key矩阵独立排序
- **目标**: 减少传输过程中的bit flip

### 3. Flit翻转优化
- 跟踪相邻flit之间的bit差异
- 统计翻转频率用于功耗分析
- 支持IEEE754浮点数bit级分析

## 调试与监控

### 调试宏定义
```cpp
LLM_INFO(x)    // Level 1: 基础信息
LLM_DEBUG(x)   // Level 2: 调试信息
LLM_TRACE(x)   // Level 3: 详细跟踪
```

### 性能指标
- 任务完成率 (tasks/cycle)
- 网络利用率
- Bit flip统计
- 延迟分析

## 文件结构
```
2508date/
├── src/
│   ├── main.cpp           # 主程序入口
│   ├── parameters.hpp     # 配置参数
│   ├── llmmac.cpp/hpp     # LLM MAC单元
│   ├── llmmacnet.cpp/hpp  # LLM网络控制器
│   ├── MAC.cpp/hpp        # CNN MAC单元
│   ├── MACnet.cpp/hpp     # CNN网络控制器
│   ├── NoC/               # NoC组件
│   └── Input/
│       └── llminput/      # LLM输入数据
│           ├── X_input.txt
│           └── Wq.txt
├── Debug/                 # 编译输出
└── output/               # 运行结果

```

## 关键发现
- LLM数据的bit分布相对均匀（17-18 bits平均值）
- 这种均匀分布限制了排序优化的效果
- 8×128配置下任务粒度更适合实际LLM推理场景

## 注意事项
1. 确保输入文件路径正确: `../src/Input/llminput/`
2. 文件缺失时程序会使用随机数据
3. 调试输出可能影响性能，生产环境建议设置DEBUG_LEVEL=1
4. NoC路由可能存在死锁，注意任务映射策略选择