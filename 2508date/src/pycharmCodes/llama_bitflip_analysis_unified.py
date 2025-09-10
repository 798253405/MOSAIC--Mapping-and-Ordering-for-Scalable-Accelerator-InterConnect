#!/usr/bin/env python3
"""
LLaMA Bit Flip Analysis Unified Tool
=====================================

统一的LLaMA attention矩阵bit翻转分析工具，整合了多个分析功能：
1. 真实LLaMA数据分布分析
2. Bit翻转优化策略对比
3. NoC传输模拟与评估
4. 排序算法效果验证

Author: YZ
Date: 2025-09
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
from enum import Enum
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# ========================= 配置类 =========================

@dataclass
class TaskConfig:
    """任务配置参数"""
    matrix_size: int = 512  # 原始矩阵大小
    pixel_size: int = 64    # 每个pixel矩阵大小
    task_rows: int = 8      # 每个task的行数
    query_cols: int = 8     # Query列数
    key_cols: int = 8       # Key列数
    link_width: int = 128   # 链路宽度(bits)
    quantization_bits: int = 8  # 量化bit数
    random_seed: Optional[int] = 42
    output_dir: str = "./output"
    
    def __post_init__(self):
        """验证配置参数"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class SortingStrategy(Enum):
    """排序策略枚举"""
    BASELINE = "baseline"
    ONES_COUNT = "ones_count"
    MAGNITUDE = "magnitude"
    FIXED_POINT = "fixed_point"
    SIGN_MAGNITUDE = "sign_magnitude"

# ========================= 数据生成与加载 =========================

class LlamaDataLoader:
    """LLaMA数据加载器"""
    
    @staticmethod
    def load_real_matrix(filename: str) -> np.ndarray:
        """加载真实LLaMA矩阵数据"""
        print(f"Loading {filename}...")
        data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            # 跳过前两行（维度信息）
            for line in lines[2:]:
                if line.strip():
                    values = line.strip().split()
                    data.extend([float(v) for v in values])
        return np.array(data, dtype=np.float32)
    
    @staticmethod
    def generate_random_matrices(size: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成随机Query和Key矩阵（Xavier初始化）"""
        std = np.sqrt(2.0 / size)
        query = np.random.normal(0, std, (size, size)).astype(np.float32)
        key = np.random.normal(0, std, (size, size)).astype(np.float32)
        return query, key

class LlamaAttentionMatrixGenerator:
    """LLaMA attention矩阵生成器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.hidden_dim = config.matrix_size
        
    def generate_qk_matrices(self, use_real_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """生成或加载Query和Key矩阵"""
        if use_real_data:
            base_path = "/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/"
            try:
                query = LlamaDataLoader.load_real_matrix(base_path + "llama_query.txt")
                key = LlamaDataLoader.load_real_matrix(base_path + "llama_key.txt")
                # Reshape to matrix if needed
                matrix_size = self.config.matrix_size
                if len(query) >= matrix_size * matrix_size:
                    query = query[:matrix_size * matrix_size].reshape(matrix_size, matrix_size)
                    key = key[:matrix_size * matrix_size].reshape(matrix_size, matrix_size)
                return query, key
            except:
                print("Failed to load real data, using random instead")
        
        return LlamaDataLoader.generate_random_matrices(self.config.matrix_size)

# ========================= 矩阵分解与处理 =========================

class PixelMatrixDecomposer:
    """Pixel矩阵分解器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.num_pixels_per_dim = config.matrix_size // config.pixel_size
        self.total_pixels = self.num_pixels_per_dim ** 2
        
    def decompose_to_pixels(self, query_matrix: np.ndarray, key_matrix: np.ndarray) -> List[Dict]:
        """将大矩阵分解为pixel子矩阵"""
        pixels = []
        
        for i in range(self.num_pixels_per_dim):
            for j in range(self.num_pixels_per_dim):
                start_row = i * self.config.pixel_size
                end_row = start_row + self.config.pixel_size
                start_col = j * self.config.pixel_size  
                end_col = start_col + self.config.pixel_size
                
                pixel_info = {
                    'pixel_id': i * self.num_pixels_per_dim + j,
                    'position': (i, j),
                    'query_pixel': query_matrix[start_row:end_row, start_col:end_col],
                    'key_pixel': key_matrix[start_row:end_row, start_col:end_col]
                }
                pixels.append(pixel_info)
        
        return pixels

class TaskProcessor:
    """Task处理器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        
    def process_pixel_to_tasks(self, pixel_info: Dict) -> List[Dict]:
        """将一个pixel矩阵分解为多个task"""
        query_pixel = pixel_info['query_pixel']
        key_pixel = pixel_info['key_pixel']
        
        tasks = []
        rows_per_task = self.config.task_rows
        num_tasks = self.config.pixel_size // rows_per_task
        
        for task_id in range(num_tasks):
            start_row = task_id * rows_per_task
            end_row = start_row + rows_per_task
            
            query_task = query_pixel[start_row:end_row, :self.config.query_cols]
            key_task = key_pixel[start_row:end_row, :self.config.key_cols]
            task_data = np.concatenate([query_task, key_task], axis=1)
            
            tasks.append({
                'pixel_id': pixel_info['pixel_id'],
                'task_id': task_id,
                'global_task_id': pixel_info['pixel_id'] * num_tasks + task_id,
                'task_data': task_data,
                'query_data': query_task,
                'key_data': key_task,
                'total_elements': task_data.size
            })
            
        return tasks

# ========================= Bit翻转分析 =========================

class BitFlipAnalyzer:
    """Bit翻转分析器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    def count_ones_ieee754(self, value: float) -> int:
        """计算IEEE754浮点数的1-bit数"""
        try:
            bytes_val = struct.pack('f', np.float32(value))
            int_val = struct.unpack('I', bytes_val)[0]
            return bin(int_val).count('1')
        except:
            return 0
    
    def quantize_data(self, data: np.ndarray) -> np.ndarray:
        """量化浮点数据为整数"""
        if data.size == 0:
            return np.array([], dtype=np.int32)
        
        max_val = np.max(np.abs(data))
        if max_val == 0:
            return np.zeros_like(data, dtype=np.int32)
        
        scale = max_val / (2**(self.config.quantization_bits-1) - 1)
        quantized = np.round(data / scale).astype(np.int32)
        quantized = np.clip(quantized, 
                          -(2**(self.config.quantization_bits-1)), 
                          2**(self.config.quantization_bits-1)-1)
        return quantized
    
    def calculate_bit_flips(self, sequence: List[Union[int, float]]) -> int:
        """计算序列传输的bit翻转次数"""
        if len(sequence) < 2:
            return 0
        
        total_flips = 0
        for i in range(1, len(sequence)):
            if isinstance(sequence[i], float):
                # 浮点数用IEEE754比较
                val1 = struct.unpack('I', struct.pack('f', np.float32(sequence[i-1])))[0]
                val2 = struct.unpack('I', struct.pack('f', np.float32(sequence[i])))[0]
            else:
                # 整数直接比较
                val1 = int(sequence[i-1])
                val2 = int(sequence[i])
            
            xor_result = val1 ^ val2
            flips = bin(xor_result).count('1')
            total_flips += flips
        
        return total_flips
    
    def sort_by_strategy(self, data: np.ndarray, strategy: SortingStrategy) -> np.ndarray:
        """根据策略排序数据"""
        flat_data = data.flatten()
        
        if strategy == SortingStrategy.BASELINE:
            return flat_data
        
        elif strategy == SortingStrategy.ONES_COUNT:
            # 按1-bit数排序
            ones_counts = [self.count_ones_ieee754(float(val)) for val in flat_data]
            sorted_indices = sorted(range(len(flat_data)), 
                                  key=lambda i: (ones_counts[i], flat_data[i]))
            return flat_data[sorted_indices]
        
        elif strategy == SortingStrategy.MAGNITUDE:
            # 按数值大小排序
            return np.sort(flat_data)
        
        elif strategy == SortingStrategy.FIXED_POINT:
            # 转换为定点数后排序
            quantized = self.quantize_data(flat_data)
            sorted_indices = np.argsort(quantized)
            return flat_data[sorted_indices]
        
        elif strategy == SortingStrategy.SIGN_MAGNITUDE:
            # 先正后负，各自按幅度排序
            positives = flat_data[flat_data >= 0]
            negatives = flat_data[flat_data < 0]
            positives_sorted = np.sort(positives)
            negatives_sorted = np.sort(negatives)[::-1]
            return np.concatenate([positives_sorted, negatives_sorted])
        
        return flat_data
    
    def analyze_task(self, task_info: Dict) -> Dict:
        """分析单个task的bit翻转情况"""
        task_data = task_info['task_data']
        results = {}
        
        # 测试所有排序策略
        for strategy in SortingStrategy:
            sorted_data = self.sort_by_strategy(task_data, strategy)
            flips = self.calculate_bit_flips(sorted_data.tolist())
            results[strategy.value] = {
                'flips': flips,
                'sequence': sorted_data[:10].tolist()  # 保存前10个用于调试
            }
        
        # 计算优化效果
        baseline_flips = results[SortingStrategy.BASELINE.value]['flips']
        for strategy in SortingStrategy:
            if strategy != SortingStrategy.BASELINE:
                strategy_flips = results[strategy.value]['flips']
                reduction = baseline_flips - strategy_flips
                percentage = (reduction / baseline_flips * 100) if baseline_flips > 0 else 0
                results[strategy.value]['reduction'] = reduction
                results[strategy.value]['percentage'] = percentage
        
        return results

# ========================= 数据分布分析 =========================

class DataDistributionAnalyzer:
    """数据分布分析器"""
    
    @staticmethod
    def analyze_distribution(data: np.ndarray, name: str) -> Dict:
        """分析数据分布特性"""
        stats = {
            'name': name,
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(np.median(data)),
            'percentiles': {
                '1%': float(np.percentile(data, 1)),
                '25%': float(np.percentile(data, 25)),
                '75%': float(np.percentile(data, 75)),
                '99%': float(np.percentile(data, 99))
            }
        }
        
        # 符号分布
        stats['positive_ratio'] = float((data > 0).sum() / len(data))
        stats['negative_ratio'] = float((data < 0).sum() / len(data))
        stats['near_zero_ratio'] = float((np.abs(data) < 0.001).sum() / len(data))
        
        # 1-bit分布（采样）
        analyzer = BitFlipAnalyzer(TaskConfig())
        sample_size = min(1000, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_indices]
        ones_counts = [analyzer.count_ones_ieee754(float(val)) for val in sample_data]
        
        stats['ones_bit_mean'] = float(np.mean(ones_counts))
        stats['ones_bit_std'] = float(np.std(ones_counts))
        
        return stats

# ========================= 完整模拟器 =========================

class LlamaAttentionSimulator:
    """LLaMA attention模拟器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.matrix_generator = LlamaAttentionMatrixGenerator(config)
        self.pixel_decomposer = PixelMatrixDecomposer(config)
        self.task_processor = TaskProcessor(config)
        self.bit_analyzer = BitFlipAnalyzer(config)
    
    def run_complete_simulation(self, use_real_data: bool = False, 
                              max_pixels: Optional[int] = None) -> Dict:
        """运行完整模拟"""
        print("=== LLaMA Attention Bit Flip Simulation ===\n")
        
        # 1. 生成/加载矩阵
        print("1. Generating/Loading matrices...")
        query_matrix, key_matrix = self.matrix_generator.generate_qk_matrices(use_real_data)
        print(f"   Matrix shape: {query_matrix.shape}")
        
        # 2. 分解为pixels
        print("2. Decomposing to pixels...")
        pixels = self.pixel_decomposer.decompose_to_pixels(query_matrix, key_matrix)
        if max_pixels:
            pixels = pixels[:max_pixels]
        print(f"   Processing {len(pixels)} pixels")
        
        # 3. 处理tasks
        print("3. Processing tasks...")
        all_tasks = []
        for pixel in pixels:
            tasks = self.task_processor.process_pixel_to_tasks(pixel)
            all_tasks.extend(tasks)
        print(f"   Total tasks: {len(all_tasks)}")
        
        # 4. 分析bit翻转
        print("4. Analyzing bit flips...")
        strategy_totals = {strategy.value: {'flips': 0, 'reduction': 0} 
                          for strategy in SortingStrategy}
        
        for i, task in enumerate(all_tasks):
            if i % 100 == 0:
                print(f"   Processing task {i}/{len(all_tasks)}...")
            
            analysis = self.bit_analyzer.analyze_task(task)
            
            for strategy in SortingStrategy:
                strategy_totals[strategy.value]['flips'] += analysis[strategy.value]['flips']
                if strategy != SortingStrategy.BASELINE:
                    strategy_totals[strategy.value]['reduction'] += analysis[strategy.value].get('reduction', 0)
        
        # 5. 计算总体统计
        baseline_total = strategy_totals[SortingStrategy.BASELINE.value]['flips']
        results = {
            'config': {
                'matrix_size': self.config.matrix_size,
                'pixel_size': self.config.pixel_size,
                'total_pixels': len(pixels),
                'total_tasks': len(all_tasks)
            },
            'strategy_results': {}
        }
        
        print("\n=== Results by Strategy ===")
        for strategy in SortingStrategy:
            strategy_name = strategy.value
            total_flips = strategy_totals[strategy_name]['flips']
            
            if strategy == SortingStrategy.BASELINE:
                reduction_pct = 0.0
            else:
                reduction = baseline_total - total_flips
                reduction_pct = (reduction / baseline_total * 100) if baseline_total > 0 else 0
            
            results['strategy_results'][strategy_name] = {
                'total_flips': total_flips,
                'reduction_percentage': reduction_pct
            }
            
            print(f"{strategy_name:15s}: {total_flips:10d} flips", end="")
            if strategy != SortingStrategy.BASELINE:
                print(f" (reduction: {reduction_pct:.2f}%)")
            else:
                print()
        
        return results

# ========================= 主要分析函数 =========================

def analyze_real_llama_distribution():
    """分析真实LLaMA数据分布"""
    print("\n=== Analyzing Real LLaMA Data Distribution ===\n")
    
    base_path = "/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/"
    
    try:
        # 加载数据
        query_data = LlamaDataLoader.load_real_matrix(base_path + "llama_query.txt")
        key_data = LlamaDataLoader.load_real_matrix(base_path + "llama_key.txt")
        value_data = LlamaDataLoader.load_real_matrix(base_path + "llama_value.txt")
        
        # 分析分布
        query_stats = DataDistributionAnalyzer.analyze_distribution(query_data, "Query")
        key_stats = DataDistributionAnalyzer.analyze_distribution(key_data, "Key")
        value_stats = DataDistributionAnalyzer.analyze_distribution(value_data, "Value")
        
        # 打印结果
        for stats in [query_stats, key_stats, value_stats]:
            print(f"\n{stats['name']} Matrix:")
            print(f"  Shape: {stats['count']} elements")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Median: {stats['median']:.6f}")
            print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
            print(f"  Near zero ratio: {stats['near_zero_ratio']:.2%}")
            print(f"  Average 1-bits: {stats['ones_bit_mean']:.2f} ± {stats['ones_bit_std']:.2f}")
        
        return query_stats, key_stats, value_stats
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def compare_real_vs_random():
    """比较真实数据和随机数据的bit翻转优化效果"""
    print("\n=== Comparing Real vs Random Data ===\n")
    
    config = TaskConfig(
        matrix_size=512,
        pixel_size=64,
        task_rows=8,
        random_seed=42
    )
    
    simulator = LlamaAttentionSimulator(config)
    
    # 测试真实数据
    print("Testing with REAL data...")
    real_results = simulator.run_complete_simulation(use_real_data=True, max_pixels=10)
    
    # 测试随机数据
    print("\nTesting with RANDOM data...")
    random_results = simulator.run_complete_simulation(use_real_data=False, max_pixels=10)
    
    # 对比结果
    print("\n=== Comparison Summary ===")
    print(f"{'Strategy':15s} {'Real Data':>15s} {'Random Data':>15s} {'Difference':>15s}")
    print("-" * 60)
    
    for strategy in SortingStrategy:
        if strategy == SortingStrategy.BASELINE:
            continue
        
        real_pct = real_results['strategy_results'][strategy.value]['reduction_percentage']
        random_pct = random_results['strategy_results'][strategy.value]['reduction_percentage']
        diff = real_pct - random_pct
        
        print(f"{strategy.value:15s} {real_pct:14.2f}% {random_pct:14.2f}% {diff:+14.2f}%")
    
    return real_results, random_results

def run_quick_test():
    """快速测试基本功能"""
    print("\n=== Quick Test ===\n")
    
    config = TaskConfig(
        matrix_size=128,  # 使用较小的矩阵加快测试
        pixel_size=64,
        task_rows=8
    )
    
    simulator = LlamaAttentionSimulator(config)
    results = simulator.run_complete_simulation(use_real_data=False, max_pixels=2)
    
    # 找出最佳策略
    best_strategy = None
    best_reduction = 0
    
    for strategy in SortingStrategy:
        if strategy == SortingStrategy.BASELINE:
            continue
        reduction = results['strategy_results'][strategy.value]['reduction_percentage']
        if reduction > best_reduction:
            best_reduction = reduction
            best_strategy = strategy.value
    
    print(f"\nBest strategy: {best_strategy} with {best_reduction:.2f}% reduction")
    
    return results

def save_results_to_file(results: Dict, filename: str = "analysis_results.json"):
    """保存结果到文件"""
    output_path = Path("./output") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

# ========================= 可视化 =========================

def plot_strategy_comparison(results_list: List[Tuple[str, Dict]]):
    """绘制策略对比图"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        strategies = [s.value for s in SortingStrategy if s != SortingStrategy.BASELINE]
        x = np.arange(len(strategies))
        width = 0.35
        
        for i, (label, results) in enumerate(results_list):
            reductions = [results['strategy_results'][s]['reduction_percentage'] 
                         for s in strategies]
            ax.bar(x + i * width, reductions, width, label=label)
        
        ax.set_xlabel('Sorting Strategy')
        ax.set_ylabel('Bit Flip Reduction (%)')
        ax.set_title('Bit Flip Reduction by Sorting Strategy')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./output/strategy_comparison.png', dpi=150)
        print("Plot saved to ./output/strategy_comparison.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping plot")

# ========================= 主函数 =========================

def main():
    """主函数：运行所有分析"""
    print("=" * 60)
    print("LLaMA Bit Flip Analysis Unified Tool")
    print("=" * 60)
    
    # 1. 快速测试
    print("\n[1] Running quick test...")
    quick_results = run_quick_test()
    
    # 2. 分析真实数据分布
    print("\n[2] Analyzing real LLaMA data distribution...")
    query_stats, key_stats, value_stats = analyze_real_llama_distribution()
    
    # 3. 对比真实vs随机数据
    print("\n[3] Comparing real vs random data...")
    real_results, random_results = compare_real_vs_random()
    
    # 4. 保存结果
    print("\n[4] Saving results...")
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'quick_test': quick_results,
        'real_data_stats': {
            'query': query_stats,
            'key': key_stats,
            'value': value_stats
        } if query_stats else None,
        'comparison': {
            'real': real_results,
            'random': random_results
        }
    }
    save_results_to_file(all_results)
    
    # 5. 可视化（如果可用）
    print("\n[5] Creating visualization...")
    plot_strategy_comparison([('Real Data', real_results), ('Random Data', random_results)])
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()