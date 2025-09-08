import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

@dataclass
class TaskConfig:
    """Task配置参数"""
    matrix_size: int = 512  # 原始矩阵大小
    pixel_size: int = 64    # 每个pixel矩阵大小
    task_rows: int = 8      # 每个task的行数
    query_cols: int = 8     # Query列数
    key_cols: int = 8       # Key列数
    link_width: int = 128   # 链路宽度(bits)
    quantization_bits: int = 8  # 量化bit数

class LlamaAttentionMatrixGenerator:
    """Llama attention矩阵生成器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.hidden_dim = config.matrix_size
        self.head_dim = 64  # 标准attention head维度
        
    def generate_qk_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成Query和Key矩阵"""
        # 模拟Llama的Xavier初始化
        std = np.sqrt(2.0 / self.hidden_dim)
        
        # Query矩阵: [seq_len, hidden_dim]
        query_matrix = np.random.normal(0, std, (self.config.matrix_size, self.config.matrix_size))
        
        # Key矩阵: [seq_len, hidden_dim] 
        key_matrix = np.random.normal(0, std, (self.config.matrix_size, self.config.matrix_size))
        
        return query_matrix, key_matrix

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
                # 计算pixel的起始和结束位置
                start_row = i * self.config.pixel_size
                end_row = start_row + self.config.pixel_size
                start_col = j * self.config.pixel_size  
                end_col = start_col + self.config.pixel_size
                
                # 提取pixel子矩阵
                query_pixel = query_matrix[start_row:end_row, start_col:end_col]
                key_pixel = key_matrix[start_row:end_row, start_col:end_col]
                
                pixel_info = {
                    'pixel_id': i * self.num_pixels_per_dim + j,
                    'position': (i, j),
                    'query_pixel': query_pixel,
                    'key_pixel': key_pixel,
                    'start_row': start_row,
                    'end_row': end_row,
                    'start_col': start_col,
                    'end_col': end_col
                }
                
                pixels.append(pixel_info)
        
        return pixels
    
    def get_pixel_requirements(self) -> Dict:
        """获取pixel矩阵的需求信息"""
        return {
            'total_pixels': self.total_pixels,
            'pixels_per_dimension': self.num_pixels_per_dim,
            'pixel_size': f"{self.config.pixel_size}x{self.config.pixel_size}",
            'input_elements_per_pixel': self.config.pixel_size * self.config.pixel_size * 2,  # Query + Key
            'output_elements_per_pixel': self.config.pixel_size * self.config.pixel_size,     # Attention output
            'total_input_elements': self.total_pixels * self.config.pixel_size * self.config.pixel_size * 2,
            'total_output_elements': self.total_pixels * self.config.pixel_size * self.config.pixel_size
        }

class TaskProcessor:
    """Task处理器 - 处理64x64矩阵的8行数据"""
    
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
            
            # 提取8行数据
            query_task = query_pixel[start_row:end_row, :self.config.query_cols]  # 8x8
            key_task = key_pixel[start_row:end_row, :self.config.key_cols]        # 8x8
            
            # 合并为8x16的数据块 (8个query + 8个key)
            task_data = np.concatenate([query_task, key_task], axis=1)  # 8x16
            
            task_info = {
                'pixel_id': pixel_info['pixel_id'],
                'task_id': task_id,
                'global_task_id': pixel_info['pixel_id'] * num_tasks + task_id,
                'task_data': task_data,
                'query_data': query_task,
                'key_data': key_task,
                'shape': task_data.shape,
                'rows': rows_per_task,
                'total_elements': task_data.size
            }
            
            tasks.append(task_info)
            
        return tasks

class BitFlipAnalyzer:
    """Bit翻转分析器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        
    def quantize_data(self, data: np.ndarray) -> np.ndarray:
        """将浮点数据量化为整数"""
        # 使用对称量化
        max_val = np.max(np.abs(data))
        scale = max_val / (2**(self.config.quantization_bits-1) - 1)
        
        if scale == 0:
            return np.zeros_like(data, dtype=np.int32)
            
        quantized = np.round(data / scale).astype(np.int32)
        quantized = np.clip(quantized, -(2**(self.config.quantization_bits-1)), 2**(self.config.quantization_bits-1)-1)
        
        return quantized
    
    def count_ones(self, value: int) -> int:
        """计算一个整数中1的bit数"""
        if value < 0:
            # 处理负数的二进制补码
            value = value & ((1 << self.config.quantization_bits) - 1)
        return bin(value).count('1')
    
    def calculate_bit_flips(self, data_sequence: List[int]) -> int:
        """计算连续传输时的bit翻转次数"""
        if len(data_sequence) < 2:
            return 0
            
        total_flips = 0
        for i in range(1, len(data_sequence)):
            # XOR操作找出不同的bit位
            xor_result = data_sequence[i-1] ^ data_sequence[i]
            # 计算翻转的bit数
            flips = bin(xor_result).count('1')
            total_flips += flips
            
        return total_flips
    
    def sort_by_ones_count(self, data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """按1的bit数排序数据"""
        quantized_data = self.quantize_data(data)
        flat_data = quantized_data.flatten()
        
        # 计算每个值的1的bit数
        ones_counts = [self.count_ones(val) for val in flat_data]
        
        # 创建排序索引
        sorted_indices = sorted(range(len(flat_data)), key=lambda i: (ones_counts[i], flat_data[i]))
        
        # 按索引排序数据
        sorted_data = flat_data[sorted_indices]
        sorted_ones_counts = [ones_counts[i] for i in sorted_indices]
        
        return sorted_data, sorted_ones_counts
    
    def analyze_task_bit_flips(self, task_info: Dict) -> Dict:
        """分析一个task的bit翻转情况"""
        task_data = task_info['task_data']
        query_data = task_info['query_data']
        key_data = task_info['key_data']
        
        # Baseline: 原始顺序
        quantized_data = self.quantize_data(task_data)
        baseline_sequence = quantized_data.flatten().tolist()
        baseline_flips = self.calculate_bit_flips(baseline_sequence)
        
        # 优化: 按1的bit数排序
        sorted_data, ones_counts = self.sort_by_ones_count(task_data)
        optimized_sequence = sorted_data.tolist()
        optimized_flips = self.calculate_bit_flips(optimized_sequence)
        
        # 分别统计Query和Key的1bit分布
        query_sorted, query_ones = self.sort_by_ones_count(query_data)
        key_sorted, key_ones = self.sort_by_ones_count(key_data)
        
        # 统计1的bit数分布
        ones_distribution = {}
        for count in ones_counts:
            ones_distribution[count] = ones_distribution.get(count, 0) + 1
            
        query_ones_distribution = {}
        for count in query_ones:
            query_ones_distribution[count] = query_ones_distribution.get(count, 0) + 1
            
        key_ones_distribution = {}
        for count in key_ones:
            key_ones_distribution[count] = key_ones_distribution.get(count, 0) + 1
        
        # 分析传输模式
        baseline_pattern = "Query-Key交织" if len(baseline_sequence) > 16 else "短序列"
        optimized_pattern = "按1bit数递增"
        
        return {
            'task_id': task_info['global_task_id'],
            'total_elements': len(baseline_sequence),
            'baseline_flips': baseline_flips,
            'optimized_flips': optimized_flips,
            'flip_reduction': baseline_flips - optimized_flips,
            'reduction_percentage': ((baseline_flips - optimized_flips) / baseline_flips * 100) if baseline_flips > 0 else 0,
            'ones_distribution': ones_distribution,
            'query_ones_distribution': query_ones_distribution,
            'key_ones_distribution': key_ones_distribution,
            'baseline_sequence': baseline_sequence[:10],  # 只保存前10个用于调试
            'optimized_sequence': optimized_sequence[:10],
            'ones_counts': ones_counts[:10],
            'baseline_pattern': baseline_pattern,
            'optimized_pattern': optimized_pattern
        }
    
    def to_fixed_point(self, data: np.ndarray, integer_bits: int = 4, fractional_bits: int = 4) -> np.ndarray:
        """转换为定点数表示"""
        scale = 2 ** fractional_bits
        fixed_data = np.round(data * scale).astype(np.int32)
        # 限制在指定位数范围内
        max_val = 2 ** (integer_bits + fractional_bits - 1) - 1
        min_val = -(2 ** (integer_bits + fractional_bits - 1))
        fixed_data = np.clip(fixed_data, min_val, max_val)
        return fixed_data
    
    def analyze_task_bit_flips_extended(self, task_info: Dict) -> Dict:
        """扩展分析，包含多种排序策略"""
        task_data = task_info['task_data']
        
        # 1. Baseline: 原始顺序
        quantized_data = self.quantize_data(task_data)
        baseline_sequence = quantized_data.flatten().tolist()
        baseline_flips = self.calculate_bit_flips(baseline_sequence)
        
        # 2. 按1的bit数排序
        sorted_by_ones, ones_counts = self.sort_by_ones_count(task_data)
        ones_sequence = sorted_by_ones.tolist()
        ones_sort_flips = self.calculate_bit_flips(ones_sequence)
        
        # 3. 按数值大小排序
        flat_quantized = quantized_data.flatten()
        magnitude_sorted = np.sort(flat_quantized)
        magnitude_sequence = magnitude_sorted.tolist()
        magnitude_sort_flips = self.calculate_bit_flips(magnitude_sequence)
        
        # 4. 按定点数排序
        fixed_point_data = self.to_fixed_point(task_data)
        fixed_point_sorted = np.sort(fixed_point_data.flatten())
        fixed_point_sequence = fixed_point_sorted.tolist()
        fixed_point_sort_flips = self.calculate_bit_flips(fixed_point_sequence)
        
        # 5. 按符号和幅度排序（先正后负，各自按幅度）
        positives = flat_quantized[flat_quantized >= 0]
        negatives = flat_quantized[flat_quantized < 0]
        positives_sorted = np.sort(positives)
        negatives_sorted = np.sort(negatives)[::-1]  # 负数从大到小
        sign_magnitude_sorted = np.concatenate([positives_sorted, negatives_sorted])
        sign_magnitude_sequence = sign_magnitude_sorted.tolist()
        sign_magnitude_sort_flips = self.calculate_bit_flips(sign_magnitude_sequence)
        
        # 找出最佳策略
        strategies = {
            'ones_sort': ones_sort_flips,
            'magnitude_sort': magnitude_sort_flips,
            'fixed_point_sort': fixed_point_sort_flips,
            'sign_magnitude_sort': sign_magnitude_sort_flips
        }
        
        best_strategy = min(strategies.items(), key=lambda x: x[1])
        
        return {
            'task_id': task_info['global_task_id'],
            'total_elements': len(baseline_sequence),
            
            # Baseline
            'baseline_flips': baseline_flips,
            'baseline_sequence': baseline_sequence,
            
            # 1bit排序
            'ones_sort_flips': ones_sort_flips,
            'ones_sort_reduction': baseline_flips - ones_sort_flips,
            'ones_sort_percentage': ((baseline_flips - ones_sort_flips) / baseline_flips * 100) if baseline_flips > 0 else 0,
            'ones_sequence': ones_sequence,
            
            # 数值排序
            'magnitude_sort_flips': magnitude_sort_flips,
            'magnitude_sort_reduction': baseline_flips - magnitude_sort_flips,
            'magnitude_sort_percentage': ((baseline_flips - magnitude_sort_flips) / baseline_flips * 100) if baseline_flips > 0 else 0,
            'magnitude_sequence': magnitude_sequence,
            
            # 定点数排序
            'fixed_point_sort_flips': fixed_point_sort_flips,
            'fixed_point_sort_reduction': baseline_flips - fixed_point_sort_flips,
            'fixed_point_sort_percentage': ((baseline_flips - fixed_point_sort_flips) / baseline_flips * 100) if baseline_flips > 0 else 0,
            'fixed_point_sequence': fixed_point_sequence,
            
            # 符号-幅度排序
            'sign_magnitude_sort_flips': sign_magnitude_sort_flips,
            'sign_magnitude_sort_reduction': baseline_flips - sign_magnitude_sort_flips,
            'sign_magnitude_sort_percentage': ((baseline_flips - sign_magnitude_sort_flips) / baseline_flips * 100) if baseline_flips > 0 else 0,
            'sign_magnitude_sequence': sign_magnitude_sequence,
            
            # 最佳策略
            'best_strategy': {
                'strategy': best_strategy[0],
                'flips': best_strategy[1],
                'reduction': baseline_flips - best_strategy[1],
                'percentage': ((baseline_flips - best_strategy[1]) / baseline_flips * 100) if baseline_flips > 0 else 0
            }
        }

class LlamaAttentionSimulator:
    """完整的Llama attention模拟器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.matrix_generator = LlamaAttentionMatrixGenerator(config)
        self.pixel_decomposer = PixelMatrixDecomposer(config)
        self.task_processor = TaskProcessor(config)
        self.bit_analyzer = BitFlipAnalyzer(config)
        
    def run_complete_simulation(self) -> Dict:
        """运行完整的模拟"""
        print("=== Llama Attention矩阵Pixel分解与Bit翻转分析 ===\n")
        
        # 1. 生成Query和Key矩阵
        print("1. 生成Query和Key矩阵...")
        query_matrix, key_matrix = self.matrix_generator.generate_qk_matrices()
        print(f"Query矩阵形状: {query_matrix.shape}")
        print(f"Key矩阵形状: {key_matrix.shape}")
        
        # 2. 分解为pixel矩阵
        print(f"\n2. 分解为pixel矩阵...")
        pixels = self.pixel_decomposer.decompose_to_pixels(query_matrix, key_matrix)
        pixel_requirements = self.pixel_decomposer.get_pixel_requirements()
        
        print(f"总pixel数量: {pixel_requirements['total_pixels']}")
        print(f"每个pixel大小: {pixel_requirements['pixel_size']}")
        print(f"每个pixel输入元素数: {pixel_requirements['input_elements_per_pixel']}")
        print(f"每个pixel输出元素数: {pixel_requirements['output_elements_per_pixel']}")
        
        # 3. 处理tasks
        print(f"\n3. 处理tasks...")
        all_tasks = []
        for pixel in pixels:
            tasks = self.task_processor.process_pixel_to_tasks(pixel)
            all_tasks.extend(tasks)
        
        print(f"总task数量: {len(all_tasks)}")
        print(f"每个task形状: {all_tasks[0]['shape']}")
        print(f"每个task元素数: {all_tasks[0]['total_elements']}")
        
        # 4. 分析bit翻转
        print(f"\n4. 分析bit翻转...")
        flip_analyses = []
        total_baseline_flips = 0
        total_optimized_flips = 0
        total_elements = 0
        
        for i, task in enumerate(all_tasks):
            analysis = self.bit_analyzer.analyze_task_bit_flips(task)
            flip_analyses.append(analysis)
            
            total_baseline_flips += analysis['baseline_flips']
            total_optimized_flips += analysis['optimized_flips']
            total_elements += analysis['total_elements']
            
            if i < 5:  # 显示前5个task的详细信息
                print(f"\nTask {i}:")
                print(f"  Baseline翻转: {analysis['baseline_flips']}")
                print(f"  优化后翻转: {analysis['optimized_flips']}")
                print(f"  翻转减少: {analysis['flip_reduction']}")
                print(f"  减少百分比: {analysis['reduction_percentage']:.2f}%")
        
        # 5. 汇总统计
        total_reduction = total_baseline_flips - total_optimized_flips
        total_reduction_percentage = (total_reduction / total_baseline_flips * 100) if total_baseline_flips > 0 else 0
        
        print(f"\n=== 总体统计 ===")
        print(f"总元素数: {total_elements}")
        print(f"总task数: {len(all_tasks)}")
        print(f"Baseline总翻转: {total_baseline_flips}")
        print(f"优化后总翻转: {total_optimized_flips}")
        print(f"总翻转减少: {total_reduction}")
        print(f"总减少百分比: {total_reduction_percentage:.2f}%")
        print(f"平均每个task减少: {total_reduction / len(all_tasks):.2f}翻转")
        
        return {
            'query_matrix': query_matrix,
            'key_matrix': key_matrix,
            'pixels': pixels,
            'pixel_requirements': pixel_requirements,
            'tasks': all_tasks,
            'flip_analyses': flip_analyses,
            'summary': {
                'total_elements': total_elements,
                'total_tasks': len(all_tasks),
                'total_baseline_flips': total_baseline_flips,
                'total_optimized_flips': total_optimized_flips,
                'total_reduction': total_reduction,
                'total_reduction_percentage': total_reduction_percentage,
                'avg_reduction_per_task': total_reduction / len(all_tasks) if all_tasks else 0
            }
        }

def demonstrate_detailed_analysis():
    """详细分析演示"""
    # 配置参数
    config = TaskConfig(
        matrix_size=512,
        pixel_size=64,
        task_rows=8,
        query_cols=8,
        key_cols=8,
        link_width=128,
        quantization_bits=8
    )
    
    # 创建模拟器
    simulator = LlamaAttentionSimulator(config)
    
    # 运行完整模拟
    results = simulator.run_complete_simulation()
    
    # 额外的分析
    print(f"\n=== 详细分析 ===")
    
    # 分析翻转减少的分布
    reductions = [analysis['flip_reduction'] for analysis in results['flip_analyses']]
    percentages = [analysis['reduction_percentage'] for analysis in results['flip_analyses']]
    
    print(f"翻转减少分布:")
    print(f"  最小减少: {min(reductions)}")
    print(f"  最大减少: {max(reductions)}")
    print(f"  平均减少: {np.mean(reductions):.2f}")
    print(f"  标准差: {np.std(reductions):.2f}")
    
    print(f"\n减少百分比分布:")
    print(f"  最小百分比: {min(percentages):.2f}%")
    print(f"  最大百分比: {max(percentages):.2f}%")
    print(f"  平均百分比: {np.mean(percentages):.2f}%")
    print(f"  标准差: {np.std(percentages):.2f}%")
    
    # 分析1的bit数分布
    all_ones_distributions = {}
    for analysis in results['flip_analyses']:
        for bits, count in analysis['ones_distribution'].items():
            all_ones_distributions[bits] = all_ones_distributions.get(bits, 0) + count
    
    print(f"\n整体1的bit数分布:")
    total_count = sum(all_ones_distributions.values())
    for bits in sorted(all_ones_distributions.keys()):
        count = all_ones_distributions[bits]
        percentage = (count / total_count) * 100
        print(f"  {bits} bit(s): {count:8d} 个 ({percentage:5.1f}%)")
    
    return results

def run_extended_sorting_test():
    """扩展排序策略测试"""
    print("=== 多种排序策略对比测试 ===")
    
    # 配置参数
    config = TaskConfig(
        matrix_size=512,
        pixel_size=64,
        task_rows=8,
        query_cols=8,
        key_cols=8,
        link_width=128,
        quantization_bits=8
    )
    
    # 创建生成器和分析器
    np.random.seed(42)
    matrix_gen = LlamaAttentionMatrixGenerator(config)
    pixel_decomposer = PixelMatrixDecomposer(config)
    task_processor = TaskProcessor(config)
    bit_analyzer = BitFlipAnalyzer(config)
    
    # 1. 生成矩阵和处理前几个task
    query_matrix, key_matrix = matrix_gen.generate_qk_matrices()
    pixels = pixel_decomposer.decompose_to_pixels(query_matrix, key_matrix)
    
    sample_tasks = []
    for i in range(min(2, len(pixels))):  # 只处理前2个pixel
        tasks = task_processor.process_pixel_to_tasks(pixels[i])
        sample_tasks.extend(tasks)
    
    print(f"测试task数量: {len(sample_tasks)}")
    
    # 2. 分析不同排序策略
    strategy_stats = {
        'baseline': [],
        'ones_sort': [],
        'magnitude_sort': [],
        'fixed_point_sort': [],
        'sign_magnitude_sort': []
    }
    
    best_strategy_count = {'ones_sort': 0, 'magnitude_sort': 0, 'fixed_point_sort': 0, 'sign_magnitude_sort': 0}
    
    print(f"\n详细分析前{min(3, len(sample_tasks))}个task:")
    
    for i, task in enumerate(sample_tasks):
        analysis = bit_analyzer.analyze_task_bit_flips_extended(task)
        
        # 收集统计数据
        strategy_stats['baseline'].append(analysis['baseline_flips'])
        strategy_stats['ones_sort'].append(analysis['ones_sort_flips'])
        strategy_stats['magnitude_sort'].append(analysis['magnitude_sort_flips'])
        strategy_stats['fixed_point_sort'].append(analysis['fixed_point_sort_flips'])
        strategy_stats['sign_magnitude_sort'].append(analysis['sign_magnitude_sort_flips'])
        
        # 统计最佳策略
        best_strategy_count[analysis['best_strategy']['strategy']] += 1
        
        if i < 3:  # 显示前3个详细结果
            print(f"\n--- Task {i} ---")
            print(f"Baseline翻转: {analysis['baseline_flips']}")
            print(f"1bit排序翻转: {analysis['ones_sort_flips']} (减少 {analysis['ones_sort_reduction']}, {analysis['ones_sort_percentage']:.1f}%)")
            print(f"数值排序翻转: {analysis['magnitude_sort_flips']} (减少 {analysis['magnitude_sort_reduction']}, {analysis['magnitude_sort_percentage']:.1f}%)")
            print(f"定点数排序翻转: {analysis['fixed_point_sort_flips']} (减少 {analysis['fixed_point_sort_reduction']}, {analysis['fixed_point_sort_percentage']:.1f}%)")
            print(f"符号幅度排序翻转: {analysis['sign_magnitude_sort_flips']} (减少 {analysis['sign_magnitude_sort_reduction']}, {analysis['sign_magnitude_sort_percentage']:.1f}%)")
            print(f"最佳策略: {analysis['best_strategy']['strategy']} (减少 {analysis['best_strategy']['reduction']} 翻转, {analysis['best_strategy']['percentage']:.1f}%)")
            
            # 显示排序前后的数据样本
            print(f"Baseline序列前8个: {analysis['baseline_sequence'][:8]}")
            print(f"1bit排序序列前8个: {analysis['ones_sequence'][:8]}")
            print(f"数值排序序列前8个: {analysis['magnitude_sequence'][:8]}")
            print(f"定点数排序序列前8个: {analysis['fixed_point_sequence'][:8]}")
            print(f"符号幅度排序序列前8个: {analysis['sign_magnitude_sequence'][:8]}")
    
    # 3. 总体统计
    print(f"\n=== 总体策略对比 ===")
    total_tasks = len(sample_tasks)
    
    for strategy in ['ones_sort', 'magnitude_sort', 'fixed_point_sort', 'sign_magnitude_sort']:
        total_baseline = sum(strategy_stats['baseline'])
        total_optimized = sum(strategy_stats[strategy])
        total_reduction = total_baseline - total_optimized
        reduction_percentage = (total_reduction / total_baseline * 100) if total_baseline > 0 else 0
        avg_reduction = total_reduction / total_tasks
        
        strategy_names = {
            'ones_sort': '1bit数排序',
            'magnitude_sort': '数值大小排序',
            'fixed_point_sort': '定点数排序',
            'sign_magnitude_sort': '符号-幅度排序'
        }
        
        print(f"\n{strategy_names[strategy]}:")
        print(f"  总翻转减少: {total_reduction}")
        print(f"  减少百分比: {reduction_percentage:.2f}%")
        print(f"  平均每task减少: {avg_reduction:.1f}")
        print(f"  获胜次数: {best_strategy_count[strategy]}/{total_tasks}")
    
    # 4. 定点数详细分析
    print(f"\n=== 定点数表示分析 ===")
    sample_task = sample_tasks[0]
    sample_data = sample_task['query_data'][:2, :2]  # 取2x2样本
    
    print(f"原始浮点数据:\n{sample_data}")
    
    # 转换为定点数
    fixed_point_data = bit_analyzer.to_fixed_point(sample_data, integer_bits=4, fractional_bits=4)
    print(f"定点数表示 (4.4格式):\n{fixed_point_data}")
    
    # 显示二进制表示
    print(f"二进制表示:")
    for i in range(2):
        for j in range(2):
            val = fixed_point_data[i, j]
            binary = format(val & 0xFF, '08b')  # 8bit二进制
            integer_part = val >> 4
            fractional_part = val & 0xF
            print(f"  [{i},{j}]: {val:3d} = {binary} (整数部分:{integer_part}, 小数部分:{fractional_part})")
    
    return {
        'strategy_stats': strategy_stats,
        'best_strategy_count': best_strategy_count,
        'total_tasks': total_tasks
    }

# 更新主测试函数
def run_quick_test():
    """快速测试版本"""
    print("=== 快速测试结果 ===")
    
    # 配置参数
    config = TaskConfig(
        matrix_size=512,
        pixel_size=64,
        task_rows=8,
        query_cols=8,
        key_cols=8,
        link_width=128,
        quantization_bits=8
    )
    
    # 创建生成器和分析器
    np.random.seed(42)
    matrix_gen = LlamaAttentionMatrixGenerator(config)
    pixel_decomposer = PixelMatrixDecomposer(config)
    task_processor = TaskProcessor(config)
    bit_analyzer = BitFlipAnalyzer(config)
    
    # 1. 生成矩阵
    query_matrix, key_matrix = matrix_gen.generate_qk_matrices()
    print(f"1. 矩阵生成:")
    print(f"   Query矩阵: {query_matrix.shape}")
    print(f"   Key矩阵: {key_matrix.shape}")
    print(f"   Query前3x3样本:\n{query_matrix[:3, :3]}")
    
    # 2. Pixel分解
    pixels = pixel_decomposer.decompose_to_pixels(query_matrix, key_matrix)
    requirements = pixel_decomposer.get_pixel_requirements()
    print(f"\n2. Pixel分解:")
    print(f"   总pixel数: {requirements['total_pixels']}")
    print(f"   每个pixel输入元素: {requirements['input_elements_per_pixel']}")
    print(f"   每个pixel输出元素: {requirements['output_elements_per_pixel']}")
    
    # 3. 处理前几个pixel的tasks
    sample_tasks = []
    for i in range(min(3, len(pixels))):  # 只处理前3个pixel
        tasks = task_processor.process_pixel_to_tasks(pixels[i])
        sample_tasks.extend(tasks)
    
    print(f"\n3. Task处理 (前3个pixel):")
    print(f"   生成的task数: {len(sample_tasks)}")
    print(f"   每个task形状: {sample_tasks[0]['shape']}")
    print(f"   每个task元素数: {sample_tasks[0]['total_elements']}")
    
    # 4. Bit翻转分析
    total_baseline = 0
    total_optimized = 0
    total_reduction = 0
    
    print(f"\n4. Bit翻转分析 (前{len(sample_tasks)}个task):")
    
    for i, task in enumerate(sample_tasks):
        analysis = bit_analyzer.analyze_task_bit_flips(task)
        
        total_baseline += analysis['baseline_flips']
        total_optimized += analysis['optimized_flips']
        total_reduction += analysis['flip_reduction']
        
        if i < 3:  # 显示前3个详细结果
            print(f"   Task {i}:")
            print(f"     Baseline翻转: {analysis['baseline_flips']}")
            print(f"     优化后翻转: {analysis['optimized_flips']}")
            print(f"     减少翻转: {analysis['flip_reduction']}")
            print(f"     减少百分比: {analysis['reduction_percentage']:.1f}%")
            print(f"     总1bit分布: {analysis['ones_distribution']}")
            print(f"     Query1bit分布: {analysis['query_ones_distribution']}")
            print(f"     Key1bit分布: {analysis['key_ones_distribution']}")
            print(f"     Baseline模式: {analysis['baseline_pattern']}")
            print(f"     优化后模式: {analysis['optimized_pattern']}")
            print(f"     ---")
    
    # 5. 总结
    avg_reduction_pct = (total_reduction / total_baseline * 100) if total_baseline > 0 else 0
    
    print(f"\n5. 总结统计:")
    print(f"   分析的task数: {len(sample_tasks)}")
    print(f"   总baseline翻转: {total_baseline}")
    print(f"   总优化翻转: {total_optimized}")
    print(f"   总减少翻转: {total_reduction}")
    print(f"   平均减少百分比: {avg_reduction_pct:.2f}%")
    print(f"   每task平均减少: {total_reduction/len(sample_tasks):.1f}")
    
    return {
        'total_baseline': total_baseline,
        'total_optimized': total_optimized,
        'total_reduction': total_reduction,
        'reduction_percentage': avg_reduction_pct,
        'num_tasks': len(sample_tasks)
    }

if __name__ == "__main__":
    # 运行基础测试
    print("运行基础1bit排序测试...")
    basic_results = run_quick_test()
    
    print("\n" + "="*60 + "\n")
    
    # 运行扩展排序策略测试
    print("运行扩展排序策略对比测试...")
    extended_results = run_extended_sorting_test()