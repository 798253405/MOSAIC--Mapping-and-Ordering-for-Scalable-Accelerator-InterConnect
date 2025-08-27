#!/usr/bin/env python3
"""
Python参考实现 - LLM Attention计算
用于验证C++仿真器的正确性
"""

import numpy as np
import math
import time


class LLMAttentionReference:
    def __init__(self):
        # 使用与C++相同的参数
        self.matrix_size = 32  # 32x32 矩阵
        self.tile_size = 4  # 4x4 tile
        self.time_slices = 2  # 2个时间片
        self.data_elements = 16  # 每个query/key向量16个元素

        self.tiles_per_dim = self.matrix_size // self.tile_size  # 8
        self.total_tiles = self.tiles_per_dim * self.tiles_per_dim  # 64
        self.total_tasks = self.matrix_size * self.matrix_size * self.time_slices  # 2048

        # 初始化matrices（使用与C++相同的随机种子）
        np.random.seed(42)
        self.attention_query_table = np.random.uniform(-1, 1, (self.matrix_size, self.matrix_size))
        self.attention_key_table = np.random.uniform(-1, 1, (self.matrix_size, self.matrix_size))
        self.attention_value_table = np.random.uniform(-1, 1, (self.matrix_size, self.matrix_size))
        self.attention_output_table = np.zeros((self.matrix_size, self.matrix_size))

        print(f"初始化完成:")
        print(f"  矩阵大小: {self.matrix_size}x{self.matrix_size}")
        print(f"  Tile大小: {self.tile_size}x{self.tile_size}")
        print(f"  时间片: {self.time_slices}")
        print(f"  总任务数: {self.total_tasks}")
        print(f"  数据元素/向量: {self.data_elements}")

    def generate_task_data(self, pixel_x, pixel_y, time_slice):
        """生成单个任务的query和key数据"""
        query_data = []
        key_data = []

        for i in range(self.data_elements):
            data_idx = (time_slice * self.data_elements + i) % self.matrix_size
            query_val = self.attention_query_table[pixel_y][(pixel_x + data_idx) % self.matrix_size]
            key_val = self.attention_key_table[pixel_y][(pixel_x + data_idx) % self.matrix_size]
            query_data.append(query_val)
            key_data.append(key_val)

        return query_data, key_data

    def compute_attention(self, query_data, key_data):
        """计算注意力值"""
        # 计算点积 Q·K
        dot_product = sum(q * k for q, k in zip(query_data, key_data))

        # 缩放 (attention_output / sqrt(d_k))
        scaled_output = dot_product / math.sqrt(self.data_elements)

        # 应用tanh激活
        attention_output = math.tanh(scaled_output)

        return attention_output

    def run_all_tasks(self):
        """运行所有任务并生成输出矩阵"""
        print("\n开始处理所有任务...")
        start_time = time.time()

        task_count = 0
        results = []

        for pixel_y in range(self.matrix_size):
            for pixel_x in range(self.matrix_size):
                pixel_results = []
                for time_slice in range(self.time_slices):
                    # 生成任务数据
                    query_data, key_data = self.generate_task_data(pixel_x, pixel_y, time_slice)

                    # 计算注意力
                    attention_output = self.compute_attention(query_data, key_data)

                    pixel_results.append(attention_output)
                    task_count += 1

                    if task_count <= 10:  # 打印前10个任务的详细信息
                        print(f"任务 {task_count}: 像素({pixel_x},{pixel_y}) 时间片{time_slice}")
                        print(f"  Query[0:5]: {query_data[:5]}")
                        print(f"  Key[0:5]: {key_data[:5]}")
                        print(f"  注意力输出: {attention_output:.6f}")

                # 对于每个像素，我们可能需要聚合多个时间片的结果
                # 这里简单地取平均值
                final_attention = sum(pixel_results) / len(pixel_results)
                self.attention_output_table[pixel_y][pixel_x] = final_attention
                results.append((pixel_x, pixel_y, final_attention))

        end_time = time.time()
        print(f"\n所有任务完成！")
        print(f"  处理任务数: {task_count}")
        print(f"  处理时间: {end_time - start_time:.3f}秒")

        return results

    def print_sample_output(self, sample_size=10):
        """打印样本输出"""
        print(f"\n样本输出矩阵 (前{sample_size}x{sample_size}):")
        for i in range(min(sample_size, self.matrix_size)):
            row_str = ""
            for j in range(min(sample_size, self.matrix_size)):
                row_str += f"{self.attention_output_table[i][j]:8.4f} "
            print(row_str)

    def save_output_to_file(self, filename="python_llm_attention_output.txt"):
        """保存输出到文件以便与C++结果比较"""
        with open(filename, 'w') as f:
            for i in range(self.matrix_size):
                row_values = []
                for j in range(self.matrix_size):
                    row_values.append(f"{self.attention_output_table[i][j]:.6f}")
                f.write(",".join(row_values) + "\n")
        print(f"\n输出已保存到: {filename}")

    def verify_specific_tasks(self):
        """验证特定任务以便与C++调试输出对比"""
        print("\n=== 验证特定任务 ===")

        # 验证任务0: 像素(0,0), 时间片0
        query_data, key_data = self.generate_task_data(0, 0, 0)
        dot_product = sum(q * k for q, k in zip(query_data, key_data))
        scaled = dot_product / math.sqrt(self.data_elements)
        attention_output = math.tanh(scaled)

        print(f"任务0 [像素(0,0), 时间片0]:")
        print(f"  Query前4个元素: {[f'{x:.6f}' for x in query_data[:4]]}")
        print(f"  Key前4个元素: {[f'{x:.6f}' for x in key_data[:4]]}")
        print(f"  点积: {dot_product:.6f}")
        print(f"  缩放后: {scaled:.6f}")
        print(f"  注意力输出: {attention_output:.6f}")

        # 验证任务1: 像素(1,0), 时间片0
        query_data, key_data = self.generate_task_data(1, 0, 0)
        dot_product = sum(q * k for q, k in zip(query_data, key_data))
        scaled = dot_product / math.sqrt(self.data_elements)
        attention_output = math.tanh(scaled)
        print(f"任务1 [像素(1,0), 时间片0]:")
        print(f"  注意力输出: {attention_output:.6f}")

        # 验证任务32: 像素(0,0), 时间片1
        query_data, key_data = self.generate_task_data(0, 0, 1)
        dot_product = sum(q * k for q, k in zip(query_data, key_data))
        scaled = dot_product / math.sqrt(self.data_elements)
        attention_output = math.tanh(scaled)
        print(f"任务32 [像素(0,0), 时间片1]:")
        print(f"  注意力输出: {attention_output:.6f}")

        # 验证前10个任务，看分布情况
        print(f"\n前10个任务的输出分布:")
        task_id = 0
        for pixel_y in range(self.matrix_size):
            for pixel_x in range(self.matrix_size):
                for time_slice in range(self.time_slices):
                    query_data, key_data = self.generate_task_data(pixel_x, pixel_y, time_slice)
                    attention_output = self.compute_attention(query_data, key_data)
                    print(f"  任务{task_id} [({pixel_x},{pixel_y}), ts{time_slice}]: {attention_output:.6f}")
                    task_id += 1
                    if task_id >= 10:
                        return

    def get_statistics(self):
        """获取统计信息"""
        output_values = self.attention_output_table.flatten()
        print(f"\n=== 统计信息 ===")
        print(f"输出值范围: {np.min(output_values):.6f} 到 {np.max(output_values):.6f}")
        print(f"平均值: {np.mean(output_values):.6f}")
        print(f"标准差: {np.std(output_values):.6f}")
        print(f"零值数量: {np.sum(output_values == 0)}")


def main():
    print("=== LLM Attention Python参考实现 ===")
    print("用于验证C++仿真器结果\n")

    # 创建LLM attention实例
    llm = LLMAttentionReference()

    # 运行所有任务
    results = llm.run_all_tasks()

    # 打印样本输出
    llm.print_sample_output(10)

    # 验证特定任务
    llm.verify_specific_tasks()

    # 获取统计信息
    llm.get_statistics()

    # 保存输出到文件
    llm.save_output_to_file()

    print("\n=== 与C++对比指南 ===")
    print("1. 运行C++仿真器，确保它保存输出到 'output/llm_attention_output.txt'")
    print("2. 比较两个输出文件的数值")
    print("3. 检查C++调试输出中的特定任务结果与上面的验证输出")
    print("4. 验证统计信息是否匹配")

    return llm


if __name__ == "__main__":
    llm_ref = main()