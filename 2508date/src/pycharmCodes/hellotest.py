#!/usr/bin/env python3
"""
Python验证脚本 - 读取C++导出的数据进行验证
确保C++和Python使用完全相同的数据进行计算
"""

import numpy as np
import math
import time
import os


class LLMAttentionVerification:
    def __init__(self):
        self.matrix_size = 32
        self.tile_size = 4
        self.time_slices = 2
        self.data_elements = 16

        self.tiles_per_dim = self.matrix_size // self.tile_size
        self.total_tiles = self.tiles_per_dim * self.tiles_per_dim
        self.total_tasks = self.matrix_size * self.matrix_size * self.time_slices

        # 从C++导出文件读取数据
        self.load_cpp_data()

        print(f"Python验证脚本初始化完成:")
        print(f"  矩阵大小: {self.matrix_size}x{self.matrix_size}")
        print(f"  Tile大小: {self.tile_size}x{self.tile_size}")
        print(f"  时间片: {self.time_slices}")
        print(f"  总任务数: {self.total_tasks}")

    def load_cpp_data(self):
        """从C++导出的文件中读取数据"""
        print("从C++导出文件读取数据...")

        try:
            # 读取Query矩阵
            self.attention_query_table = self.load_matrix_from_file("../output/cpp_query_matrix.txt")
            print(f"成功加载Query矩阵: {self.attention_query_table.shape}")

            # 读取Key矩阵
            self.attention_key_table = self.load_matrix_from_file("../output/cpp_key_matrix.txt")
            print(f"成功加载Key矩阵: {self.attention_key_table.shape}")

            # 读取Value矩阵
            self.attention_value_table = self.load_matrix_from_file("../output/cpp_value_matrix.txt")
            print(f"成功加载Value矩阵: {self.attention_value_table.shape}")

            # 验证数据的前几个值
            print(f"\n数据验证:")
            print(f"Query[0][0:3]: {self.attention_query_table[0, 0:3]}")
            print(f"Key[0][0:3]: {self.attention_key_table[0, 0:3]}")

        except FileNotFoundError as e:
            print(f"错误：找不到C++导出的数据文件: {e}")
            print("请确保先运行C++程序生成数据文件")
            print("需要的文件:")
            print("  - output/cpp_query_matrix.txt")
            print("  - output/cpp_key_matrix.txt")
            print("  - output/cpp_value_matrix.txt")
            raise

    def load_matrix_from_file(self, filename):
        """从文件加载矩阵数据"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到文件: {filename}")

        matrix_data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 按逗号分割并转换为浮点数
                    row = [float(x) for x in line.split(',')]
                    matrix_data.append(row)

        return np.array(matrix_data)

    def generate_task_data(self, pixel_x, pixel_y, time_slice):
        """使用与C++相同的算法生成任务数据"""
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
        """使用与C++相同的算法计算注意力"""
        # 计算点积 Q·K
        dot_product = sum(q * k for q, k in zip(query_data, key_data))

        # 缩放 (attention_output / sqrt(d_k))
        scaled_output = dot_product / math.sqrt(self.data_elements)

        # 应用tanh激活
        attention_output = math.tanh(scaled_output)

        return attention_output, dot_product, scaled_output

    def verify_with_cpp_tasks(self):
        """验证前几个任务与C++结果的一致性"""
        print("\n=== 验证Python计算与C++的一致性 ===")

        # 验证前10个任务
        task_id = 0
        results = []

        for pixel_y in range(self.matrix_size):
            for pixel_x in range(self.matrix_size):
                for time_slice in range(self.time_slices):
                    # 生成任务数据（使用与C++相同的算法）
                    query_data, key_data = self.generate_task_data(pixel_x, pixel_y, time_slice)

                    # 计算注意力（使用与C++相同的算法）
                    attention_output, dot_product, scaled = self.compute_attention(query_data, key_data)

                    # 记录结果
                    result = {
                        'task_id': task_id,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'time_slice': time_slice,
                        'query_data': query_data,
                        'key_data': key_data,
                        'dot_product': dot_product,
                        'scaled': scaled,
                        'attention_output': attention_output
                    }
                    results.append(result)

                    # 打印前10个任务的详细信息
                    if task_id < 10:
                        print(f"任务 {task_id} [({pixel_x},{pixel_y}), ts={time_slice}]:")
                        print(f"  Query[0:4]: {[f'{x:.10f}' for x in query_data[:4]]}")
                        print(f"  Key[0:4]: {[f'{x:.10f}' for x in key_data[:4]]}")
                        print(f"  点积: {dot_product:.10f}")
                        print(f"  缩放后: {scaled:.10f}")
                        print(f"  注意力输出: {attention_output:.10f}")
                        print()

                    task_id += 1
                    if task_id >= 10:  # 只验证前10个任务
                        break
                if task_id >= 10:
                    break
            if task_id >= 10:
                break

        return results

    def run_full_computation(self):
        """运行完整的注意力计算"""
        print("\n=== 运行完整的注意力计算 ===")
        start_time = time.time()

        # 初始化输出矩阵
        attention_output_table = np.zeros((self.matrix_size, self.matrix_size))

        task_count = 0
        all_results = []

        for pixel_y in range(self.matrix_size):
            for pixel_x in range(self.matrix_size):
                pixel_results = []
                for time_slice in range(self.time_slices):
                    query_data, key_data = self.generate_task_data(pixel_x, pixel_y, time_slice)
                    attention_output, _, _ = self.compute_attention(query_data, key_data)
                    pixel_results.append(attention_output)
                    task_count += 1

                # 对于每个像素，取多个时间片的平均值
                final_attention = sum(pixel_results) / len(pixel_results)
                attention_output_table[pixel_y][pixel_x] = final_attention
                all_results.append((pixel_x, pixel_y, final_attention))

        end_time = time.time()

        print(f"完整计算完成！")
        print(f"  处理任务数: {task_count}")
        print(f"  处理时间: {end_time - start_time:.3f}秒")

        # 保存Python结果
        self.save_python_results(attention_output_table)

        return attention_output_table, all_results

    def save_python_results(self, attention_output_table):
        """保存Python计算结果"""
        # 保存输出矩阵
        with open("../output/python_verification_output.txt", 'w') as f:
            for i in range(self.matrix_size):
                row_values = []
                for j in range(self.matrix_size):
                    row_values.append(f"{attention_output_table[i][j]:.6f}")
                f.write(",".join(row_values) + "\n")

        print("Python结果已保存到 output/python_verification_output.txt")

    def compare_with_cpp_output(self):
        """与C++输出结果进行比较"""
        print("\n=== 比较Python与C++的输出结果 ===")

        try:
            # 读取C++输出
            cpp_output = self.load_matrix_from_file("../output/llm_attention_output.txt")
            print(f"成功加载C++输出矩阵: {cpp_output.shape}")

            # 运行Python计算
            python_output, _ = self.run_full_computation()

            # 计算差异
            diff = np.abs(cpp_output - python_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\n结果比较:")
            print(f"  最大差异: {max_diff:.10f}")
            print(f"  平均差异: {mean_diff:.10f}")
            print(f"  差异标准差: {np.std(diff):.10f}")

            # 打印样本比较
            print(f"\n样本比较 (前5x5):")
            print("C++结果:")
            print(cpp_output[:5, :5])
            print("\nPython结果:")
            print(python_output[:5, :5])
            print("\n差异:")
            print(diff[:5, :5])

            # 检查是否匹配
            tolerance = 1e-6
            if max_diff < tolerance:
                print(f"\n✅ 结果匹配！(差异 < {tolerance})")
            else:
                print(f"\n❌ 结果不匹配！(最大差异: {max_diff} >= {tolerance})")

                # 找出差异最大的位置
                max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"最大差异位置: [{max_diff_pos[0]}, {max_diff_pos[1]}]")
                print(f"  C++值: {cpp_output[max_diff_pos]:.10f}")
                print(f"  Python值: {python_output[max_diff_pos]:.10f}")
                print(f"  差异: {diff[max_diff_pos]:.10f}")

            return max_diff < tolerance

        except FileNotFoundError as e:
            print(f"错误：找不到C++输出文件: {e}")
            print("请确保C++程序已运行完成并生成了输出文件")
            return False

    def load_and_compare_cpp_verification(self):
        """加载并比较C++的验证文件"""
        print("\n=== 与C++验证文件对比 ===")

        try:
            with open("../output/cpp_verification.txt", 'r') as f:
                cpp_verification = f.read()

            print("C++验证文件内容:")
            print(cpp_verification[:1000] + "..." if len(cpp_verification) > 1000 else cpp_verification)

        except FileNotFoundError:
            print("找不到C++验证文件 output/cpp_verification.txt")

    def print_sample_output(self, output_matrix, sample_size=10):
        """打印样本输出"""
        print(f"\n样本输出矩阵 (前{sample_size}x{sample_size}):")
        for i in range(min(sample_size, self.matrix_size)):
            row_str = ""
            for j in range(min(sample_size, self.matrix_size)):
                row_str += f"{output_matrix[i][j]:8.4f} "
            print(row_str)


def main():
    print("=== LLM Attention Python验证脚本 ===")
    print("使用C++导出的数据进行验证\n")

    try:
        # 创建验证实例
        verifier = LLMAttentionVerification()

        # 验证前几个任务的计算过程
        task_results = verifier.verify_with_cpp_tasks()

        # 加载并比较C++验证文件
        verifier.load_and_compare_cpp_verification()

        # 比较完整的输出结果
        match_result = verifier.compare_with_cpp_output()

        print(f"\n=== 验证总结 ===")
        if match_result:
            print("✅ C++和Python结果完全一致！")
        else:
            print("❌ C++和Python结果存在差异，需要进一步调试")

        print("\n=== 使用指南 ===")
        print("1. 确保C++程序已运行并生成以下文件：")
        print("   - output/cpp_query_matrix.txt")
        print("   - output/cpp_key_matrix.txt")
        print("   - output/cpp_value_matrix.txt")
        print("   - output/llm_attention_output.txt")
        print("   - output/cpp_verification.txt")
        print("2. 比较 output/python_verification_output.txt 和 output/llm_attention_output.txt")
        print("3. 检查验证文件中的详细计算过程")

        return verifier

    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    verifier = main()