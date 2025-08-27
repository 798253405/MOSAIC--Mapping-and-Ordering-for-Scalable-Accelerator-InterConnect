#!/usr/bin/env python3
"""
小矩阵Python验证脚本 - 4x4可调试版本
与C++小矩阵版本完全对应，便于逐步调试
修复版本：tile_size=4, time_slices=1, 路径../output
"""

import numpy as np
import math
import time
import os


class SmallMatrixLLMVerification:
    def __init__(self):
        # 小矩阵参数 - 与修改后的C++完全一致
        self.matrix_size = 4
        self.tile_size = 4  # 改为4x4 tile（整个矩阵作为一个tile）
        self.time_slices = 1  # 改为1个时间片
        self.data_elements = self.matrix_size  # 4 elements per vector

        self.tiles_per_dim = self.matrix_size // self.tile_size  # 1
        self.total_tiles = self.tiles_per_dim * self.tiles_per_dim  # 1
        self.total_tasks = self.matrix_size * self.matrix_size * self.time_slices  # 4*4*1 = 16

        print(f"=== 小矩阵LLM Python验证脚本 ===")
        print(f"矩阵大小: {self.matrix_size}x{self.matrix_size}")
        print(f"Tile大小: {self.tile_size}x{self.tile_size}")
        print(f"时间片: {self.time_slices}")
        print(f"总任务数: {self.total_tasks}")
        print(f"数据元素/向量: {self.data_elements}")

        # 从C++导出文件读取数据
        self.load_cpp_data()

    def load_cpp_data(self):
        """从C++导出的文件中读取数据"""
        print("\n从C++导出文件读取数据...")

        base_path = "../output"  # 修正路径

        try:
            # 读取矩阵数据
            self.attention_query_table = self.load_matrix_from_file(f"{base_path}/cpp_query_matrix.txt")
            print(f"✓ 成功加载Query矩阵: {self.attention_query_table.shape}")

            self.attention_key_table = self.load_matrix_from_file(f"{base_path}/cpp_key_matrix.txt")
            print(f"✓ 成功加载Key矩阵: {self.attention_key_table.shape}")

            self.attention_value_table = self.load_matrix_from_file(f"{base_path}/cpp_value_matrix.txt")
            print(f"✓ 成功加载Value矩阵: {self.attention_value_table.shape}")

            # 打印完整的小矩阵用于验证
            print(f"\n=== Python读取的Query矩阵 ===")
            for i in range(self.matrix_size):
                print(f"Row {i}: {self.attention_query_table[i]}")

            print(f"\n=== Python读取的Key矩阵 ===")
            for i in range(self.matrix_size):
                print(f"Row {i}: {self.attention_key_table[i]}")

        except FileNotFoundError as e:
            print(f"错误：找不到C++导出的数据文件: {e}")
            print(f"请确保C++程序已运行并在 {base_path}/ 生成了数据文件")
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
                    row = [float(x) for x in line.split(',')]
                    matrix_data.append(row)

        return np.array(matrix_data)

    def generate_task_data(self, pixel_x, pixel_y, time_slice):
        """使用与C++完全相同的算法生成任务数据"""
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
        """使用与C++完全相同的算法计算注意力"""
        # 计算点积 Q·K
        dot_product = 0.0
        for i, (q, k) in enumerate(zip(query_data, key_data)):
            product = q * k
            dot_product += product
            print(f"      Q[{i}] * K[{i}] = {q:.6f} * {k:.6f} = {product:.6f}")

        # 缩放 (attention_output / sqrt(d_k))
        scaled_output = dot_product / math.sqrt(len(query_data))

        # 应用tanh激活
        attention_output = math.tanh(scaled_output)

        return attention_output, dot_product, scaled_output

    def verify_all_tasks_detailed(self):
        """详细验证所有任务"""
        print(f"\n=== 详细验证所有 {self.total_tasks} 个任务 ===")

        # 初始化输出矩阵
        python_output = np.zeros((self.matrix_size, self.matrix_size))

        task_id = 0
        all_results = []

        # 注意：与C++匹配的任务生成顺序（时间片在外层）
        for time_slice in range(self.time_slices):
            for pixel_y in range(self.matrix_size):
                for pixel_x in range(self.matrix_size):
                    print(f"\n--- 任务 {task_id}: 像素({pixel_x},{pixel_y}), 时间片{time_slice} ---")

                    # 生成任务数据（使用与C++相同的算法）
                    query_data, key_data = self.generate_task_data(pixel_x, pixel_y, time_slice)

                    print(f"  Query数据: {[f'{x:.6f}' for x in query_data]}")
                    print(f"  Key数据:   {[f'{x:.6f}' for x in key_data]}")

                    # 计算注意力（使用与C++相同的算法）
                    attention_output, dot_product, scaled = self.compute_attention(query_data, key_data)

                    print(f"  点积: {dot_product:.10f}")
                    print(f"  缩放后: {scaled:.10f}")
                    print(f"  Tanh输出: {attention_output:.10f}")

                    # 由于只有1个时间片，直接存储结果
                    python_output[pixel_y, pixel_x] = attention_output

                    all_results.append({
                        'task_id': task_id,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'time_slice': time_slice,
                        'attention_output': attention_output
                    })

                    task_id += 1

        return python_output, all_results

    def compare_with_cpp_output(self):
        """与C++输出进行比较"""
        print(f"\n=== 与C++输出进行比较 ===")

        base_path = "../output"  # 修正路径

        try:
            # 读取C++输出
            cpp_output = self.load_matrix_from_file(f"{base_path}/llm_attention_output.txt")
            print(f"✓ 成功加载C++输出矩阵: {cpp_output.shape}")

            print(f"\n=== C++输出矩阵 ===")
            for i in range(self.matrix_size):
                print(f"Row {i}: {cpp_output[i]}")

            # 计算Python结果
            python_output, all_results = self.verify_all_tasks_detailed()

            print(f"\n=== Python计算的输出矩阵 ===")
            for i in range(self.matrix_size):
                print(f"Row {i}: {python_output[i]}")

            # 比较差异
            diff = np.abs(cpp_output - python_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\n=== 比较结果 ===")
            print(f"最大差异: {max_diff:.10f}")
            print(f"平均差异: {mean_diff:.10f}")

            # 详细差异矩阵
            print(f"\n=== 差异矩阵 ===")
            for i in range(self.matrix_size):
                print(f"Row {i}: {diff[i]}")

            tolerance = 1e-6
            if max_diff < tolerance:
                print(f"\n✅ 结果匹配！(差异 < {tolerance})")
                match_result = True
            else:
                print(f"\n❌ 结果不匹配 (差异 >= {tolerance})")
                match_result = False

                # 找出差异最大的位置
                max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"最大差异位置: ({max_diff_pos[1]}, {max_diff_pos[0]})")
                print(f"C++值: {cpp_output[max_diff_pos]:.10f}")
                print(f"Python值: {python_output[max_diff_pos]:.10f}")

            # 保存Python结果
            self.save_python_results(python_output)

            return match_result

        except FileNotFoundError as e:
            print(f"错误：找不到C++输出文件: {e}")
            return False

    def save_python_results(self, python_output):
        """保存Python计算结果"""
        base_path = "../output"  # 修正路径

        with open(f"{base_path}/python_small_matrix_output.txt", 'w') as f:
            for i in range(self.matrix_size):
                row_values = []
                for j in range(self.matrix_size):
                    row_values.append(f"{python_output[i][j]:.10f}")
                f.write(",".join(row_values) + "\n")

        print(f"Python结果已保存到 {base_path}/python_small_matrix_output.txt")

    def analyze_cpp_tasks_file(self):
        """分析C++任务文件"""
        print(f"\n=== 分析C++任务文件 ===")

        base_path = "../output"  # 修正路径
        tasks_file = f"{base_path}/cpp_tasks.txt"

        if os.path.exists(tasks_file):
            print("C++任务文件存在，分析前几个任务...")

            with open(tasks_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')

                task_count = 0
                for i, line in enumerate(lines):
                    if line.startswith('Task '):
                        task_count += 1
                        if task_count <= 3:  # 只打印前3个任务
                            print(f"\n{line}")
                            # 打印接下来的几行任务详情
                            for j in range(1, 6):
                                if i + j < len(lines) and lines[i + j].strip():
                                    print(lines[i + j])

                print(f"\n总共找到 {task_count} 个任务")
        else:
            print("C++任务文件不存在")

    def analyze_cpp_verification_file(self):
        """分析C++验证文件"""
        print(f"\n=== 分析C++验证文件 ===")

        base_path = "../output"  # 修正路径
        verify_file = f"{base_path}/cpp_verification.txt"

        if os.path.exists(verify_file):
            with open(verify_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')

                # 打印文件头部信息
                print("C++验证文件头部信息:")
                for line in lines[:10]:
                    if line.strip():
                        print(line)

                # 找到并打印第一个任务的完整计算过程
                print("\n第一个任务的计算过程:")
                task_found = False
                for i, line in enumerate(lines):
                    if line.startswith('Task 0 '):
                        task_found = True
                        # 打印这个任务的所有行
                        j = i
                        while j < len(lines) and (j == i or not lines[j].startswith('Task ')):
                            if lines[j].strip():
                                print(lines[j])
                            j += 1
                        break

                if not task_found:
                    print("未找到任务计算详情")
        else:
            print("C++验证文件不存在")

    def comprehensive_debug(self):
        """综合调试分析"""
        print(f"\n{'=' * 60}")
        print(f"=== 综合调试分析 ===")
        print(f"{'=' * 60}")

        # 1. 分析C++文件
        self.analyze_cpp_tasks_file()
        self.analyze_cpp_verification_file()

        # 2. 与C++输出比较
        is_match = self.compare_with_cpp_output()

        return is_match


def main():
    print("=== 小矩阵LLM Attention Python验证脚本 ===")
    print("配置: 4x4矩阵, 4x4 tile, 1个时间片\n")

    try:
        # 创建验证实例
        verifier = SmallMatrixLLMVerification()

        # 综合调试
        is_match = verifier.comprehensive_debug()

        print(f"\n{'=' * 60}")
        print(f"=== 调试总结 ===")
        print(f"{'=' * 60}")

        if is_match:
            print(f"✅ C++和Python结果完全一致！")
            print("验证成功：两边的注意力计算实现相同")
        else:
            print(f"❌ C++和Python结果存在差异")
            print("\n调试建议:")
            print("1. 检查C++中的任务生成顺序")
            print("2. 验证C++的attention_output_table更新逻辑")
            print("3. 确认C++的浮点数精度设置")
            print("4. 检查内存节点的结果处理逻辑")

        print(f"\n=== 生成的文件位置 ===")
        print("检查 ../output/ 目录下的以下文件:")
        print("- cpp_query_matrix.txt (C++生成的Query矩阵)")
        print("- cpp_key_matrix.txt (C++生成的Key矩阵)")
        print("- cpp_tasks.txt (C++任务详情)")
        print("- cpp_verification.txt (C++计算验证)")
        print("- llm_attention_output.txt (C++最终输出)")
        print("- python_small_matrix_output.txt (Python计算结果)")

        return verifier

    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    verifier = main()