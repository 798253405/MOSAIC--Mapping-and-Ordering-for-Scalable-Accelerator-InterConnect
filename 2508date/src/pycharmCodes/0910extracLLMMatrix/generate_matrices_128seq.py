#!/usr/bin/env python3
"""
生成128序列版本的矩阵文件：
1. X_input.txt - 输入矩阵 [128×4096] (从8改为128)
2. Wq.txt - Query权重矩阵 [128×4096] (不变)

用于C++读取并计算 Q = X @ Wq^T = [128×128]
"""

import numpy as np
import struct
import json
import os

def extract_wq_from_model():
    """
    从LLaMA模型文件中提取Query权重矩阵
    提取第0层第0个attention头的权重
    """
    # 使用完整路径
    model_dir = '/home/yz/myprojects/2025/202508/Llama-2-7b-pruned50-retrained/'
    model_file = os.path.join(model_dir, 'model-00001-of-00003.safetensors')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file):
        print(f"警告: 找不到模型文件 {model_file}")
        print("使用随机初始化的Wq矩阵代替")
        np.random.seed(100)  # 不同的种子以区分
        wq = np.random.randn(128, 4096).astype(np.float32) * 0.01
        # 添加一些稀疏性
        mask = np.random.random((128, 4096)) < 0.1  # 10%稀疏
        wq[mask] = 0
        return wq
    
    print(f"   从模型文件读取: {model_file}")
    
    # 读取SafeTensors文件的header
    with open(model_file, 'rb') as f:
        # 前8字节是header长度
        header_size = struct.unpack('<Q', f.read(8))[0]
        # 读取JSON格式的header
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
    
    # 获取Q投影权重的信息
    q_proj_name = 'model.layers.0.self_attn.q_proj.weight'
    tensor_info = header[q_proj_name]
    start_offset, end_offset = tensor_info['data_offsets']
    
    # 读取权重数据
    with open(model_file, 'rb') as f:
        f.seek(data_start + start_offset)
        weight_bytes = f.read(end_offset - start_offset)
    
    # BFloat16转换为Float32
    # BFloat16是16位浮点数，实际是Float32的高16位
    bf16_array = np.frombuffer(weight_bytes, dtype=np.uint16)
    float32_array = np.zeros(len(bf16_array), dtype=np.float32)
    float32_bytes = float32_array.view(np.uint8)
    bf16_bytes_view = bf16_array.view(np.uint8)
    float32_bytes[2::4] = bf16_bytes_view[0::2]  # 低字节
    float32_bytes[3::4] = bf16_bytes_view[1::2]  # 高字节
    
    # 重塑为完整矩阵 [4096, 4096]
    full_matrix = float32_array.reshape([4096, 4096])
    
    # 提取第0个attention头的权重 [128, 4096]
    # LLaMA-7B有32个头，每个头128维
    wq = full_matrix[:128, :]
    
    return wq

def create_input_matrix_128seq():
    """
    创建输入矩阵X [128×4096]
    模拟128个token的嵌入向量（增加到128个序列）
    """
    # 使用固定种子确保可重复性
    np.random.seed(42)
    
    # 生成基础随机矩阵，标准差0.02（LLaMA的典型值）
    X = np.random.randn(128, 4096).astype(np.float32) * 0.02
    
    # 模拟真实嵌入的特征：某些维度更活跃
    for i in range(128):
        # 随机选择10%的维度增强
        active_dims = np.random.choice(4096, size=400, replace=False)
        X[i, active_dims] *= 2.5
        
        # 为了更真实，添加位置编码的影响
        # 序列位置越靠后，某些维度的值略有增加
        position_effect = (i / 128) * 0.01
        X[i, :100] += position_effect  # 前100维受位置影响
    
    # 添加一些结构化模式（模拟attention patterns）
    # 每16个序列有相似的模式（模拟重复的结构）
    for group in range(8):  # 128/16 = 8组
        start_idx = group * 16
        end_idx = start_idx + 16
        # 这组序列共享一些特征
        shared_features = np.random.randn(256).astype(np.float32) * 0.03
        X[start_idx:end_idx, 1000:1256] += shared_features
    
    return X

def main():
    print("="*60)
    print("生成128序列版本的Attention计算矩阵")
    print("="*60)
    
    # 1. 创建输入矩阵（128序列）
    print("\n1. 创建输入矩阵X (128序列版本)...")
    X = create_input_matrix_128seq()
    print(f"   形状: {X.shape}")
    print(f"   范围: [{X.min():.4f}, {X.max():.4f}]")
    print(f"   均值: {X.mean():.6f}")
    print(f"   标准差: {X.std():.6f}")
    
    # 2. 提取Query权重
    print("\n2. 提取/生成Query权重Wq...")
    Wq = extract_wq_from_model()
    print(f"   形状: {Wq.shape}")
    zeros = (Wq == 0).sum()
    print(f"   稀疏度: {zeros/Wq.size*100:.1f}%")
    print(f"   范围: [{Wq.min():.4f}, {Wq.max():.4f}]")
    
    # 3. 创建输出目录
    output_dir = '../../Input/llminput/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n3. 输出目录: {os.path.abspath(output_dir)}")
    
    # 4. 保存矩阵
    print("\n4. 保存矩阵文件...")
    
    # 保存X_input.txt
    x_path = os.path.join(output_dir, 'X_input_128seq.txt')
    np.savetxt(x_path, X, fmt='%.8f', delimiter=' ')
    print(f"   ✓ X_input_128seq.txt [{X.shape[0]}×{X.shape[1]}]")
    
    # 保存Wq.txt
    wq_path = os.path.join(output_dir, 'Wq_128seq.txt')
    np.savetxt(wq_path, Wq, fmt='%.8f', delimiter=' ')
    print(f"   ✓ Wq_128seq.txt [{Wq.shape[0]}×{Wq.shape[1]}]")
    
    # 5. 验证计算
    print("\n5. 验证矩阵乘法:")
    Q = X @ Wq.T  # [128×4096] @ [4096×128] = [128×128]
    print(f"   Q = X @ Wq^T")
    print(f"   [{X.shape[0]}×{X.shape[1]}] @ [{Wq.shape[1]}×{Wq.shape[0]}] = [{Q.shape[0]}×{Q.shape[1]}]")
    
    # 打印Q矩阵的统计信息
    print("\n6. Query矩阵Q的计算结果 (128×128):")
    print("="*60)
    print(f"   形状: {Q.shape}")
    print(f"   总元素数: {Q.size} (原8×128版本的16倍)")
    print(f"   数值范围: [{Q.min():.6f}, {Q.max():.6f}]")
    print(f"   均值: {Q.mean():.6f}")
    print(f"   标准差: {Q.std():.6f}")
    
    # 分析Q矩阵的分布
    print("\n   Q矩阵分布分析:")
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        val = np.percentile(Q, p)
        print(f"   {p:3d}% 分位数: {val:10.6f}")
    
    # 打印前5行前8列的结果
    print("\n   Q矩阵前5行×前8列:")
    for i in range(min(5, Q.shape[0])):
        print(f"   行{i:3d}: ", end="")
        for j in range(min(8, Q.shape[1])):
            print(f"{Q[i,j]:8.5f}", end=" ")
        print("...")
    
    # 保存Q矩阵供验证
    print("\n7. 保存Q矩阵供C++验证:")
    q_result_path = os.path.join(output_dir, 'Q_result_python_128seq.txt')
    np.savetxt(q_result_path, Q, fmt='%.8f', delimiter=' ')
    print(f"   ✓ Q_result_python_128seq.txt [{Q.shape[0]}×{Q.shape[1]}]")
    print("   (C++计算结果应该与此文件一致)")
    
    # 任务数量分析
    print("\n8. 任务数量分析:")
    original_tasks = 8 * 128  # 原版本
    new_tasks = 128 * 128     # 新版本
    print(f"   原版本 (8×128):   {original_tasks:6d} 个输出元素")
    print(f"   新版本 (128×128): {new_tasks:6d} 个输出元素")
    print(f"   增长倍数: {new_tasks/original_tasks:.1f}x")
    
    # 内存估算
    print("\n9. 内存需求估算 (float32):")
    x_mem = X.nbytes / (1024*1024)  # MB
    wq_mem = Wq.nbytes / (1024*1024)
    q_mem = Q.nbytes / (1024*1024)
    total_mem = x_mem + wq_mem + q_mem
    print(f"   X_input: {x_mem:.2f} MB")
    print(f"   Wq:      {wq_mem:.2f} MB")
    print(f"   Q输出:   {q_mem:.2f} MB")
    print(f"   总计:    {total_mem:.2f} MB")
    
    print("\n完成！生成了128序列版本的矩阵文件。")
    print("注意：C++代码需要相应修改 matrixOutputPixels_inputsequencelength = 128")

if __name__ == "__main__":
    main()