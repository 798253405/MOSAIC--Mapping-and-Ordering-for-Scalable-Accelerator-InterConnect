#!/usr/bin/env python3
"""
生成两个矩阵文件：
1. X_input.txt - 输入矩阵 [8×4096]
2. Wq.txt - Query权重矩阵 [128×4096]

用于C++读取并计算 Q = X @ Wq^T
"""

import numpy as np
import struct
import json

def extract_wq_from_model():
    """
    从LLaMA模型文件中提取Query权重矩阵
    提取第0层第0个attention头的权重
    """
    # 读取SafeTensors文件的header
    with open('model-00001-of-00003.safetensors', 'rb') as f:
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
    with open('model-00001-of-00003.safetensors', 'rb') as f:
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

def create_input_matrix():
    """
    创建输入矩阵X [8×4096]
    模拟8个token的嵌入向量
    """
    # 使用固定种子确保可重复性
    np.random.seed(42)
    
    # 生成基础随机矩阵，标准差0.02（LLaMA的典型值）
    X = np.random.randn(8, 4096).astype(np.float32) * 0.02
    
    # 模拟真实嵌入的特征：某些维度更活跃
    for i in range(8):
        # 随机选择10%的维度增强
        active_dims = np.random.choice(4096, size=400, replace=False)
        X[i, active_dims] *= 2.5
    
    return X

def main():
    print("="*50)
    print("生成Attention计算所需的矩阵")
    print("="*50)
    
    # 1. 创建输入矩阵
    print("\n1. 创建输入矩阵X...")
    X = create_input_matrix()
    print(f"   形状: {X.shape}")
    print(f"   范围: [{X.min():.4f}, {X.max():.4f}]")
    
    # 2. 提取Query权重
    print("\n2. 提取Query权重Wq...")
    Wq = extract_wq_from_model()
    print(f"   形状: {Wq.shape}")
    zeros = (Wq == 0).sum()
    print(f"   稀疏度: {zeros/Wq.size*100:.1f}%")
    
    # 3. 保存矩阵
    print("\n3. 保存矩阵文件...")
    
    # 保存X_input.txt
    np.savetxt('X_input.txt', X, fmt='%.8f', delimiter=' ')
    print(f"   ✓ X_input.txt [{X.shape[0]}×{X.shape[1]}]")
    
    # 保存Wq.txt
    np.savetxt('Wq.txt', Wq, fmt='%.8f', delimiter=' ')
    print(f"   ✓ Wq.txt [{Wq.shape[0]}×{Wq.shape[1]}]")
    
    # 4. 验证计算
    print("\n4. 验证计算:")
    Q = X @ Wq.T  # [8×4096] @ [4096×128] = [8×128]
    print(f"   Q = X @ Wq^T")
    print(f"   [{X.shape[0]}×{X.shape[1]}] @ [{Wq.shape[1]}×{Wq.shape[0]}] = [{Q.shape[0]}×{Q.shape[1]}]")
    
    # 打印Q矩阵的详细结果
    print("\n5. Query矩阵Q的计算结果:")
    print("="*50)
    print(f"   形状: {Q.shape}")
    print(f"   数值范围: [{Q.min():.6f}, {Q.max():.6f}]")
    print(f"   均值: {Q.mean():.6f}")
    print(f"   标准差: {Q.std():.6f}")
    
    # 打印前5行前8列的结果
    print("\n   Q矩阵前5行×前8列:")
    for i in range(min(5, Q.shape[0])):
        print(f"   行{i}: ", end="")
        for j in range(min(8, Q.shape[1])):
            print(f"{Q[i,j]:8.5f}", end=" ")
        print("...")
    
    # 保存Q矩阵供验证
    print("\n6. 保存Q矩阵供C++验证:")
    np.savetxt('Q_result_python.txt', Q, fmt='%.8f', delimiter=' ')
    print(f"   ✓ Q_result_python.txt [{Q.shape[0]}×{Q.shape[1]}]")
    print("   (C++计算结果应该与此文件一致)")
    
    print("\n完成！")

if __name__ == "__main__":
    main()