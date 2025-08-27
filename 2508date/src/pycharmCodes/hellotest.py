import numpy as np
import time

# ========================================================================
# == 1. C-like Python 代码 (模拟C++逻辑)
# ========================================================================

# --- Llama 2 (7B) 模型及硬件参数设定 ---
DIM_MODEL = 4096
NUM_HEAD = 32
SEQUENCE_LENGTH = 512
D_HEAD = DIM_MODEL // NUM_HEAD  # 4096 / 32 = 128

# --- 硬件假设 ---
# 1. 基础计算单元：每个Node节点可以执行一个 8x8 的MAC操作
MAC_OPERATION_DIM = 8
# 2. 并行架构：我们有一个 32x32 的计算节点阵列 (Node Array)
NODE_ARRAY_DIM = 32


def tiled_qk_transpose_multiply_c_like(Q, K_T):
    """
    模拟在32x32的NoC阵列上对QKᵀ乘法进行分块计算。
    此函数严格模仿C++版本的循环结构。
    """
    # 初始化Scores矩阵为0
    Scores = np.zeros((SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=np.float32)

    # 遍历 32x32 的计算节点阵列 (Node Array)
    # 这两层循环模拟了1024个节点的并行工作
    for node_row in range(NODE_ARRAY_DIM):
        for node_col in range(NODE_ARRAY_DIM):

            # --- 当前进入了一个特定Node(node_row, node_col)的处理流程 ---
            # 该Node负责计算 2x2 = 4 个 8x8 的子块 (Node Tiles)
            # 这4个任务由该节点串行完成
            for task_row in range(2):
                for task_col in range(2):

                    # 计算这个 8x8 任务块在全局 512x512 Scores矩阵中的起始坐标
                    i_start = (node_row * 2 + task_row) * MAC_OPERATION_DIM
                    j_start = (node_col * 2 + task_col) * MAC_OPERATION_DIM

                    # --- 当前进入了一个Node Tile (8x8) 的处理流程 ---

                    # 沿共享维度（k_dim, 对应d_head）进行遍历和累加 (时间映射)
                    for k_start in range(0, D_HEAD, MAC_OPERATION_DIM):

                        # --- 当前进入了一个Time Slice的处理流程 ---
                        # 每个 8x8 Node Tile的计算需要 128 / 8 = 16 个Time Slices。

                        i_end = i_start + MAC_OPERATION_DIM
                        j_end = j_start + MAC_OPERATION_DIM
                        k_end = k_start + MAC_OPERATION_DIM

                        # 在Node节点内部执行 8x8 的点积和累加操作
                        # 这是最底层的循环，完全模仿C/C++
                        for i in range(i_start, i_end):
                            for j in range(j_start, j_end):
                                partial_sum = 0.0
                                for k in range(k_start, k_end):
                                    partial_sum += Q[i, k] * K_T[k, j]
                                Scores[i, j] += partial_sum
    return Scores


# ========================================================================
# == 2. 验证与Debug代码 (使用标准NumPy)
# ========================================================================

def validate_with_numpy(Q, K_T, c_like_result):
    """
    使用NumPy高效计算结果，并与C-like版本的结果进行对比验证。
    """
    print("\n--- 开始验证 ---")

    # 1. 使用NumPy的np.dot()高效计算标准答案
    print("正在使用 NumPy np.dot() 计算标准答案...")
    start_time = time.time()
    numpy_result = np.dot(Q, K_T)
    end_time = time.time()
    print(f"NumPy 计算耗时: {end_time - start_time:.4f} 秒")

    # 2. 比较两个结果
    # np.allclose() 用于比较两个浮点数数组是否在容差范围内“足够接近”
    print("正在比较 C-like 版本的结果和 NumPy 标准答案...")
    are_equal = np.allclose(c_like_result, numpy_result, rtol=1e-5, atol=1e-8)

    if are_equal:
        print("\n✅ 验证通过！C-like版本的计算结果与NumPy完全一致。")
    else:
        print("\n❌ 验证失败！两个版本的计算结果不一致。")
        # 计算并打印差异
        difference = np.abs(c_like_result - numpy_result)
        print(f"  最大差异值: {np.max(difference)}")
        print(f"  差异发生位置 (第一个): {np.unravel_index(np.argmax(difference), difference.shape)}")

    print("--- 验证结束 ---\n")


if __name__ == "__main__":
    # --- 准备数据 ---
    # 使用NumPy创建与C++版本中相同维度的矩阵
    print("正在准备输入数据 (Q 和 K_T 矩阵)...")
    # 为了使结果可验证，使用随机数种子
    np.random.seed(42)
    Q = np.random.rand(SEQUENCE_LENGTH, D_HEAD).astype(np.float32)
    K_T = np.random.rand(D_HEAD, SEQUENCE_LENGTH).astype(np.float32)

    print(f"输入 Q 矩阵维度: {Q.shape}")
    print(f"输入 K_T 矩阵维度: {K_T.shape}")
    print("-" * 20)

    # --- 执行C-like模拟计算 ---
    print("\n正在执行 C-like 版本的模拟计算 (这会比较慢)...")
    start_time = time.time()
    c_like_scores = tiled_qk_transpose_multiply_c_like(Q, K_T)
    end_time = time.time()
    print(f"C-like 版本计算耗时: {end_time - start_time:.4f} 秒")
    print(f"输出 Scores 矩阵维度: {c_like_scores.shape}")

    # --- 执行验证 ---
    validate_with_numpy(Q, K_T, c_like_scores)

