/**
 * @file yzllmieee754.cpp
 * @brief LLM IEEE754 Bit Manipulation and Sorting Implementation
 * 
 * 实现LLM特定的IEEE754浮点数排序优化算法。
 * 这些函数从llmmac.cpp中提取，专门处理Transformer Attention计算的数据优化。
 * 
 * @author YZ
 * @date 2025
 */

#include "yzllmieee754.hpp"
#include "parameters.hpp"
#include <cmath>
#include <cstdint>

namespace YzLLMIEEE754 {

void llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload) {
    // Step 1: 检查payload格式
    // LLM payload格式: [query数据(64个), key数据(64个)]
    if (payload.size() < 128) {
        std::cerr << "WARNING: Payload too small: " << payload.size() << " < 128" << std::endl;
        return;
    }
    
    // Step 2: 提取Query和Key数据段
    int data_size = 64;  // 每个矩阵64个元素
    std::deque<float> query_data(payload.begin(), payload.begin() + data_size);
    std::deque<float> key_data(payload.begin() + data_size, payload.begin() + data_size * 2);
    
    // Step 3: 根据宏定义应用排序优化
    #ifdef YZSeperatedOrdering_reArrangeInput
        // 分离排序模式
        sortSeparated(query_data, key_data, 8, 8);
    #elif defined(YzAffiliatedOrdering)
        // 关联排序模式
        sortAffiliated(query_data, key_data, 8, 8);
    #endif
    // 否则不排序
    
    // Step 4: 矩阵重组 - 按列主序填充
    int rownum_per_col = 8;
    int querycolnum_per_row = 8;
    int keycolnum_per_row = 8;
    int totalcolnum_per_row = querycolnum_per_row + keycolnum_per_row;
    
    std::vector<std::deque<float>> query_rows(rownum_per_col);
    std::vector<std::deque<float>> key_rows(rownum_per_col);
    
    // 填充Query矩阵（按列主序）
    for (int col_index = 0; col_index < querycolnum_per_row; col_index++) {
        for (int row_index = 0; row_index < rownum_per_col; row_index++) {
            int idx = col_index * rownum_per_col + row_index;
            if (idx < query_data.size()) {
                query_rows[row_index].push_back(query_data[idx]);
            } else {
                query_rows[row_index].push_back(0.0f);  // 填充零
            }
        }
    }
    
    // 填充Key矩阵（按列主序）
    for (int col_index = 0; col_index < keycolnum_per_row; col_index++) {
        for (int row_index = 0; row_index < rownum_per_col; row_index++) {
            int idx = col_index * rownum_per_col + row_index;
            if (idx < key_data.size()) {
                key_rows[row_index].push_back(key_data[idx]);
            } else {
                key_rows[row_index].push_back(0.0f);  // 填充零
            }
        }
    }
    
    // Step 5: 按行组合query和key数据
    std::vector<std::deque<float>> combined_flits(rownum_per_col);
    for (int row = 0; row < rownum_per_col; ++row) {
        // 先添加query第row行的所有列元素
        combined_flits[row].insert(combined_flits[row].end(), 
            query_rows[row].begin(), query_rows[row].end());
        // 再添加key第row行的所有列元素
        combined_flits[row].insert(combined_flits[row].end(), 
            key_rows[row].begin(), key_rows[row].end());
        
        // 验证flit大小
        if (combined_flits[row].size() != totalcolnum_per_row) {
            std::cerr << "Error: flit size not equal to expected 16 elements. "
                     << "Flit " << row << " size: " << combined_flits[row].size()
                     << " Expected: " << totalcolnum_per_row << std::endl;
            assert(false && "Error: flit size not equal to expected 16 elements");
            return;
        }
    }
    
    // Step 6: 将重组后的数据写回原始容器
    payload.clear();
    for (const auto &flit : combined_flits) {
        for (const auto &element : flit) {
            payload.push_back(element);
        }
    }
}

void sortSeparated(std::deque<float>& query_data, 
                   std::deque<float>& key_data,
                   int cols, int rows) {
    // 分别对Query和Key进行独立排序
    sortMatrixByColumns(query_data, cols, rows);
    sortMatrixByColumns(key_data, cols, rows);
}

void sortAffiliated(std::deque<float>& query_data,
                    std::deque<float>& key_data,
                    int cols, int rows) {
    // 验证输入
    if (key_data.empty() || query_data.empty()) {
        std::cerr << "ERROR: sortAffiliated - key_data or query_data is empty!" << std::endl;
        assert(false && "sortAffiliated: empty data provided");
        return;
    }
    if (key_data.size() != query_data.size()) {
        std::cerr << "ERROR: sortAffiliated - key_data size (" << key_data.size() 
                  << ") != query_data size (" << query_data.size() << ")" << std::endl;
        assert(false && "sortAffiliated: size mismatch between key and query");
        return;
    }
    
    // Step 1: 创建索引数组，用于跟踪元素原始位置
    std::vector<int> indices(key_data.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    // Step 2: 根据key的bit数对索引排序
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
#ifdef FIXED_POINT_SORTING
        return compareFloatsByFixed17Ones(key_data[i], key_data[j]);
#else
        return compareFloatsByOnes(key_data[i], key_data[j]);
#endif
    });
    
    // Step 3: 根据排序后的索引重新排列query和key
    std::deque<float> sortedKeys;
    std::deque<float> sortedQueries;
    for (int idx : indices) {
        sortedKeys.push_back(key_data[idx]);
        sortedQueries.push_back(query_data[idx]);
    }
    
    // Step 4: 将排序后的数据写回原容器
    query_data = sortedQueries;
    key_data = sortedKeys;
}

void sortMatrixByColumns(std::deque<float>& data, int cols, int rows) {
    if (data.empty()) return;
    
    // Step 1: 对整个数据进行全局排序
#ifdef FIXED_POINT_SORTING
    std::sort(data.begin(), data.end(), compareFloatsByFixed17Ones);
#else
    std::sort(data.begin(), data.end(), compareFloatsByOnes);
#endif
    
    // Step 2: 按列主序重新排列到矩阵
    std::vector<std::deque<float>> matrix_rows(rows);
    int row_index = 0;
    int col_index = 0;
    
    for (float num : data) {
        matrix_rows[row_index].push_back(num);
        col_index++;
        if (col_index == cols) {
            row_index++;
            col_index = 0;
        }
        if (row_index == rows) {
            row_index = 0;
        }
    }
    
    // Step 3: 按行序写回数据
    data.clear();
    for (const auto& row : matrix_rows) {
        for (float val : row) {
            data.push_back(val);
        }
    }
}

void printDetailedData(const std::deque<float>& data, 
                      const std::string& name,
                      int max_elements) {
    std::cout << "\n=== " << name << " Debug Info ===\n";
    std::cout << "Total elements: " << data.size() << "\n";
    
    int print_count = std::min((int)data.size(), max_elements);
    for (int i = 0; i < print_count; i++) {
        float value = data[i];
        std::string bit_repr = float_to_ieee754(value);
        int bit_count = countOnesInIEEE754(value);
        
        std::cout << "[" << i << "] Value: " << std::fixed << std::setprecision(6) << value
                  << " | Bits: " << bit_repr 
                  << " | 1-Count: " << bit_count << "\n";
    }
    if ((int)data.size() > max_elements) {
        std::cout << "... (showing first " << max_elements << " of " << data.size() << " elements)\n";
    }
    std::cout << std::endl;
}

void analyzeBitStatistics(const std::deque<float>& query_data,
                          const std::deque<float>& key_data,
                          bool show_details) {
    std::cout << "\n--- Bit Statistics Analysis ---" << std::endl;
    
    // Calculate bit counts
    std::vector<int> q_bit_counts, k_bit_counts;
    int q_total = 0, k_total = 0;
    int q_min = 32, q_max = 0;
    int k_min = 32, k_max = 0;
    
    for (const auto& val : query_data) {
        int bits = countOnesInIEEE754(val);
        q_bit_counts.push_back(bits);
        q_total += bits;
        q_min = std::min(q_min, bits);
        q_max = std::max(q_max, bits);
    }
    
    for (const auto& val : key_data) {
        int bits = countOnesInIEEE754(val);
        k_bit_counts.push_back(bits);
        k_total += bits;
        k_min = std::min(k_min, bits);
        k_max = std::max(k_max, bits);
    }
    
    // Display statistics
    std::cout << "Query Statistics:" << std::endl;
    std::cout << "  Size: " << query_data.size() << " elements" << std::endl;
    std::cout << "  Bit count range: [" << q_min << ", " << q_max << "]" << std::endl;
    std::cout << "  Average bits: " << (q_total / (float)query_data.size()) << std::endl;
    
    std::cout << "Key Statistics:" << std::endl;
    std::cout << "  Size: " << key_data.size() << " elements" << std::endl;
    std::cout << "  Bit count range: [" << k_min << ", " << k_max << "]" << std::endl;
    std::cout << "  Average bits: " << (k_total / (float)key_data.size()) << std::endl;
    
    if (show_details && query_data.size() <= 16) {
        std::cout << "\nQuery bit counts: ";
        for (int bits : q_bit_counts) {
            std::cout << bits << " ";
        }
        std::cout << "\nKey bit counts: ";
        for (int bits : k_bit_counts) {
            std::cout << bits << " ";
        }
        std::cout << std::endl;
    }
}

int calculateBitFlips(const std::deque<float>& data) {
    if (data.size() < 2) return 0;
    
    int total_flips = 0;
    for (size_t i = 1; i < data.size(); i++) {
        // 将两个浮点数转换为整数表示，计算XOR，然后统计1的个数
        union {
            float f;
            uint32_t i;
        } prev, curr;
        
        prev.f = data[i-1];
        curr.f = data[i];
        
        // XOR得到不同的bit位，然后统计1的个数（即Hamming距离）
        uint32_t xor_result = prev.i ^ curr.i;
        int hamming_distance = __builtin_popcount(xor_result);
        
        total_flips += hamming_distance;
    }
    return total_flips;
}

void verifyOptimization(const std::deque<float>& original_data,
                        const std::deque<float>& sorted_data,
                        const std::string& name) {
    int original_flips = calculateBitFlips(original_data);
    int sorted_flips = calculateBitFlips(sorted_data);
    
    float reduction = (1.0f - (float)sorted_flips / original_flips) * 100.0f;
    
    std::cout << "\n=== " << name << " Optimization Results ===" << std::endl;
    std::cout << "Original bit flips: " << original_flips << std::endl;
    std::cout << "Sorted bit flips: " << sorted_flips << std::endl;
    std::cout << "Reduction: " << std::fixed << std::setprecision(2) 
              << reduction << "%" << std::endl;
    
    if (reduction < 0) {
        std::cout << "WARNING: Sorting increased bit flips!" << std::endl;
    }
}

} // namespace YzLLMIEEE754