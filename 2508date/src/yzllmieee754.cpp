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
    // LLM payload格式: [input数据(64个), query权重(64个)]
    
    // Step 2: 提取Input和Query数据段
    int data_size = 64;  // 每个矩阵64个元素
    std::deque<float> input_data(payload.begin(), payload.begin() + data_size);
    std::deque<float> query_weights(payload.begin() + data_size, payload.begin() + data_size * 2);
    
    // Debug: 只在有非零数据时打印排序前的值
    static int debug_count = 0;
    bool has_nonzero = false;
    for (int i = 0; i < input_data.size() && !has_nonzero; i++) {
        if (input_data[i] != 0.0f) has_nonzero = true;
    }
    
    if (debug_count < 3) {
        if (has_nonzero) {
            std::cout << "[DEBUG-SORT-BEFORE] Payload size=" << payload.size() 
                      << " First 5 input: ";
            for (int i = 0; i < 5 && i < input_data.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << input_data[i] << " ";
            }
            std::cout << "\n[DEBUG-SORT-BEFORE] First 5 query: ";
            for (int i = 0; i < 5 && i < query_weights.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << query_weights[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "[DEBUG-SORT] Called with all-zero payload (size=" << payload.size() << ")" << std::endl;
        }
    }
    
    // Step 3: 根据宏定义应用排序优化
    #ifdef YZSeperatedOrdering_reArrangeInput
        // 分离排序模式 - 分别对两个数据段排序
        sortMatrixByColumns(input_data, 8, 8);
        sortMatrixByColumns(query_weights, 8, 8);
    #elif defined(YzAffiliatedOrdering)
        // 关联排序模式 - 基于query的1-bit数排序，同时保持配对关系
        // Step 1: 创建索引数组，跟踪元素位置
        std::vector<int> indices(query_weights.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        // Debug: 打印排序前query的bit分布
        if (has_nonzero && debug_count < 3) {
            std::cout << "[DEBUG-PRESORT] Query 1-bit distribution before sort: ";
            for (int i = 0; i < 8 && i < query_weights.size(); i++) {
                int bits = countOnesInIEEE754(query_weights[i]);
                std::cout << bits << " ";
            }
            std::cout << "..." << std::endl;
        }
        
        // Step 2: 基于query权重的1-bit数对索引排序
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return compareFloatsByOnes(query_weights[i], query_weights[j]);
        });
        
        // Step 3: 根据排序后的索引重新排列（保持配对）
        std::deque<float> sorted_input;
        std::deque<float> sorted_query;
        for (int idx : indices) {
            sorted_input.push_back(input_data[idx]);
            sorted_query.push_back(query_weights[idx]);
        }
        
        // Debug: 打印排序后的bit分布
        if (has_nonzero && debug_count < 3) {
            std::cout << "[DEBUG-SORTED] Query 1-bit distribution after sort: ";
            for (int i = 0; i < 8 && i < sorted_query.size(); i++) {
                int bits = countOnesInIEEE754(sorted_query[i]);
                std::cout << bits << " ";
            }
            std::cout << "..." << std::endl;
        }
        
        // Step 4: 按列重组成8x8矩阵
        // 创建8x8矩阵
        std::vector<std::vector<float>> input_matrix(8, std::vector<float>(8, 0.0f));
        std::vector<std::vector<float>> query_matrix(8, std::vector<float>(8, 0.0f));
        
        // 将排序后的数据转为deque便于pop
        std::deque<float> sorted_input_dq(sorted_input.begin(), sorted_input.end());
        std::deque<float> sorted_query_dq(sorted_query.begin(), sorted_query.end());
        
        // 按列填充矩阵：for i in cols, for j in rows
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 8; row++) {
                if (!sorted_input_dq.empty()) {
                    input_matrix[row][col] = sorted_input_dq.front();
                    sorted_input_dq.pop_front();
                } else {
                    input_matrix[row][col] = 0.0f;
                }
                
                if (!sorted_query_dq.empty()) {
                    query_matrix[row][col] = sorted_query_dq.front();
                    sorted_query_dq.pop_front();
                } else {
                    query_matrix[row][col] = 0.0f;
                }
            }
        }
        
        // Step 5: 将矩阵按行读出，重新组成线性数据
        input_data.clear();
        query_weights.clear();
        
        // Debug: 打印重组后的矩阵（显示每列是否有序）
        if (has_nonzero && debug_count < 3) {
            std::cout << "[DEBUG-MATRIX] Query matrix columns (should be sorted):" << std::endl;
            for (int col = 0; col < 8; col++) {
                std::cout << "  Col " << col << ": ";
                for (int row = 0; row < 8; row++) {
                    int bits = countOnesInIEEE754(query_matrix[row][col]);
                    std::cout << bits << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // 先读出所有input行
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                input_data.push_back(input_matrix[row][col]);
            }
        }
        
        // 再读出所有query行
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                query_weights.push_back(query_matrix[row][col]);
            }
        }
#endif
    // 否则不排序
    
    // Debug: 只在有非零数据时打印排序后的值
    if (debug_count < 3) {
        if (has_nonzero) {
            std::cout << "[DEBUG-SORT-AFTER] First 5 input: ";
            for (int i = 0; i < 5 && i < input_data.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << input_data[i] << " ";
            }
            std::cout << "\n[DEBUG-SORT-AFTER] First 5 query: ";
            for (int i = 0; i < 5 && i < query_weights.size(); i++) {
                std::cout << std::fixed << std::setprecision(3) << query_weights[i] << " ";
            }
            std::cout << std::endl;
        }
        debug_count++;
    }
    
    // Step 6: 拼接两个矩阵（左边input，右边query）
    payload.clear();
    
    // 先添加input数据
    for (const auto& val : input_data) {
        payload.push_back(val);
    }
    
    // 再添加query权重数据
    for (const auto& val : query_weights) {
        payload.push_back(val);
    }
}



void sortMatrixByColumns(std::deque<float>& dq, int colnum_per_row, int rownum_per_col) {
    if (dq.empty()) return;

    // Step 1: 对整个数据进行全局排序（与CNN完全相同）
    // 将矩阵展开为一行：B0, B1, B2, ..., B31
#ifdef FIXED_POINT_SORTING
    std::sort(dq.begin(), dq.end(), compareFloatsByFixed17Ones);
#else
    std::sort(dq.begin(), dq.end(), compareFloatsByOnes);
#endif

	// put the sorted number back to one row. Make sure the order is col-major
	//
		std::vector<std::deque<float>> rows(rownum_per_col); //rownum_per_col = the number of flits
		// Fill rows with elements from dq
		int row_index = 0;
		int col_index = 0;
		for (float num : dq) {
			rows[row_index].push_back(num);
			col_index++;
			if (col_index == colnum_per_row) {
				row_index++;
				col_index = 0;
			}
			if (row_index == rownum_per_col) { // reach max row number,reset
				row_index = 0;
			}
		}
		dq.clear();
		for (const auto &row : rows) {
			for (const auto &element : row) {
				dq.push_back(element);
			}
		}
}


void sortAffiliated(std::deque<float>& first_data,
                    std::deque<float>& second_data,
                    int cols, int rows) {
    // Debug: 只在有非零数据时显示
    static int sort_real_count = 0;
    bool has_nonzero = false;
    for (int i = 0; i < second_data.size() && !has_nonzero; i++) {
        if (second_data[i] != 0.0f) has_nonzero = true;
    }
    
    if (has_nonzero && sort_real_count < 3) {
        std::cout << "[SORT-AFFILIATED-BEFORE] Real data #" << ++sort_real_count 
                  << ", size=" << first_data.size() << std::endl;
        std::cout << "  Query first 5 values and 1-bits: ";
        for (int i = 0; i < 5 && i < second_data.size(); i++) {
            int bits = countOnesInIEEE754(second_data[i]);
            std::cout << std::fixed << std::setprecision(3) << second_data[i] << "(" << bits << ") ";
        }
        std::cout << std::endl;
    }
    
    // 验证输入
    if (first_data.empty() || second_data.empty()) {
        std::cerr << "ERROR: sortAffiliated - first_data or second_data is empty!" << std::endl;
        assert(false && "sortAffiliated: empty data provided");
        return;
    }
    if (first_data.size() != second_data.size()) {
        std::cerr << "ERROR: sortAffiliated - first_data size (" << first_data.size() 
                  << ") != second_data size (" << second_data.size() << ")" << std::endl;
        assert(false && "sortAffiliated: size mismatch between first and second data");
        return;
    }
    
    // Step 1: 创建索引数组，用于跟踪元素原始位置
    std::vector<int> indices(first_data.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    // Step 2: 根据second_data（query权重）的1-bit数对索引排序
    // 这样query权重会按1的个数排序，input会跟随query一起移动
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
#ifdef FIXED_POINT_SORTING
        return compareFloatsByFixed17Ones(second_data[i], second_data[j]);
#else
        return compareFloatsByOnes(second_data[i], second_data[j]);
#endif
    });
    
    // Step 3: 根据排序后的索引重新排列两个数据集，保持配对关系
    std::deque<float> sortedFirst;
    std::deque<float> sortedSecond;
    for (int idx : indices) {
        sortedFirst.push_back(first_data[idx]);
        sortedSecond.push_back(second_data[idx]);
    }
    
    // Step 4: 将排序后的数据写回原容器
    first_data = sortedFirst;
    second_data = sortedSecond;
    
    // Debug: 显示排序后的结果
    if (has_nonzero && sort_real_count <= 3) {
        std::cout << "[SORT-AFFILIATED-AFTER] Query after sorting: ";
        for (int i = 0; i < 5 && i < second_data.size(); i++) {
            int bits = countOnesInIEEE754(second_data[i]);
            std::cout << std::fixed << std::setprecision(3) << second_data[i] << "(" << bits << ") ";
        }
        std::cout << std::endl;
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
