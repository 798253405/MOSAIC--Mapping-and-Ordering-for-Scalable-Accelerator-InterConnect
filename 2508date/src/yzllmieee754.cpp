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
    // 创建8x8矩阵
    std::vector<std::vector<float>> input_matrix(8, std::vector<float>(8, 0.0f));
    std::vector<std::vector<float>> query_matrix(8, std::vector<float>(8, 0.0f));
    // Debug: 只在有非零数据时打印排序前的值
    static int debug_count = 0;
    bool has_nonzero = false;
    for (int i = 0; i < input_data.size() && !has_nonzero; i++) {
        if (input_data[i] != 0.0f) has_nonzero = true;
    }

    // Step 3: 根据宏定义应用排序优化
    #ifdef YZSeperatedOrdering_reArrangeInput
    std::vector<int> indices(query_weights.size());
          for (size_t i = 0; i < indices.size(); ++i) {
              indices[i] = i;
          }
  // Step 2: 基于query权重的1-bit数对索引排序
		 std::sort(indices.begin(), indices.end(), [&](int i, int j) {
			 return compareFloatsByOnes(query_weights[i], query_weights[j]);
		 });
		 std::deque<float> sorted_query;
		   for (int idx : indices) {
			            sorted_query.push_back(query_weights[idx]);
			        }
		 //For input
		   std::vector<int> indicesInput(input_data.size());
		          for (size_t i = 0; i < indicesInput.size(); ++i) {
		        	  indicesInput[i] = i;
		          }

		 std::sort(indicesInput.begin(), indicesInput.end(), [&](int i, int j) {
			 return compareFloatsByOnes(input_data[i], input_data[j]);
		 });
		 std::deque<float> sorted_input;
		   for (int idx : indicesInput) {
			   sorted_input.push_back(input_data[idx]);
				        }
		 // Step 4: 按列重组成8x8矩阵
		        // 按列填充矩阵：for i in cols, for j in rows
		        for (int col = 0; col < 8; col++) {
		            for (int row = 0; row < 8; row++) {
		                if (!sorted_input.empty()) {
		                    input_matrix[row][col] = sorted_input.front();
		                    sorted_input.pop_front();
		                } else {
		                    input_matrix[row][col] = 0.0f;
		                }

		                if (!sorted_query.empty()) {
		                    query_matrix[row][col] = sorted_query.front();
		                    sorted_query.pop_front();
		                } else {
		                    query_matrix[row][col] = 0.0f;
		                }
		            }
		        }

		        input_data.clear();
		        query_weights.clear();

        // 分离排序模式 - 分别对两个数据段排序
    #elif defined(YzAffiliatedOrdering)
        // 关联排序模式 - 基于query的1-bit数排序，同时保持配对关系
        // Step 1: 创建索引数组，跟踪元素位置
        std::vector<int> indices(query_weights.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
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
        
        // Step 4: 按列重组成8x8矩阵
        // 按列填充矩阵：for i in cols, for j in rows
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 8; row++) {
                if (!sorted_input.empty()) {
                    input_matrix[row][col] = sorted_input.front();
                    sorted_input.pop_front();
                } else {
                    input_matrix[row][col] = 0.0f;
                }
                
                if (!sorted_query.empty()) {
                    query_matrix[row][col] = sorted_query.front();
                    sorted_query.pop_front();
                } else {
                    query_matrix[row][col] = 0.0f;
                }
            }
        }
        
        input_data.clear();
        query_weights.clear();


#else
        for (int col = 0; col < 8; col++) {
                    for (int row = 0; row < 8; row++) {
                        if (!input_data.empty()) {
                            input_matrix[row][col] = input_data.front();
                            input_data.pop_front();
                        } else {
                            input_matrix[row][col] = 0.0f;
                        }

                        if (! query_weights.empty()) {
                            query_matrix[row][col] =  query_weights.front();
                            query_weights.pop_front();
                        } else {
                            query_matrix[row][col] = 0.0f;
                        }
                    }
                }
#endif
    // 否则不排序
        //
	std::vector<std::deque<float>> combined_rows(8);
	for (int i = 0; i < 8; ++i) { // 先写死 8
		for  (int j = 0; j < 8; ++j) {
			combined_rows[i].push_back(input_matrix[i][j]);
		}
		for  (int j = 0; j < 8; ++j){
			combined_rows[i].push_back(query_matrix[i][j]);
		}
	}
    // Step 6: 拼接两个矩阵（左边input，右边query）
    payload.clear();
    for (const auto &row : combined_rows) {
    		for (const auto &element : row) {
    			payload.push_back(element);
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

void llmReqRestReorderingFunc(std::deque<float>& payload, float value) {
    // 确保payload大小为16
    if (payload.size() != 16) {
        payload.resize(16, 0.0f);
    }
    
    // Type 0和Type 3消息的优化策略：
    // 1. 第0位设置为指定值（task_id或结果值）
    // 2. 其余15个位置填充0（已经是0的保持不变）
    payload[0] = value;
    
    // 如果启用了排序优化，可以对padding进行优化
    // 但由于都是0，实际上不需要排序
    #ifdef YZSeperatedOrdering_reArrangeInput
        // 对于Type 0/3，padding都是0，无需排序
        // 保持原样即可
    #elif defined(YzAffiliatedOrdering)
        // 对于Type 0/3，padding都是0，无需排序
        // 保持原样即可
    #endif
    
    // Debug输出（可选）
    static int debug_count = 0;
    if (debug_count < 3 && value != 0) {
        std::cout << "[DEBUG-LLM-REORDER] Type 0/3 message: value=" << value 
                  << ", payload size=" << payload.size() << std::endl;
        debug_count++;
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
