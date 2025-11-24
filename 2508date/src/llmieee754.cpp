/**
 * @file llmieee754.cpp
 * @brief LLM IEEE754 Bit Manipulation and Sorting Implementation
 *
 * LLMIEEE754。
 * llmmac.cpp，Transformer Attention。
 *
 * @date 2025
 */

#include "llmieee754.hpp"
#include "parameters.hpp"
#include <cmath>
#include <cstdint>

namespace authorLLMIEEE754 {

void llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload) {
    // Step 1: payload
    // LLM payload: [input(64), query(64)]

    // Step 2: InputQuery
    int data_size = 64;  // 64
    std::deque<float> input_data(payload.begin(), payload.begin() + data_size);
    std::deque<float> query_weights(payload.begin() + data_size, payload.begin() + data_size * 2);
    // 8x8
    std::vector<std::vector<float>> input_matrix(8, std::vector<float>(8, 0.0f));
    std::vector<std::vector<float>> query_matrix(8, std::vector<float>(8, 0.0f));
    // Debug:
    static int debug_count = 0;
    bool has_nonzero = false;
    for (int i = 0; i < input_data.size() && !has_nonzero; i++) {
        if (input_data[i] != 0.0f) has_nonzero = true;
    }

    // Step 3:
    #ifdef AUTHORSeperatedOrdering_reArrangeInput
    std::vector<int> indices(query_weights.size());
          for (size_t i = 0; i < indices.size(); ++i) {
              indices[i] = i;
          }
  // Step 2: query1-bit
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
		 // Step 4: 8x8
		        // ：for i in cols, for j in rows
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

        //  -
    #elif defined(AuthorAffiliatedOrdering)
        //  - query1-bit，
        // Step 1: ，
        std::vector<int> indices(query_weights.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        // Step 2: query1-bit
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return compareFloatsByOnes(query_weights[i], query_weights[j]);
        });
        // Step 3: （）
        std::deque<float> sorted_input;
        std::deque<float> sorted_query;
        for (int idx : indices) {
            sorted_input.push_back(input_data[idx]);
            sorted_query.push_back(query_weights[idx]);
        }

        // Step 4: 8x8
        // ：for i in cols, for j in rows
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
        //
	std::vector<std::deque<float>> combined_rows(8);
	for (int i = 0; i < 8; ++i) { //  8
		for  (int j = 0; j < 8; ++j) {
			combined_rows[i].push_back(input_matrix[i][j]);
		}
		for  (int j = 0; j < 8; ++j){
			combined_rows[i].push_back(query_matrix[i][j]);
		}
	}
    // Step 6: （input，query）
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

void analauthoreBitStatistics(const std::deque<float>& query_data,
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
    // payload16
    if (payload.size() != 16) {
        payload.resize(16, 0.0f);
    }

    // Type 0Type 3：
    // 1. 0（task_id）
    // 2. 150（0）
    payload[0] = value;

    // ，padding
    // 0，
    #ifdef AUTHORSeperatedOrdering_reArrangeInput
        // Type 0/3，padding0，
    #elif defined(AuthorAffiliatedOrdering)
        // Type 0/3，padding0，
    #endif

    // Debug（）
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
        // ，XOR，1
        union {
            float f;
            uint32_t i;
        } prev, curr;

        prev.f = data[i-1];
        curr.f = data[i];

        // XORbit，1（Hamming）
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

} // namespace authorLLMIEEE754
