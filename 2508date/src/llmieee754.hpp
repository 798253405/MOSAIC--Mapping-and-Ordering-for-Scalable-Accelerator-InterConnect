/**

 * @brief LLM IEEE754 Bit Manipulation and Sorting Utilities
 *
 * LLMIEEE754bit。
 * NoCbit，。
 *
 * ：
 * ----------
 * 1. （Separated Ordering）
 *    - QueryKey
 *    - IEEE754 1-bit
 *    - bit
 *
 * 2. （Affiliated Ordering）
 *    - Query-Key
 *    - Keybit，Query
 *    -
 *
 * ：
 * ---------
 * - ： [query(64), key(64)]
 * - ：8x8，
 * - ：flits，flit 16
 *
 * @date 2025
 */

#ifndef AUTHORLLMIEEE754_HPP_
#define AUTHORLLMIEEE754_HPP_

#include <deque>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "IEEE754.hpp"  // IEEE754

namespace authorLLMIEEE754 {

/**
 * @brief LLM -
 *
 * payload，。
 * 。
 *
 * @param payload /，128float（64 query + 64 key）
 */
void llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload);

/**
 * @brief LLM - QueryKey
 *
 * IEEE754 1-bit。
 * Query-Key。
 *
 * @param query_data Query（64）
 * @param key_data Key（64）
 * @param cols （8）
 * @param rows （8）
 */


/**
 * @brief LLM - Query-Key
 *
 * Keybit，QueryKey。
 * attention。
 *
 * @param query_data Query（64）
 * @param key_data Key（64）
 * @param cols （8）
 * @param rows （8）
 */
void sortAffiliated(std::deque<float>& query_data,
                    std::deque<float>& key_data,
                    int cols = 8, int rows = 8);

/**
 * @brief
 *
 * 。
 * ，sortSeparated。
 *
 * @param data
 * @param cols
 * @param rows
 */

/**
 * @brief LLM（）
 *
 * 、IEEE7541-bit。
 *
 * @param data
 * @param name
 * @param max_elements
 */
void printDetailedData(const std::deque<float>& data,
                      const std::string& name,
                      int max_elements = 8);

/**
 * @brief bit
 *
 * bit、。
 *
 * @param query_data Query
 * @param key_data Key
 * @param show_details
 */
void analauthoreBitStatistics(const std::deque<float>& query_data,
                          const std::deque<float>& key_data,
                          bool show_details = false);

/**
 * @brief bit
 *
 * bit。
 *
 * @param data
 * @return bit
 */
int calculateBitFlips(const std::deque<float>& data);

/**
 * @brief LLM/ - Type 0Type 3
 *
 * Type 0（）Type 3（）。
 * 16，0，padding。
 *
 * @param payload 16floatpayload
 * @param value 0（Type 0task_id，Type 3）
 */
void llmReqRestReorderingFunc(std::deque<float>& payload, float value = 0.0f);

/**
 * @brief
 *
 * bit。
 *
 * @param original_data
 * @param sorted_data
 * @param name
 */
void verifyOptimization(const std::deque<float>& original_data,
                        const std::deque<float>& sorted_data,
                        const std::string& name);

} // namespace authorLLMIEEE754

#endif /* AUTHORLLMIEEE754_HPP_ */
