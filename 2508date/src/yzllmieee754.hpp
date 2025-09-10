/**
 * @file yzllmieee754.hpp
 * @brief LLM IEEE754 Bit Manipulation and Sorting Utilities
 * 
 * 专门为LLM模式提供的IEEE754浮点数bit操作和排序优化功能。
 * 主要目标是通过智能排序减少NoC传输中的bit翻转，提高能效。
 * 
 * 排序策略：
 * ----------
 * 1. 分离排序（Separated Ordering）
 *    - Query和Key独立排序
 *    - 按IEEE754 1-bit数量排序
 *    - 最大化减少bit翻转
 * 
 * 2. 关联排序（Affiliated Ordering）  
 *    - 保持Query-Key语义关联
 *    - Key按bit数排序，Query跟随
 *    - 平衡优化与语义保持
 * 
 * 数据重组：
 * ---------
 * - 输入：线性数组 [query(64), key(64)]
 * - 重组：8x8矩阵，按列主序排列
 * - 输出：按行组合的flits，每flit 16个元素
 * 
 * @author YZ
 * @date 2025
 */

#ifndef YZLLMIEEE754_HPP_
#define YZLLMIEEE754_HPP_

#include <deque>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "yzIEEE754.hpp"  // 基础IEEE754函数

namespace YzLLMIEEE754 {

/**
 * @brief LLM数据重组与排序优化 - 主入口函数
 * 
 * 将线性payload数据重组为矩阵格式，并应用排序优化。
 * 根据编译时宏定义自动选择排序策略。
 * 
 * @param payload 输入/输出数据容器，包含128个float（64 query + 64 key）
 */
void llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload);

/**
 * @brief LLM分离排序 - Query和Key独立排序
 * 
 * 每个矩阵独立按IEEE754 1-bit数量排序。
 * 适用于不需要保持Query-Key关联的场景。
 * 
 * @param query_data Query数据（64个元素）
 * @param key_data Key数据（64个元素）
 * @param cols 矩阵列数（默认8）
 * @param rows 矩阵行数（默认8）
 */
void sortSeparated(std::deque<float>& query_data, 
                   std::deque<float>& key_data,
                   int cols = 8, int rows = 8);

/**
 * @brief LLM关联排序 - 保持Query-Key配对
 * 
 * Key按bit数排序，Query保持与Key的对应关系。
 * 适用于需要保持attention语义的场景。
 * 
 * @param query_data Query数据（64个元素）
 * @param key_data Key数据（64个元素）
 * @param cols 矩阵列数（默认8）
 * @param rows 矩阵行数（默认8）
 */
void sortAffiliated(std::deque<float>& query_data,
                    std::deque<float>& key_data,
                    int cols = 8, int rows = 8);

/**
 * @brief 单矩阵按列主序排序
 * 
 * 对单个矩阵进行全局排序后按列主序重排。
 * 内部函数，被sortSeparated调用。
 * 
 * @param data 待排序数据
 * @param cols 矩阵列数
 * @param rows 矩阵行数
 */
void sortMatrixByColumns(std::deque<float>& data, int cols, int rows);

/**
 * @brief 打印LLM数据详细信息（调试用）
 * 
 * 显示数据值、IEEE754位表示和1-bit计数。
 * 
 * @param data 数据容器
 * @param name 数据名称
 * @param max_elements 最多显示元素数
 */
void printDetailedData(const std::deque<float>& data, 
                      const std::string& name,
                      int max_elements = 8);

/**
 * @brief 分析数据bit统计信息
 * 
 * 计算并显示bit分布、平均值等统计信息。
 * 
 * @param query_data Query数据
 * @param key_data Key数据
 * @param show_details 是否显示详细信息
 */
void analyzeBitStatistics(const std::deque<float>& query_data,
                          const std::deque<float>& key_data,
                          bool show_details = false);

/**
 * @brief 计算bit翻转成本
 * 
 * 计算按当前顺序传输数据的总bit翻转数。
 * 
 * @param data 数据序列
 * @return 总bit翻转数
 */
int calculateBitFlips(const std::deque<float>& data);

/**
 * @brief 验证排序优化效果
 * 
 * 比较排序前后的bit翻转减少率。
 * 
 * @param original_data 原始数据
 * @param sorted_data 排序后数据
 * @param name 数据集名称
 */
void verifyOptimization(const std::deque<float>& original_data,
                        const std::deque<float>& sorted_data,
                        const std::string& name);

} // namespace YzLLMIEEE754

#endif /* YZLLMIEEE754_HPP_ */