/**
 * @file llmmac.hpp
 * @brief LLM MAC计算单元头文件 - Transformer Attention处理器
 * 
 * 定义了LLM模式下的MAC计算单元类，实现Transformer架构的Attention计算。
 * 与CNN MAC不同，LLM MAC采用状态机驱动的异步处理模式。
 * 
 * 状态机定义：
 * -----------
 * State 0 (IDLE):    空闲状态，检查任务队列
 * State 1 (REQUEST): 请求状态，发送type 0消息获取数据
 * State 2 (WAIT):    等待状态，等待type 1响应数据
 * State 3 (COMPUTE): 计算状态，执行attention计算
 * 
 * 消息类型定义：
 * ------------
 * Type 0: 数据请求 - MAC向内存节点请求Query/Key数据
 * Type 1: 数据响应 - 内存节点返回排序后的数据
 * Type 2: 中间结果 - LLM模式中未使用
 * Type 3: 最终结果 - Attention计算的最终输出
 * 
 * 主要功能：
 * ---------
 * - 任务队列管理：维护待处理的attention任务
 * - 数据请求发送：向内存节点请求所需数据
 * - 排序优化处理：对数据应用bit翻转优化排序
 * - Attention计算：Q*K^T/sqrt(d_k)及softmax
 * - 结果输出管理：将计算结果发送到目标节点
 * 
 * 排序策略：
 * ---------
 * - 分离排序：Query和Key独立排序，最大化减少bit翻转
 * - 关联排序：保持Query-Key配对，维护语义关联
 * 
 * @see llmmacnet.hpp - LLM网络管理器
 * @see yzIEEE754.hpp - IEEE754位操作函数
 * 
 * @author LLM Version, YZ (comments)
 * @date 2024-12-19 (original), 2025 (updated)
 */

#ifndef LLMMAC_HPP_
#define LLMMAC_HPP_

/*
 * Message Type (`msgtype`) Usage Summary:
 *
 * This table explains how different message types are used across the CNN and LLM simulation modes.
 *
 * Type | In CNN Mode                   | In LLM Mode
 * -----|-------------------------------|-------------------------------------------------
 *  0   | Request for data from MAC     | Request for data from MAC
 *  1   | Response with data from Memory| Response with data from Memory
 *  2   | FINAL RESULT from MAC         | Intermediate result (Defined but UNUSED)
 *  3   | Unused                        | FINAL AGGREGATED RESULT from MAC
 *
 */

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <deque>
#include <cmath>
#include <cassert>
#include <algorithm>  // 添加 std::min
#include <map>        // 添加 std::map
#include "parameters.hpp"
#include "NoC/Packet.hpp"
#include "NoC/NI.hpp"
#include "yzllmieee754.hpp"  // LLM专用IEEE754排序优化
// 注意: llmmacnet.hpp 会在 .cpp 文件中包含，避免循环依赖

#if defined DATEMC2_4X4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // (2,1) and (2,3) in 4x4 grid

#elif defined DATEMC8_8X8
	#define MEM_NODES 8
	// 2x2 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		17, 19,   // Tile(0,0): (2,1), (2,3)
		21, 23,   // Tile(0,1): (2,5), (2,7)
		49, 51,   // Tile(1,0): (6,1), (6,3)
		53, 55    // Tile(1,1): (6,5), (6,7)
	};

#elif defined DATEMC32_16X16
	#define MEM_NODES 32
	// 4x4 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		// Row 0 tiles (y=2)
		33, 35,   37, 39,   41, 43,   45, 47,
		// Row 1 tiles (y=6)
		97, 99,   101, 103, 105, 107, 109, 111,
		// Row 2 tiles (y=10)
		161, 163, 165, 167, 169, 171, 173, 175,
		// Row 3 tiles (y=14)
		225, 227, 229, 231, 233, 235, 237, 239
	};

#elif defined DATEMC128_32X32
	#define MEM_NODES 128
	// 8x8 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		// Tile row 0
		65,  67,  69,  71,  73,  75,  77,  79,
		81,  83,  85,  87,  89,  91,  93,  95,
		// Tile row 1
		193, 195, 197, 199, 201, 203, 205, 207,
		209, 211, 213, 215, 217, 219, 221, 223,
		// Tile row 2
		321, 323, 325, 327, 329, 331, 333, 335,
		337, 339, 341, 343, 345, 347, 349, 351,
		// Tile row 3
		449, 451, 453, 455, 457, 459, 461, 463,
		465, 467, 469, 471, 473, 475, 477, 479,
		// Tile row 4
		577, 579, 581, 583, 585, 587, 589, 591,
		593, 595, 597, 599, 601, 603, 605, 607,
		// Tile row 5
		705, 707, 709, 711, 713, 715, 717, 719,
		721, 723, 725, 727, 729, 731, 733, 735,
		// Tile row 6
		833, 835, 837, 839, 841, 843, 845, 847,
		849, 851, 853, 855, 857, 859, 861, 863,
		// Tile row 7
		961, 963, 965, 967, 969, 971, 973, 975,
		977, 979, 981, 983, 985, 987, 989, 991
	};
#endif

using namespace std;

extern long long  packet_id;
extern unsigned int cycles;
extern vector<vector<int>> DNN_latency;
extern double samplingWindowDelay[TOT_NUM];

class LLMMACnet;  // 前向声明
class Packet;

class LLMMAC
{
	public:
		/** @brief LLMMAC - LLM-specific MAC unit
		 */
		LLMMAC(int t_id, LLMMACnet* t_net, int t_NI_id);

		LLMMACnet* net;
		int selfMACid;
		int fn;
		int pecycle;

		/*
		 * LLMMAC State Machine Explanation:
		 * The `selfstatus` variable controls the state of a single MAC unit.
		 *
		 * State | Name (名称)    | Duration (周期)                  | Description (描述)
		 * ------|----------------|----------------------------------|------------------------------------------------------------------------------------------------
		 *   0   | IDLE (空闲)    | 1 cycle                          | Transitional state. If tasks are available, moves to REQUEST.
		 *       |                |                                  | (过渡状态。若任务可用，则切换到REQUEST状态。)
		 *   1   | REQUEST (请求) | 1 cycle                          | Transitional state. Sends a Type 0 data request to memory.
		 *       |                |                                  | (过渡状态。向内存发送Type 0数据请求。)
		 *   2   | WAITING (等待) | Variable (Network-dependent)     | Waits for the Type 1 response packet from memory. Duration depends on NoC latency.
		 *       |                |                                  | (可变长状态,网络依赖。等待内存返回Type 1响应包。时长取决于NoC延迟。)
		 *   3   | COMPUTE (计算) | 1 cycle + computation delay      | Transitional state. Calculates partial sum, then moves to state 4 while setting
		 *       |                | (1周期 + 计算延迟)               | a `pecycle` timer that stalls the MAC (~40 cycles).
		 *       |                |                                  | (过渡状态。计算部分和，然后切换到状态4，同时设置一个pecycle定时器来暂停MAC。)
		 *   4   | COMPLETE (完成)| 1 cycle                          | Transitional state after computation delay. Decides whether to move to IDLE (0) or FINISHED (5).
		 *       |                |                                  | (计算延迟后的过渡状态。决策是切换到IDLE(0)还是FINISHED(5)。)
		 *   5   | FINISHED (结束)| Permanent (永久)                 | Terminal state. The MAC has completed all tasks and remains inactive.
		 *       |                |                                  | (终点状态。MAC已完成所有任务并保持非活动状态。)
		 *
		 */
		int selfstatus;
		
		/**
		 * @brief 当前正在处理的任务ID（原名request）
		 * 
		 * 数据流程详解：
		 * ================
		 * 
		 * 1. 任务ID的来源 (State 1: REQUEST)
		 * ------------------------------------
		 * - 从llmtasktable队列中取出: currentRequestedTaskIDd = llmtasktable.front()
		 * - 任务ID范围: 0 到 1,048,575 (总共262,144像素 × 4个子块)
		 * - 任务ID编码: pixel_id * LLM_SUBCHUNKS_PER_PIXEL + subchunk_id
		 *   例如: 任务ID 1025 = 像素256的第1个子块 (256*4+1)
		 * 
		 * 2. 发送数据请求 (State 1: REQUEST)
		 * ------------------------------------
		 * - 将任务ID作为请求包发送到内存节点
		 * - llmInject(type=0, ..., currentRequestedTaskIDd, ...)
		 * - 内存节点收到后，根据ID查找对应数据
		 * 
		 * 3. 内存节点处理 (Memory Node)
		 * -------------------------------
		 * - 接收type 0请求，提取任务ID
		 * - 从all_tasks数组中查找: task = all_tasks[task_id]
		 * - 提取该任务的Query和Key数据(各64个float)
		 * - 应用排序优化(如果启用)
		 * - 创建type 1响应包，包含排序后的数据
		 * 
		 * 4. 接收响应数据 (State 2: WAIT)
		 * ---------------------------------
		 * - 收到type 1响应后，设置currentRequestedTaskIDd = -1
		 * - 表示数据已到达，进入计算阶段
		 * 
		 * 5. 任务ID的编解码
		 * -----------------
		 * - 解码: pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL, subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL
		 * - 每个像素需要4个任务完成才能得到最终结果
		 * - 用于聚合4个子块的部分和
		 */
		int currentRequestedTaskIDd;  // 当前正在处理的任务ID，-1表示空闲
		int inPETaskIDFromResp;  // 当前正在处理的任务ID，-1表示空闲
		/**
		 * @brief 保存的任务ID，用于发送结果（原名tmp_requestID）
		 * 
		 * 作用：
		 * - 在State 3/4使用此ID发送结果和更新输出表
		 * - 保持任务ID贯穿整个处理流程
		 */
		int send;
		int NI_id;

		// LLM-specific data structures - 只有Input和Query
		deque<float> input_data;     // Input vectors (输入数据)
		deque<float> query_data;     // Query weight vectors (Query权重)
		deque<float> input_buffer;   // Input buffer for received data

		// LLM attention parameters

		int current_subchunk_id;                  // Current time slice  == Current subchunk being processed
		int dest_mem_id;                 // Memory node ID
		
		// Partial sum aggregation for pixels
		std::map<int, std::vector<float>> pixel_partial_sums;    // pixel_id -> [LLM_SUBCHUNKS_PER_PIXEL partial sums]
		int current_pixel_id;                          // Current pixel being processed

		// Monitoring structures for SAMOS vs actual latency tracking
		struct LatencyMonitoring {
			// Per-MAC statistics
			int task_count;                    // Number of tasks processed
			double sampled_latency_avg;        // Average latency from SAMOS sampling
			double actual_latency_sum;         // Sum of actual latencies
			double actual_latency_min;         // Minimum actual latency
			double actual_latency_max;         // Maximum actual latency

			// Sampling phase data (from SAMOS)
			double samos_expected_latency;     // Expected latency from SAMOS sampling

			// For periodic reporting
			int last_report_task_count;        // Task count at last report

			LatencyMonitoring() :
				task_count(0),
				sampled_latency_avg(0.0),
				actual_latency_sum(0.0),
				actual_latency_min(1e9),
				actual_latency_max(0.0),
				samos_expected_latency(0.0),
				last_report_task_count(0) {}
		};

		LatencyMonitoring latency_monitor;

		deque<int> llmPEExpectedtasktable;

		LLMMAC* nextLLMMAC;
		
		// Timing tracking for task phases with packet travel details
		struct TaskTiming {
			int task_id;
			
			// Request packet timing
			int request_send_cycle;      // When request was sent
			int request_arrive_cycle;    // When request arrived at memory
			int request_hops;            // Number of hops for request
			
			// Response packet timing  
			int response_send_cycle;     // When memory sent response
			int response_arrive_cycle;   // When response arrived at MAC
			int response_hops;           // Number of hops for response
			
			// Computation timing
			int compute_start_cycle;
			int compute_end_cycle;
			
			// Result packet timing
			int result_send_cycle;       // When result was sent
			int result_arrive_cycle;     // When result arrived at memory (if tracked)
			int result_hops;            // Number of hops for result
			
			TaskTiming() : task_id(-1), 
			               request_send_cycle(0), request_arrive_cycle(0), request_hops(0),
			               response_send_cycle(0), response_arrive_cycle(0), response_hops(0),
			               compute_start_cycle(0), compute_end_cycle(0),
			               result_send_cycle(0), result_arrive_cycle(0), result_hops(0) {}
		};
		
		std::vector<TaskTiming> task_timings;
		TaskTiming current_task_timing;

		// Core functions
		bool llmPEInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src, int task_id);
		bool llmMemNodeInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src, int task_id);


		void llmPEReceiveResp(Message* re_msg);
		void llmRunOneStep();


		// State management
		bool llmIsWaitingForData();
		void llmResetForNextTask();
		// 注意: llmReshapeFlatToQueryKeyMatrix 已移至 yzllmieee754.hpp/cpp

		~LLMMAC();
};



// Hierarchical debug macros based on LLM_DEBUG_LEVEL from parameters.hpp
// With cycle info and system time (for runtime use)
#define LLM_INFO(x) do { \
    if (LLM_DEBUG_LEVEL >= 1) { \
        std::cout << "[" << getCurrentTimeStr() << "] [INFO @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_DEBUG(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[DEBUG @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[TRACE @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

// Without cycle info (for initialization)
#define LLM_INFO_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 1) { \
        std::cout << "[" << getCurrentTimeStr() << "] [INFO @init] " << x << std::endl; \
    } \
} while(0)

#define LLM_DEBUG_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[DEBUG @init] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[TRACE @init] " << x << std::endl; \
    } \
} while(0)

// Helper function
template<class C, typename T>
bool contains(C &&c, T e) {
	return find(begin(c), end(c), e) != end(c);
}

#endif /* LLMMAC_HPP_ */
