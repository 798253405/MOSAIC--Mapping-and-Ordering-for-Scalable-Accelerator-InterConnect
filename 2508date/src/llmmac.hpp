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

#if defined MemNode2_4X4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // 4*4

#elif defined MemNode4_4X4
#define MEM_NODES 4
	// 4×4：TL(1,1),BL(3,1), TR(1,3),  BR(3,3)
	const int dest_list[] = {5, 13, 7, 15}; // 4*4

#elif defined  MemNode4_8X8
	#define MEM_NODES 4
	// 8×8：象限中心 -> (2,2),(6,2),(2,6),(6,6)
	const int dest_list[] = {18, 50, 22, 54}; // 8*8

#elif defined MemNode4_16X16
	#define MEM_NODES 4
	 // 16×16：象限中心 -> (4,4),(12,4),(4,12),(12,12)   // 顺序：TL, BL, TR, BR
	 // 节点ID = xid*16 + yid
	 const int dest_list[] = {68, 196, 76, 204}; // 16*16

#elif defined MemNode4_32X32
	#define MEM_NODES 4
	// 32×32：象限中心 -> (8,8),24,8),(8,24),((24,24)
	const int dest_list[] = {264, 776, 280, 792};
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
		 * - 从llmtasktable队列中取出: current_processing_task_id = llmtasktable.front()
		 * - 任务ID范围: 0 到 1,048,575 (总共262,144像素 × 4个子块)
		 * - 任务ID编码: pixel_id * LLM_SUBCHUNKS_PER_PIXEL + subchunk_id
		 *   例如: 任务ID 1025 = 像素256的第1个子块 (256*4+1)
		 * 
		 * 2. 发送数据请求 (State 1: REQUEST)
		 * ------------------------------------
		 * - 将任务ID作为请求包发送到内存节点
		 * - llmInject(type=0, ..., current_processing_task_id, ...)
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
		 * - 收到type 1响应后，设置current_processing_task_id = -1
		 * - 表示数据已到达，进入计算阶段
		 * - saved_task_id_for_result保存原始ID用于后续
		 * 
		 * 5. 任务ID的编解码
		 * -----------------
		 * - 解码: pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL, subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL
		 * - 每个像素需要4个任务完成才能得到最终结果
		 * - 用于聚合4个子块的部分和
		 */
		int current_processing_task_id;  // 当前正在处理的任务ID，-1表示空闲
		
		/**
		 * @brief 保存的任务ID，用于发送结果（原名tmp_requestID）
		 * 
		 * 作用：
		 * - 在State 1保存任务ID: saved_task_id_for_result = current_processing_task_id
		 * - 在State 3/4使用此ID发送结果和更新输出表
		 * - 保持任务ID贯穿整个处理流程
		 */
		int saved_task_id_for_result;    // 保存的任务ID，用于发送结果时使用

		int send;
		int NI_id;

		// LLM-specific data structures - 只有Input和Query
		deque<float> input_data;     // Input vectors (输入数据)
		deque<float> query_data;     // Query weight vectors (Query权重)
		deque<float> input_buffer;   // Input buffer for received data

		// LLM attention parameters

		int curTimeSliceID;                  // Current time slice (0-1)
		int dest_mem_id;                 // Memory node ID

		float attention_output;          // Computed attention output
		
		// Partial sum aggregation for pixels
		std::map<int, std::vector<float>> pixel_partial_sums;    // pixel_id -> [LLM_SUBCHUNKS_PER_PIXEL partial sums]
		int current_pixel_id;                          // Current pixel being processed
		int current_subchunk_id;                       // Current subchunk being processed

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


		void llmNonMemMACReceiveResp(Message* re_msg);
		void llmRunOneStep();

		// LLM-specific attention computation
		void llmComputeAttention();
		void llmComputeQueryKeyDot();
		void llmComputeValueWeightedSum();

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
