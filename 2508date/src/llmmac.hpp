/*
 * LLMMAC.hpp
 *
 *  Created on: Dec 19, 2024
 *      Author: LLM Version
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
		int request;
		int tmp_requestID;

		int send;
		int NI_id;

		// LLM-specific data structures
		deque<float> query_data;     // Query vectors (acts like CNN inputs)
		deque<float> key_data;       // Key vectors (acts like CNN weights)
		deque<float> value_data;     // Value vectors
		deque<float> input_buffer;   // Input buffer for received data

		// LLM attention parameters
		int tile_x_start, tile_y_start;  // Starting position of this tile
		int tile_Pixels_size;            // Size of tile (4x4)
		int time_slice;                  // Current time slice (0-1)
		int dest_mem_id;                 // Memory node ID

		float attention_output;          // Computed attention output
		
		// Partial sum aggregation for pixels
		std::map<int, std::vector<float>> pixel_partial_sums;    // pixel_id -> [4 partial sums]
		std::map<int, int> pixel_subchunks_received;             // pixel_id -> count of received subchunks
		int current_pixel_id;                          // Current pixel being processed
		int current_subchunk_id;                       // Current subchunk being processed

		deque<int> llmtasktable;

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
		bool llmInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src);
		void llmReceive(Message* re_msg);
		void llmRunOneStep();

		// LLM-specific attention computation
		void llmComputeAttention();
		void llmComputeQueryKeyDot();
		void llmApplySoftmax();
		void llmComputeValueWeightedSum();

		// State management
		bool llmIsWaitingForData();
		void llmResetForNextTask();
		void llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload);  // LLM payload ordering
		void sortMatrix_LLMSeparated(std::deque<float>& data, int colnum_per_row, int rownum_per_col);
		void sortMatrix_LLMAffiliated(std::deque<float>& query_data, std::deque<float>& key_data, int colnum_per_row, int rownum_per_col);
		void llmPrintDetailedData(const std::deque<float>& data, const std::string& name, int max_elements = 8);  // Debug print function

		~LLMMAC();
};

#endif /* LLMMAC_HPP_ */
