/**
 * @file llmmac.hpp
 * @brief LLM MAC - Transformer Attention
 *
 * LLMMAC，TransformerAttention。
 * CNN MAC，LLM MAC。
 *
 * ：
 * -----------
 * State 0 (IDLE):    ，
 * State 1 (REQUEST): ，type 0
 * State 2 (WAIT):    ，type 1
 * State 3 (COMPUTE): ，attention
 *
 * ：
 * ------------
 * Type 0:  - MACQuery/Key
 * Type 1:  -
 * Type 2:  - LLM
 * Type 3:  - Attention
 *
 * ：
 * ---------
 * - ：attention
 * - ：
 * - ：bit
 * - Attention：Q*K^T/sqrt(d_k)softmax
 * - ：
 *
 * ：
 * ---------
 * - ：QueryKey，bit
 * - ：Query-Key，
 *
 * @see llmmacnet.hpp - LLM
 * @see authorIEEE754.hpp - IEEE754
 *
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
#include <algorithm>  //  std::min
#include <map>        //  std::map
#include "parameters.hpp"
#include "NoC/Packet.hpp"
#include "NoC/NI.hpp"
#include "llmieee754.hpp"  // LLMIEEE754
// : llmmacnet.hpp  .cpp ，

#if defined NOCSIZEMC2_4X4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // (2,1) and (2,3) in 4x4 grid

#elif defined NOCSIZEMC8_8X8
	#define MEM_NODES 8
	// 2x2 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		17, 19,   // Tile(0,0): (2,1), (2,3)
		21, 23,   // Tile(0,1): (2,5), (2,7)
		49, 51,   // Tile(1,0): (6,1), (6,3)
		53, 55    // Tile(1,1): (6,5), (6,7)
	};

#elif defined NOCSIZEMC32_16X16
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

#elif defined NOCSIZEMC128_32X32
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

class LLMMACnet;
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
		 * State | Name ()    | Duration ()                  | Description ()
		 * ------|----------------|----------------------------------|------------------------------------------------------------------------------------------------
		 *   0   | IDLE ()    | 1 cycle                          | Transitional state. If tasks are available, moves to REQUEST.
		 *       |                |                                  | (。，REQUEST。)
		 *   1   | REQUEST () | 1 cycle                          | Transitional state. Sends a Type 0 data request to memory.
		 *       |                |                                  | (。Type 0。)
		 *   2   | WAITING () | Variable (Network-dependent)     | Waits for the Type 1 response packet from memory. Duration depends on NoC latency.
		 *       |                |                                  | (,。Type 1。NoC。)
		 *   3   | COMPUTE () | 1 cycle + computation delay      | Transitional state. Calculates partial sum, then moves to state 4 while setting
		 *       |                | (1 + )               | a `pecycle` timer that stalls the MAC (~40 cycles).
		 *       |                |                                  | (。，4，pecycleMAC。)
		 *   4   | COMPLETE ()| 1 cycle                          | Transitional state after computation delay. Decides whether to move to IDLE (0) or FINISHED (5).
		 *       |                |                                  | (。IDLE(0)FINISHED(5)。)
		 *   5   | FINISHED ()| Permanent ()                 | Terminal state. The MAC has completed all tasks and remains inactive.
		 *       |                |                                  | (。MAC。)
		 *
		 */
		int selfstatus;

		/**
		 * @brief ID（request）
		 *
		 * ：
		 * ================
		 *
		 * 1. ID (State 1: REQUEST)
		 * ------------------------------------
		 * - llmtasktable: currentRequestedTaskIDd = llmtasktable.front()
		 * - ID: 0  1,048,575 (262,144 × 4)
		 * - ID: pixel_id * LLM_SUBCHUNKS_PER_PIXEL + subchunk_id
		 *   : ID 1025 = 2561 (256*4+1)
		 *
		 * 2.  (State 1: REQUEST)
		 * ------------------------------------
		 * - ID
		 * - llmInject(type=0, ..., currentRequestedTaskIDd, ...)
		 * - ，ID
		 *
		 * 3.  (Memory Node)
		 * -------------------------------
		 * - type 0，ID
		 * - all_tasks: task = all_tasks[task_id]
		 * - QueryKey(64float)
		 * - ()
		 * - type 1，
		 *
		 * 4.  (State 2: WAIT)
		 * ---------------------------------
		 * - type 1，currentRequestedTaskIDd = -1
		 * - ，
		 *
		 * 5. ID
		 * -----------------
		 * - : pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL, subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL
		 * - 4
		 * - 4
		 */
		int currentRequestedTaskIDd;  // ID，-1
		int inPETaskIDFromResp;  // ID，-1
		/**
		 * @brief ID，（tmp_requestID）
		 *
		 * ：
		 * - State 3/4ID
		 * - ID
		 */
		int send;
		int NI_id;

		// LLM-specific data structures - InputQuery
		deque<float> input_data;     // Input vectors ()
		deque<float> query_data;     // Query weight vectors (Query)
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

#ifdef binaryroutingSwitch
		int lastResponseRouting;  // response packetrouting (1 or 2)
#endif

#ifdef fireAdvance
		// Fire advance：request
		int total_tasks;              // （）
		int requests_sent;            // request
		int responses_received;       // response
		int tasks_completed;

		int fire_advance_counter;     // ：responserequest
		bool fire_advance_armed;      // fire advance

		int computing_task_id;        // task ID（-1）
#endif

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
		// : llmReshapeFlatToQueryKeyMatrix  llmieee754.hpp/cpp

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
