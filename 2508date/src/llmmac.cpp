/**
 * @file llmmac.cpp
 * @brief LLM MAC - Attention
 *
 * LLMMAC，Transformer
 * Attention。，。
 *
 * ========================================================================
 * llmmacnet.cpp7
 * ========================================================================
 *
 * Step 0: （）
 * ----------------------------------------
 * ：LLMMAC::LLMMAC() [54-83]
 * - MAC IDNIID
 * - dest_mem_id（）
 * -
 * - selfstatus0（IDLE）
 *
 * Step 1: 【】
 * ----------------------------------------
 * ：LLMMAC::llmRunOneStep() [319-479]
 *
 * ：
 *   State 0 (IDLE):    ，
 *   State 1 (REQUEST):
 *   State 2 (WAIT):
 *   State 3 (COMPUTE):
 *
 * ：
 *   IDLE → REQUEST:    llmPEExpectedtasktable [336-342]
 *   REQUEST → WAIT:     [373]
 *   WAIT → COMPUTE:    processCNNPacket() [520]
 *   COMPUTE → IDLE:     [476]
 *
 * Step 2: （1）
 * ----------------------------------------
 * ：LLMMAC::llmInject() [212-260]
 * ：selfstatus == 1
 * ：
 *   - llmtasktableID [349]
 *   - type 0 [228]
 *   -  [229-232]
 *   - NoC [238]
 *
 * Step 3: （）
 * ----------------------------------------
 * ：LLMMAC::processCNNPacket() [262-301]
 * 3.1 payload：
 *   - 132float [278]
 *   - [0-3]：magic、size、chunk_id、pixel_id
 *   - Query[4-67]：64
 *   - Key[68-131]：64
 *
 * 3.2 ：
 *   ：authorLLMIEEE754::llmReshapeFlatToQueryKeyMatrix() [llmieee754.cpp]
 *   - ：AUTHORSeperatedOrdering_reArrangeInput
 *   - ：AuthorAffiliatedOrdering
 *
 * Step 4: （2）
 * ----------------------------------------
 * ：LLMMAC::processCNNPacket() [487-545]
 * ：type 1
 * ：
 *   -  [489]
 *   - payload [507,514]
 *   - infeature/weight [507,514]
 *   - selfstatus = 3 [520]
 *
 * Step 5: （3）
 * ----------------------------------------
 * ：LLMMAC::compute() [547-608]
 * ：selfstatus == 3
 * ：
 *   - MAC：outfeature += weight[k] * infeature[i] [568]
 *   - ：tanh(outfeature) [573]
 *   -  [585]
 *   - ：selfstatus = 4 [577]
 *
 * Step 6:
 * ----------------------------------------
 * ：LLMMAC::sendOutput() [610-640]
 * ：
 *   - type 2 [615]
 *   -  [622]
 *   - NoC [630]
 *   - IDLE：selfstatus = 0 [635]
 *
 * ========================================================================
 * ========================================================================
 *
 * （Separated Ordering）：
 * - QueryKey，1-bit
 * - ：bit
 * - ：Query-Key
 *
 * （Affiliated Ordering）：
 * - Key，Query
 * - ：attention
 * - ：bit
 *
 * ========================================================================
 * ========================================================================
 *
 * - llmtasktable: ，ID
 * - infeature: （Query）
 * - weight: （Key）
 * - outfeature:
 * - selfstatus: （0-3）
 * - pecycle: PE
 *
 * @date 2025
 */

//  llmmac.cpp - 4x4
#include "llmmac.hpp"
#include "llmmacnet.hpp"
#include "mc_mapping.hpp"
#include "IEEE754.hpp"  // For bit-count sorting functions
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cassert>

// Hierarchical debug macros based on LLM_DEBUG_LEVEL from parameters.hpp
#include "parameters.hpp"

// Helper function to get current time string
static inline std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm* timeinfo = localtime(&time_t);
    char buffer[10];
    strftime(buffer, sizeof(buffer), "%H:%M", timeinfo);
    return std::string(buffer);
}

// With cycle info and system time (for runtime use)
#define LLM_INFO(x) do { \
    if (LLM_DEBUG_LEVEL >= 1) { \
        std::cout << "[" << getCurrentTimeStr() << "] [MAC-INFO @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_DEBUG(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[MAC-DEBUG @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[MAC-TRACE @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

// Without cycle info (for initialization)
#define LLM_DEBUG_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[MAC-DEBUG @init] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[MAC-TRACE @init] " << x << std::endl; \
    } \
} while(0)

LLMMAC::LLMMAC(int t_id, LLMMACnet *t_net, int t_NI_id) {
	selfMACid = t_id;
	net = t_net;
	NI_id = t_NI_id;

	input_data.clear();
	query_data.clear();
	// Key
	input_buffer.clear();

	fn = -1;
	currentRequestedTaskIDd = -1;
	nextLLMMAC = NULL;
	pecycle = 0;
	selfstatus = 0;
	send = 0;
	current_pixel_id = -1;
	current_subchunk_id = -1;
	pixel_partial_sums.clear();

	current_subchunk_id = 0;

	// Find destination memory ID
	dest_mem_id = get_mc_for_pe(NI_id, X_NUM, Y_NUM);

	// Debug output for first few MACs
	if (selfMACid < 16) {
		int xid = NI_id / X_NUM;
		int yid = NI_id % X_NUM;
		std::cout << "204LLMMAC[" << selfMACid << "] at (" << xid << "," << yid
		          << ") -> MC " << dest_mem_id << std::endl;
	}

	llmPEExpectedtasktable.clear();

	// Initialize latency monitoring
	latency_monitor = LatencyMonitoring();

#ifdef binaryroutingSwitch
	lastResponseRouting = 1;  // 1，response packetrouting 1
#endif

#ifdef fireAdvance
	total_tasks = 0;
	requests_sent = 0;
	responses_received = 0;
	tasks_completed = 0;
	computing_task_id = -1;
	fire_advance_counter = 0;
	fire_advance_armed = false;
#endif
}

bool LLMMAC::llmMemNodeInject(int type, int d_id, int  tllm_eleNum, float t_output, NI* t_NI, int p_id, int mac_src,int task_id) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length =  tllm_eleNum -4 ;// 132 or 128
	msg.QoS = 0;

	msg.data.clear();
	msg.destination = d_id;
	// Type 3 (final result) should be processed immediately, not delayed
	msg.out_cycle = (type == 3) ? cycles : pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id;
	msg.source_id = NI_id;
	msg.msgtype = type;
	msg.authorMSGPayload.clear();
	current_pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL;
	current_subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL;

  if (msg.msgtype == 1) { // Response with data - all_tasks
		msg.authorMSGPayload.clear();
		// netall_tasks
		if (net && task_id >= 0 && task_id < static_cast<int>(net->all_tasks.size())) {
			const LLMMACnet::LLMTask& task = net->all_tasks[task_id];
			// Input
			msg.authorMSGPayload.insert(msg.authorMSGPayload.end(),
									task.input_data.begin(),
									task.input_data.end());
			// Query
			msg.authorMSGPayload.insert(msg.authorMSGPayload.end(),
									task.query_data.begin(),
									task.query_data.end());
		} else {

			assert(false && "ERROR: Invalid task_id - task not found in all_tasks!");
		}

		// flitpaddingflit
		int flitNumSinglePacket = (msg.authorMSGPayload.size() - 1 + payloadElementNum) / payloadElementNum;
		//cout <<"int flitNumSinglePacket "<<  flitNumSinglePacket<<" msg.authorMSGPayload.size( "<<msg.authorMSGPayload.size() <<" msg.msgdata_length " <<  msg.msgdata_length <<endl;
		// paddingflit（CNN）
		std::fill_n(std::back_inserter(msg.authorMSGPayload),
					(flitNumSinglePacket * payloadElementNum - msg.authorMSGPayload.size()),
					0.0f);

		// （）
		static int inject_count = 0;
		authorLLMIEEE754::llmReshapeFlatToQueryKeyMatrix(msg.authorMSGPayload);
	}

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;

#ifdef binaryroutingSwitch
	// Response packets alternate between routing 1 and 2
	if (packet->message.msgtype == 1) {  // msgtype 1 = response packets
		// Use the next routing mode (toggle between 1 and 2)
		packet->xyroutingBool = (lastResponseRouting == 1) ? 2 : 1;
		// Update for next response packet
		lastResponseRouting = packet->xyroutingBool;
	}
	// Request packets keep default (0)
#endif

	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);

	return true;
}

bool LLMMAC::llmPEInject(int type, int d_id, int  tllm_eleNum, float t_output, NI* t_NI, int p_id, int mac_src,int task_id) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length =  tllm_eleNum -4 ;// 132 or 128
	msg.QoS = 0;
	//  task_id  pixel_id  subchunk_id
	current_pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL;
	current_subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL;
	if ( type == 3) { //type == 2 result。
		// （type 2  type 3），
		// current_pixel_id
		//  8×128 (8×128)
		// pixel_id: 0-1023 (8*128=1024)

		int pixel_x = current_pixel_id % net->matrixOutputPixels_queryoutputdim;  //  (0-127)
		int pixel_y = current_pixel_id / net->matrixOutputPixels_queryoutputdim;  //  (0-7)
		msg.data.assign(1, t_output);
		msg.data.push_back( pixel_x);
		msg.data.push_back( pixel_y);
		// current_subchunk_id  State 1  task_id
		msg.data.push_back( current_subchunk_id);  // Use subchunk_id instead of ts

	} else if (type == 0) {
		// Type 0: Request message - msg.data[0] must contain task_id for Memory node to retrieve the correct task
		msg.data.assign(1, task_id);  // msg.data[0] = task_id (0！)
		int pixel_x = current_pixel_id % net->matrixOutputPixels_queryoutputdim;  //  (0-127)
		int pixel_y = current_pixel_id / net->matrixOutputPixels_queryoutputdim;  //  (0-7)
		msg.data.push_back(pixel_x);
		msg.data.push_back(pixel_y);
		// current_subchunk_id  State 1  task_id
		msg.data.push_back(current_subchunk_id);
	} else {
		assert(false && "ERROR:PE LLM2，1Mem");
	}


	msg.destination = d_id;
	// Type 3 (final result) should be processed immediately, not delayed
	msg.out_cycle = (type == 3) ? cycles : pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id;
	msg.source_id = NI_id;
	msg.msgtype = type;
	msg.authorMSGPayload.clear();

#ifdef LLM_OPTIMIZED_TYPE03_HANDLING
	// ：Type 0/3 16
	if (msg.msgtype == 0) { // Request
		// Request message: 16，0task_id（，）
		msg.authorMSGPayload.assign(payloadElementNum, 0);
		// Type 0/3
		authorLLMIEEE754::llmReqRestReorderingFunc(msg.authorMSGPayload, 0.0f);
	} else if (msg.msgtype == 3) { // Result (type 2 intermediate, type 3 final)
		// Result message: 16，0
		msg.authorMSGPayload.assign(payloadElementNum, 0);
		// Type 0/3，
		authorLLMIEEE754::llmReqRestReorderingFunc(msg.authorMSGPayload, t_output);
		// std::cout << "[LLM-INJECT-TYPE3] Injecting Type 3 to NI " << NI_id
		//           << " dest=" << d_id << " value=" << t_output << std::endl;
	}
	else {
		assert(false && "ERROR:PE LLM2，1Mem");
	}
	// Type 0Type 3（16），padding

#else
	// ：128
	if (msg.msgtype == 0) { // Request
		// Request message padding
		msg.authorMSGPayload.assign(payloadElementNum, 0);
	} else if (msg.msgtype == 3) { // Result (type 2 intermediate, type 3 final)
		msg.authorMSGPayload.assign(payloadElementNum, 0);
		msg.authorMSGPayload[0] = t_output;
		// std::cout << "[LLM-INJECT-TYPE3] Injecting Type 3 to NI " << NI_id
		//           << " dest=" << d_id << " value=" << t_output << std::endl;
	}
	else {
		assert(false && "ERROR:PE LLM2，1Mem");
	}

	// flitpaddingflit
	int flitNumSinglePacket = (msg.authorMSGPayload.size() - 1 + payloadElementNum) / payloadElementNum;
	// paddingflit（CNN）
	std::fill_n(std::back_inserter(msg.authorMSGPayload),
				(flitNumSinglePacket * payloadElementNum - msg.authorMSGPayload.size()),
				0.0f);
	// （）
	// ，
	// Type 3 messages only have 1 element, but sorting won't hurt
	authorLLMIEEE754::llmReshapeFlatToQueryKeyMatrix(msg.authorMSGPayload);
#endif

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;

#ifdef binaryroutingSwitch
	// req packets alternate between routing 1 and 2
	if (packet->message.msgtype == 0) {  // 0=request msgtype 1 = response packets
		// Use the next routing mode (toggle between 1 and 2)
		packet->xyroutingBool = (lastResponseRouting == 1) ? 2 : 1;
		// Update for next response packet
		lastResponseRouting = packet->xyroutingBool;
	}
	// Request packets keep default (0)
#endif

	// if (msg.msgtype == 3) {
	// 	std::cout << "[LLM-ENQUEUE-TYPE3] Enqueuing Type 3 to vnet=" << packet->vnet
	// 	          << " (buffer[0]) at NI " << NI_id << " -> dest " << msg.destination
	// 	          << " send_out_time=" << packet->send_out_time
	// 	          << " out_cycle=" << msg.out_cycle << std::endl;
	// }

	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);
	return true;
}

void LLMMAC::llmRunOneStep() {
	static int total_run_count = 0;
	total_run_count++;
	if ((int)pecycle < (int)cycles) {
		// State 0: IDLE
		if (selfstatus == 0) {
			if (llmPEExpectedtasktable.size() == 0) { //。pe。
				selfstatus = 0;
				pecycle = cycles;
			} else {
				pecycle = cycles;
				selfstatus = 1;
			}
		}
		// State 1: REQUEST
		// - Purpose: Send a request (Type 0) to the memory controller for the current task's data.
		// - Duration: 1 cycle. This state is transitional.
		else if (selfstatus == 1) {
			currentRequestedTaskIDd = llmPEExpectedtasktable.front();  // ID
			llmPEExpectedtasktable.pop_front();
			//if (selfMACid == 0 && currentRequestedTaskIDd < 100) {
			//	cout << "Line471: MAC 0 processing task_id=" << currentRequestedTaskIDd
			//	     << " at cycle=" << cycles << endl;
			//}

			//  task_id  pixel_id  subchunk_id
			current_pixel_id = currentRequestedTaskIDd / LLM_SUBCHUNKS_PER_PIXEL;
			current_subchunk_id = currentRequestedTaskIDd % LLM_SUBCHUNKS_PER_PIXEL;

			// Start timing for new task
			current_task_timing = TaskTiming();
			current_task_timing.task_id = currentRequestedTaskIDd;
			current_task_timing.request_send_cycle = cycles;
			int signal_id_to_send = packet_id + currentRequestedTaskIDd;
			//if (currentRequestedTaskIDd == 34880 || currentRequestedTaskIDd == 43840) {
			//	cout << "Line481: MAC " << selfMACid << " sending request: currentRequestedTaskIDd=" << currentRequestedTaskIDd
			//	     << " packet_id=" << packet_id << " signal_id=" << signal_id_to_send << endl;
			//}
			llmPEInject(0, dest_mem_id, 1, 0/*output is 0*/, net->vcNetwork->NI_list[NI_id],
					  signal_id_to_send, selfMACid, currentRequestedTaskIDd);
			//bool LLMMAC::llmPEInject(int type, int d_id, int  tllm_eleNum, float t_output, NI* t_NI, int p_id, int mac_src,int task_id)
			// Calculate expected hops for request (Manhattan distance in mesh)
			int src_x = NI_id % X_NUM;
			int src_y = NI_id / X_NUM;
			int dst_x = dest_mem_id % X_NUM;
			int dst_y = dest_mem_id / X_NUM;
			current_task_timing.request_hops = abs(dst_x - src_x) + abs(dst_y - src_y);

#ifdef fireAdvance
			requests_sent++;
#endif

			selfstatus = 2;
			pecycle = cycles;
		}
		// State 2: WAITING
		// - Purpose: Wait for the memory controller to send back the requested data (Type 1).
		// - Duration: Variable. Depends on the network travel time for the request packet to reach memory
		//             and the response packet to return.
		else if (selfstatus == 2) {
#ifndef fireAdvance
			if (currentRequestedTaskIDd >= 0) {
				pecycle = cycles;
				selfstatus = 2;
				//std::cout << "llmmac439 selfstatus [DATA-CHECK] MAC "  << selfMACid  << " taskwearedoing now " << currentRequestedTaskIDd<<endl;
				return;
			}
#else
			// Fire advance：response
			if (computing_task_id < 0) {
				// response，
				pecycle = cycles;
				selfstatus = 2;
				return;
			}
#endif
			// Track response arrival time (this is when we process it)
			current_task_timing.response_arrive_cycle = cycles;
			selfstatus = 3;
			pecycle = cycles;
			return;
		}
		// State 3: COMPUTE
		else if (selfstatus == 3) { // currentRequestedTaskIDd ，state3。
			// Track computation start
			current_task_timing.compute_start_cycle = cycles;

			// Partial sum  llmPEReceiveResp

			int calc_time = ((query_data.size()-1) / PE_NUM_OP + 1) * 20;
			selfstatus = 4;
			pecycle = cycles + calc_time;

			// Track computation end and result send
			current_task_timing.compute_end_cycle = cycles + calc_time;
			current_task_timing.result_send_cycle = cycles + calc_time;



			// Check if all subchunks for this pixel are complete
			// 1：，(63)
			if (current_subchunk_id == LLM_SUBCHUNKS_PER_PIXEL-1) {
				// Aggregate all partial sums
				float total_sum = 0.0f;
				int valid_count = 0;

				// Debug: Print all partial sums for pixel[0][0]
				// if (current_pixel_id == 0) {
				// std::cout << "[DEBUG-PIXEL0-AGGREGATION] Aggregating pixel[0][0]:" << std::endl;
				// }

				for (int i = 0; i < LLM_SUBCHUNKS_PER_PIXEL; i++) {
					if (pixel_partial_sums[current_pixel_id].size() > i) {
						float partial = pixel_partial_sums[current_pixel_id][i];
						total_sum += partial;
						valid_count++;
					}
				}


				// Calculate pixel coordinates for debug output
				int pixel_x = current_pixel_id % net->query_output_dim;
				int pixel_y = current_pixel_id / net->query_output_dim;

				// Send Type 3 final aggregated result
				// std::cout << "[LLM-AGGREGATION] MAC " << selfMACid
				//           << " sending Type 3 for pixel " << current_pixel_id
				//           << " (x=" << pixel_x << ",y=" << pixel_y
				//           << ") with sum=" << total_sum
				//           << " from NI " << NI_id << " to dest " << dest_mem_id << std::endl;

				llmPEInject(3, dest_mem_id, 1, total_sum,
						  net->vcNetwork->NI_list[NI_id], packet_id + inPETaskIDFromResp, selfMACid, inPETaskIDFromResp);

				// Clean up aggregation data for this pixel
				pixel_partial_sums.erase(current_pixel_id);
			}  // End of "if all subchunks complete"

			// Calculate result packet hops (same as request)
			current_task_timing.result_hops = current_task_timing.request_hops;

			// Fire advancestate 4，
			// Save completed task timing (moved from state 4)
			task_timings.push_back(current_task_timing);

			// Update sampling window delay and monitoring for SAMOS mapping
			#ifdef AUTHORSAMOSSampleMapping
			if (net && net->mapping_again == 1) {  // Only during sampling phase
				// Calculate total latency for this task
				int total_latency = current_task_timing.compute_end_cycle - current_task_timing.request_send_cycle;

				samplingWindowDelay[selfMACid] += total_latency;

				// Store the expected latency from SAMOS sampling
				if (latency_monitor.samos_expected_latency == 0.0) {
					// Store the average latency from sampling phase
					latency_monitor.samos_expected_latency = total_latency;
				}
			} else {
				// During actual execution phase - track the real latency
				int actual_latency = current_task_timing.compute_end_cycle - current_task_timing.request_send_cycle;

				// Update monitoring statistics
				latency_monitor.task_count++;
				latency_monitor.actual_latency_sum += actual_latency;
				latency_monitor.actual_latency_min = std::min(latency_monitor.actual_latency_min, (double)actual_latency);
				latency_monitor.actual_latency_max = std::max(latency_monitor.actual_latency_max, (double)actual_latency);
			}
			#endif

#ifdef fireAdvance
			tasks_completed++;
			computing_task_id = -1;  // ，
#endif

			return;
		}
		// State 4: COMPLETE
		// - Purpose: Finalize a single sub-task's computation and decide the next state.
		// - Duration: 1 cycle. This state is transitional.
		else if (selfstatus == 4) {
			// ：State 3（line 561-593），Fire AdvanceState 4
			// Statistics code moved to State 3 (lines 561-593) because Fire Advance bypasses State 4
			// double-counting（514 vs 512）
			// Removed duplicate statistics to avoid double-counting (which caused the 514 vs 512 discrepancy)

			this->send = 0;

#ifndef fireAdvance
			// ：
			if (this->llmPEExpectedtasktable.size() == 0) {
				// State 5: FINISHED
				// - Purpose: A final, static state indicating this MAC has completed all its tasks.
				// - Duration: Stays in this state until the simulation ends.
				this->selfstatus = 5;

				#ifdef AUTHORSAMOSSampleMapping
				// Print final summary when MAC finishes all tasks
				if (latency_monitor.task_count > 0) {
					double actual_avg = latency_monitor.actual_latency_sum / latency_monitor.task_count;
					double samos_expected = 0.0;
					if (latency_monitor.samos_expected_latency == 0.0 && samplingTasksPerMAC > 0) {
						samos_expected = samplingWindowDelay[selfMACid] / samplingTasksPerMAC;
					} else {
						samos_expected = latency_monitor.samos_expected_latency;
					}

					std::cout << "[LATENCY-MONITOR-FINAL] MAC " << selfMACid
					          << " COMPLETED " << latency_monitor.task_count << " tasks:"
					          << " SAMOS_expected=" << std::fixed << std::setprecision(2) << samos_expected
					          << " Actual_avg=" << actual_avg
					          << " Actual_min=" << latency_monitor.actual_latency_min
					          << " Actual_max=" << latency_monitor.actual_latency_max
					          << " Final_Discrepancy=" << (actual_avg - samos_expected)
					          << " (" << (100.0 * (actual_avg - samos_expected) / samos_expected) << "%)"
					          << std::endl;
				}
				#endif

			} else {
				this->selfstatus = 0;
			}
#else
			// Fire advance：
			if (tasks_completed >= total_tasks) {
				this->selfstatus = 5;

				#ifdef AUTHORSAMOSSampleMapping
				// Print final summary when MAC finishes all tasks
				if (latency_monitor.task_count > 0) {
					double actual_avg = latency_monitor.actual_latency_sum / latency_monitor.task_count;
					double samos_expected = 0.0;
					if (latency_monitor.samos_expected_latency == 0.0 && samplingTasksPerMAC > 0) {
						samos_expected = samplingWindowDelay[selfMACid] / samplingTasksPerMAC;
					} else {
						samos_expected = latency_monitor.samos_expected_latency;
					}

					std::cout << "[LATENCY-MONITOR-FINAL] MAC " << selfMACid
					          << " COMPLETED " << latency_monitor.task_count << " tasks:"
					          << " SAMOS_expected=" << std::fixed << std::setprecision(2) << samos_expected
					          << " Actual_avg=" << actual_avg
					          << " Actual_min=" << latency_monitor.actual_latency_min
					          << " Actual_max=" << latency_monitor.actual_latency_max
					          << " Final_Discrepancy=" << (actual_avg - samos_expected)
					          << " (" << (100.0 * (actual_avg - samos_expected) / samos_expected) << "%)"
					          << std::endl;
				}
				#endif

				// Fire advance
				std::cout << "[FIRE-ADVANCE-FINAL] MAC " << selfMACid
				          << " stats: sent=" << requests_sent
				          << " received=" << responses_received
				          << " completed=" << tasks_completed
				          << " (total=" << total_tasks << ")" << std::endl;

			} else if (computing_task_id >= 0) {
				this->selfstatus = 4;
			} else if (responses_received < requests_sent) {
				// outstanding requests，WAITING
				this->selfstatus = 2;
			} else {
				// IDLErequests
				this->selfstatus = 0;
			}
#endif

			llmResetForNextTask();
			this->pecycle = cycles + 1;
			return;
		}
	}

#ifdef fireAdvance
	// ===== Fire Advance Logic =====
	// cyclefire advance（MAC）
	if (fire_advance_armed && fire_advance_counter > 0) {
		fire_advance_counter--;

		if (fire_advance_counter == 0) {
			// ，request
			fire_advance_armed = false;

//			std::cout << "[FIRE-ADVANCE-TRIGGER] MAC " << selfMACid
//			          << " @cycle=" << cycles
//			          << " state=" << selfstatus
//			          << " sent=" << requests_sent
//			          << " total=" << total_tasks
//			          << " queue=" << llmPEExpectedtasktable.size()
//			          << " computing_id=" << computing_task_id
//			          << " pecycle=" << pecycle
//			          << std::endl;

			// ：、REQUEST
			if (requests_sent < total_tasks &&
			    llmPEExpectedtasktable.size() > 0 &&
			    selfstatus != 1) {

//				std::cout << "[FIRE-ADVANCE-SEND] MAC " << selfMACid
//				          << " forcing REQUEST state from state " << selfstatus
//				          << " (was scheduled for cycle " << pecycle << ")"
//				          << std::endl;

				// REQUEST，MAC
				selfstatus = 1;
				pecycle = cycles;
			} else {
//				std::cout << "[FIRE-ADVANCE-BLOCKED] MAC " << selfMACid
//				          << " cannot send: sent=" << requests_sent
//				          << " total=" << total_tasks
//				          << " queue=" << llmPEExpectedtasktable.size()
//				          << " state=" << selfstatus
//				          << std::endl;
			}
		}
	}
#endif
}

void LLMMAC::llmPEReceiveResp(Message* re_msg) {
	if (re_msg->msgtype == 1) {
		// Track when response actually arrives at MAC
		current_task_timing.response_arrive_cycle = cycles;
		// Calculate response hops (same as request typically in symmetric routing)
		current_task_timing.response_hops = current_task_timing.request_hops;
		inPETaskIDFromResp =  re_msg->signal_id;
		//cout<<"  currentRequestedTaskIDd "<<currentRequestedTaskIDd <<" inPETaskIDFromResp "<<inPETaskIDFromResp<<endl;

#ifndef fireAdvance
		assert(inPETaskIDFromResp ==currentRequestedTaskIDd && "currentRequestedTaskIDd shouldsame inPETaskIDFromRespSigID");
#else
		// Fire advance：
		responses_received++;

		// task_idpixelsubchunk
		current_pixel_id = inPETaskIDFromResp / LLM_SUBCHUNKS_PER_PIXEL;
		current_subchunk_id = inPETaskIDFromResp % LLM_SUBCHUNKS_PER_PIXEL;
#endif

		//  input  query
		input_data.clear();
		query_data.clear();

		// Memory，header！
		// Payload: [64input] + [64query]
		int data_size = 64;  // subchunk64
		int indexPayload=0;
		for(int i = 0; i < 8;i++ ){
			for(int j = 0; j < 8; j++ ){
				input_data.push_back(re_msg->authorMSGPayload[indexPayload]);
				indexPayload ++;
			}
			for(int j = 0; j < 8; j++ ){
				query_data.push_back(re_msg->authorMSGPayload[indexPayload]);
				indexPayload ++;
			}
		}

		//  partial sum
		float partial_sum = 0.0f;
		for (int i = 0; i < input_data.size() && i < query_data.size(); i++) {
			partial_sum += input_data[i] * query_data[i];
		}

		// Debug: Print computation details for first few pixels
		if (current_pixel_id <= 2) {  // Print for pixel[0][0], [0][1], [0][2]
			int px = current_pixel_id % net->query_output_dim;
			int py = current_pixel_id / net->query_output_dim;
			// Print first few elements for each pixel's first subchunk
			if (current_subchunk_id == 0) {
				std::cout << "[DEBUG-DATA] Pixel[" << py << "][" << px << "] subchunk 0:" << std::endl;
				std::cout << "  First 10 input values: ";
				for (int i = 0; i < 10 && i < input_data.size(); i++) {
					std::cout << std::fixed << std::setprecision(6) << input_data[i] << " ";
				}
				std::cout << std::endl;
				std::cout << "  First 10 query values: ";
				for (int i = 0; i < 10 && i < query_data.size(); i++) {
					std::cout << std::fixed << std::setprecision(6) << query_data[i] << " ";
				}
				std::cout << std::endl;

				// Manual calculation of first few products
				std::cout << "  First 5 products: ";
				for (int i = 0; i < 5 && i < input_data.size() && i < query_data.size(); i++) {
					float product = input_data[i] * query_data[i];
					std::cout << "(" << input_data[i] << "*" << query_data[i] << "=" << product << ") ";
				}
				std::cout << std::endl;
			}
		}

		//  partial sum
		if (pixel_partial_sums[current_pixel_id].size() < LLM_SUBCHUNKS_PER_PIXEL) {
			pixel_partial_sums[current_pixel_id].resize(LLM_SUBCHUNKS_PER_PIXEL, 0.0f);
		}
		pixel_partial_sums[current_pixel_id][current_subchunk_id] = partial_sum;

#ifndef fireAdvance
		currentRequestedTaskIDd = -1;  // ID，//currentrequestedtask
#else
		// computing_task_id，
		computing_task_id = inPETaskIDFromResp;
		currentRequestedTaskIDd = -1;

		// fire advance：request
		// ：fire advance，
		if (requests_sent < total_tasks) {
			fire_advance_counter = FIRE_ADVANCE_DELAY;
			fire_advance_armed = true;

			#ifdef AUTHORSAMOSSampleMapping
			// debug
			// bool in_sampling_phase = (net && net->mapping_again == 1);
//			std::cout << "[FIRE-ADVANCE-ARM] MAC " << selfMACid
//			          << " @cycle=" << cycles
//			          << " armed counter=" << FIRE_ADVANCE_DELAY
//			          << " for task_id=" << inPETaskIDFromResp
//			          << " (sent=" << requests_sent << "/" << total_tasks << ")"
//			          << (in_sampling_phase ? " [SAMPLING]" : " [PHASE2]")
//			          << std::endl;
			#else
//			std::cout << "[FIRE-ADVANCE-ARM] MAC " << selfMACid
//			          << " @cycle=" << cycles
//			          << " armed counter=" << FIRE_ADVANCE_DELAY
//			          << " for task_id=" << inPETaskIDFromResp
//			          << " (sent=" << requests_sent << "/" << total_tasks << ")"
//			          << std::endl;
			#endif
		}
#endif
	}
	else {
		// ：Type 1，

		assert(false && "LLMMAC received unexpected message type when expecting Type 1 response");
	}
}





bool LLMMAC::llmIsWaitingForData() {
	return (selfstatus == 2 && currentRequestedTaskIDd >= 0);
}

void LLMMAC::llmResetForNextTask() {
	input_data.clear();
	query_data.clear();
	input_buffer.clear();
	// Note: Don't clear pixel_partial_sums here as we need them for aggregation across tasks
	// Only clear them when a pixel is complete and sent
}







// : llmPrintDetailedData  llmieee754.cpp


// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
