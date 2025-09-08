// 小矩阵版本的 llmmac.cpp - 4x4可调试
#include "llmmac.hpp"
#include "llmmacnet.hpp"
#include "yzIEEE754.hpp"  // For bit-count sorting functions
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

	query_data.clear();
	key_data.clear();
	value_data.clear();
	input_buffer.clear();

	fn = -1;
	request = -1;
	tmp_requestID = -1;
	attention_output = 0.0;
	nextLLMMAC = NULL;
	pecycle = 0;
	selfstatus = 0;
	send = 0;
	current_pixel_id = -1;
	current_subchunk_id = -1;
	pixel_partial_sums.clear();
	pixel_subchunks_received.clear();

	// 修改：4x4 tile for 4x4 matrix
	tile_Pixels_size = 4;  // 整个矩阵作为一个tile
	time_slice = 0;

	// 计算tile位置 - 对于4x4 tile，只有一个tile
	int tiles_per_row = 1;  // 4/4 = 1
	int tile_id = 0;  // 只有1个tile
	tile_x_start = 0;
	tile_y_start = 0;

	// Find destination memory ID
	int xid = NI_id / X_NUM;
	int yid = NI_id % X_NUM;

#if defined MemNode2_4X4
	dest_mem_id = dest_list[(yid / 2)];
#elif defined MemNode4_4X4
	if (xid <= 1 && yid <= 1) {
		dest_mem_id = dest_list[0];
	} else if (xid >= 2 && yid <= 1) {
		dest_mem_id = dest_list[1];
	} else if (xid <= 1 && yid >= 2) {
		dest_mem_id = dest_list[2];
	} else if ((xid >= 2 && yid >= 2)) {
		dest_mem_id = dest_list[3];
	} else {
		cout << "Error in LLMMAC constructor";
	}
#elif defined MemNode4_8X8
	const int mid = X_NUM / 2;
	if (xid < mid && yid < mid) {
		dest_mem_id = dest_list[0];
	} else if (xid >= mid && yid < mid) {
		dest_mem_id = dest_list[1];
	} else if (xid < mid && yid >= mid) {
		dest_mem_id = dest_list[2];
	} else if (xid >= mid && yid >= mid) {
		dest_mem_id = dest_list[3];
	} else {
		cout << "Error in LLMMAC constructor";
	}
#elif defined MemNode4_16X16
	const int mid = X_NUM / 2;
	if (xid < mid && yid < mid) {
		dest_mem_id = dest_list[0];
	} else if (xid >= mid && yid < mid) {
		dest_mem_id = dest_list[1];
	} else if (xid < mid && yid >= mid) {
		dest_mem_id = dest_list[2];
	} else if (xid >= mid && yid >= mid) {
		dest_mem_id = dest_list[3];
	} else {
		cout << "Error in LLMMAC constructor";
	}
#elif defined MemNode4_32X32
	const int mid = X_NUM / 2;
	if (xid < mid && yid < mid) {
		dest_mem_id = dest_list[0];
	} else if (xid >= mid && yid < mid) {
		dest_mem_id = dest_list[1];
	} else if (xid < mid && yid >= mid) {
		dest_mem_id = dest_list[2];
	} else if (xid >= mid && yid >= mid) {
		dest_mem_id = dest_list[3];
	} else {
		cout << "Error in LLMMAC constructor";
	}
#endif

	llmtasktable.clear();

	if (selfMACid < 10 && LLM_DEBUG_LEVEL >= 2) {
		LLM_DEBUG_INIT("LLMMAC " << selfMACid << " created: NI_id=" << NI_id
		          << " dest_mem_id=" << dest_mem_id << " position=(" << xid << "," << yid << ")");
	}
}

bool LLMMAC::llmInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length = data_length;
	msg.QoS = 0;

	if (type == 2 || type == 3) {
		// 对于结果消息（type 2中间结果 或 type 3最终结果），获取正确的像素坐标
		// 从current_pixel_id计算坐标
		int pixel_x = current_pixel_id % net->matrixOutputPixels_size;
		int pixel_y = current_pixel_id / net->matrixOutputPixels_size;

		msg.data.assign(1, t_output);
		msg.data.push_back((float)pixel_x);
		msg.data.push_back((float)pixel_y);
		msg.data.push_back((float)current_subchunk_id);  // Use subchunk_id instead of ts

		// 关键调试信息
		if (LLM_DEBUG_LEVEL >= 2) {
			std::cout << "[CRITICAL @" << cycles << "] MAC " << selfMACid << " sending " 
			          << (type == 3 ? "FINAL" : "intermediate") << " result:" << std::endl;
			std::cout << "  Task ID: " << tmp_requestID << std::endl;
			std::cout << "  Pixel: (" << pixel_x << "," << pixel_y << ")" << std::endl;
			std::cout << "  Time slice: " << current_subchunk_id << std::endl;
			std::cout << "  Attention value: " << std::fixed << std::setprecision(10) << t_output << std::endl;
			std::cout << "  Destination: " << d_id << std::endl;
		}
	} else if (type == 0) {
		// Request message - 传递task ID
		msg.data.assign(1, t_output);  // t_output contains the task ID for type 0
		msg.data.push_back(tile_x_start);
		msg.data.push_back(tile_y_start);
		msg.data.push_back(time_slice);
	} else {
		msg.data.assign(1, t_output);
		msg.data.push_back(tile_x_start);
		msg.data.push_back(tile_y_start);
		msg.data.push_back(time_slice);
	}

	msg.destination = d_id;
	msg.out_cycle = pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id;
	msg.source_id = NI_id;
	msg.msgtype = type;


	msg.yzMSGPayload.clear();

	if (msg.msgtype == 0) { // Request
		// Request message padding
		msg.yzMSGPayload.assign(payloadElementNum, 0);
#ifdef PADDING_RANDOM
		// Use random padding instead of zeros
		static bool warning_printed = false;
		if (!warning_printed) {
			cout << "WARNING: PADDING_RANDOM enabled in LLM mode - using random values for padding instead of zeros" << endl;
			warning_printed = true;
		}
		for (int i = 0; i < payloadElementNum; i++) {
			msg.yzMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
		if (selfMACid < 10) {
			LLM_DEBUG("MAC " << selfMACid << " sending request (type 0) to " << d_id << " for task " << t_output);
		}
	} else if (msg.msgtype == 2 || msg.msgtype == 3) { // Result (type 2 intermediate, type 3 final)
		// Result message padding
		msg.yzMSGPayload.assign(payloadElementNum, 0);
#ifdef PADDING_RANDOM
		// Use random padding instead of zeros  
		for (int i = 1; i < payloadElementNum; i++) { // i从1开始，保留[0]位置给t_output
			msg.yzMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
		msg.yzMSGPayload[0] = t_output;
		if (selfMACid < 10) {
			LLM_DEBUG("MAC " << selfMACid << " sending " << (msg.msgtype == 3 ? "FINAL" : "intermediate")
			          << " result (type " << msg.msgtype << ") to " << d_id
			          << " pixel(" << msg.data[1] << "," << msg.data[2] << ") ts=" << msg.data[3]
			          << " value: " << t_output);
		}
	} else if (msg.msgtype == 1) { // Response with data
		// TEST: Use completely random data instead of real data
		// #define TEST_RANDOM_RESPONSE  // 注释掉：不要每次都生成新的随机数据
		#ifdef TEST_RANDOM_RESPONSE
			// Fill with 128 random floats for testing
			static bool random_test_printed = false;
			if (!random_test_printed) {
				std::cout << "\n=== WARNING: TEST_RANDOM_RESPONSE enabled ===" << std::endl;
				std::cout << "Using completely random data for response payload!" << std::endl;
				std::cout << "This is for testing sorting effectiveness on random data.\n" << std::endl;
				random_test_printed = true;
			}
			
			msg.yzMSGPayload.clear();
			for (int i = 0; i < 128; i++) {
				// Generate random float between -1.0 and 1.0
				float random_val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
				msg.yzMSGPayload.push_back(random_val);
			}
		#else
			// Normal mode: Skip metadata like CNN - only send pure data
			// input_buffer: [metadata(4) + query(64) + key(64)] = 132 elements
			// We skip first 4, send only [query(64) + key(64)] = 128 elements
			msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), input_buffer.begin() + 4,
									input_buffer.end());
		#endif

		// Calculate flits: 128/16 = 8 flits exactly (no padding needed!)
		int flitNumSinglePacket = (msg.yzMSGPayload.size() + payloadElementNum - 1) / payloadElementNum;
		
		// Print flit information
		if (selfMACid < 10) {
			std::cout << "[LLM Flit Info] MAC " << selfMACid 
			          << " Payload size: " << msg.yzMSGPayload.size() 
			          << " floats, Flits needed: " << flitNumSinglePacket
			          << " (each flit = " << payloadElementNum << " floats)" << std::endl;
		}
		
		std::fill_n(std::back_inserter(msg.yzMSGPayload),
					(flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size()),
					0.0f);

		// Always apply the reshape for debugging (but only sort if ordering is enabled)
		llmReshapeFlatToQueryKeyMatrix(msg.yzMSGPayload);

		if (selfMACid < 10) {
			LLM_DEBUG("MAC " << selfMACid << " sending response (type 1) to " << d_id
			          << " payload size: " << msg.yzMSGPayload.size());
		}
	}

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;
	
	// Debug: verify packet creation for type 3
	if (msg.msgtype == 3) {
		LLM_DEBUG("Created type 3 packet: vnet=" << packet->vnet 
		          << " type=" << packet->type 
		          << " dest=" << packet->destination[0] << "," << packet->destination[1] << "," << packet->destination[2]
		          << " msg_dest=" << msg.destination);
	}
	
	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);

	return true;
}

void LLMMAC::llmRunOneStep() {
	static int total_run_count = 0;
	total_run_count++;

	// 为小矩阵提供更详细的调试
	if (selfMACid < 10 && total_run_count % 10000 == 0) {
		LLM_DEBUG("MAC " << selfMACid << " status: " << selfstatus
		          << " tasks: " << llmtasktable.size()
		          << " request: " << request << " cycle: " << pecycle << "/" << cycles);
	}

	if ((int)pecycle < (int)cycles) {
		// State 0: IDLE
		if (selfstatus == 0) {
			if (llmtasktable.size() == 0) {
				selfstatus = 0;
				pecycle = cycles;
			} else {
				if (selfMACid < 10) {
					LLM_DEBUG("MAC " << selfMACid << " transitioning IDLE->REQUEST with "
					          << llmtasktable.size() << " tasks, next task: " << llmtasktable.front());
				}
				pecycle = cycles;
				selfstatus = 1;
			}
		}
		// State 1: REQUEST
		// - Purpose: Send a request (Type 0) to the memory controller for the current task's data.
		// - Duration: 1 cycle. This state is transitional.
		else if (selfstatus == 1) {
			request = llmtasktable.front();
			tmp_requestID = request;
			llmtasktable.pop_front();
			
			// Start timing for new task
			current_task_timing = TaskTiming();
			current_task_timing.task_id = request;
			current_task_timing.request_send_cycle = cycles;

			if (selfMACid < 10) {
				LLM_DEBUG("MAC " << selfMACid << " sending request for task " << request << " at cycle " << cycles);
			}

			llmInject(0, dest_mem_id, 1, request, net->vcNetwork->NI_list[NI_id],
					  packet_id + request, selfMACid);
			
			// Calculate expected hops for request (Manhattan distance in mesh)
			int src_x = NI_id % X_NUM;
			int src_y = NI_id / X_NUM;
			int dst_x = dest_mem_id % X_NUM;
			int dst_y = dest_mem_id / X_NUM;
			current_task_timing.request_hops = abs(dst_x - src_x) + abs(dst_y - src_y);
			selfstatus = 2;
			pecycle = cycles;
		}
		// State 2: WAITING
		// - Purpose: Wait for the memory controller to send back the requested data (Type 1).
		// - Duration: Variable. Depends on the network travel time for the request packet to reach memory
		//             and the response packet to return.
		else if (selfstatus == 2) {
			if (request >= 0) {
				pecycle = cycles;
				selfstatus = 2;
				return;
			}

			if (input_buffer.size() < 4) {
				LLM_DEBUG("ERROR: MAC " << selfMACid << " input buffer size " << input_buffer.size() << " < 4");
				return;
			}

			// Track response arrival time (this is when we process it)
			current_task_timing.response_arrive_cycle = cycles;
			
			if (selfMACid < 10) {
				LLM_DEBUG("MAC " << selfMACid << " received response, processing task " << tmp_requestID);
				LLM_DEBUG("Input buffer size: " << input_buffer.size());
			}

			fn = input_buffer[0];
			int data_size = input_buffer[1];
			current_subchunk_id = input_buffer[2];  // subchunk ID
			current_pixel_id = input_buffer[3];      // pixel ID
			time_slice = current_subchunk_id;        // time_slice = subchunk_id

			// 提取数据 - 64x64 subchunk版本
			if (input_buffer.size() >= 4 + data_size * 2) {
				query_data.assign(input_buffer.begin() + 4, input_buffer.begin() + 4 + data_size);
				key_data.assign(input_buffer.begin() + 4 + data_size, input_buffer.begin() + 4 + data_size * 2);
				
				// 注意: 数据已经在发送端排序，这里不需要再排序

				if (selfMACid < 10) {
					LLM_DEBUG("MAC " << selfMACid << " extracted data for task " << tmp_requestID
					          << " - Query size: " << query_data.size()
					          << ", Key size: " << key_data.size());

					if (LLM_DEBUG_LEVEL >= 3) {
						std::cout << "Query data: ";
						for (int i = 0; i < query_data.size(); i++) {
							std::cout << std::fixed << std::setprecision(6) << query_data[i];
							if (i < query_data.size() - 1) std::cout << ",";
						}
						std::cout << std::endl;

						std::cout << "Key data: ";
						for (int i = 0; i < key_data.size(); i++) {
							std::cout << std::fixed << std::setprecision(6) << key_data[i];
							if (i < key_data.size() - 1) std::cout << ",";
						}
						std::cout << std::endl;
					}
				}
			} else {
				LLM_DEBUG("ERROR: MAC " << selfMACid << " insufficient input buffer size: "
				          << input_buffer.size() << " (need at least " << (4 + data_size * 2) << ")");
			}

			attention_output = 0.0;
			selfstatus = 3;
			pecycle = cycles;
			return;
		}
		// State 3: COMPUTE
		else if (selfstatus == 3) {
			// Track computation start
			current_task_timing.compute_start_cycle = cycles;
			
			if (selfMACid < 10) {
				LLM_DEBUG("MAC " << selfMACid << " computing attention for task " << tmp_requestID);
			}

			// Compute partial sum for this 64x64 subchunk
			float partial_sum = 0.0f;
			for (int i = 0; i < query_data.size(); i++) {
				partial_sum += query_data[i] * key_data[i];
			}

			// Store partial sum for aggregation
			if (pixel_partial_sums[current_pixel_id].size() < 4) {
				pixel_partial_sums[current_pixel_id].resize(4, 0.0f);
			}
			pixel_partial_sums[current_pixel_id][current_subchunk_id] = partial_sum;
			pixel_subchunks_received[current_pixel_id]++;

			if (LLM_DEBUG_LEVEL >= 2) {
				std::cout << "[PARTIAL-SUM @" << cycles << "] MAC " << selfMACid
				          << " computed subchunk " << current_subchunk_id
				          << " for pixel " << current_pixel_id
				          << " partial sum: " << std::fixed << std::setprecision(10) << partial_sum
				          << " (" << pixel_subchunks_received[current_pixel_id] << "/4 received)" << std::endl;
			}

			int calc_time = (query_data.size() / PE_NUM_OP + 1) * 20;
			selfstatus = 4;
			pecycle = cycles + calc_time;
			
			// Track computation end and result send
			current_task_timing.compute_end_cycle = cycles + calc_time;
			current_task_timing.result_send_cycle = cycles + calc_time;

			// Check if all 4 subchunks for this pixel are complete
			if (pixel_subchunks_received[current_pixel_id] == 4) {
				// Aggregate all partial sums
				float total_sum = 0.0f;
				for (int i = 0; i < 4; i++) {
					total_sum += pixel_partial_sums[current_pixel_id][i];
				}
				
				// Apply scaling and activation
				float scaled = total_sum / sqrt(128.0f);  // sqrt of full vector size
				attention_output = tanh(scaled);
				
				if (LLM_DEBUG_LEVEL >= 2) {
					std::cout << "[AGGREGATE @" << cycles << "] MAC " << selfMACid
					          << " pixel " << current_pixel_id << " complete:"
					          << " sum=" << total_sum
					          << " scaled=" << scaled  
					          << " final=" << attention_output << std::endl;
				}
				
				// Send final aggregated result (type 3)
				llmInject(3, dest_mem_id, 1, attention_output,
						  net->vcNetwork->NI_list[NI_id], packet_id + tmp_requestID, selfMACid);
				
				// Clean up aggregation data for this pixel
				pixel_partial_sums.erase(current_pixel_id);
				pixel_subchunks_received.erase(current_pixel_id);
			}  // End of "if all 4 subchunks complete"
			
			// Calculate result packet hops (same as request)
			current_task_timing.result_hops = current_task_timing.request_hops;
			
			// WORKAROUND: Directly update output table due to NoC routing issues
			// This simulates the result reaching memory instantly
			if (net && tmp_requestID >= 0 && tmp_requestID < net->all_tasks.size()) {
				int pixel_x = net->all_tasks[tmp_requestID].pixel_x;
				int pixel_y = net->all_tasks[tmp_requestID].pixel_y;
				if (pixel_x >= 0 && pixel_x < net->matrixOutputPixels_size && 
				    pixel_y >= 0 && pixel_y < net->matrixOutputPixels_size) {
					net->attention_output_table[pixel_y][pixel_x] = attention_output;
					if (LLM_DEBUG_LEVEL >= 2) {
						std::cout << "[DIRECT-UPDATE @" << cycles << "] MAC " << selfMACid 
						          << " directly updated output[" << pixel_y << "][" << pixel_x 
						          << "] = " << attention_output << std::endl;
					}
				}
			}
			return;
		}
		// State 4: COMPLETE
		// - Purpose: Finalize a single sub-task's computation and decide the next state.
		// - Duration: 1 cycle. This state is transitional.
		else if (selfstatus == 4) {
			if (selfMACid < 10) {
				LLM_DEBUG("MAC " << selfMACid << " task " << tmp_requestID << " completed, remaining tasks: "
				          << llmtasktable.size());
			}
			
			// Save completed task timing
			task_timings.push_back(current_task_timing);
			
			// Update sampling window delay for SAMOS mapping
			#ifdef YZSAMOSSampleMapping
			if (net && net->mapping_again == 1) {  // Only during sampling phase
				// Calculate total latency for this task
				int total_latency = current_task_timing.compute_end_cycle - current_task_timing.request_send_cycle;
				samplingWindowDelay[selfMACid] += total_latency;
				
				if (selfMACid < 10) {
					LLM_DEBUG("[SAMOS] MAC " << selfMACid << " task latency: " << total_latency 
					          << ", accumulated: " << samplingWindowDelay[selfMACid]);
				}
			}
			#endif

			this->send = 0;
			if (this->llmtasktable.size() == 0) {
				// State 5: FINISHED
				// - Purpose: A final, static state indicating this MAC has completed all its tasks.
				// - Duration: Stays in this state until the simulation ends.
				this->selfstatus = 5;
				if (selfMACid < 10) {
					LLM_DEBUG("MAC " << selfMACid << " all tasks completed");
				}
			} else {
				this->selfstatus = 0;
			}

			llmResetForNextTask();
			this->pecycle = cycles + 1;
			return;
		}
	}
}

void LLMMAC::llmComputeAttention() {
	attention_output = 0.0;

	if (query_data.size() != key_data.size() || query_data.size() == 0) {
		LLM_DEBUG("ERROR: MAC " << selfMACid << " data size mismatch or empty");
		return;
	}

	// 计算 Q·K (dot product)
	float dot_product = 0.0;
	for (int i = 0; i < query_data.size(); i++) {
		float product = query_data[i] * key_data[i];
		dot_product += product;
		if (selfMACid < 10) {
			LLM_DEBUG("  Q[" << i << "] * K[" << i << "] = " << query_data[i]
			          << " * " << key_data[i] << " = " << product);
		}
	}

	// 缩放 (attention_output / sqrt(d_k))
	float scaled = dot_product / sqrt((float)query_data.size());

	// 应用 tanh 激活
	attention_output = tanh(scaled);

	// 详细调试输出
	if (LLM_DEBUG_LEVEL >= 3) {
		std::cout << "[ATTENTION-CALC @" << cycles << "] MAC " << selfMACid << " task " << tmp_requestID << ":" << std::endl;
		std::cout << "  Vector size: " << query_data.size() << std::endl;
		std::cout << "  Dot product: " << std::fixed << std::setprecision(10) << dot_product << std::endl;
		std::cout << "  Scaled (dot/sqrt(" << query_data.size() << ")): " << std::fixed << std::setprecision(10) << scaled << std::endl;
		std::cout << "  Final output (tanh): " << std::fixed << std::setprecision(10) << attention_output << std::endl;
	}
}

void LLMMAC::llmComputeQueryKeyDot() {
	float dot_product = 0.0;
	size_t min_size = std::min(query_data.size(), key_data.size());
	for (size_t i = 0; i < min_size; i++) {
		dot_product += query_data[i] * key_data[i];
	}
	attention_output = dot_product;
}

void LLMMAC::llmApplySoftmax() {
	if (attention_output > 10.0) attention_output = 10.0;
	if (attention_output < -10.0) attention_output = -10.0;
	attention_output = 1.0 / (1.0 + exp(-attention_output));
}

void LLMMAC::llmComputeValueWeightedSum() {
	attention_output = attention_output * 1.0; // Placeholder
}

bool LLMMAC::llmIsWaitingForData() {
	return (selfstatus == 2 && request >= 0);
}

void LLMMAC::llmResetForNextTask() {
	query_data.clear();
	key_data.clear();
	value_data.clear();
	input_buffer.clear();
	attention_output = 0.0;
	// Note: Don't clear pixel_partial_sums here as we need them for aggregation across tasks
	// Only clear them when a pixel is complete and sent
}

void LLMMAC::llmReshapeFlatToQueryKeyMatrix(std::deque<float>& payload) {
	// === Step 4 & 5: 矩阵重组与排序 ===
	// 功能：将线性payload数据重组为8x8矩阵并应用排序优化
	// Commented out reshape function call tracking for cleaner output
	static int call_count = 0;
	call_count++;
	// if (call_count <= 10) {
	//	std::cout << "\n=== llmReshapeFlatToQueryKeyMatrix CALLED (call #" << call_count << ") ===" << std::endl;
	//	std::cout << "    Payload size: " << payload.size() << " floats" << std::endl;
	// }
	
	// Step 4.1: 检查payload格式
	// 新的LLM payload格式（跳过元数据）: [query数据(64个), key数据(64个)]
	// 总共128个float元素（纯数据，无元数据）
	if (payload.size() < 128) {
		std::cout << "WARNING: Payload too small: " << payload.size() << " < 128" << std::endl;
		return;
	}
	
	// Step 4.2: 设置数据大小（固定值，因为没有元数据了）
	int data_size = 64;  // 每个矩阵64个元素
	
	// Step 4.3: 验证payload完整性
	// 需要: 64(query) + 64(key) = 128个元素
	if (payload.size() < data_size * 2) return;
	
	// Step 4.4: 提取Query和Key数据段
	// Query数据: payload[0]到payload[63] (64个元素)
	std::deque<float> query_data(payload.begin(), 
	                             payload.begin() + data_size);
	// Key数据: payload[64]到payload[127] (64个元素)
	std::deque<float> key_data(payload.begin() + data_size,
	                           payload.begin() + data_size * 2);
	
	// Debug: Print raw data and bit statistics - only for ordered cases
	#ifdef YzAffiliatedOrdering
	if (call_count <= 3) {
		std::cout << "\n--- RAW DATA ANALYSIS (call #" << call_count << ") ---" << std::endl;
		
		// Print first 16 values
		std::cout << "Query raw values (first 16): ";
		for (int i = 0; i < 16; i++) {
			std::cout << std::fixed << std::setprecision(3) << query_data[i] << " ";
		}
		std::cout << std::endl;
		
		std::cout << "Key raw values (first 16): ";
		for (int i = 0; i < 16; i++) {
			std::cout << std::fixed << std::setprecision(3) << key_data[i] << " ";
		}
		std::cout << std::endl;
		
		// Analyze bit counts
		int q_bit_counts[64], k_bit_counts[64];
		for (int i = 0; i < 64; i++) {
			q_bit_counts[i] = countOnesInIEEE754(query_data[i]);
			k_bit_counts[i] = countOnesInIEEE754(key_data[i]);
		}
		

		// Show bit count distribution
		std::cout << "\nQuery bit counts (first 16): ";
		for (int i = 0; i < 16; i++) {
			std::cout << q_bit_counts[i] << " ";
		}
		std::cout << std::endl;
		
		std::cout << "Key bit counts (first 16): ";
		for (int i = 0; i < 16; i++) {
			std::cout << k_bit_counts[i] << " ";
		}
		std::cout << std::endl;
	}

	// Step 5: 根据配置应用排序算法
#ifdef YZSeperatedOrdering_reArrangeInput
	// Step 5.1: 分离排序 - Query和Key各自独立按列排序
	// 每列按1-bit数量升序排列，减少列内bit翻转
	sortMatrix_LLMSeparated(query_data, 8, 8);  // 将64个元素视为8x8矩阵
	sortMatrix_LLMSeparated(key_data, 8, 8);
#elif defined(YzAffiliatedOrdering)
	// Step 5.2: 关联排序 - Query跟随Key的bit数排序
	// Key按bit数排序，Query保持与Key的对应关系
	sortMatrix_LLMAffiliated(query_data, key_data, 8, 8);
#endif

	#endif

	// Define row and column dimensions for LLM matrices
	int rownum_per_col = 8;  // 8 rows in the matrix
	int querycolnum_per_row = 8;  // 8 columns for query matrix
	int keycolnum_per_row = 8;    // 8 columns for key matrix
	int totalcolnum_per_row = querycolnum_per_row + keycolnum_per_row;  // Total 16 columns per row when combined

	std::vector<std::deque<float>> query_rows(rownum_per_col); // one row contains "colnum_per_row" elements //for example， overall 40 = 5row *8 elements。 Inside one row， the left 4 elements are query the right 4 elements are key
	std::vector<std::deque<float>> key_rows(rownum_per_col); // one row contains "colnum_per_row" elements
	// convert 1 row to matrix. and  padding zero elements from flatten payload to matrix payload
	// Calculate the number of columns required  // to understand, assumming 4 rows, 16 cols
	for (int col_index = 0; col_index < querycolnum_per_row; col_index++) {	// first col, 4 element, then next col 4elemetn, next col 4 elements...
		for (int row_index = 0; row_index < rownum_per_col; row_index++) {
			if (col_index * rownum_per_col + row_index < query_data.size())// fill elements
					{
				query_rows[row_index].push_back(
						query_data[col_index * rownum_per_col + row_index]);
			}

			else {  //else padding zeros
				query_rows[row_index].push_back(0.0f);
			}
			//std::cout << querycolnum_per_row << " " << col_index	<< " ieee754line99 combined rows ok " << col_index << std::endl;
		}
	}
	// Calculate the number of columns required  // to understand, assumming 4 rows, 16 cols
	for (int col_index = 0; col_index < keycolnum_per_row; col_index++) { // first col, 4 element, then next col 4elemetn, next col 4 elements...
		for (int row_index = 0; row_index < rownum_per_col; row_index++) {
			if (col_index * rownum_per_col + row_index < key_data.size()) // fill elements
					{
				key_rows[row_index].push_back(
						key_data[col_index * rownum_per_col + row_index]);
			}

			else {  //else padding zeros
				key_rows[row_index].push_back(0.0f);
			}
			//	std::cout << " querycolnum_per_row " << querycolnum_per_row << " col_index " << col_index	<< " ieee754line99 combined rows ok " << col_index << std::endl;
		}
	}
	// Step 6.1: 按行组合query和key数据
	// 每个flit格式：[query第i行的8个元素] + [key第i行的8个元素]
	// 注意：由于数据已按列主序填充，第0行包含各列的第0个元素（bit数最多的元素）
	std::vector<std::deque<float>> combined_flits(rownum_per_col);  // 8个flits

	for (int row = 0; row < rownum_per_col; ++row) { // 遍历每一行
		// 先添加query第row行的所有列元素
		combined_flits[row].insert(combined_flits[row].end(), 
			query_rows[row].begin(), query_rows[row].end());
		// 再添加key第row行的所有列元素
		combined_flits[row].insert(combined_flits[row].end(), 
			key_rows[row].begin(), key_rows[row].end());
		//std::cout << combined_flits.size() << " " << combined_flits[row].size() 	<< "  combined_flits.size()lineafterinsert" << std::endl;
		if (combined_flits[row].size() != totalcolnum_per_row) { // Each flit should have 16 elements (8 query + 8 key)
			std::cerr
					<< "Error: flit size not equal to expected 16 elements "
					<< "Flit " << row << " size: " << combined_flits[row].size()
					<< " Expected: " << totalcolnum_per_row
					<< std::endl;
			assert(
					false
							&& "Error: flit size not equal to expected 16 elements");
			return;
		}
	}

	// original dq is: k*k inputs, k*k weights, 1 bias. For example, 25 inputs and 25 weights.

#ifdef printCombinedMatrix
	int printnextRowIndex1 = 0;
	std::cout << "dq:before halfinput half weight " << std::endl;
	for (const auto &element : dq) {
				std::cout << std::setw(10) << element << " ";
				printnextRowIndex1++;
				if (printnextRowIndex1 == totalcolnum_per_row) {
					std::cout << std::endl;
					printnextRowIndex1 = 0;
				}
			}
			std::cout << std::endl;
#endif
	// Step 6.2: 将重组后的数据写回原始容器
	// 这些数据将通过NoC传输（Step 7）
	payload.clear(); // 清空原始数据
	for (const auto &flit : combined_flits) {
		for (const auto &element : flit) {
			payload.push_back(element);  // 按flit顺序展开存储
		}
	}

}




void LLMMAC::sortMatrix_LLMAffiliated(std::deque<float>& query_data, std::deque<float>& key_data,
                                          int colnum_per_row, int rownum_per_col) {
	// === LLM版本：全局关联排序后按列填充（根据Key排序） ===
	if (key_data.empty() || query_data.empty()) {
		std::cerr << "ERROR: sortMatrix_LLMAffiliated - key_data or query_data is empty!" << std::endl;
		assert(false && "sortMatrix_LLMAffiliated: empty data provided");
	}
	if (key_data.size() != query_data.size()) {
		std::cerr << "ERROR: sortMatrix_LLMAffiliated - key_data size (" << key_data.size() 
		          << ") != query_data size (" << query_data.size() << ")" << std::endl;
		assert(false && "sortMatrix_LLMAffiliated: size mismatch between key and query");
	}

	// Step 1: 创建索引数组，根据Key进行全局排序
	std::vector<int> indices(key_data.size());
	for (size_t i = 0; i < key_data.size(); i++) {
		indices[i] = i;
	}

	// 不直接移动数据，而是排序索引，这样query可以跟随相同顺序
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
#ifdef FIXED_POINT_SORTING
		return compareFloatsByFixed17Ones(key_data[i], key_data[j]);  // 定点数比较
#else
		return compareFloatsByOnes(key_data[i], key_data[j]);         // 浮点数bit数比较
#endif
	});

	// Step 2: 根据排序后的索引重新排列query和key
	// query和key保持配对关系，都按key的bit数顺序排列
	std::deque<float> sortedKeys;
	std::deque<float> sortedQueries;
	for (int idx : indices) {
		sortedKeys.push_back(key_data[idx]);      // key按bit数排序
		sortedQueries.push_back(query_data[idx]); // query跟随key的顺序
	}

	// Step 3: 将排序后的数据写回原容器（与CNN保持一致）
	// 不进行矩阵重组，保持一维数组形式
	key_data  = sortedKeys;
	query_data = sortedQueries;
}



void LLMMAC::sortMatrix_LLMSeparated(std::deque<float>& data, int colnum_per_row, int rownum_per_col) {
	// === LLM版本：全局排序（与CNN保持一致） ===
	if (data.empty()) return;

	// Step 1: 对整个数据进行全局排序（降序）
	std::vector<float> sorted_data(data.begin(), data.end());
#ifdef FIXED_POINT_SORTING
	std::sort(sorted_data.begin(), sorted_data.end(), compareFloatsByFixed17Ones);
#else
	std::sort(sorted_data.begin(), sorted_data.end(), compareFloatsByOnes);
#endif

	// Step 2: 将排序后的数据写回原容器（与CNN保持一致）
	// 不进行矩阵重组，保持一维数组形式，后续在矩阵组合时按列主序填充
	data.clear();
	data.insert(data.end(), sorted_data.begin(), sorted_data.end());
}





void LLMMAC::llmReceive(Message* re_msg) {
	if (re_msg->msgtype == 1) {
		// Track when response actually arrives at MAC
		current_task_timing.response_arrive_cycle = cycles;
		
		// Calculate response hops (same as request typically in symmetric routing)
		current_task_timing.response_hops = current_task_timing.request_hops;
		
		input_buffer.clear();
		input_buffer.assign(re_msg->yzMSGPayload.begin(), re_msg->yzMSGPayload.end());
		request = -1;

		if (selfMACid < 10) {
			LLM_DEBUG("MAC " << selfMACid << " received response message, buffer size: "
			          << input_buffer.size());
			LLM_DEBUG("Message payload size: " << re_msg->yzMSGPayload.size());

			if (input_buffer.size() >= 8) {
				LLM_DEBUG("Buffer content: fn=" << input_buffer[0]
				          << " data_size=" << input_buffer[1]
				          << " time_slice=" << input_buffer[2]
				          << " pixel_id=" << input_buffer[3]);
				LLM_DEBUG("First 4 data values: " << input_buffer[4] << ", "
				          << input_buffer[5] << ", " << input_buffer[6] << ", " << input_buffer[7]);
			}
		}
	}
}

// LLM-specific sorting functions for IEEE754 bit-count optimization

void LLMMAC::llmPrintDetailedData(const std::deque<float>& data, const std::string& name, int max_elements) {
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


// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
