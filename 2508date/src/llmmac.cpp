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
		// Include ALL data from input_buffer (metadata + query + key)
		msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), input_buffer.begin(),
								input_buffer.end());

		int flitNumSinglePacket = (msg.yzMSGPayload.size()) / (payloadElementNum) + 1;
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
	static int call_count = 0;
	call_count++;
	if (call_count <= 5) {
		std::cout << "=== llmReshapeFlatToQueryKeyMatrix CALLED (call #" << call_count << ") ===" << std::endl;
	}
	
	// LLM payload contains: [metadata(4), query_data, key_data]
	// Extract data size from metadata
	if (payload.size() < 8) return; // Need at least metadata + some data
	
	int data_size = (int)payload[1];  // payload[1] contains query_data.size()
	int metadata_size = 4;
	
	// Verify we have enough data for query + key
	if (payload.size() < metadata_size + data_size * 2) return;
	
	// Extract query and key data sections
	std::deque<float> query_data(payload.begin() + metadata_size, 
	                             payload.begin() + metadata_size + data_size);
	std::deque<float> key_data(payload.begin() + metadata_size + data_size,
	                           payload.begin() + metadata_size + data_size * 2);

	// Apply sorting based on configuration
#ifdef YZSeperatedOrdering_reArrangeInput
	// Separated ordering: sort query and key independently
	sortMatrix_LLMSeparated(query_data, 8, 8);  // Assuming 8x8 matrix for 64 elements
	sortMatrix_LLMSeparated(key_data, 8, 8);
#elif defined(YzAffiliatedOrdering)
	// Affiliated ordering: sort query and key together
	sortMatrix_LLMAffiliated(query_data, key_data, 8, 8);
#endif

	// CNN-style matrix reorganization - row by row combination
	// After sorting, query_data and key_data are in row-major format (8x8 each)
	// Combine them row by row: [Query row i (8 elements) + Key row i (8 elements)] = 16 elements per row
	
	const int rows = 8;
	const int cols = 8;
	
	// Extract rows from query and key data
	std::vector<std::deque<float>> query_rows(rows);
	std::vector<std::deque<float>> key_rows(rows);
	
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			int idx = row * cols + col;
			if (idx < query_data.size()) {
				query_rows[row].push_back(query_data[idx]);
			}
			if (idx < key_data.size()) {
				key_rows[row].push_back(key_data[idx]);
			}
		}
	}
	
	// Combine rows: each row = [query_row + key_row]
	std::vector<std::deque<float>> combined_rows(rows);
	for (int i = 0; i < rows; i++) {
		// Add query row (8 elements)
		combined_rows[i].insert(combined_rows[i].end(), 
		                       query_rows[i].begin(), query_rows[i].end());
		// Add key row (8 elements)
		combined_rows[i].insert(combined_rows[i].end(), 
		                       key_rows[i].begin(), key_rows[i].end());
	}
	
	// Write back to payload
	int write_idx = metadata_size;
	for (const auto& row : combined_rows) {
		for (const auto& element : row) {
			if (write_idx < payload.size()) {
				payload[write_idx++] = element;
			}
		}
	}
	
	// Debug: Print packet structure and bit flips for first packet
	static int packet_count = 0;
	packet_count++;
	if (packet_count == 1) {  // Only analyze first packet
		std::cout << "\n=== Packet #" << packet_count << " Flit Structure ===" << std::endl;
		std::cout << "Total payload size: " << payload.size() << " elements" << std::endl;
		
		// Calculate number of flits
		int elements_per_flit = 16;
		int num_flits = (payload.size() + elements_per_flit - 1) / elements_per_flit;
		std::cout << "Number of flits: " << num_flits << std::endl;
		
		// Store all flits for bit flip analysis
		std::vector<std::vector<uint32_t>> flit_bits;
		
		// Print each flit's content and convert to bits
		for (int flit = 0; flit < num_flits; flit++) {
			std::cout << "\nFlit " << flit << ": ";
			int start_idx = flit * elements_per_flit;
			int end_idx = std::min(start_idx + elements_per_flit, (int)payload.size());
			
			// Collect values in this flit
			std::vector<float> flit_values;
			std::vector<uint32_t> flit_bit_repr;
			
			for (int i = start_idx; i < end_idx; i++) {
				flit_values.push_back(payload[i]);
				// Convert float to bit representation
				uint32_t bits = *reinterpret_cast<uint32_t*>(&payload[i]);
				flit_bit_repr.push_back(bits);
			}
			
			// Pad with zeros if needed
			while (flit_bit_repr.size() < 16) {
				flit_bit_repr.push_back(0);
			}
			
			flit_bits.push_back(flit_bit_repr);
			
			// Sort to check ordering
			std::vector<float> sorted_values = flit_values;
			std::sort(sorted_values.begin(), sorted_values.end());
			
			// Print first few values
			std::cout << "[";
			for (int i = 0; i < std::min(4, (int)flit_values.size()); i++) {
				std::cout << flit_values[i];
				if (i < 3 && i < flit_values.size()-1) std::cout << ", ";
			}
			if (flit_values.size() > 4) std::cout << "...";
			std::cout << "] (" << flit_values.size() << " elements)";
			
			// Check if sorted
			bool is_sorted = (flit_values == sorted_values);
			std::cout << " - " << (is_sorted ? "SORTED" : "NOT SORTED");
		}
		
		// Calculate bit flips between consecutive flits
		std::cout << "\n\n=== Bit Flips Between Consecutive Flits ===" << std::endl;
		int total_bit_flips = 0;
		for (int i = 1; i < flit_bits.size(); i++) {
			int flips = 0;
			for (int j = 0; j < 16; j++) {
				uint32_t xor_result = flit_bits[i-1][j] ^ flit_bits[i][j];
				// Count 1s in XOR result
				while (xor_result) {
					flips += xor_result & 1;
					xor_result >>= 1;
				}
			}
			std::cout << "Flit " << (i-1) << " -> Flit " << i << ": " << flips << " bit flips" << std::endl;
			total_bit_flips += flips;
		}
		std::cout << "\nTotal bit flips in packet: " << total_bit_flips << std::endl;
		std::cout << "Average bit flips per transition: " << (float)total_bit_flips / (flit_bits.size() - 1) << std::endl;
		
		std::cout << "\n" << std::endl;
	}
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

void LLMMAC::sortMatrix_LLMSeparated(std::deque<float>& data, int colnum_per_row, int rownum_per_col) {
	// Sort data independently by bit count (exactly like CNN's rearrangeDeque)
	//std::cout << "[DEBUG] sortMatrix_LLMSeparated called for MAC " << selfMACid << ", data size: " << data.size() << std::endl;
	if (data.empty()) return;
	
	// Step 1: Sort the entire deque based on bit counts
#ifdef FIXED_POINT_SORTING
	std::sort(data.begin(), data.end(), compareFloatsByFixed17Ones);
#else
	std::sort(data.begin(), data.end(), compareFloatsByOnes);
#endif

	// Step 2: Reorganize into col-major format (like CNN does)
	// Fill column by column: first column gets smallest values
	std::vector<std::deque<float>> rows(rownum_per_col);
	int idx = 0;
	
	// Fill by columns: for each column, fill all rows
	for (int col = 0; col < colnum_per_row; col++) {
		for (int row = 0; row < rownum_per_col; row++) {
			if (idx < data.size()) {
				rows[row].push_back(data[idx++]);
			}
		}
	}
	
	// Step 3: Write back to data in row order
	data.clear();
	for (const auto &row : rows) {
		for (const auto &element : row) {
			data.push_back(element);
		}
	}
}

void LLMMAC::sortMatrix_LLMAffiliated(std::deque<float>& query_data, std::deque<float>& key_data, 
                                          int colnum_per_row, int rownum_per_col) {
	// Sort by key_data bit count, query_data follows same order (like CNN's rearrangeDequeAccordingly)
	//std::cout << "[DEBUG] sortMatrix_LLMAffiliated called for MAC " << selfMACid << ", query size: " << query_data.size() << ", key size: " << key_data.size() << std::endl;
	if (key_data.empty() || query_data.empty()) return;
	if (key_data.size() != query_data.size()) return;
	
	// Create indices and sort by key_data bit count
	std::vector<int> indices(key_data.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	
	// Sort indices based on key_data bit count (key acts like weight)
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
#ifdef FIXED_POINT_SORTING
		return compareFloatsByFixed17Ones(key_data[i], key_data[j]);
#else
		return compareFloatsByOnes(key_data[i], key_data[j]);
#endif
	});
	
	// Rearrange both query and key data based on sorted indices
	std::deque<float> sorted_query;
	std::deque<float> sorted_key;
	for (int idx : indices) {
		sorted_query.push_back(query_data[idx]);
		sorted_key.push_back(key_data[idx]);
	}
	
	// Now reorganize into col-major format (like CNN does)
	// Step 2a: Reorganize query data - fill column by column
	std::vector<std::deque<float>> query_rows(rownum_per_col);
	int idx = 0;
	for (int col = 0; col < colnum_per_row; col++) {
		for (int row = 0; row < rownum_per_col; row++) {
			if (idx < sorted_query.size()) {
				query_rows[row].push_back(sorted_query[idx++]);
			}
		}
	}
	
	// Step 2b: Reorganize key data - fill column by column  
	std::vector<std::deque<float>> key_rows(rownum_per_col);
	idx = 0;
	for (int col = 0; col < colnum_per_row; col++) {
		for (int row = 0; row < rownum_per_col; row++) {
			if (idx < sorted_key.size()) {
				key_rows[row].push_back(sorted_key[idx++]);
			}
		}
	}
	
	// Step 3: Write back to data in row order
	query_data.clear();
	for (const auto &row : query_rows) {
		for (const auto &element : row) {
			query_data.push_back(element);
		}
	}
	
	key_data.clear();
	for (const auto &row : key_rows) {
		for (const auto &element : row) {
			key_data.push_back(element);
		}
	}
}

// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
