// 小矩阵版本的 llmmac.cpp - 4x4可调试
#include "llmmac.hpp"
#include "llmmacnet.hpp"
#include "yzIEEE754.hpp"  // For bit-count sorting functions
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <chrono>

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

	// 修改：4x4 tile for 4x4 matrix
	tile_size = 4;  // 整个矩阵作为一个tile
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

	routing_table.clear();

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
		int current_task_id = tmp_requestID;
		if (current_task_id >= 0 && net && current_task_id < net->all_tasks.size()) {
			int pixel_x = net->all_tasks[current_task_id].pixel_x;
			int pixel_y = net->all_tasks[current_task_id].pixel_y;
			int ts = net->all_tasks[current_task_id].time_slice;

			msg.data.assign(1, t_output);
			msg.data.push_back((float)pixel_x);
			msg.data.push_back((float)pixel_y);
			msg.data.push_back((float)ts);

			// 关键调试信息
			if (LLM_DEBUG_LEVEL >= 2) {
				std::cout << "[CRITICAL @" << cycles << "] MAC " << selfMACid << " sending " 
				          << (type == 3 ? "FINAL" : "intermediate") << " result:" << std::endl;
				std::cout << "  Task ID: " << current_task_id << std::endl;
				std::cout << "  Pixel: (" << pixel_x << "," << pixel_y << ")" << std::endl;
				std::cout << "  Time slice: " << ts << std::endl;
				std::cout << "  Attention value: " << std::fixed << std::setprecision(10) << t_output << std::endl;
				std::cout << "  Destination: " << d_id << std::endl;
			}
		} else {
			LLM_DEBUG("ERROR: Invalid task ID " << current_task_id);
			msg.data.assign(1, t_output);
			msg.data.push_back(0);
			msg.data.push_back(0);
			msg.data.push_back(time_slice);
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
		msg.yzMSGPayload.assign(payloadElementNum, 0);
		if (selfMACid < 10) {
			LLM_DEBUG("MAC " << selfMACid << " sending request (type 0) to " << d_id << " for task " << t_output);
		}
	} else if (msg.msgtype == 2 || msg.msgtype == 3) { // Result (type 2 intermediate, type 3 final)
		msg.yzMSGPayload.assign(payloadElementNum, 0);
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

#ifdef flitLevelFlippingSwitch
		// Apply LLM ordering to response payload  
		llmApplyOrdering(msg.yzMSGPayload);
#endif

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
		          << " tasks: " << routing_table.size()
		          << " request: " << request << " cycle: " << pecycle << "/" << cycles);
	}

	if ((int)pecycle < (int)cycles) {
		// State 0: IDLE
		if (selfstatus == 0) {
			if (routing_table.size() == 0) {
				selfstatus = 0;
				pecycle = cycles;
			} else {
				if (selfMACid < 10) {
					LLM_DEBUG("MAC " << selfMACid << " transitioning IDLE->REQUEST with "
					          << routing_table.size() << " tasks, next task: " << routing_table.front());
				}
				pecycle = cycles;
				selfstatus = 1;
			}
		}
		// State 1: REQUEST
		else if (selfstatus == 1) {
			request = routing_table.front();
			tmp_requestID = request;
			routing_table.pop_front();
			
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
			time_slice = input_buffer[2];

			// 提取数据 - 小矩阵版本
			if (input_buffer.size() >= 4 + data_size * 2) {
				query_data.assign(input_buffer.begin() + 4, input_buffer.begin() + 4 + data_size);
				key_data.assign(input_buffer.begin() + 4 + data_size, input_buffer.begin() + 4 + data_size * 2);

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

			llmComputeAttention();

			// 验证计算结果
			if (LLM_DEBUG_LEVEL >= 2) {
				std::cout << "[COMPUTE-VERIFY @" << cycles << "] MAC " << selfMACid
				          << " task " << tmp_requestID
				          << " computed attention: " << std::fixed << std::setprecision(10)
				          << attention_output << std::endl;
			}

			int calc_time = (query_data.size() / PE_NUM_OP + 1) * 20;
			selfstatus = 4;
			pecycle = cycles + calc_time;
			
			// Track computation end and result send
			current_task_timing.compute_end_cycle = cycles + calc_time;
			current_task_timing.result_send_cycle = cycles + calc_time;

			// Changed: Send type 3 (final result) instead of type 2 (intermediate)
			// Since we're computing the final attention value for each pixel
			llmInject(3, dest_mem_id, 1, attention_output,
					  net->vcNetwork->NI_list[NI_id], packet_id + tmp_requestID, selfMACid);
			
			// Calculate result packet hops (same as request)
			current_task_timing.result_hops = current_task_timing.request_hops;
			
			// WORKAROUND: Directly update output table due to NoC routing issues
			// This simulates the result reaching memory instantly
			if (net && tmp_requestID >= 0 && tmp_requestID < net->all_tasks.size()) {
				int pixel_x = net->all_tasks[tmp_requestID].pixel_x;
				int pixel_y = net->all_tasks[tmp_requestID].pixel_y;
				if (pixel_x >= 0 && pixel_x < net->matrix_size && 
				    pixel_y >= 0 && pixel_y < net->matrix_size) {
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
				          << routing_table.size());
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
			if (this->routing_table.size() == 0) {
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
}

void LLMMAC::llmApplyOrdering(std::deque<float>& payload) {
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

	// Debug: Print original data for first few MACs
	if (selfMACid < 2) {
		std::cout << "\n=== MAC " << selfMACid << " Payload Ordering Debug ===\n";
		llmPrintDetailedData(query_data, "Query Data (BEFORE sorting)", 6);
		llmPrintDetailedData(key_data, "Key Data (BEFORE sorting)", 6);
	}

#ifdef reArrangeInput
	// Mode 1: Independent sorting - query and key sorted separately by bitcount
	llmRearrangeDeque(query_data, data_size, 1);  // query acts like "input"
	llmRearrangeDeque(key_data, data_size, 1);    // key acts like "weight"
#else
	// Mode 2: Correlated sorting - sort by key bitcount, query follows same order
	llmRearrangeDequeAccordingly(query_data, key_data, data_size, 1);
#endif

	// Debug: Print sorted data for first few MACs
	if (selfMACid < 2) {
		llmPrintDetailedData(query_data, "Query Data (AFTER sorting)", 6);
		llmPrintDetailedData(key_data, "Key Data (AFTER sorting)", 6);
		std::cout << "=== End MAC " << selfMACid << " Debug ===\n" << std::endl;
	}

	// Put the sorted data back into payload
	std::copy(query_data.begin(), query_data.end(), payload.begin() + metadata_size);
	std::copy(key_data.begin(), key_data.end(), payload.begin() + metadata_size + data_size);
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

void LLMMAC::llmRearrangeDeque(std::deque<float>& data, int colnum_per_row, int rownum_per_col) {
	// Sort data independently by IEEE754 bit count (like CNN's rearrangeDeque)
	if (data.empty()) return;
	
	// Create indices and sort by bit count
	std::vector<int> indices(data.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	
	// Sort indices based on IEEE754 bit count comparison
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return compareFloatsByOnes(data[i], data[j]);
	});
	
	// Rearrange data based on sorted indices
	std::deque<float> sorted_data;
	for (int idx : indices) {
		sorted_data.push_back(data[idx]);
	}
	
	data = sorted_data;
}

void LLMMAC::llmRearrangeDequeAccordingly(std::deque<float>& query_data, std::deque<float>& key_data, 
                                          int colnum_per_row, int rownum_per_col) {
	// Sort by key_data bit count, query_data follows same order (like CNN's rearrangeDequeAccordingly)
	if (key_data.empty() || query_data.empty()) return;
	if (key_data.size() != query_data.size()) return;
	
	// Create indices and sort by key_data bit count
	std::vector<int> indices(key_data.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	
	// Sort indices based on key_data IEEE754 bit count (key acts like weight)
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return compareFloatsByOnes(key_data[i], key_data[j]);
	});
	
	// Rearrange both query and key data based on sorted indices
	std::deque<float> sorted_query;
	std::deque<float> sorted_key;
	for (int idx : indices) {
		sorted_query.push_back(query_data[idx]);
		sorted_key.push_back(key_data[idx]);
	}
	
	query_data = sorted_query;
	key_data = sorted_key;
}

// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
