// 修复的 llmmac.cpp
#include "llmmac.hpp"
#include <ctime>
#include <iomanip>

// Debug macro switch
#define LLM_DEBUG_PRINT
#ifdef LLM_DEBUG_PRINT
    #define LLM_DEBUG(x) do { \
        time_t now = time(0); \
        struct tm* timeinfo = localtime(&now); \
        std::cout << "[" << std::setfill('0') << std::setw(2) << timeinfo->tm_hour << ":" \
                  << std::setw(2) << timeinfo->tm_min << ":" << std::setw(2) << timeinfo->tm_sec \
                  << "] " << x << std::endl; \
    } while(0)
#else
    #define LLM_DEBUG(x) do {} while(0)
#endif

LLMMAC::LLMMAC(int t_id, LLMMACnet *t_net, int t_NI_id) {
	selfMACid = t_id;
	net = t_net;
	NI_id = t_NI_id;

	// Clear all data structures
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

	// Initialize LLM-specific parameters
	tile_size = 16;
	time_slice = 0;

	// Calculate tile position based on NI_id
	int tile_id = NI_id;
	int tiles_per_row = 8;  // 对应 32x32 NoC with 4x4 tiles (32/4=8)
	tile_x_start = (tile_id % tiles_per_row) * tile_size;
	tile_y_start = (tile_id / tiles_per_row) * tile_size;

	// Find destination memory ID based on position
	int xid = NI_id / X_NUM;
	int yid = NI_id % X_NUM;

#if defined MemNode2_4x4
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
		cout << "Error in LLMMAC constructor line 66";
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

	// Debug output for first few MAC units
	if (selfMACid < 5) {
		LLM_DEBUG("LLMMAC " << selfMACid << " created: NI_id=" << NI_id
		          << " dest_mem_id=" << dest_mem_id << " position=(" << xid << "," << yid << ")");
	}
}

bool LLMMAC::llmInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length = data_length;

	msg.QoS = 0;

	msg.data.assign(1, t_output);
	msg.data.push_back(tile_x_start);
	msg.data.push_back(tile_y_start);
	msg.data.push_back(time_slice);

	msg.destination = d_id;
	msg.out_cycle = pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id;
	msg.source_id = NI_id;
	msg.type = type;

	msg.yzMSGPayload.clear();

	if (msg.type == 0) { // Request
		msg.yzMSGPayload.assign(payloadElementNum, 0);
	} else if (msg.type == 2) { // Result
		msg.yzMSGPayload.assign(payloadElementNum, 0);
		msg.yzMSGPayload[0] = t_output;
	} else if (msg.type == 1) { // Response with data
		msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), input_buffer.begin() + 4,
								input_buffer.end());

		int flitNumSinglePacket = (msg.yzMSGPayload.size()) / (payloadElementNum) + 1;
		std::fill_n(std::back_inserter(msg.yzMSGPayload),
					(flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size()),
					0.0f);
	} else {
		LLM_DEBUG("Unknown message type: " << msg.type);
	}

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;
	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);

	return true;
}

void LLMMAC::llmRunOneStep() {
	static int total_run_count = 0;
	static int active_macs_last_report = 0;
	total_run_count++;

	// Debug: Print first few MAC units' status
	if (selfMACid < 3 && total_run_count % 100000 == 0) {
		LLM_DEBUG("MAC " << selfMACid << " status: " << selfstatus
		          << " tasks: " << routing_table.size()
		          << " request: " << request << " cycle: " << pecycle << "/" << cycles);
	}

	if (pecycle < cycles) {
		// State 0: IDLE - check if we have tasks to process
		if (selfstatus == 0) {
			if (routing_table.size() == 0) {
				selfstatus = 0;
				pecycle = cycles;
			} else {
				if (selfMACid < 3) {
					LLM_DEBUG("MAC " << selfMACid << " transitioning from IDLE to REQUEST with "
					          << routing_table.size() << " tasks");
				}
				pecycle = cycles;
				selfstatus = 1; // Go to request state
			}
		}
		// State 1: REQUEST - send request for data
		else if (selfstatus == 1) {
			request = routing_table.front();
			tmp_requestID = request;
			routing_table.pop_front();

			if (selfMACid < 3) {
				LLM_DEBUG("MAC " << selfMACid << " sending request for task " << request);
			}

			// Send request to memory
			llmInject(0, dest_mem_id, 1, request, net->vcNetwork->NI_list[NI_id],
					  packet_id + request, selfMACid);
			selfstatus = 2; // Wait for response
			pecycle = cycles;
		}
		// State 2: WAITING - wait for response from memory
		else if (selfstatus == 2) {
			if (request >= 0) {
				// Still waiting for response
				pecycle = cycles;
				selfstatus = 2;
				return;
			}

			// Response received, process the data
			if (input_buffer.size() < 4) {
				LLM_DEBUG("ERROR: MAC " << selfMACid << " input buffer size " << input_buffer.size() << " < 4");
				return;
			}

			if (selfMACid < 3) {
				LLM_DEBUG("MAC " << selfMACid << " received response, processing data");
			}

			// Extract LLM attention data from input_buffer
			fn = input_buffer[0]; // Function type (attention operation)
			int data_size = input_buffer[1]; // Size of query/key/value data
			time_slice = input_buffer[2]; // Current time slice
			int pixel_id = input_buffer[3]; // Pixel ID within tile

			// Extract query, key, value data (16 + 16 = 32 elements total)
			if (input_buffer.size() >= 4 + 32) {
				query_data.assign(input_buffer.begin() + 4, input_buffer.begin() + 4 + 16);
				key_data.assign(input_buffer.begin() + 4 + 16, input_buffer.begin() + 4 + 32);
			} else {
				LLM_DEBUG("ERROR: MAC " << selfMACid << " insufficient input buffer size: "
				          << input_buffer.size());
			}

			attention_output = 0.0;
			selfstatus = 3; // Go to compute state
			pecycle = cycles;
			return;
		}
		// State 3: COMPUTE - perform LLM attention computation
		else if (selfstatus == 3) {
			if (selfMACid < 3) {
				LLM_DEBUG("MAC " << selfMACid << " computing attention");
			}

			llmComputeAttention();

			// Calculate computation time based on data size
			int calc_time = (32 / PE_NUM_OP + 1) * 20; // 调整计算时间

			selfstatus = 4; // Ready for output
			pecycle = cycles + calc_time;

			// Send result back to memory
			llmInject(2, dest_mem_id, 1, attention_output,
					  net->vcNetwork->NI_list[NI_id], packet_id + tmp_requestID, selfMACid);
			return;
		}
		// State 4: COMPLETE - task completed
		else if (selfstatus == 4) {
			if (selfMACid < 3) {
				LLM_DEBUG("MAC " << selfMACid << " task completed, remaining tasks: "
				          << routing_table.size());
			}

			this->send = 0;
			if (this->routing_table.size() == 0) {
				this->selfstatus = 5; // All tasks completed
				if (selfMACid < 3) {
					LLM_DEBUG("MAC " << selfMACid << " all tasks completed");
				}
			} else {
				this->selfstatus = 0; // Back to idle for next task
			}

			llmResetForNextTask();
			this->pecycle = cycles + 1;
			return;
		}
	}
}

void LLMMAC::llmComputeAttention() {
	// Simple attention computation: dot product of query and key vectors
	attention_output = 0.0;

	// Ensure we have data
	if (query_data.size() < 16 || key_data.size() < 16) {
		LLM_DEBUG("ERROR: MAC " << selfMACid << " insufficient data for attention computation");
		return;
	}

	// Compute Q·K (dot product)
	for (int i = 0; i < 16; i++) {
		attention_output += query_data[i] * key_data[i];
	}

	// Apply simple scaling (attention_output / sqrt(d_k))
	attention_output = attention_output / sqrt(16.0);

	// Apply tanh activation (simplified softmax)
	attention_output = tanh(attention_output);
}

void LLMMAC::llmComputeQueryKeyDot() {
	// Detailed Q·K computation if needed for more complex attention
	float dot_product = 0.0;
	for (int i = 0; i < query_data.size() && i < key_data.size(); i++) {
		dot_product += query_data[i] * key_data[i];
	}
	attention_output = dot_product;
}

void LLMMAC::llmApplySoftmax() {
	// Simplified softmax - just normalize
	if (attention_output > 10.0) attention_output = 10.0;
	if (attention_output < -10.0) attention_output = -10.0;
	attention_output = 1.0 / (1.0 + exp(-attention_output));
}

void LLMMAC::llmComputeValueWeightedSum() {
	// If we had value vectors, we would compute weighted sum here
	// For now, just apply the attention weight
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

void LLMMAC::llmReceive(Message* re_msg) {
	// Handle received messages (responses from memory)
	if (re_msg->type == 1) {
		// Response with data
		input_buffer.clear();
		input_buffer.assign(re_msg->yzMSGPayload.begin(), re_msg->yzMSGPayload.end());
		request = -1; // Mark request as fulfilled

		if (selfMACid < 3) {
			LLM_DEBUG("MAC " << selfMACid << " received response message, buffer size: "
			          << input_buffer.size());
		}
	}
}

// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
