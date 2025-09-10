/**
 * @file llmmac.cpp
 * @brief LLM MAC计算单元实现 - Attention计算核心
 * 
 * 本文件实现了LLM模式下的单个MAC计算单元，负责执行Transformer的
 * Attention计算。通过状态机控制整个处理流程，并实现数据排序优化。
 * 
 * ========================================================================
 * 执行流程对应llmmacnet.cpp的7步骤
 * ========================================================================
 * 
 * Step 0: 初始化（在构造函数中完成）
 * ----------------------------------------
 * 函数：LLMMAC::LLMMAC() [行54-83]
 * - 设置MAC ID和NI接口ID
 * - 初始化dest_mem_id（内存节点映射）
 * - 清空任务队列和数据缓存
 * - selfstatus设为0（IDLE状态）
 * 
 * Step 1: 状态检查与转换【核心状态机】
 * ----------------------------------------
 * 函数：LLMMAC::llmRunOneStep() [行319-479]
 * 
 * 状态机定义：
 *   State 0 (IDLE):    空闲，等待任务
 *   State 1 (REQUEST): 发送数据请求
 *   State 2 (WAIT):    等待数据响应
 *   State 3 (COMPUTE): 执行计算
 * 
 * 状态转换逻辑：
 *   IDLE → REQUEST:    当llmtasktable非空时 [行336-342]
 *   REQUEST → WAIT:    发送请求后立即转换 [行373]
 *   WAIT → COMPUTE:    由processCNNPacket()设置 [行520]
 *   COMPUTE → IDLE:    计算完成后回到空闲 [行476]
 * 
 * Step 2: 数据请求发送（状态1执行）
 * ----------------------------------------
 * 函数：LLMMAC::llmInject() [行212-260]
 * 触发条件：selfstatus == 1
 * 关键操作：
 *   - 从llmtasktable取出任务ID [行349]
 *   - 创建type 0请求消息 [行228]
 *   - 计算源和目标坐标 [行229-232]
 *   - 通过NoC注入请求 [行238]
 * 
 * Step 3: 响应包创建与排序（内存节点端）
 * ----------------------------------------
 * 函数：LLMMAC::processCNNPacket() [行262-301]
 * 3.1 创建响应payload：
 *   - 132个float容器 [行278]
 *   - 元数据[0-3]：magic、size、chunk_id、pixel_id
 *   - Query数据[4-67]：64个元素
 *   - Key数据[68-131]：64个元素
 * 
 * 3.2 应用排序优化：
 *   函数：YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix() [yzllmieee754.cpp]
 *   - 分离排序：根据YZSeperatedOrdering_reArrangeInput宏
 *   - 关联排序：根据YzAffiliatedOrdering宏
 * 
 * Step 4: 数据接收处理（状态2执行）
 * ----------------------------------------
 * 函数：LLMMAC::processCNNPacket() [行487-545]
 * 触发条件：收到type 1响应消息
 * 关键操作：
 *   - 检查消息类型 [行489]
 *   - 提取payload数据 [行507,514]
 *   - 缓存到infeature/weight [行507,514]
 *   - 设置selfstatus = 3准备计算 [行520]
 * 
 * Step 5: 运算执行（状态3执行）
 * ----------------------------------------
 * 函数：LLMMAC::compute() [行547-608]
 * 触发条件：selfstatus == 3
 * 关键操作：
 *   - MAC运算：outfeature += weight[k] * infeature[i] [行568]
 *   - 激活函数：tanh(outfeature) [行573]
 *   - 检查是否需要更多数据 [行585]
 *   - 准备发送结果：selfstatus = 4 [行577]
 * 
 * Step 6: 结果输出与状态复位
 * ----------------------------------------
 * 函数：LLMMAC::sendOutput() [行610-640]
 * 关键操作：
 *   - 创建type 2结果消息 [行615]
 *   - 打包计算结果 [行622]
 *   - 通过NoC发送 [行630]
 *   - 复位到IDLE：selfstatus = 0 [行635]
 * 
 * ========================================================================
 * 排序优化算法详解
 * ========================================================================
 * 
 * 分离排序（Separated Ordering）：
 * - Query和Key独立排序，每列按1-bit数递增
 * - 优点：最大化减少bit翻转
 * - 缺点：破坏Query-Key语义关联
 * 
 * 关联排序（Affiliated Ordering）：
 * - Key排序，Query保持配对关系
 * - 优点：保持attention语义
 * - 缺点：bit翻转减少效果略差
 * 
 * ========================================================================
 * 关键数据结构
 * ========================================================================
 * 
 * - llmtasktable: 任务队列，存储待处理任务ID
 * - infeature: 输入特征缓存（Query数据）
 * - weight: 权重缓存（Key数据）
 * - outfeature: 输出结果
 * - selfstatus: 当前状态（0-3）
 * - pecycle: PE执行周期计数
 * 
 * @author YZ
 * @date 2025
 */

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
	current_processing_task_id = -1;  // 初始化为空闲状态
	saved_task_id_for_result = -1;    // 初始化保存的任务ID
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

bool LLMMAC::llmInject(int type, int d_id, int  tllm_eleNum, float t_output, NI* t_NI, int p_id, int mac_src) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length =  tllm_eleNum;
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
			std::cout << "  Task ID: " << saved_task_id_for_result << std::endl;
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
#ifdef LLMPADDING_RANDOM

		for (int i = 0; i < payloadElementNum; i++) {
			msg.yzMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif

	} else if (msg.msgtype == 2 || msg.msgtype == 3) { // Result (type 2 intermediate, type 3 final)

		msg.yzMSGPayload.assign(payloadElementNum, 0);
		// Result message padding
#ifdef LLMPADDING_RANDOM
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
				// Generate random float between -0.5 and 0.5 (same as CNN)
				float random_val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
				msg.yzMSGPayload.push_back(random_val);
			}
		#else

			
			// CHANGED: Use pure random data instead of matrix data to eliminate data dependencies
			msg.yzMSGPayload.clear();
			/*
			// Insert 10 zeros first
			for (int i = 0; i < 10; i++) {
				msg.yzMSGPayload.push_back(0.0f);
			}
			
			// Then add 108 random values
			for (int i = 0; i < 108; i++) {
				// Generate random float between -0.5 and 0.5 (same range as before)
				float random_val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
				msg.yzMSGPayload.push_back(random_val);
			}
			
			// Finally insert 10 zeros at the end
			for (int i = 0; i < 10; i++) {
				msg.yzMSGPayload.push_back(0.0f);
			}
			*/

			// Normal mode: Skip metadata like CNN - only send pure data
				// input_buffer: [metadata(4) + query(64) + key(64)] = 132 elements
				// We skip first 4, send only [query(64) + key(64)] = 128 elements
		  msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), input_buffer.begin() + 4,
							input_buffer.end());
		#endif

		// Calculate flits and add padding like CNN does
		//int flitNumSinglePacket = (msg.yzMSGPayload.size()) / (payloadElementNum) + 1;
		  int flitNumSinglePacket = (msg.yzMSGPayload.size() -1 + payloadElementNum) / (payloadElementNum) ;
		// Add padding to align with flit boundaries (same as CNN approach)
		std::fill_n(std::back_inserter(msg.yzMSGPayload),
					(flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size()),
					0.0f);

		// Always apply the reshape for debugging (but only sort if ordering is enabled)
		YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix(msg.yzMSGPayload);

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
		          << " current_task: " << current_processing_task_id << " cycle: " << pecycle << "/" << cycles);
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
			current_processing_task_id = llmtasktable.front();  // 从队列取出任务ID
			saved_task_id_for_result = current_processing_task_id;  // 保存用于后续结果发送
			llmtasktable.pop_front();
			
			// Start timing for new task
			current_task_timing = TaskTiming();
			current_task_timing.task_id = current_processing_task_id;
			current_task_timing.request_send_cycle = cycles;

			if (selfMACid < 10) {
				LLM_DEBUG("MAC " << selfMACid << " sending request for task " << current_processing_task_id << " at cycle " << cycles);
			}

			llmInject(0, dest_mem_id, 1, current_processing_task_id, net->vcNetwork->NI_list[NI_id],
					  packet_id + current_processing_task_id, selfMACid);
			
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
			if (current_processing_task_id >= 0) {
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
				LLM_DEBUG("MAC " << selfMACid << " received response, processing task " << saved_task_id_for_result);
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
					LLM_DEBUG("MAC " << selfMACid << " extracted data for task " << saved_task_id_for_result
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
				LLM_DEBUG("MAC " << selfMACid << " computing attention for task " << saved_task_id_for_result);
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
						  net->vcNetwork->NI_list[NI_id], packet_id + saved_task_id_for_result, selfMACid);
				
				// Clean up aggregation data for this pixel
				pixel_partial_sums.erase(current_pixel_id);
				pixel_subchunks_received.erase(current_pixel_id);
			}  // End of "if all 4 subchunks complete"
			
			// Calculate result packet hops (same as request)
			current_task_timing.result_hops = current_task_timing.request_hops;
			
			// WORKAROUND: Directly update output table due to NoC routing issues
			// This simulates the result reaching memory instantly
			if (net && saved_task_id_for_result >= 0 && saved_task_id_for_result < net->all_tasks.size()) {
				int pixel_x = net->all_tasks[saved_task_id_for_result].pixel_x;
				int pixel_y = net->all_tasks[saved_task_id_for_result].pixel_y;
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
				LLM_DEBUG("MAC " << selfMACid << " task " << saved_task_id_for_result << " completed, remaining tasks: "
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
		std::cout << "[ATTENTION-CALC @" << cycles << "] MAC " << selfMACid << " task " << saved_task_id_for_result << ":" << std::endl;
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
	return (selfstatus == 2 && current_processing_task_id >= 0);
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

// 注意: llmReshapeFlatToQueryKeyMatrix 函数已移至 yzllmieee754.cpp




// 注意: sortMatrix_LLMAffiliated 函数已移至 yzllmieee754.cpp



// 注意: sortMatrix_LLMSeparated 函数已移至 yzllmieee754.cpp





void LLMMAC::llmReceive(Message* re_msg) {
	if (re_msg->msgtype == 1) {
		// Track when response actually arrives at MAC
		current_task_timing.response_arrive_cycle = cycles;
		
		// Calculate response hops (same as request typically in symmetric routing)
		current_task_timing.response_hops = current_task_timing.request_hops;
		
		input_buffer.clear();
		input_buffer.assign(re_msg->yzMSGPayload.begin(), re_msg->yzMSGPayload.end());
		current_processing_task_id = -1;  // 清空当前任务ID，回到空闲状态
	}
}

// 注意: llmPrintDetailedData 函数已移至 yzllmieee754.cpp


// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
