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
 *   IDLE → REQUEST:    当llmPEExpectedtasktable非空时 [行336-342]
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

	input_data.clear();
	query_data.clear();
	// Key已移除
	input_buffer.clear();

	fn = -1;
	currentRequestedTaskIDd = -1;  // 初始化为空闲状态
	nextLLMMAC = NULL;
	pecycle = 0;
	selfstatus = 0;
	send = 0;
	current_pixel_id = -1;
	current_subchunk_id = -1;
	pixel_partial_sums.clear();
	
	current_subchunk_id = 0;

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

	llmPEExpectedtasktable.clear();
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
	msg.yzMSGPayload.clear();
	current_pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL;
	current_subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL;

  if (msg.msgtype == 1) { // Response with data - 从all_tasks读取真实数据
		msg.yzMSGPayload.clear();
		// 从net的all_tasks获取真实数据而非随机生成
		if (net && task_id >= 0 && task_id < static_cast<int>(net->all_tasks.size())) {
			const LLMMACnet::LLMTask& task = net->all_tasks[task_id];
			// 添加Input数据
			msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), 
									task.input_data.begin(), 
									task.input_data.end());
			// 添加Query权重数据
			msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), 
									task.query_data.begin(), 
									task.query_data.end());
		} else {

			assert(false && "ERROR: Invalid task_id - task not found in all_tasks!");
		}

		// 计算flit数量并添加padding以对齐flit边界
		int flitNumSinglePacket = (msg.yzMSGPayload.size() - 1 + payloadElementNum) / payloadElementNum;
		//cout <<"int flitNumSinglePacket "<<  flitNumSinglePacket<<" msg.yzMSGPayload.size( "<<msg.yzMSGPayload.size() <<" msg.msgdata_length " <<  msg.msgdata_length <<endl;
		// 添加padding对齐到flit边界（与CNN相同的方法）
		std::fill_n(std::back_inserter(msg.yzMSGPayload),
					(flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size()),
					0.0f);

		// 应用排序优化（对所有消息类型）
		static int inject_count = 0;
		YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix(msg.yzMSGPayload);
	}

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;
	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);

	return true;
}

bool LLMMAC::llmPEInject(int type, int d_id, int  tllm_eleNum, float t_output, NI* t_NI, int p_id, int mac_src,int task_id) {
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src;
	msg.msgdata_length =  tllm_eleNum -4 ;// 132 or 128
	msg.QoS = 0;
	// 从 task_id 计算 pixel_id 和 subchunk_id
	current_pixel_id = task_id / LLM_SUBCHUNKS_PER_PIXEL;
	current_subchunk_id = task_id % LLM_SUBCHUNKS_PER_PIXEL;
	if ( type == 3) { //type == 2 不发result。
		// 对于结果消息（type 2中间结果 或 type 3最终结果），获取正确的像素坐标
		// 从current_pixel_id计算坐标
		// 输出矩阵是 8×128 (8行×128列)
		// pixel_id范围: 0-1023 (总共8*128=1024个像素)

		int pixel_x = current_pixel_id % net->matrixOutputPixels_queryoutputdim;  // 列坐标 (0-127)
		int pixel_y = current_pixel_id / net->matrixOutputPixels_queryoutputdim;  // 行坐标 (0-7)
		msg.data.assign(1, t_output);
		msg.data.push_back( pixel_x);
		msg.data.push_back( pixel_y);
		// current_subchunk_id 已经在 State 1 中从 task_id 正确计算
		msg.data.push_back( current_subchunk_id);  // Use subchunk_id instead of ts

	} else if (type == 0) {
		// Type 0: Request message - msg.data[0] must contain task_id for Memory node to retrieve the correct task
		msg.data.assign(1, task_id);  // msg.data[0] = task_id (不是0！)
		int pixel_x = current_pixel_id % net->matrixOutputPixels_queryoutputdim;  // 列坐标 (0-127)
		int pixel_y = current_pixel_id / net->matrixOutputPixels_queryoutputdim;  // 行坐标 (0-7)
		msg.data.push_back(pixel_x);
		msg.data.push_back(pixel_y);
		// current_subchunk_id 已经在 State 1 中从 task_id 正确计算
		msg.data.push_back(current_subchunk_id);      //  时间片
	} else {
		assert(false && "ERROR:PE LLM2不发，1则应该是Mem");
	}


	msg.destination = d_id;
	// Type 3 (final result) should be processed immediately, not delayed
	msg.out_cycle = (type == 3) ? cycles : pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id;
	msg.source_id = NI_id;
	msg.msgtype = type;
	msg.yzMSGPayload.clear();
	
#ifdef LLM_OPTIMIZED_TYPE03_HANDLING
	// 优化版本：Type 0/3 消息正确处理为16个元素
	if (msg.msgtype == 0) { // Request
		// Request message: 16个元素，第0位是task_id（但这里不设置，由调用者处理）
		msg.yzMSGPayload.assign(payloadElementNum, 0);
		// 使用专门的Type 0/3处理函数
		YzLLMIEEE754::llmReqRestReorderingFunc(msg.yzMSGPayload, 0.0f);
	} else if (msg.msgtype == 3) { // Result (type 2 intermediate不发, type 3 final)
		// Result message: 16个元素，第0位是结果值
		msg.yzMSGPayload.assign(payloadElementNum, 0);
		// 使用专门的Type 0/3处理函数，设置结果值
		YzLLMIEEE754::llmReqRestReorderingFunc(msg.yzMSGPayload, t_output);
		// std::cout << "[LLM-INJECT-TYPE3] Injecting Type 3 to NI " << NI_id 
		//           << " dest=" << d_id << " value=" << t_output << std::endl;
	}
	else {
		assert(false && "ERROR:PE LLM2不发，1则应该是Mem");
	}
	// Type 0和Type 3消息已经是正确大小（16个元素），无需额外padding
	
#else
	// 原版本：错误地对所有消息类型应用128元素排序
	if (msg.msgtype == 0) { // Request
		// Request message padding
		msg.yzMSGPayload.assign(payloadElementNum, 0);
	} else if (msg.msgtype == 3) { // Result (type 2 intermediate不发, type 3 final)
		msg.yzMSGPayload.assign(payloadElementNum, 0);
		msg.yzMSGPayload[0] = t_output;
		// std::cout << "[LLM-INJECT-TYPE3] Injecting Type 3 to NI " << NI_id 
		//           << " dest=" << d_id << " value=" << t_output << std::endl;
	}
	else {
		assert(false && "ERROR:PE LLM2不发，1则应该是Mem");
	}
	
	// 计算flit数量并添加padding以对齐flit边界
	int flitNumSinglePacket = (msg.yzMSGPayload.size() - 1 + payloadElementNum) / payloadElementNum;
	// 添加padding对齐到flit边界（与CNN相同的方法）
	std::fill_n(std::back_inserter(msg.yzMSGPayload),
				(flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size()),
				0.0f);
	// 应用排序优化（如果启用了排序宏）
	// 对所有类型消息进行排序，保持代码通用性
	// Type 3 messages only have 1 element, but sorting won't hurt
	YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix(msg.yzMSGPayload);
#endif

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;
	
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
			if (llmPEExpectedtasktable.size() == 0) { //一般是跑完了就一直等。比如快的pe跑完了。
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
			currentRequestedTaskIDd = llmPEExpectedtasktable.front();  // 从队列取出任务ID
			llmPEExpectedtasktable.pop_front();
			//if (selfMACid == 0 && currentRequestedTaskIDd < 100) {
			//	cout << "Line471: MAC 0 processing task_id=" << currentRequestedTaskIDd 
			//	     << " at cycle=" << cycles << endl;
			//}
			
			// 从 task_id 计算 pixel_id 和 subchunk_id
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
			selfstatus = 2;
			pecycle = cycles;
		}
		// State 2: WAITING
		// - Purpose: Wait for the memory controller to send back the requested data (Type 1).
		// - Duration: Variable. Depends on the network travel time for the request packet to reach memory
		//             and the response packet to return.
		else if (selfstatus == 2) {
			if (currentRequestedTaskIDd >= 0) {
				pecycle = cycles;
				selfstatus = 2;
				//std::cout << "llmmac439 selfstatus [DATA-CHECK] MAC "  << selfMACid  << " taskwearedoing now " << currentRequestedTaskIDd<<endl;
				return;
			}
			// Track response arrival time (this is when we process it)
			current_task_timing.response_arrive_cycle = cycles;
			selfstatus = 3;
			pecycle = cycles;
			return;
		}
		// State 3: COMPUTE
		else if (selfstatus == 3) { // currentRequestedTaskIDd 有值的时候，跳转到state3了。
			// Track computation start
			current_task_timing.compute_start_cycle = cycles;
			
			// Partial sum 已经在 llmPEReceiveResp 中计算并存储
			// 这里只需要模拟计算延迟

			int calc_time = ((query_data.size()-1) / PE_NUM_OP + 1) * 20;
			selfstatus = 4;
			pecycle = cycles + calc_time;
			
			// Track computation end and result send
			current_task_timing.compute_end_cycle = cycles + calc_time;
			current_task_timing.result_send_cycle = cycles + calc_time;



			// Check if all subchunks for this pixel are complete
			// 方案1：假设按顺序处理，最后一个子块(63)到达时聚合
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
			
			return;
		}
		// State 4: COMPLETE
		// - Purpose: Finalize a single sub-task's computation and decide the next state.
		// - Duration: 1 cycle. This state is transitional.
		else if (selfstatus == 4) {
			// Save completed task timing
			task_timings.push_back(current_task_timing);
			
			// Update sampling window delay for SAMOS mapping
			#ifdef YZSAMOSSampleMapping
			if (net && net->mapping_again == 1) {  // Only during sampling phase
				// Calculate total latency for this task
				int total_latency = current_task_timing.compute_end_cycle - current_task_timing.request_send_cycle;
				//if (selfMACid == 0 && samplingWindowDelay[0] > 300000) {
				//	cout << "Line588: MAC 0 large delay! Before=" << samplingWindowDelay[0] 
				//	     << " adding=" << total_latency << " task_id=" << current_task_timing.task_id << endl;
				//}
				samplingWindowDelay[selfMACid] += total_latency;
			}
			#endif

			this->send = 0;
			if (this->llmPEExpectedtasktable.size() == 0) {
				// State 5: FINISHED
				// - Purpose: A final, static state indicating this MAC has completed all its tasks.
				// - Duration: Stays in this state until the simulation ends.
				this->selfstatus = 5;

			} else {
				this->selfstatus = 0;
			}

			llmResetForNextTask();
			this->pecycle = cycles + 1;
			return;
		}
	}
}

void LLMMAC::llmPEReceiveResp(Message* re_msg) {
	if (re_msg->msgtype == 1) {
		// Track when response actually arrives at MAC
		current_task_timing.response_arrive_cycle = cycles;
		// Calculate response hops (same as request typically in symmetric routing)
		current_task_timing.response_hops = current_task_timing.request_hops;
		inPETaskIDFromResp =  re_msg->signal_id;
		//cout<<"  currentRequestedTaskIDd "<<currentRequestedTaskIDd <<" inPETaskIDFromResp "<<inPETaskIDFromResp<<endl;
		assert(inPETaskIDFromResp ==currentRequestedTaskIDd && "currentRequestedTaskIDd shouldsame inPETaskIDFromRespSigID");
		
		// 从响应消息中提取 input 和 query 数据
		input_data.clear();
		query_data.clear();
		
		// Memory直接发送数据，没有header！
		// Payload格式: [64个input数据] + [64个query数据]
		int data_size = 64;  // 每个subchunk包含64个元素
		int indexPayload=0;
		for(int i = 0; i < 8;i++ ){
			for(int j = 0; j < 8; j++ ){
				input_data.push_back(re_msg->yzMSGPayload[indexPayload]);
				indexPayload ++;
			}
			for(int j = 0; j < 8; j++ ){
				query_data.push_back(re_msg->yzMSGPayload[indexPayload]);
				indexPayload ++;
			}
		}
		
		// 直接在这里计算 partial sum
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
		
		// 存储 partial sum 用于聚合
		if (pixel_partial_sums[current_pixel_id].size() < LLM_SUBCHUNKS_PER_PIXEL) {
			pixel_partial_sums[current_pixel_id].resize(LLM_SUBCHUNKS_PER_PIXEL, 0.0f);
		}
		pixel_partial_sums[current_pixel_id][current_subchunk_id] = partial_sum;
		currentRequestedTaskIDd = -1;  // 清空当前任务ID，回到空闲状态//准确的说应该是currentrequestedtask到了
	}
	else {
		// 错误：期望Type 1响应，但收到了其他类型

		assert(false && "LLMMAC received unexpected message type when expecting Type 1 response");
	}
}





bool LLMMAC::llmIsWaitingForData() {
	return (selfstatus == 2 && currentRequestedTaskIDd >= 0);
}

void LLMMAC::llmResetForNextTask() {
	input_data.clear();
	query_data.clear();
	// Key已移除
	input_buffer.clear();
	// Note: Don't clear pixel_partial_sums here as we need them for aggregation across tasks
	// Only clear them when a pixel is complete and sent
}







// 注意: llmPrintDetailedData 函数已移至 yzllmieee754.cpp


// Destructor
LLMMAC::~LLMMAC() {
	// Cleanup if needed
}
