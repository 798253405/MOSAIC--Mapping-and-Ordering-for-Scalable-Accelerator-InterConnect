#include <cstdlib>  // For std::exit()

/**
 * @file llmmacnet.cpp
 * @brief LLM (Large Language Model) MAC网络实现 - 优化版本
 * 
 * 本文件实现了LLM模式下的MAC网络管理器，专门处理Transformer架构的
 * Attention计算。主要目标是通过数据排序优化减少NoC传输中的bit翻转。
 * 
 * ==================================================================
 * LLM处理流程 - 7大步骤详解
 * ==================================================================
 * 
 * Step 0: 初始化 - 数据加载与任务创建
 * ---------------------------------
 * 0.1 全部数据加载 [llmmacnet.cpp]
 *     函数：LLMMACnet::llmLoadRealMatrices() [行273-391]
 *     关键代码：
 *       llm_query_matrix.resize(matrix_size);  // 行279
 *       llm_key_matrix.resize(matrix_size);    // 行280
 *       std::ifstream file(filepath);          // 行304
 *       llm_query_matrix[i][j] = value;        // 行338
 *     功能：
 *     - 初始化512x512矩阵尺寸
 *     - 分配Query/Key/Value/Output矩阵存储空间
 *     - 从文件加载真实LLaMA数据或生成随机数据
 *     - 处理维度不匹配时的循环填充
 * 
 * 0.2 子任务创建 [llmmacnet.cpp]
 *     函数：LLMMACnet::llmGenerateAllTasks() [行536-659]
 *     关键代码：
 *       all_tasks.reserve(matrixOutputPixels * 4);  // 行549
 *       for(int pixel_y=0; pixel_y<matrixOutputPixels_size; pixel_y++)  // 行553
 *       for(int subchunk_id=0; subchunk_id<4; subchunk_id++)  // 行556
 *       task.query_data[data_idx] = llm_query_matrix[src_y][src_x];  // 行586
 *     功能：
 *     - 将512x512大矩阵分解为262,144个像素
 *     - 每像素生成4个子任务（2x2 subchunks）
 *     - 总计生成1,048,576个任务
 *     - 提取每任务的64个Query和64个Key元素
 * 
 * Step 1: 状态检查与转换
 * ---------------------------------
 * 函数：LLMMAC::llmRunOneStep() [llmmac.cpp 行319-479]
 * 关键代码与状态：
 *   if (selfstatus == 0) {  // IDLE状态 行332
 *     if (llmtasktable.size() > 0) selfstatus = 1;  // 有任务则转REQUEST 行342
 *   }
 *   else if (selfstatus == 1) {  // REQUEST状态 行348
 *     request = llmtasktable.front();  // 获取任务 行349
 *     llmInject(0, dest_mem_id, ...);  // 发送请求 行362
 *     selfstatus = 2;  // 转WAIT状态 行373
 *   }
 *   else if (selfstatus == 2) {  // WAIT状态 行378
 *     // 等待数据到达，由processCNNPacket()设置selfstatus=3
 *   }
 *   else if (selfstatus == 3) {  // COMPUTE状态 行443
 *     compute();  // 执行计算 行454
 *     selfstatus = 0;  // 完成后回IDLE 行476
 *   }
 * 功能：
 *   - 状态机控制：IDLE(0) -> REQUEST(1) -> WAIT(2) -> COMPUTE(3) -> IDLE(0)
 *   - 检查任务队列，管理任务执行流程
 *   - 跟踪每个状态的转换时间点
 * 
 * Step 2: 数据请求发送 (状态1: REQUEST)
 * ---------------------------------
 * 函数：LLMMAC::llmInject() [llmmac.cpp 行93-136]
 * 
 * Type 0请求消息数据结构：
 * ========================
 * msg.msgtype = 0;  // 请求类型
 * msg.data字段更新（llmmac.cpp行126-129）：
 *   msg.data[0] = current_processing_task_id;  // 任务ID（原名request）
 *   msg.data[1] = tile_x_start;  // tile起始X坐标
 *   msg.data[2] = tile_y_start;  // tile起始Y坐标  
 *   msg.data[3] = time_slice;     // 时间片(=subchunk_id)
 * 
 * current_processing_task_id说明：
 * - 变量原名：request
 * - 取值范围：0 到 1,048,575 (262,144像素 × 4个子块)
 * - 编码规则：pixel_id * 4 + subchunk_id
 * - 大小：int类型，通过float传输
 * 
 * 触发条件：
 *   selfstatus == 1 且 llmtasktable非空
 * 功能：
 * - 从任务队列取出任务ID
 * - 创建type 0请求消息
 * - 通过NoC发送到内存节点
 * 
 * Step 3: 响应包创建与排序 (内存节点处理)
 * ---------------------------------
 * 3.1 内存节点处理请求
 *     位置：LLMMACnet::llmRunOneStep() [llmmacnet.cpp 行1215-1524]
 *     功能：
 *     - 接收MAC发来的type 0请求
 *     - 从请求包的data[0]提取任务ID
 *     - 根据任务ID从all_tasks数组查找对应的Query/Key数据
 *     - 创建type 1响应消息
 *     - 准备数据容器tmpLLMMAC->input_buffer：[元数据(4) + query(64) + key(64)]
 *     - 排序处理（可选）：通过YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix()应用
 * 
 * 3.2 应用排序优化
 *     主函数：YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix() [yzllmieee754.cpp]
 *     
 * ========================================================================
 * 分离排序模式详细步骤（Separated Ordering）- 以4x8矩阵为例
 * ========================================================================
 * 
 * 输入：Query[4x8]=32个float, Key[4x8]=32个float
 * 
 * Step 1: 数据布局
 * ----------------
 * 原始Query矩阵（按行存储）：
 *   [Q0  Q1  Q2  Q3  Q4  Q5  Q6  Q7 ]  <- Row0
 *   [Q8  Q9  Q10 Q11 Q12 Q13 Q14 Q15]  <- Row1
 *   [Q16 Q17 Q18 Q19 Q20 Q21 Q22 Q23]  <- Row2
 *   [Q24 Q25 Q26 Q27 Q28 Q29 Q30 Q31]  <- Row3
 *   
 * Step 2: 展开为一维数组并全局排序
 * ----------------------------------
 * a) 展开：[Q0, Q1, Q2, ..., Q31] 共32个元素
 * 
 * b) 计算每个元素的1-bit数并排序：
 *    假设排序后：[B0, B1, B2, ..., B31]
 *    其中B0的1-bit数最少，B31的1-bit数最多
 * 
 * Step 3: 按列主序重新填充矩阵
 * ----------------------------
 * 使用排序后的数组B[]，按列填充：
 * 
 * x = 0
 * for col in 0..7:
 *     for row in 0..3:
 *         Query[row][col] = B[x]
 *         x = x + 1
 * 
 * 填充后的矩阵：
 *   Col0    Col1    Col2    ...  Col7
 * Row0: B[0]    B[4]    B[8]        B[28]
 * Row1: B[1]    B[5]    B[9]        B[29]
 * Row2: B[2]    B[6]    B[10]       B[30]
 * Row3: B[3]    B[7]    B[11]       B[31]
 * 
 * Key矩阵同样处理
 * 
 * Step 4: NoC传输顺序
 * --------------------
 * 传输序列: Col0[0]→Col0[1]→...→Col7[3]
 * 相邻传输bit翻转最小
 * 
 * ========================================================================
 * 关联排序模式详细步骤（Affiliated Ordering）- 以4x8矩阵为例
 * ========================================================================
 * 
 * Step 1: 创建Q-K配对（32对）
 * ----------------------------
 * Pair[0]=(Q0,K0), Pair[1]=(Q1,K1), ..., Pair[31]=(Q31,K31)
 * 
 * Step 2: 计算Key的1-bit数作为排序键
 * -----------------------------------
 * 示例：
 * Pair[0]: K0=0.5 → 9个1-bit
 * Pair[1]: K1=0.125 → 8个1-bit  
 * Pair[2]: K2=1.0 → 10个1-bit
 * 
 * Step 3: 按Key的1-bit数排序所有对
 * ---------------------------------
 * 排序前: [(Q0,K0,9), (Q1,K1,8), (Q2,K2,10)]
 * 排序后: [(Q1,K1,8), (Q0,K0,9), (Q2,K2,10)]
 * Query跟随Key移动，保持配对
 * 
 * Step 4: 重组为4x8矩阵
 * ---------------------
 * 排序后的对按顺序填充矩阵位置
 * 保持Query-Key语义对应关系
 *
 *     辅助函数 [yzIEEE754.cpp]：
 *     - countOnesInIEEE754() [行76-89]: 计算浮点数1-bit数
 *     - compareFloatsByOnes() [行125-127]: 比较函数
 * 
 * Step 4: 数据接收处理 (状态2: WAIT)
 * ---------------------------------
 * 函数：LLMMAC::() [llmmac.cpp 行262-317]
 * 关键代码：
 *   if(re_msg->msgtype == 1) {  // 行264
 *   input_buffer = re_msg->yzMSGPayload;  // 行281
 *   llmReshapeFlatToQueryKeyMatrix(input_buffer);  // 行293
 *   selfstatus = 3;  // 准备计算 行315
 * 功能：
 * - 接收type 1响应消息
 * - 提取payload数据到input_buffer
 * - 调用reshape函数重组数据
 * - 触发状态转换到COMPUTE
 * 
 * Step 5: 运算执行与新请求发送 (状态3: COMPUTE)
 * ---------------------------------
 * 函数：LLMMAC::llmComputeAttention() [llmmac.cpp 行795-868]
 * 关键代码：
 *   llmComputeQueryKeyDot();  // Q*K^T计算 行809
 *   llmComputeValueWeightedSum();  // 与Value相乘 行815
 *   attention_output = result;  // 保存结果 行853
 * 功能：
 * - 执行attention三步计算
 * - Q*K^T矩阵乘法，除以sqrt(d_k)
 * - Softmax归一化获得注意力权重
 * - 权重与Value矩阵相乘得最终输出
 * 
 * Step 6: 结果输出与状态复位
 * ---------------------------------
 * 函数：LLMMAC::llmInject() with type=3 [llmmac.cpp 行212-260]
 * 关键代码：
 *   msg.msgtype = 3;  // LLM最终结果消息 行228
 *   msg.yzMSGPayload.push_back(attention_output);  // 添加结果 行235
 *   t_NI->inject(s_id, d_id, Message_cnt, NMessage);  // 发送 行238
 *   selfstatus = 0;  // 回到空闲状态 行257
 * 功能：
 * - 创建type 3最终结果消息（LLM专用）
 * - 打包attention计算结果
 * - 通过NoC发送到内存节点
 * - MAC状态复位，准备处理下一任务
 * 
 * ==================================================================
 * 数据格式与优化策略
 * ==================================================================
 * 
 * Payload结构（132个float）：
 * [0-3]   : 元数据（magic, size, chunk_id, pixel_id）
 * [4-67]  : Query数据（64个元素）
 * [68-131]: Key数据（64个元素）
 * 
 * 半半重组格式（CNN兼容）：
 * 每行：[Query_row[0-7] | Key_row[0-7]] = 16元素
 * 8行x16列 = 128元素矩阵
 * 
 * Bit翻转优化效果：
 * - 基准线（无排序）：1620.5 bit flips
 * - 分离排序：1595 bit flips (1.57%改善)
 * - 关联排序：保持语义同时减少翻转
 * 
 * ==================================================================
 * 配置选项 (parameters.hpp)
 * ==================================================================
 * 
 * - YZLLMSwitchON: 启用LLM模式
 * - case4_seperratedordering: 使用分离排序
 * - case3_affiliatedordering: 使用关联排序
 * - LLM_CNN_HALFHALF_RESHAPE: 启用半半重组
 * - YzAffiliatedOrdering: 激活排序优化
 * 
 * ==================================================================
 * 与CNN模式的关键区别
 * ==================================================================
 * 
 * LLM特点：
 * - 任务级并行（百万级子任务）
 * - 动态数据访问模式
 * - Attention矩阵每次不同
 * - Bit flipping较多（需要排序优化）
 * 
 * CNN特点：
 * - 层级顺序处理
 * - 固定卷积核复用
 * - 规则的滑动窗口
 * - Bit flipping较少（天然随机分布）
 * 
 * @author YZ
 * @date 2025
 */

#include "llmmacnet.hpp"
#include "llmmac.hpp"
#include "yzIEEE754.hpp"  // For float_to_ieee754 and countOnesInIEEE754
#include <cassert>
#include <ctime>
#include <iomanip>
#include <climits>
#include <chrono>
#include <cmath>  // For sqrt and ceil functions
#include <sstream>  // For std::istringstream
// Helper function to get current time string
static inline std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm* timeinfo = localtime(&time_t);
    char buffer[10];
    strftime(buffer, sizeof(buffer), "%H:%M", timeinfo);
    return std::string(buffer);
}
LLMMACnet::LLMMACnet(int mac_num, int t_pe_x, int t_pe_y, VCNetwork *t_Network) {
	macNum = mac_num;
	LLMMAC_list.reserve(mac_num);
	pe_x = t_pe_x;
	pe_y = t_pe_y;
	vcNetwork = t_Network;

	current_layer = 0;
	total_layers = 1;

	// Use configuration from parameters.hpp
	#if LLM_TEST_CASE == 1
	// Test Case 1: Small matrix test
	
	#elif LLM_TEST_CASE == 2
	#define LLM_SUBCHUNKS_PER_PIXEL 64
	// Test Case 2: Real matrix 8×128 output
	// Set actual dimensions for the real matrices
	// X_input (8×4096) @ Wq^T (4096×128) = Q (8×128)

	input_sequence_length = 8;     // X_input has 8 rows
	input_hidden_dim = 4096;       // X_input has 4096 columns
	query_output_dim = 128;        // Wq produces 128-dim query vectors
	time_slices = LLM_SUBCHUNKS_PER_PIXEL;  // Each pixel has N time slices (subchunks)
	matrixOutputPixels_inputsequencelength = 8;  // 8 rows output matrix (from X_input rows)
	matrixOutputPixels_queryoutputdim = 128;

	// Calculate derived parameters
	// Each pixel generates N tasks (subchunks)
	tasks_per_pixel = LLM_SUBCHUNKS_PER_PIXEL;  // Use configured subchunks per pixel
	int elements_per_task = 128;  // 64 query + 64 key per task
	total_task_slicedPixels = input_sequence_length * query_output_dim * time_slices;  // 8 * 128 * 64 = 65536 tasks
	#endif


	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;


	for (int i = 0; i < macNum; i++) {
		int temp_ni_id = i % TOT_NUM;
		LLMMAC *newLLMMAC = new LLMMAC(i, this, temp_ni_id);
		LLMMAC_list.push_back(newLLMMAC);
	}


	// Initialize matrices - try loading from files first
	if (!llmReadSavedMatrix()) {
		// If loading fails, initialize with random matrices
		llmInitializeRandomMatrices();
	}
	// Generate all tasks
	llmGenerateAllTasks();
	// Test bit representation functions with demo data
	std::cout << "\n=== LLM Initialization: Demo Data Debug ===\n";
	layer_latency.clear();
}

void LLMMACnet::llmNetRunStep() {
	static int run_step_count = 0;
	run_step_count++;

	for (int i = 0; i < macNum; i++) {
		LLMMAC_list[i]->llmRunOneStep();
	}



	/**
	 * @brief 内存节点包处理循环 - LLM模式的核心数据交换
	 *
	 * 包类型处理详解：
	 * ================
	 *
	 * Type 0 (数据请求包)：
	 * --------------------
	 * - 来源：MAC单元在State 1发送
	 * - 内容：包含任务ID (task_id)
	 * - 处理：内存节点查找对应数据并返回Type 1响应
	 * - 流程：MAC→Memory
	 *
	 * Type 1 (数据响应包)：
	 * --------------------
	 * - 来源：内存节点响应Type 0请求
	 * - 内容：132个float (4个header + 64个Query + 64个Key)
	 * - 处理：MAC接收后进入计算状态
	 * - 流程：Memory→MAC
	 *
	 * Type 2 (中间结果包)：
	 * --------------------
	 * - 来源：MAC计算的部分结果（调试用）
	 * - 内容：[result_value, pixel_x, pixel_y, time_slice]
	 * - 处理：仅记录日志，不影响最终输出
	 * - 流程：MAC→Memory
	 *
	 * Type 3 (最终结果包)：
	 * --------------------
	 * - 来源：MAC聚合4个子块后的最终结果
	 * - 内容：[final_value, pixel_x, pixel_y, time_slice]
	 * - 处理：更新Q_resOutput_matrix矩阵
	 * - 流程：MAC→Memory
	 */

	// Memory operations handling
	int pbuffer_size;
	int src, pid_signal_id, mem_id, src_mac;
	LLMMAC *tmpLLMMAC;
	Packet *tmpPacket;
	NI *tmpNI;

	// 遍历所有内存节点，检查其包缓冲区
	for (int memidx = 0; memidx < MEM_NODES; memidx++) {
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];

		// Process ALL message types in buffer[0]
		pbuffer_size = tmpNI->packet_buffer_out[0].size();

		// 检查缓冲区中的每个包
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();

			// ===== Type 0: 处理数据请求包 =====
			// MC 收到MAC发送的请求 0 ，内存需要返回对应的Query-Key数据，就是MC 收0 发1
			if (tmpPacket->message.msgtype == 0) {
				if (tmpPacket->message.out_cycle >= cycles) {
					tmpNI->packet_buffer_out[0].pop_front();
					tmpNI->packet_buffer_out[0].push_back(tmpPacket);
					continue;  //跳过后续对当前包，而是检查下一个包。
				}

				// 提取请求包的源信息
				pid_signal_id = tmpPacket->message.signal_id; // 包的信号ID
				src = tmpPacket->message.source_id;          // 源节点ID
				src_mac = tmpPacket->message.mac_id;         // 发送请求的MAC ID



				tmpLLMMAC = LLMMAC_list[src_mac];            // 获取对应的MAC对象，可以给这mac直接提供运算数据，而packet传输是单独的。

				// 验证MAC确实在等待数据（State 2: WAITING）
				if (tmpLLMMAC->selfstatus == 2) {
					/**
					 * @brief 内存节点处理Type 0请求 - 详细流程
					 *
					 * Step 3.1: 提取任务ID
					 * =====================
					 * Type 0包结构：
					 * - msgtype: 0 (请求类型)
					 * - source_id: 发送MAC的节点ID
					 * - mac_id: 发送MAC的编号
					 * - data[0]: task_id (核心数据)
					 *
					 * 任务ID编码规则：
					 * - task_id = pixel_id * 4 + subchunk_id
					 * - 范围: 0 到 1,048,575 (262,144像素 × 4子块)
					 *
					 * 示例：
					 * - task_id = 1025
					 * - pixel_id = 1025 / 4 = 256
					 * - subchunk_id = 1025 % 4 = 1
					 */
					int task_id = tmpPacket->message.data[0];  // 从Type 0请求包的data[0]字段提取任务ID

					// 记录请求到达内存的时间（用于性能分析）
					tmpLLMMAC->current_task_timing.request_arrive_cycle = cycles;

					/**
					 * Step 3.2: 查找任务数据
					 * ======================
					 * 数据查找过程：
					 * 1. 验证task_id合法性（0 <= task_id < all_tasks.size()）
					 * 2. 从all_tasks向量中索引对应任务
					 * 3. all_tasks在llmGenerateAllTasks()中预先生成
					 * 4. 每个LLMTask包含：
					 *    - query_data: 64个float（从512×512矩阵提取）
					 *    - key_data: 64个float（从512×512矩阵提取）
					 *    - 元信息: pixel坐标、子块ID等
					 */
					if (task_id >= 0 && task_id < all_tasks.size()) {
						LLMTask& task = all_tasks[task_id];  // 直接索引访问，O(1)时间复杂度

						/**
						 * Step 3.3: 准备数据容器（input_buffer）
						 * =======================================
						 *
						 * 数据容器位置：tmpLLMMAC->input_buffer
						 * - 这是MAC对象的临时缓冲区
						 * - 用于组装Type 1响应包的payload
						 *
						 * 容器结构（总计132个float）：
						 * ┌────────────────────────────────┐
						 * │ Header (4 floats)              │
						 * ├────────────────────────────────┤
						 * │ [0]: 1.0 (LLM模式标志)          │
						 * │ [1]: 64 (数据块大小)            │
						 * │ [2]: subchunk_id (0-3)         │
						 * │ [3]: pixel_id (像素标识)        │
						 * ├────────────────────────────────┤
						 * │ Query Data (64 floats)         │
						 * ├────────────────────────────────┤
						 * │ [4-67]: Query向量元素           │
						 * ├────────────────────────────────┤
						 * │ Key Data (64 floats)           │
						 * ├────────────────────────────────┤
						 * │ [68-131]: Key向量元素           │
						 * └────────────────────────────────┘
						 */

						// 清空缓冲区，准备新数据
						tmpLLMMAC->input_buffer.clear();

						// 添加4个Header元数据
						tmpLLMMAC->input_buffer.push_back(1.0f);              // [0] 函数标志(1.0=LLM模式)
						tmpLLMMAC->input_buffer.push_back(64);                // [1] 数据大小(每个矩阵64元素)
						tmpLLMMAC->input_buffer.push_back(task.subchunk_id);  // [2] 子块ID(0-3)
						tmpLLMMAC->input_buffer.push_back(task.pixel_id);     // [3] 像素ID(用于结果聚合)

						/**
						 * Step 3.4: 数据排序处理
						 * ======================
						 *
						 * 排序位置和时机：
						 * 1. 原始数据生成时（llmGenerateAllTasks）：
						 *    - 从512×512矩阵提取64个元素到task.query_data/key_data
						 *    - 此时数据保持原始顺序
						 *
						 * 2. 当前位置（内存节点响应）：
						 *    - 可选择在此处应用排序优化
						 *    - 通过YzLLMIEEE754::llmReshapeFlatToQueryKeyMatrix()
						 *    - 根据宏定义选择排序策略：
						 *      * YZSeperatedOrdering_reArrangeInput: 分离排序
						 *      * YzAffiliatedOrdering: 关联排序
						 *      * 无定义: 不排序
						 *
						 * 3. MAC接收端（llmNonMemMACReceiveResp）：
						 *    - 接收已排序/未排序的数据
						 *    - 直接用于Attention计算
						 *
						 * 注意：当前实现中，排序主要在数据传输时通过
						 * llmReshapeFlatToQueryKeyMatrix应用到payload
						 */

						// 复制任务数据（保持原始数据不变）
						std::deque<float> query_data_copy(task.query_data.begin(), task.query_data.end());
						std::deque<float> key_data_copy(task.query_data.begin(), task.query_data.end());

						// 将Query和Key数据添加到缓冲区
						// Query: input_buffer[4] 到 input_buffer[67]
						tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
							query_data_copy.begin(), query_data_copy.end());

						// Key: input_buffer[68] 到 input_buffer[131]
						tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
							key_data_copy.begin(), key_data_copy.end());



						/**
						 * Step 6: 创建Type 1响应包并注入NoC
						 * =====================================
						 *
						 * 传输机制：
						 * - 数据被切分为多个flit，每个flit 512位(16个float)
						 * - 128个float需要 128/16 = 8个flit
						 * - 使用虚拟通道(VC)避免死锁
						 *
						 * 延迟计算：
						 * - 内存访问延迟 = (数据量 * MEM_read_delay) + CACHE_DELAY
						 * - 网络传输延迟取决于路由距离和网络拥塞
						 */


						//下面才是和真是网络相关
						/* CNN 参考代码  收0发 1
					MAC_list[mem_id]->inbuffer.clear();
					MAC_list[mem_id]->inbuffer = MAC_list[src_mac]->inbuffer;
					MAC_list[mem_id]->inject(1, src, tmpMAC->inbuffer.size(),
							o_fnReluOrPool, vcNetwork->NI_list[mem_id],
							pidSignalID, src_mac);
						 */
						// 计算内存访问延迟
						int mem_delay = static_cast<int>(ceil((task.query_data.size() * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
						LLMMAC_list[mem_id]->pecycle = cycles + mem_delay;

						// 记录响应发送时间（用于延迟分析）
						tmpLLMMAC->current_task_timing.response_send_cycle = cycles + mem_delay;

						// 将数据复制到内存节点的缓冲区（跳过前4个元数据）
						LLMMAC_list[mem_id]->input_buffer.clear();
						// 复制时跳过前4个元数据 [mode, size, subchunk_id, pixel_id]
						//LLMMAC_list[mem_id]->input_buffer.assign( tmpLLMMAC->input_buffer.begin() + 4, 	tmpLLMMAC->input_buffer.end());
						//int actual_payload_size = LLMMAC_list[mem_id]->input_buffer.size();  // 应该是128
						LLMMAC_list[mem_id]->input_buffer =tmpLLMMAC->input_buffer;
						int actual_payload_size = LLMMAC_list[mem_id]->input_buffer.size();
						// Pass task_id (not payload size) as third parameter for type 1 messages
						LLMMAC_list[mem_id]->llmMemNodeInject(1, src, actual_payload_size,
							1.0f, vcNetwork->NI_list[mem_id], pid_signal_id, src_mac, task_id); //传递task_id
						// 注入Type 1响应包到NoC网络
						// 参数说明：
						// - 1: msgtype (Type 1 = 数据响应)
						// - src: 目标MAC的节点ID
						// - actual_payload_size: 实际payload大小(128个float)
						// - 1.0f: 数据值（此处未使用）
						// - mem_id: 发送方内存节点的网络接口
						// - pid_signal_id: 包ID（用于匹配请求-响应）
						// - src_mac: 目标MAC的ID

					}
					tmpNI->packet_buffer_out[0].pop_front();
				}
			}
			// ===== Type 2 & 3: 处理结果包 =====
			else if (tmpPacket->message.msgtype == 2 || tmpPacket->message.msgtype == 3) {
				// 检查包是否应该在当前周期处理
				if (tmpPacket->message.out_cycle >= cycles) {
					tmpNI->packet_buffer_out[0].pop_front();
					tmpNI->packet_buffer_out[0].push_back(tmpPacket);
					continue;
				}

				/**
				 * Type 2/3包数据格式：
				 * - data[0]: result_value (计算结果)
				 * - data[1]: pixel_x (像素X坐标)
				 * - data[2]: pixel_y (像素Y坐标)
				 * - data[3]: time_slice (时间片/子块ID)
				 */
				int msg_type = tmpPacket->message.msgtype;
				src = tmpPacket->message.source_id;
				src_mac = tmpPacket->message.mac_id;
				tmpLLMMAC = LLMMAC_list[src_mac];

				// 验证数据完整性
				if (tmpPacket->message.data.size() >= 4) {
					float result_value = tmpPacket->message.data[0];  // Attention计算结果
					int pixel_x = tmpPacket->message.data[1];         // 输出矩阵X坐标
					int pixel_y = tmpPacket->message.data[2];         // 输出矩阵Y坐标
					int time_slice = tmpPacket->message.data[3];      // 子块ID (0-3)

					if (msg_type == 2) {
						/**
						 * Type 2: 中间结果包（调试用）
						 * - 每个子块的部分结果
						 * - 不影响最终输出
						 * - 用于验证计算过程
						 */

					}
					else if (msg_type == 3) {
						/**
						 * Type 3: 最终结果包
						 * - 4个子块聚合后的最终Attention值
						 * - 更新Q_resOutput_matrix矩阵
						 * - 标记任务完成
						 */
						std::cout << "[FINAL-UPDATE] Memory " << mem_id
						          << " received FINAL result from MAC " << src_mac << std::endl;
						std::cout << "  Pixel: (" << pixel_x << "," << pixel_y << ")" << std::endl;
						std::cout << "  Final value: " << std::fixed << std::setprecision(10) << result_value << std::endl;

						if (pixel_x >= 0 && pixel_x < query_output_dim &&
						    pixel_y >= 0 && pixel_y < input_sequence_length) {

							// 保存旧值用于比较
							float old_value = Q_resOutput_matrix[pixel_y][pixel_x];

							// 更新输出矩阵
							Q_resOutput_matrix[pixel_y][pixel_x] = result_value;

							// 增加已执行任务计数
							executed_tasks++;

							std::cout << "[TABLE-UPDATE] Updated output table:" << std::endl;
							std::cout << "  Position [" << pixel_y << "][" << pixel_x << "]" << std::endl;
							std::cout << "  Old value: " << std::fixed << std::setprecision(10) << old_value << std::endl;
							std::cout << "  New value: " << std::fixed << std::setprecision(10) << result_value << std::endl;
							std::cout << "  Verification: " << Q_resOutput_matrix[pixel_y][pixel_x] << std::endl;
							std::cout << "  Tasks completed: " << executed_tasks << "/" << total_task_slicedPixels << std::endl;

						} else {
							std::cout << "[ERROR] Invalid pixel coordinates: ("
							          << pixel_x << "," << pixel_y << ")" << std::endl;
						}
					}
				}

				if (tmpLLMMAC->selfstatus == 5) {
					tmpLLMMAC->send = 3;
				}
				tmpNI->packet_buffer_out[0].pop_front();
			}
			else {
				// Other message types - just cycle them
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
			}
		}
	}

	//NonMemMAC  Handle responses (type 1 messages)
	for (int i = 0; i < TOT_NUM; i++) {
		if (llmIsMemoryNode(i)) continue;

		tmpNI = this->vcNetwork->NI_list[i];
		pbuffer_size = tmpNI->packet_buffer_out[0].size();

		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			if (tmpPacket->message.msgtype != 1) {
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}

			src_mac = tmpPacket->message.mac_id;
			tmpLLMMAC = LLMMAC_list[src_mac];
			tmpLLMMAC->llmNonMemMACReceiveResp(&tmpPacket->message);
			tmpNI->packet_buffer_out[0].pop_front();
		}
	}
}


// ==================================================
// Step 1: 数据生成与加载
// 功能：从文件加载真实LLaMA矩阵或生成随机数据
// ==================================================
bool LLMMACnet::llmLoadRealMatrices(const std::string& input_dir) {
	try {
		// Step 1.1: 初始化矩阵尺寸为512x512
		const int matrix_size =  512;  // 512
		
		// Step 1.2: 分配矩阵存储空间
		input_matrix.resize(matrix_size);
		query_weight_matrix.resize(matrix_size);     // Query矩阵
		Q_resOutput_matrix.resize(matrix_size); //q 输出矩阵
		// Step 1.3: 尝试打开LLaMA数据文件
		std::ifstream query_file(input_dir + "llama_query_8x4096.txt");
		if (!query_file.is_open()) {
			// Try 2048x128 version
			query_file.open(input_dir + "llama_query_2048x128.txt");
			if (!query_file.is_open()) {
				std::cerr << "Failed to open query file from " << input_dir << std::endl;
				return false;
			}
		}
		
		// Step 1.4: 读取文件维度信息（第一行）
		int file_rows, file_cols;
		query_file >> file_rows >> file_cols;  // 读取行数和列数
		std::cerr << "Loading Query matrix: " << file_rows << "x" << file_cols << std::endl;
		
		// 跳过第一行剩余内容
		std::string dummy;
		std::getline(query_file, dummy);
		
		/**
		 * @brief 维度不匹配时的循环填充机制
		 * 
		 * 问题背景：
		 * ==========
		 * - 目标矩阵：512×512 (NoC仿真需要)
		 * - 输入文件：可能是 128×128, 512×128, 2048×128 等不同维度
		 * - 需求：将任意维度扩展到512×512，保持数据分布特性
		 * 
		 * 循环填充策略：
		 * ==============
		 * 
		 * 1. 列维度填充（当file_cols < 512）：
		 * ------------------------------------
		 * 示例：file_cols = 128, matrix_size = 512
		 * 
		 * 原始数据：[A0, A1, A2, ..., A127]
		 * 填充后：  [A0, A1, ..., A127, A0, A1, ..., A127, A0, A1, ..., A127, A0, A1, ..., A127]
		 *           |<--- 原始128 --->|<--- 复制128 --->|<--- 复制128 --->|<--- 复制128 --->|
		 * 
		 * 实现方式：input_matrix[i][j] = input_matrix[i][j % file_cols]
		 * - j=128时: 128 % 128 = 0, 复制A0
		 * - j=129时: 129 % 128 = 1, 复制A1
		 * - j=255时: 255 % 128 = 127, 复制A127
		 * - j=256时: 256 % 128 = 0, 再次复制A0
		 * 
		 * 2. 行维度填充（当file_rows < 512）：
		 * ------------------------------------
		 * 示例：file_rows = 128, matrix_size = 512
		 * 
		 * 原始数据：Row[0] 到 Row[127]
		 * 填充后：  Row[0-127], Row[0-127], Row[0-127], Row[0-127]
		 *           共4次完整复制，形成512行
		 * 
		 * 实现方式：input_matrix[i][j] = input_matrix[i % file_rows][j]
		 * - i=128时: 128 % 128 = 0, 复制Row[0]
		 * - i=255时: 255 % 128 = 127, 复制Row[127]
		 * - i=256时: 256 % 128 = 0, 再次复制Row[0]
		 * 
		 * 3. 组合效果：
		 * ------------
		 * 128×128 → 512×512：数据被复制16次(4×4)
		 * 512×128 → 512×512：列方向复制4次
		 * 2048×128 → 512×512：行截断到512，列复制4次
		 * 
		 * 优点：
		 * ------
		 * - 保持数据分布：循环复制不改变数值范围和统计特性
		 * - 空间局部性：相邻数据保持相关性
		 * - 计算简单：模运算效率高
		 * - 避免零填充：不会稀释数据密度
		 */
		
		// Step 1.5: 逐行读取矩阵数据并存储
		for (int i = 0; i < matrix_size && i < file_rows; i++) {
			input_matrix[i].resize(matrix_size);
			std::string line;
			if (!std::getline(query_file, line)) break;
			
			std::istringstream iss(line);
			for (int j = 0; j < matrix_size; j++) {
				float value;
				if (j < file_cols && (iss >> value)) {
					// 情况1：在文件列范围内，直接读取实际数据
					input_matrix[i][j] = value;
				} else if (file_cols > 0) {
					// 情况2：超出文件列范围，循环复制已有列数据
					// 使用模运算实现循环：j=256时复制j=0的数据(256%128=0)
					input_matrix[i][j] = input_matrix[i][j % file_cols];
				} else {
					// 情况3：文件列数为0（异常情况），填充0
					input_matrix[i][j] = 0.0f;
				}
			}
		}
		
		// Step 1.6: 如果行数不足512，循环复制已有行
		for (int i = file_rows; i < matrix_size; i++) {
			input_matrix[i].resize(matrix_size);
			for (int j = 0; j < matrix_size; j++) {
				if (file_rows > 0) {
					// 循环复制：第128行复制第0行，第256行再次复制第0行
					// 保证数据分布的周期性重复
					input_matrix[i][j] = input_matrix[i % file_rows][j];
				} else {
					// 异常情况：无有效行数据
					input_matrix[i][j] = 0.0f;
				}
			}
		}
		query_file.close();
		
		// Step 1.7: 加载Key矩阵（流程与Query相同）
		std::ifstream key_file(input_dir + "llama_key_128x4096.txt");
		if (!key_file.is_open()) {
			// 如果128x4096文件不存在，尝试2048x128版本
			key_file.open(input_dir + "llama_key_2048x128.txt");
			if (!key_file.is_open()) {
				std::cerr << "Failed to open key file from " << input_dir << std::endl;
				return false;
			}
		}
		
		// Read dimensions from first line
		key_file >> file_rows >> file_cols;
		std::cerr << "Loading Key matrix: " << file_rows << "x" << file_cols << std::endl;
		
		// Skip rest of first line
		std::getline(key_file, dummy);
		
		/**
		 * Key矩阵循环填充
		 * ==================
		 * 与Query矩阵采用相同的循环填充策略
		 * 保证Query-Key对应位置的数据扩展方式一致
		 * 这对Attention计算的正确性至关重要
		 */
		for (int i = 0; i < matrix_size && i < file_rows; i++) {
			query_weight_matrix[i].resize(matrix_size);
			std::string line;
			if (!std::getline(key_file, line)) break;
			
			std::istringstream iss(line);
			for (int j = 0; j < matrix_size; j++) {
				float value;
				if (j < file_cols && (iss >> value)) {
					// 直接存储文件中的原始数据
					query_weight_matrix[i][j] = value;
				} else if (file_cols > 0) {
					// 列循环填充：第j列 = 第(j % file_cols)列
					// 保持Key与Query的对齐关系
					query_weight_matrix[i][j] = query_weight_matrix[i][j % file_cols];
				} else {
					query_weight_matrix[i][j] = 0.0f;
				}
			}
		}
		
		// 行循环填充：确保512行完整数据
		for (int i = file_rows; i < matrix_size; i++) {
			query_weight_matrix[i].resize(matrix_size);
			for (int j = 0; j < matrix_size; j++) {
				if (file_rows > 0) {
					query_weight_matrix[i][j] = query_weight_matrix[i % file_rows][j];
				} else {
					query_weight_matrix[i][j] = 0.0f;
				}
			}
		}
		key_file.close();
		
		// Initialize value and output tables to zero
		for (int i = 0; i < matrix_size; i++) {
			Q_resOutput_matrix[i].resize(matrix_size, 0.0f);
			Q_resOutput_matrix[i].resize(matrix_size, 0.0f);
		}
		
		std::cerr << "Successfully loaded real LLaMA matrices!" << std::endl;
		return true;
		
	} catch (const std::exception& e) {
		std::cerr << "Error loading LLaMA matrices: " << e.what() << std::endl;
		return false;
	}
}
void LLMMACnet::setInputMatrices(const vector<vector<float>>& X_input, 
                                const vector<vector<float>>& Wq) {
	// 直接设置输入矩阵
	input_matrix = X_input;           // 8×4096
	query_weight_matrix = Wq;         // 128×4096
	
	// 计算Q矩阵作为输出: Q = X @ Wq^T (8×128)
	Q_resOutput_matrix.resize(input_sequence_length);
	for (int i = 0; i < input_sequence_length; i++) {
		Q_resOutput_matrix[i].resize(query_output_dim, 0.0f);
		for (int j = 0; j < query_output_dim; j++) {
			float sum = 0.0f;
			for (int k = 0; k < input_hidden_dim; k++) {
				sum += input_matrix[i][k] * query_weight_matrix[j][k];
			}
			Q_resOutput_matrix[i][j] = sum;
		}
	}
	
	std::cout << "[LLMMACnet] Input matrices set: X_input(" << input_matrix.size() 
	          << "x" << (input_matrix.empty() ? 0 : input_matrix[0].size()) 
	          << "), Wq(" << query_weight_matrix.size() 
	          << "x" << (query_weight_matrix.empty() ? 0 : query_weight_matrix[0].size()) 
	          << "), Q_output(" << Q_resOutput_matrix.size()
	          << "x" << (Q_resOutput_matrix.empty() ? 0 : Q_resOutput_matrix[0].size()) 
	          << ")" << std::endl;
}

bool LLMMACnet::llmReadSavedMatrix() {

	// Try to load real matrices from llminput directory
	// Eclipse runs from project root, command line from Debug folder
	std::string input_dir = "src/Input/llminput/";
	
	// Check if running from Debug directory
	std::ifstream test_from_debug("../src/Input/llminput/X_input.txt");
	if (test_from_debug.is_open()) {
		input_dir = "../src/Input/llminput/";
		test_from_debug.close();
	}
	std::string x_input_file = input_dir + "X_input.txt";
	std::string wq_file = input_dir + "Wq.txt";
	
	// Check if files exist
	std::ifstream test_x(x_input_file);
	std::ifstream test_wq(wq_file);
	
	if (test_x.is_open() && test_wq.is_open()) {
		test_x.close();
		test_wq.close();
		
		
		// Load X_input (8x4096)
		input_matrix.resize(input_sequence_length); //变成8行
		std::ifstream x_file(x_input_file);
		if (!x_file.is_open()) {
			std::cerr << "FATAL ERROR: Cannot open X_input file: " << x_input_file << std::endl;
			std::exit(1);
		}
		
		for (int i = 0; i < input_sequence_length; i++) {
			input_matrix[i].resize(input_hidden_dim);
			for (int j = 0; j < input_hidden_dim; j++) {
				if (!(x_file >> input_matrix[i][j])) {
					std::cerr << "FATAL ERROR: Failed to read X_input[" << i << "][" << j 
					          << "] from " << x_input_file << std::endl;
					std::cerr << "Expected " << input_sequence_length << "x" << input_hidden_dim 
					          << " = " << (input_sequence_length * input_hidden_dim) << " values" << std::endl;
					std::cerr << "Only read " << (i * input_hidden_dim + j) << " values" << std::endl;
					x_file.close();
					std::exit(1);
				}
			}
		}
		x_file.close();
		std::cout << "Successfully loaded X_input matrix (" << input_sequence_length 
		          << "x" << input_hidden_dim << ")" << std::endl;
		
		// Load Wq (128x4096)
		query_weight_matrix.resize(query_output_dim);
		std::ifstream wq_stream(wq_file);
		if (!wq_stream.is_open()) {
			std::cerr << "FATAL ERROR: Cannot open Wq file: " << wq_file << std::endl;
			std::exit(1);
		}
		
		for (int i = 0; i < query_output_dim; i++) {
			query_weight_matrix[i].resize(input_hidden_dim);
			for (int j = 0; j < input_hidden_dim; j++) {
				if (!(wq_stream >> query_weight_matrix[i][j])) {
					std::cerr << "FATAL ERROR: Failed to read Wq[" << i << "][" << j 
					          << "] from " << wq_file << std::endl;
					std::cerr << "Expected " << query_output_dim << "x" << input_hidden_dim 
					          << " = " << (query_output_dim * input_hidden_dim) << " values" << std::endl;
					std::cerr << "Only read " << (i * input_hidden_dim + j) << " values" << std::endl;
					wq_stream.close();
					std::exit(1);
				}
			}
		}
		wq_stream.close();
		std::cout << "Successfully loaded Wq matrix (" << query_output_dim 
		          << "x" << input_hidden_dim << ")" << std::endl;
		
		// Compute Q = X @ Wq^T
		Q_resOutput_matrix.resize(input_sequence_length);
		for (int i = 0; i < input_sequence_length; i++) {
			Q_resOutput_matrix[i].resize(query_output_dim, 0.0f);
			for (int j = 0; j < query_output_dim; j++) {
				float sum = 0.0f;
				for (int k = 0; k < input_hidden_dim; k++) {
					sum += input_matrix[i][k] * query_weight_matrix[j][k];
				}
				Q_resOutput_matrix[i][j] = sum;
			}
		}
		
		// Print verification values
		std::cout << "Loaded real matrices from " << input_dir << std::endl;
		std::cout << "First 5 values from X_input[0]: ";
		for (int i = 0; i < 5 && i < input_hidden_dim; i++) {
			std::cout << input_matrix[0][i] << " ";
		}
		std::cout << std::endl;
		
		// Debug: Print Q_resOutput_matrix after computation
		std::cout << "Q_resOutput_matrix after initialization (first 3x3):" << std::endl;
		for (int i = 0; i < 3 && i < input_sequence_length; i++) {
			for (int j = 0; j < 3 && j < query_output_dim; j++) {
				std::cout << "Q[" << i << "][" << j << "]=" << Q_resOutput_matrix[i][j] << " ";
			}
			std::cout << std::endl;
		}
		
		std::cout << "First 5 values from Wq[0]: ";
		for (int i = 0; i < 5 && i < input_hidden_dim; i++) {
			std::cout << query_weight_matrix[0][i] << " ";
		}
		std::cout << std::endl;
		return true;  // Successfully loaded from files
	} else {
		// Close any open files
		if (test_x.is_open()) test_x.close();
		if (test_wq.is_open()) test_wq.close();
		
		std::cout << "Matrix files not found at " << input_dir << std::endl;
		assert( false && "Matrix files must be readed");
	}
		return false;  // Failed to load from files
}

void LLMMACnet::llmGenerateAllTasks() {
	all_tasks.clear();
	
	// Step 2.1: 计算任务参数 - 使用8×128维度
	int task_id = 0;

	int  total_task_slicedPixels =  matrixOutputPixels_inputsequencelength * matrixOutputPixels_queryoutputdim  * this->tasks_per_pixel;       // 总共 4096个任务 //matrixOutputPixels_inputsequencelength = 8;  // 8 rows output matrix (from X_input rows) 	matrixOutputPixels_queryoutputdim = 128;
	
	all_tasks.reserve( total_task_slicedPixels);
	// Step 2.2: 根据像素生成任务

	// 使用构造函数中设置的pixels_to_test值
	int pixel_count = 0;
	// Step 2.3: 遍历所有像素位置 - 使用8×128维度
	for (int pixel_y = 0; pixel_y < input_sequence_length && pixel_count < total_task_slicedPixels; pixel_y++) {
		for (int pixel_x = 0; pixel_x < query_output_dim && pixel_count < total_task_slicedPixels; pixel_x++) {
			int pixel_id = pixel_y * query_output_dim + pixel_x;  // 计算像素的线性ID
			pixel_count++;
			// Step 2.4: 为每个像素生成 N 个子任务 (N = tasks_per_pixel)
			for (int subchunk_id = 0; subchunk_id < this->tasks_per_pixel; subchunk_id++) {
				LLMTask task;
				task.task_id = pixel_id * this->tasks_per_pixel + subchunk_id;  // 全局任务ID = 像素ID*N + 子块ID
				task.pixel_id = pixel_id;                   // 所属像素ID
				task.pixel_x = pixel_x;                     // 像素X坐标
				task.pixel_y = pixel_y;                     // 像素Y坐标
				task.time_slice = subchunk_id;              // 时间片 = 子块ID
				task.subchunk_id = subchunk_id;             // 子块ID (0-3)

				int elements_per_chunk = input_hidden_dim / this->tasks_per_pixel;  // 4096/64 = 64
				task.input_offset = subchunk_id * elements_per_chunk;  // 0, 1024, 2048, 3072
				task.query_offset = subchunk_id * elements_per_chunk;  // 同样的偏移
				
				// Step 2.6: 从矩阵中提取Input和Query元素
				task.input_data.clear();
				task.query_data.clear();
				task.input_data.reserve(elements_per_chunk);  // 预分配空间
				task.query_data.reserve(elements_per_chunk);
				
				/**
				 * @brief 从大矩阵提取子块数据 - 详细维度解析
				 * 
				 * 矩阵维度说明：
				 * ==============
				 * 
				 * 1. 源矩阵（input_matrix/query_weight_matrix）：
				 *    - 维度：512×512
				 *    - 总元素：262,144
				 *    - 索引：[row][col]，范围 [0-511][0-511]
				 * 
				 * 2. 目标向量（task.query_data/task.query_data）：
				 *    - 维度：64×1（一维向量）
				 *    - 总元素：64
				 *    - 索引：[0-63]
				 * 
				 * 3. 像素坐标映射：
				 *    - pixel_x, pixel_y：范围 [0-511]
				 *    - 每个像素对应输出矩阵的一个位置
				 *    - 像素(x,y)需要计算128×128的点积
				 * 
				 * 4. 子块划分（128×128 → 4个64×64）：
				 *    - subchunk_id=0: Query[0:63] × Key[0:63]     (左上)
				 *    - subchunk_id=1: Query[0:63] × Key[64:127]   (右上)
				 *    - subchunk_id=2: Query[64:127] × Key[0:63]   (左下)
				 *    - subchunk_id=3: Query[64:127] × Key[64:127] (右下)
				 * 
				 * 数据提取示例（权重模式）：
				 * ========================
				 * 
				 * 假设：pixel_x=10, pixel_y=20, subchunk_id=2
				 * - query_offset = (2/2)*64 = 64
				 * - input_offset = (2%2)*64 = 0
				 * 
				 * 循环i从0到63：
				 * i=0时：
				 *   weight_query_idx = (10 + 64 + 0) % 512 = 74
				 *   weight_key_idx = (10 + 0 + 0) % 512 = 10
				 *   task.query_data[0] = input_matrix[20][74]
				 *   task.query_data[0] = query_weight_matrix[20][10]
				 * 
				 * i=63时：
				 *   weight_query_idx = (10 + 64 + 63) % 512 = 137
				 *   weight_key_idx = (10 + 0 + 63) % 512 = 73
				 *   task.query_data[63] = input_matrix[20][137]
				 *   task.query_data[63] = query_weight_matrix[20][73]
				 * 
				 * 矩阵访问模式：
				 * =============
				 * input_matrix[pixel_y][weight_query_idx]
				 *                       ↑        ↑
				 *                    行索引    列索引
				 *                   (0-511)   (0-511)
				 * 
				 * - 行索引(pixel_y)：决定从哪一行提取数据
				 * - 列索引(weight_query_idx)：决定从该行的哪些列提取64个元素
				 * - 模运算：确保索引不越界，实现循环访问
				 */
				
				// Step 2.7: 提取具体数据 - 使用Input和Query矩阵
				// 对于8×4096的input和128×4096的query，我们需要：
			//	 pixel_y ∈ [0, 7]    → 对应输出矩阵 Q 的行索引
			//	  pixel_x ∈ [0, 127]  → 对应输出矩阵 Q 的列索引

				for (int i = 0; i < elements_per_chunk; i++) {
					// 从input_matrix[pixel_y]提取数据 (第pixel_y行)
					int input_idx = task.input_offset + i;
					if (input_idx < input_hidden_dim && pixel_y < input_sequence_length) {
						task.input_data.push_back(input_matrix[pixel_y][input_idx]); //input_matrix[pixel_y][input_idx] - 正确，取 X 的第 pixel_y 行 x是8行4096列。 这一行分成很多小短行，一行是64列。

					} else {
						// 索引超出范围，使用assert报错
						std::cerr << "ERROR: Index out of bounds - input_idx=" << input_idx 
						          << " (max=" << input_hidden_dim << "), pixel_y=" << pixel_y 
						          << " (max=" << input_sequence_length << ")" << std::endl;
						assert(false && "Input matrix index out of bounds - no padding supported in LLM mode");
					}
					
					// 从query_weight_matrix[pixel_x]提取数据 (第pixel_x行)
					int query_idx = task.query_offset + i;
					if (query_idx < input_hidden_dim && pixel_x < query_output_dim) {
						task.query_data.push_back(query_weight_matrix[pixel_x][query_idx]);
					} else {
						// 索引超出范围，使用assert报错
						std::cerr << "ERROR: Index out of bounds - input_idx=" << input_idx 
						          << " (max=" << input_hidden_dim << "), pixel_y=" << pixel_y 
						          << " (max=" << input_sequence_length << ")" << std::endl;
						assert(false && "Input matrix index out of bounds - no padding supported in LLM mode");
					}
				}
				// 打印Input和Query数据（用于debug）
				// 只打印第一个任务，且只打印一次
				if (task.task_id == 0 && cycles < 3) {
					std::cout << "\n llmmacnet === Task " << task.task_id << " Input data (first 8 values) ===" << std::endl;
					for (int i = 0; i < std::min(100, (int)task.input_data.size()); i++) {
						std::cout << std::fixed << std::setprecision(3) 
						         << std::setw(8) << task.input_data[i] << " ";
					}
					std::cout << std::endl;
					
					std::cout << "\n=== Task " << task.task_id << " Query weights (first 8 values) ===" << std::endl;
					for (int i = 0; i < std::min(20, (int)task.query_data.size()); i++) {
						std::cout << std::fixed << std::setprecision(3) 
						         << std::setw(8) << task.query_data[i] << " ";
					}
					std::cout << std::endl;
				}
				all_tasks.push_back(task);
			}
		}
	}
	// Update total_task_slicedPixels to reflect actual number of tasks
	assert(total_task_slicedPixels == all_tasks.size() && "Total sliced pixels must equal number of tasks");
}

void LLMMACnet::llmInitializeRandomMatrices() {
	// Initialize with random data
	srand(0);
	input_matrix.resize(input_sequence_length);
	query_weight_matrix.resize(query_output_dim);
	Q_resOutput_matrix.resize(input_sequence_length);

	// Initialize input matrix with random values
	for (int i = 0; i < input_sequence_length; i++) {
		input_matrix[i].resize(input_hidden_dim);
		for (int j = 0; j < input_hidden_dim; j++) {
			input_matrix[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
		}
		Q_resOutput_matrix[i].resize(query_output_dim, 0.0f);
	}

	// Initialize query weight matrix with random values
	for (int i = 0; i < query_output_dim; i++) {
		query_weight_matrix[i].resize(input_hidden_dim);
		for (int j = 0; j < input_hidden_dim; j++) {
			query_weight_matrix[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
		}
	}

	// Compute Q = X @ Wq^T
	for (int i = 0; i < input_sequence_length; i++) {
		for (int j = 0; j < query_output_dim; j++) {
			float sum = 0.0f;
			for (int k = 0; k < input_hidden_dim; k++) {
				sum += input_matrix[i][k] * query_weight_matrix[j][k];
			}
			Q_resOutput_matrix[i][j] = sum;
		}
	}

	// Debug: Print Q_resOutput_matrix after computation
	std::cout << "Q_resOutput_matrix after random initialization (first 3x3):" << std::endl;
	for (int i = 0; i < 3 && i < input_sequence_length; i++) {
		for (int j = 0; j < 3 && j < query_output_dim; j++) {
			std::cout << "Q[" << i << "][" << j << "]=" << Q_resOutput_matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}
// 辅助函数
bool LLMMACnet::llmIsMemoryNode(int node_id) {
	for (int i = 0; i < MEM_NODES; i++) {
		if (dest_list[i] == node_id) {
			return true;
		}
	}
	return false;
}

void LLMMACnet::llmXMapping(int total_pixels) {
	// 计算总任务数（像素数 * N）
	this->llmOutputPixelMappingTable.clear();
	this->llmOutputPixelMappingTable.resize(macNum);
	this->llmTaskMappingTable.clear();
	this->llmTaskMappingTable.resize(macNum);
	vector<int> available_macs;
	for (int i = 0; i < macNum; i++) {
		int ni_id = i % TOT_NUM;
		if (!llmIsMemoryNode(ni_id)) {
			available_macs.push_back(i);
		}
	}
	// 像素级轮询分配，每个像素的N个task分配到同一节点
	for (int pixel_id = 0; pixel_id < total_pixels; pixel_id++) {
		int mac_id = available_macs[pixel_id % available_macs.size()];
		// 记录像素分配
		this->llmOutputPixelMappingTable[mac_id].push_back(pixel_id);
		// 该像素的N个task(subchunk)都分配给同一个节点（便于聚合）
		for (int subchunk_id = 0; subchunk_id < this->tasks_per_pixel; subchunk_id++) {
			int task_id = pixel_id * this->tasks_per_pixel + subchunk_id;  // Fixed mapping formula
			this->llmTaskMappingTable[mac_id].push_back(task_id);
		}
	}
}
void LLMMACnet::llmCheckStatus() {
	static int status_check_count = 0;
	status_check_count++;
	// Progress reporting at different levels
	if (ready_flag == 0) {
		if (mapping_again == 0) {
			this->vcNetwork->resetVNRoundRobin();
		}
		// SAMOS mapping logic for LLM (pixel-based)
		#ifdef YZSAMOSSampleMapping
		// Calculate how many pixels per MAC for sampling window
		int available_macs = macNum - MEM_NODES;  // Exclude memory nodes
		int total_pixels = total_task_slicedPixels / this->tasks_per_pixel;  // Convert tasks to pixels (4 tasks per pixel)
		
		if (total_pixels / available_macs < samplingWindowLength) {
			// If pixels are fewer than sampling window, use normal row mapping
			LLM_DEBUG("[SAMOS] Layer has fewer pixels than sampling window!");
			LLM_DEBUG("  Total pixels: " << total_pixels << ", Available MACs: " << available_macs);
			LLM_DEBUG("  Pixels per MAC: " << (total_pixels / available_macs) << " < " << samplingWindowLength);
			LLM_DEBUG("  Using row mapping instead of SAMOS");
			this->llmXMapping(total_pixels);
		} else {
			if (mapping_again == 0) {
				// First phase: run sampling window (pixel-based)
				int sampling_pixels = available_macs * samplingWindowLength;
				LLM_DEBUG("[SAMOS] Starting sampling phase");
				LLM_DEBUG("  Sampling pixels: " << sampling_pixels << " (" << available_macs 
				          << " MACs * " << samplingWindowLength << " window)");
				LLM_DEBUG("  This generates " << (sampling_pixels * this->tasks_per_pixel) << " tasks");
				
				// Reset sampling statistics
				std::fill_n(samplingWindowDelay, TOT_NUM, 0);
				
				// Map sampling window pixels using row mapping
				this->llmXMapping(sampling_pixels);
				mapping_again = 1;  // Mark that sampling is being done
				
			} else if (mapping_again == 2) {
				// Second phase: map remaining pixels based on sampling results
				int sampling_pixels = available_macs * samplingWindowLength;
				int remaining_pixels = total_pixels - sampling_pixels;
				int remaining_tasks = remaining_pixels * this->tasks_per_pixel;
				
				std::cout << "[SAMOS DEBUG] Phase 2 mapping:" << std::endl;
				std::cout << "  Total pixels: " << total_pixels << std::endl;
				std::cout << "  Available MACs: " << available_macs << std::endl;
				std::cout << "  Sampling window: " << samplingWindowLength << std::endl;
				std::cout << "  Sampling pixels: " << sampling_pixels << std::endl;
				std::cout << "  Remaining pixels: " << remaining_pixels << std::endl;
				std::cout << "  Remaining tasks: " << remaining_tasks << std::endl;
				std::cout << "  Current packet_id: " << packet_id << std::endl;
				
				// Update packet_id based on sampling phase tasks
				packet_id = packet_id + sampling_pixels * this->tasks_per_pixel;
				
				LLM_DEBUG("[SAMOS] Applying SAMOS mapping for remaining pixels");
				LLM_DEBUG("  Remaining pixels: " << remaining_pixels);
				
				// Use SAMOS mapping based on latency measurements
				int start_pixel_id = sampling_pixels;
				std::cout << "[SAMOS DEBUG] Pixel IDs will range from " << start_pixel_id 
				          << " to " << (start_pixel_id + remaining_pixels - 1) << std::endl;
				std::cout << "[SAMOS DEBUG] Task IDs will range from " << (start_pixel_id * this->tasks_per_pixel) 
				          << " to " << ((start_pixel_id + remaining_pixels) * this->tasks_per_pixel - 1) << std::endl;
				
				this->llmSAMOSTaskMapping(remaining_pixels, start_pixel_id);
				
				LLM_DEBUG("[SAMOS] Second phase mapping complete");
				mapping_again = 0;  // Reset for next layer
			} else {
				LLM_INFO("[SAMOS] ERROR: Invalid mapping_again state: " << mapping_again);
			}
		}
		#endif
		// Normal mapping without SAMOS
		#ifdef rowmapping
		int total_pixels = total_task_slicedPixels / this->tasks_per_pixel;  // Convert tasks to pixels
		this->llmXMapping(total_pixels);  // Pass pixel count, not task count
		#endif

		int active_macs = 0;
		for (int i = 0; i < macNum; i++) {
			if (llmTaskMappingTable[i].size() == 0) {
				this->LLMMAC_list[i]->selfstatus = 5;
				this->LLMMAC_list[i]->send = 3;
			} else {
				// 分配task IDs而不是pixel IDs
				this->LLMMAC_list[i]->llmPEExpectedtasktable.assign(
					llmTaskMappingTable[i].begin(), llmTaskMappingTable[i].end());
				active_macs++;
			}
		}
		ready_flag = 1;
		return;
	}

	int finished_count = 0;
	int active_count = 0;
	int assigned_macs = 0;
	
	for (int i = 0; i < macNum; i++) {
		// Only count MACs that actually have tasks assigned
		if (llmTaskMappingTable[i].size() > 0) {
			assigned_macs++;
			if (LLMMAC_list[i]->selfstatus == 5 && LLMMAC_list[i]->send == 3) {
				finished_count++;
			} else if (LLMMAC_list[i]->selfstatus != 5) {
				active_count++;
			}
		}
	}
	// Complete when no MACs are active (all have finished their tasks)
	// Add a delay to ensure messages are delivered through the network
	static int completion_wait_cycles = 0;
	
	if (assigned_macs > 0 && active_count == 0) {
		completion_wait_cycles++;
		
		// Wait for 100 cycles after all MACs finish to ensure messages are delivered
		// This is sufficient since we're directly updating the output table
		if (completion_wait_cycles < 100) {
			return;
		}
		
		#ifdef YZSAMOSSampleMapping
		// Check if we just completed sampling phase
		if (mapping_again == 1) {
			// Sampling phase complete, now do SAMOS mapping for remaining tasks
			std::cout << "[LLM-SAMOS] Sampling phase complete at cycle " << cycles << std::endl;
			std::cout << "  Collected latency data, now applying SAMOS mapping" << std::endl;
			
			// Reset for second phase
			completion_wait_cycles = 0;
			ready_flag = 0;
			mapping_again = 2;  // Move to SAMOS mapping phase
			return;
		}
		#endif
		

		// Print timing statistics
		llmPrintTimingStatistics();

		layer_latency.push_back(cycles);
		ready_flag = 2;
		
		// Adjust packet_id for next layer based on actual tasks processed
		#ifdef YZSAMOSSampleMapping
		int available_macs = macNum - MEM_NODES;
		int total_pixels = total_task_slicedPixels;
		if (total_pixels / available_macs < samplingWindowLength) {
			// Used normal mapping, add all tasks (pixels * 4)
			packet_id = packet_id + total_pixels * 4;
		} else {
			// Used SAMOS mapping, already adjusted during mapping
			// No need to adjust here as it was done incrementally
		}
		#else
		// Normal mapping: total_task_slicedPixels now represents pixels, so multiply by 4 for actual tasks
		packet_id = packet_id + total_task_slicedPixels;
		#endif
		last_layer_packet_id = packet_id;
		return;
	}

	ready_flag = 1;
}


int LLMMACnet::llmSAMOSTaskMapping(int pixel_count, int start_pixel_id) {
	// Clear and prepare mapping tables
	this->llmOutputPixelMappingTable.clear();
	this->llmOutputPixelMappingTable.resize(macNum);
	this->llmTaskMappingTable.clear();
	this->llmTaskMappingTable.resize(macNum);
	// 1) Collect compute nodes (exclude memory nodes)
	std::vector<int> pe_ids;
	pe_ids.reserve(macNum);
	for (int id = 0; id < macNum; ++id) {
		if (!contains(dest_list, id))
			pe_ids.push_back(id);
	}
	if (pe_ids.empty() || pixel_count <= 0)
		return 0;

	// 2) Calculate average latency for each node (from sampling window)
	double sum_lat = 0.0;
	int nz = 0;
	for (int id : pe_ids) {
		double lat = double(samplingWindowDelay[id]) / std::max(1, samplingWindowLength);
		if (lat > 0.0) {
			sum_lat += lat;
			++nz;
		}
	}
	const double default_lat = (nz > 0) ? (sum_lat / nz) : 1.0;
	const double eps = 1e-12;

	struct NodeW {
		int id;
		double w;     // Weight = 1/latency
		double want;  // Ideal allocation
		int alloc;    // Actual integer allocation
		double frac;  // Fractional remainder
	};

	std::vector<NodeW> nodes;
	nodes.reserve(pe_ids.size());

	double sumW = 0.0;
	for (int id : pe_ids) {
		double lat = double(samplingWindowDelay[id]) / std::max(1, samplingWindowLength);
		if (lat <= 0.0)
			lat = default_lat;
		double w = 1.0 / (lat + eps);
		nodes.push_back({id, w, 0.0, 0, 0.0});
		sumW += w;
	}

	if (sumW <= 0.0) { // Extreme fallback: uniform distribution
		int base = pixel_count / int(nodes.size());
		int rem = pixel_count - base * int(nodes.size());
		int current_pixel_id = start_pixel_id;
		for (auto &n : nodes) {
			for (int k = 0; k < base; ++k) {
				// 记录像素分配
				this->llmOutputPixelMappingTable[n.id].push_back(current_pixel_id);
				// 该像素的4个task都分配给同一个节点
				for (int chunk_id = 0; chunk_id < this->tasks_per_pixel; chunk_id++) {
					int task_id = current_pixel_id * this->tasks_per_pixel + chunk_id;
					this->llmTaskMappingTable[n.id].push_back(task_id);
				}
				current_pixel_id++;
			}
		}
		for (int i = 0; i < rem; ++i) {
			this->llmOutputPixelMappingTable[nodes[i].id].push_back(current_pixel_id);
			for (int chunk_id = 0; chunk_id < this->tasks_per_pixel; chunk_id++) {
				int task_id = current_pixel_id * this->tasks_per_pixel + chunk_id;
				this->llmTaskMappingTable[nodes[i].id].push_back(task_id);
			}
			current_pixel_id++;
		}
		return 0;
	}

	// 3) Hamilton's method (largest remainder)
	int allocated = 0;
	for (auto &n : nodes) {
		double exact = pixel_count * (n.w / sumW);
		n.want = exact;
		n.alloc = int(std::floor(exact));
		n.frac = exact - n.alloc;
		allocated += n.alloc;
	}
	int remainder = pixel_count - allocated;

	// Allocate remaining pixels to nodes with largest fractional parts
	std::sort(nodes.begin(), nodes.end(), [](const NodeW &a, const NodeW &b) {
		return a.frac > b.frac;
	});
	for (int i = 0; i < remainder; ++i)
		nodes[i % nodes.size()].alloc++;

	// 4) Generate pixel and task mapping
	int current_pixel_id = start_pixel_id;
	for (auto &n : nodes) {
		for (int pixel_idx = 0; pixel_idx < n.alloc; ++pixel_idx) {
			// 记录像素分配
			this->llmOutputPixelMappingTable[n.id].push_back(current_pixel_id);

			// 每个像素生成4/64/,,,个任务，都分配给同一个节点（便于聚合）
			for (int chunk_id = 0; chunk_id < this->tasks_per_pixel; chunk_id++) {
				int task_id = current_pixel_id * this->tasks_per_pixel + chunk_id;
				this->llmTaskMappingTable[n.id].push_back(task_id);
			}
			current_pixel_id++;
		}
	}


	return 0;
}


void LLMMACnet::llmPrintTimingStatistics() {
	std::cout << "\n=== Task Timing Statistics ===" << std::endl;
	
	// Collect timing data from all MACs
	std::vector<int> all_request_travel_times;
	std::vector<int> all_response_travel_times;
	std::vector<int> all_compute_times;
	std::vector<int> all_result_travel_times;
	std::vector<int> all_total_times;
	std::vector<int> all_request_hops;
	std::vector<int> all_response_hops;
	std::vector<int> all_result_hops;
	
	// Per-MAC statistics with tracking of maximum
	std::cout << "\n--- Per-MAC Timing Statistics ---" << std::endl;
	
	// Variables to track the MAC with maximum average total time
	int max_avg_mac_id = -1;
	float max_avg_total_time = 0;
	float max_avg_req_travel = 0;
	float max_avg_resp_travel = 0;
	float max_avg_compute = 0;
	float max_avg_req_hops = 0;
	float max_avg_resp_hops = 0;
	float max_avg_res_hops = 0;
	int max_avg_ni_id = 0;
	int max_avg_task_count = 0;
	
	for (int i = 0; i < macNum; i++) {
		if (LLMMAC_list[i]->task_timings.size() == 0) continue;
		
		int mac_req_travel = 0, mac_resp_travel = 0, mac_comp_total = 0;
		int mac_req_hops = 0, mac_resp_hops = 0, mac_res_hops = 0;
		int task_count = LLMMAC_list[i]->task_timings.size();
		
		for (const auto& timing : LLMMAC_list[i]->task_timings) {
			// Request travel time = arrival at memory - send from MAC
			int req_travel = timing.request_arrive_cycle - timing.request_send_cycle;
			// Response travel time = arrival at MAC - send from memory
			int resp_travel = timing.response_arrive_cycle - timing.response_send_cycle;
			// Compute time
			int comp_time = timing.compute_end_cycle - timing.compute_start_cycle;
			// Total end-to-end time
			int total_time = timing.compute_end_cycle - timing.request_send_cycle;
			
			mac_req_travel += req_travel;
			mac_resp_travel += resp_travel;
			mac_comp_total += comp_time;
			mac_req_hops += timing.request_hops;
			mac_resp_hops += timing.response_hops;
			mac_res_hops += timing.result_hops;
			
			all_request_travel_times.push_back(req_travel);
			all_response_travel_times.push_back(resp_travel);
			all_compute_times.push_back(comp_time);
			all_total_times.push_back(total_time);
			all_request_hops.push_back(timing.request_hops);
			all_response_hops.push_back(timing.response_hops);
			all_result_hops.push_back(timing.result_hops);
		}
		
		if (task_count > 0) {
			float avg_total = (float)(mac_req_travel + mac_resp_travel + mac_comp_total)/task_count;
			
			// Check if this MAC has the maximum average total time
			if (avg_total > max_avg_total_time) {
				max_avg_total_time = avg_total;
				max_avg_mac_id = i;
				max_avg_req_travel = (float)mac_req_travel/task_count;
				max_avg_resp_travel = (float)mac_resp_travel/task_count;
				max_avg_compute = (float)mac_comp_total/task_count;
				max_avg_req_hops = (float)mac_req_hops/task_count;
				max_avg_resp_hops = (float)mac_resp_hops/task_count;
				max_avg_res_hops = (float)mac_res_hops/task_count;
				max_avg_ni_id = LLMMAC_list[i]->NI_id;
				max_avg_task_count = task_count;
			}
			
			std::cout << "MAC " << i << " (NI_id=" << LLMMAC_list[i]->NI_id 
			          << ", Tasks: " << task_count << "):" << std::endl;
			std::cout << "  Request Packet: Travel=" << (float)mac_req_travel/task_count 
			          << " cycles, Hops=" << (float)mac_req_hops/task_count << std::endl;
			std::cout << "  Response Packet: Travel=" << (float)mac_resp_travel/task_count 
			          << " cycles, Hops=" << (float)mac_resp_hops/task_count << std::endl;
			std::cout << "  Computation: " << (float)mac_comp_total/task_count << " cycles" << std::endl;
			std::cout << "  Result Packet: Hops=" << (float)mac_res_hops/task_count 
			          << " (travel time not tracked due to NoC issue)" << std::endl;
			std::cout << "  Total End-to-End: " << avg_total << " cycles" << std::endl;
		}
	}
	
	// Print MAC with maximum average total time
	if (max_avg_mac_id != -1) {
		std::cout << "\n*** MAC WITH MAXIMUM AVERAGE TOTAL TIME ***" << std::endl;
		std::cout << "MAC ID: " << max_avg_mac_id << " (NI_id=" << max_avg_ni_id << ")" << std::endl;
		std::cout << "Position: (" << (max_avg_ni_id % X_NUM) << ", " << (max_avg_ni_id / X_NUM) << ")" << std::endl;
		std::cout << "Tasks Processed: " << max_avg_task_count << std::endl;
		std::cout << "\nTiming Breakdown:" << std::endl;
		std::cout << "  Average Total Time: " << max_avg_total_time << " cycles" << std::endl;
		std::cout << "  - Request Travel: " << max_avg_req_travel << " cycles (Hops: " << max_avg_req_hops << ")" << std::endl;
		std::cout << "  - Response Travel: " << max_avg_resp_travel << " cycles (Hops: " << max_avg_resp_hops << ")" << std::endl;
		std::cout << "  - Computation: " << max_avg_compute << " cycles" << std::endl;
		std::cout << "  - Result Hops: " << max_avg_res_hops << std::endl;
		std::cout << "\nPercentage Breakdown:" << std::endl;
		std::cout << "  Request: " << (max_avg_req_travel * 100.0 / max_avg_total_time) << "%" << std::endl;
		std::cout << "  Response: " << (max_avg_resp_travel * 100.0 / max_avg_total_time) << "%" << std::endl;
		std::cout << "  Compute: " << (max_avg_compute * 100.0 / max_avg_total_time) << "%" << std::endl;
		std::cout << "  Queueing/Other: " << ((max_avg_total_time - max_avg_req_travel - max_avg_resp_travel - max_avg_compute) * 100.0 / max_avg_total_time) << "%" << std::endl;
	}
	
	// Network-wide statistics
	std::cout << "\n--- Network-wide Timing Statistics ---" << std::endl;
	
	if (all_request_travel_times.size() > 0) {
		int total_req_travel = 0, total_resp_travel = 0, total_comp = 0, total_all = 0;
		int total_req_hops = 0, total_resp_hops = 0, total_res_hops = 0;
		int min_req = INT_MAX, min_resp = INT_MAX, min_comp = INT_MAX, min_total = INT_MAX;
		int max_req = 0, max_resp = 0, max_comp = 0, max_total = 0;
		
		for (size_t i = 0; i < all_request_travel_times.size(); i++) {
			total_req_travel += all_request_travel_times[i];
			total_resp_travel += all_response_travel_times[i];
			total_comp += all_compute_times[i];
			total_all += all_total_times[i];
			total_req_hops += all_request_hops[i];
			total_resp_hops += all_response_hops[i];
			total_res_hops += all_result_hops[i];
			
			min_req = std::min(min_req, all_request_travel_times[i]);
			min_resp = std::min(min_resp, all_response_travel_times[i]);
			min_comp = std::min(min_comp, all_compute_times[i]);
			min_total = std::min(min_total, all_total_times[i]);
			
			max_req = std::max(max_req, all_request_travel_times[i]);
			max_resp = std::max(max_resp, all_response_travel_times[i]);
			max_comp = std::max(max_comp, all_compute_times[i]);
			max_total = std::max(max_total, all_total_times[i]);
		}
		
		int task_count = all_request_travel_times.size();
		std::cout << "Total Tasks Completed: " << task_count << std::endl;
		
		std::cout << "\n=== REQUEST PACKET ===" << std::endl;
		std::cout << "  Travel Time: Avg=" << (float)total_req_travel/task_count 
		          << " cycles, Min=" << min_req << ", Max=" << max_req << std::endl;
		std::cout << "  Hop Count: Avg=" << (float)total_req_hops/task_count 
		          << ", Total=" << total_req_hops << std::endl;
		std::cout << "  Cycles per Hop: " << (total_req_hops > 0 ? (float)total_req_travel/total_req_hops : 0) 
		          << std::endl;
		
		std::cout << "\n=== RESPONSE PACKET ===" << std::endl;
		std::cout << "  Travel Time: Avg=" << (float)total_resp_travel/task_count 
		          << " cycles, Min=" << min_resp << ", Max=" << max_resp << std::endl;
		std::cout << "  Hop Count: Avg=" << (float)total_resp_hops/task_count 
		          << ", Total=" << total_resp_hops << std::endl;
		std::cout << "  Cycles per Hop: " << (total_resp_hops > 0 ? (float)total_resp_travel/total_resp_hops : 0) 
		          << std::endl;
		
		std::cout << "\n=== COMPUTATION ===" << std::endl;
		std::cout << "  Time: Avg=" << (float)total_comp/task_count 
		          << " cycles, Min=" << min_comp << ", Max=" << max_comp << std::endl;
		
		std::cout << "\n=== RESULT PACKET ===" << std::endl;
		std::cout << "  Hop Count: Avg=" << (float)total_res_hops/task_count 
		          << ", Total=" << total_res_hops << std::endl;
		std::cout << "  (Travel time not tracked due to NoC routing issue)" << std::endl;
		
		std::cout << "\n=== END-TO-END LATENCY ===" << std::endl;
		std::cout << "  Total: Avg=" << (float)total_all/task_count 
		          << " cycles, Min=" << min_total << ", Max=" << max_total << std::endl;
		
		// Time breakdown percentage
		std::cout << "\nTime Breakdown (Average):" << std::endl;
		std::cout << "  Request Travel: " << (total_req_travel*100.0/total_all) << "%" << std::endl;
		std::cout << "  Response Travel: " << (total_resp_travel*100.0/total_all) << "%" << std::endl;
		std::cout << "  Computation: " << (total_comp*100.0/total_all) << "%" << std::endl;
		std::cout << "  Unaccounted (queueing/waiting): " 
		          << ((total_all - total_req_travel - total_resp_travel - total_comp)*100.0/total_all) << "%" << std::endl;
	} else {
		std::cout << "No timing data available!" << std::endl;
	}
	
	std::cout << "\n=== End of Timing Statistics ===" << std::endl;
}

// Destructor
LLMMACnet::~LLMMACnet() {
	LLMMAC *llmmac;
	while (LLMMAC_list.size() != 0) {
		llmmac = LLMMAC_list.back();
		LLMMAC_list.pop_back();
		delete llmmac;
	}
}
