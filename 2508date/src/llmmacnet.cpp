#include <cstdlib>  // For std::exit()



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
	// Test Case 2: Real matrix 8×128 output
	// Set actual dimensions for the real matrices
	// X_input (8×4096) @ Wq^T (4096×128) = Q (8×128)

	input_sequence_length = 128;     // X_input has 8 rows
	input_hidden_dim = 4096;       // X_input has 4096 columns
	query_output_dim = 128;        // Wq produces 128-dim query vectors
	time_slices = LLM_SUBCHUNKS_PER_PIXEL;  // Each pixel has N time slices (subchunks)
	matrixOutputPixels_inputsequencelength = input_sequence_length;  // 8 rows output matrix (from X_input rows)
	matrixOutputPixels_queryoutputdim = query_output_dim;

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

		// Debug: Check for Type 3 packets in buffer
		// if (pbuffer_size > 0) {
		// 	for (int k = 0; k < pbuffer_size; k++) {
		// 		Packet* debugPkt = tmpNI->packet_buffer_out[0][k];
		// 		if (debugPkt && debugPkt->message.msgtype == 3) {
		// 			std::cout << "[DEBUG-BUFFER] Type 3 packet found in Memory " << mem_id 
		// 			          << " buffer at position " << k 
		// 			          << " (buffer size=" << pbuffer_size << ")" << std::endl;
		// 		}
		// 	}
		// }
		// Process packets in buffer[0] (Type 0 requests)
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			
			// Process Type 0: Data request packets from MAC to MEM
			if (tmpPacket->message.msgtype == 0) {
				if (tmpPacket->message.out_cycle >= cycles) {
					tmpNI->packet_buffer_out[0].pop_front();
					tmpNI->packet_buffer_out[0].push_back(tmpPacket);
					continue;
				}

				// Extract request information
				pid_signal_id = tmpPacket->message.signal_id;
				src = tmpPacket->message.source_id;
				src_mac = tmpPacket->message.mac_id;
				tmpLLMMAC = LLMMAC_list[src_mac];

				// Verify MAC is waiting for data (State 2: WAITING)
				if (tmpLLMMAC->selfstatus == 2) {
					int task_id = tmpPacket->message.data[0];  // Extract task ID from request
					
					// std::cout << "[LLM-TYPE0] Memory " << mem_id << " received request from MAC " 
					//           << src_mac << " for task " << task_id << std::endl;

					// Record timing for performance analysis
					tmpLLMMAC->current_task_timing.request_arrive_cycle = cycles;


					if (task_id >= 0 && task_id < all_tasks.size()) {
						LLMTask& task = all_tasks[task_id];  // 直接索引访问，O(1)时间复杂度


						// 清空缓冲区，准备新数据
						tmpLLMMAC->input_buffer.clear();

						// 添加4个Header元数据
						tmpLLMMAC->input_buffer.push_back(1.0f);              // [0] 函数标志(1.0=LLM模式)
						tmpLLMMAC->input_buffer.push_back(64);                // [1] 数据大小(每个矩阵64元素)
						tmpLLMMAC->input_buffer.push_back(task.subchunk_id);  // [2] 子块ID(0-3)
						tmpLLMMAC->input_buffer.push_back(task.pixel_id);     // [3] 像素ID(用于结果聚合)



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
					}
					tmpNI->packet_buffer_out[0].pop_front();
				}
			}

			else {
				// Other message types - just cycle them in buffer[0]
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
			}
		}
	}

	// Process Type 1 messages from MEM to MAC (data responses)
	// Non-memory nodes receive Type 1 responses in buffer[0]
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
			
			// Debug output for Type 1 reception
			// std::cout << "[LLM-TYPE1] MAC " << src_mac << " receiving data response" << std::endl;
			
			tmpLLMMAC->llmPEReceiveResp(&tmpPacket->message);
			tmpNI->packet_buffer_out[0].pop_front();
		}
	}


	// Process Type 3 messages from MAC to MEM (final aggregated results)
	// Similar to CNN's Type 2 handling in buffer[1]
	for (int memidx = 0; memidx < MEM_NODES; memidx++) {
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];
		pbuffer_size = tmpNI->packet_buffer_out[1].size();
		
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[1].front();
			
			// Skip non-Type 3 messages
			if (tmpPacket->message.msgtype != 3) {
				tmpNI->packet_buffer_out[1].pop_front();
				tmpNI->packet_buffer_out[1].push_back(tmpPacket);
				continue;
			}
			
			// Extract source information
			src = tmpPacket->message.source_id;
			src_mac = tmpPacket->message.mac_id;
			tmpLLMMAC = LLMMAC_list[src_mac];
			
			// Extract result data from packet
			float result_value = tmpPacket->message.data[0];  // Attention computation result
			int pixel_x = tmpPacket->message.data[1];         // Output matrix X coordinate
			int pixel_y = tmpPacket->message.data[2];         // Output matrix Y coordinate
			int time_slice = tmpPacket->message.data[3];      // Subchunk ID (for debugging)
			
			// Validate coordinates and update output tables
			if (pixel_x >= 0 && pixel_x < query_output_dim &&
			    pixel_y >= 0 && pixel_y < input_sequence_length) {
				
				// Update both output matrices
				Q_resOutput_matrix[pixel_y][pixel_x] = result_value;      // Reference matrix
				attention_output_table[pixel_y][pixel_x] = result_value;  // Actual output table
				
				// Increment completed task counter
				executed_tasks++;
				
				// Debug output for first few pixels
				if (pixel_y == 0 && pixel_x <= 2) {
					float expected = 0.0f;
					if (pixel_x == 0) expected = 0.01544952f;
					else if (pixel_x == 1) expected = -0.01119441f;
					else if (pixel_x == 2) expected = 0.00336472f;
					
					std::cout << "[DEBUG-RECEIVED] Memory node received pixel[" << pixel_y << "][" << pixel_x << "] = " 
					          << std::fixed << std::setprecision(8) << result_value 
					          << " (expected: " << expected << ", diff: " << (result_value - expected) << ")" << std::endl;
				}
			} else {
				// std::cout << "[ERROR] Invalid pixel coordinates: (" 
				//           << pixel_x << "," << pixel_y << ")" << std::endl;
			}
			
			// Update MAC status if finished
			if (tmpLLMMAC->selfstatus == 5) {
				tmpLLMMAC->send = 3;
			}
			
			// Remove processed packet from buffer
			tmpNI->packet_buffer_out[1].pop_front();
		}
	}
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
	
	if(input_sequence_length ==128){
		x_input_file = input_dir + "X_input_128seq.txt";
		wq_file = input_dir + "Wq_128seq.txt";
	}

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
		attention_output_table.resize(input_sequence_length);  // 初始化attention_output_table
		for (int i = 0; i < input_sequence_length; i++) {
			Q_resOutput_matrix[i].resize(query_output_dim, 0.0f);
			attention_output_table[i].resize(query_output_dim, 0.0f);  // 初始化为0
			/*
			for (int j = 0; j < query_output_dim; j++) {
				float sum = 0.0f;
				for (int k = 0; k < input_hidden_dim; k++) {
					sum += input_matrix[i][k] * query_weight_matrix[j][k];
				}
				Q_resOutput_matrix[i][j] = sum;
			}
			*///这里只是debug，通过运算结果看看文件io正确。
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
				 *    - 维度：
				 *    - 总元素
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
				 *   weight_query_idx = (10 + 64 + 0)  = 74
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
					std::cout << "\n llmmacnet === Task " << task.task_id << " Input data (first 20 values) ===" << std::endl;
					for (int i = 0; i < std::min(20, (int)task.input_data.size()); i++) {
						std::cout << std::fixed << std::setprecision(3) 
						         << std::setw(8) << task.input_data[i] << " ";
					}
					std::cout << std::endl;
					
					std::cout << "\n=== Task " << task.task_id << " Query weights (first 20 values) ===" << std::endl;
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
	attention_output_table.resize(input_sequence_length);  // 初始化attention_output_table

	// Initialize input matrix with random values
	for (int i = 0; i < input_sequence_length; i++) {
		input_matrix[i].resize(input_hidden_dim);
		for (int j = 0; j < input_hidden_dim; j++) {
			input_matrix[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
		}
		Q_resOutput_matrix[i].resize(query_output_dim, 0.0f);
		attention_output_table[i].resize(query_output_dim, 0.0f);  // 初始化为0
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
		for (int subchunk_id = 0; subchunk_id < LLM_SUBCHUNKS_PER_PIXEL; subchunk_id++) {
			int task_id = pixel_id * LLM_SUBCHUNKS_PER_PIXEL + subchunk_id;  // Fixed mapping formula
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
		int total_pixels = 	matrixOutputPixels_inputsequencelength *  matrixOutputPixels_queryoutputdim  ;
		
		if (total_pixels / available_macs < samplingTasksPerMAC) {
			// If pixels are fewer than sampling window, use normal row mapping
			this->llmXMapping(total_pixels);
		} else {
			if (mapping_again == 0) {
				// First phase: run sampling window (pixel-based)
				int sampling_pixels = available_macs * samplingTasksPerMAC;
				
				// Reset sampling statistics
				std::fill_n(samplingWindowDelay, TOT_NUM, 0);
				
				// Map sampling window pixels using row mapping
				this->llmXMapping(sampling_pixels);
				mapping_again = 1;  // Mark that sampling is being done
				
			} else if (mapping_again == 2) {
				// Second phase: map remaining pixels based on sampling results
				int sampling_pixels = available_macs * samplingTasksPerMAC;
				int remaining_pixels = total_pixels - sampling_pixels;
				int remaining_tasks = remaining_pixels * this->tasks_per_pixel;

				// Update packet_id based on sampling phase tasks
				//cout << "Line960: Before update packet_id=" << packet_id << " sampling_pixels=" << sampling_pixels << " tasks_per_pixel=" << this->tasks_per_pixel << endl;
				//packet_id = packet_id + sampling_pixels * this->tasks_per_pixel;
				//cout << "Line962: After update packet_id=" << packet_id << endl;
				
				// Print sampling delay measurements
				cout << "\n=== Sampling Window Delay Measurements ===" << endl;
				for (int i = 0; i < macNum; i++) {
					if (!llmIsMemoryNode(i)) {
						cout << "MAC " << i << ": samplingWindowDelay=" << samplingWindowDelay[i] 
						     << " (avg=" << (double)samplingWindowDelay[i]/samplingTasksPerMAC << ")" << endl;
					}
				}
				cout << "samplingTasksPerMAC=" << samplingTasksPerMAC << endl;
				cout << "=== End Sampling Measurements ===" << endl;
				
				// Use SAMOS mapping based on latency measurements
				int start_pixel_id = sampling_pixels;
				//cout << "Line964: start_pixel_id=" << start_pixel_id << " remaining_pixels=" << remaining_pixels << endl;
				this->llmSAMOSTaskMapping(remaining_pixels, start_pixel_id);

				mapping_again = 0;  // Reset for next layer
			} else {
			}
		}
		#endif
		// Normal mapping without SAMOS
		#if defined(rowmapping)
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
				//cout << "MAC " << i << " assigned " << llmTaskMappingTable[i].size() << " tasks" << endl;
			}
		}
		cout<<"cyclesdebug" <<cycles <<" "<<this->LLMMAC_list[0]->llmPEExpectedtasktable.size() <<endl;
		//assert( false && "debug ");

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
		cout << "[DEBUG-COMPLETION] All MACs inactive. Wait cycle: " << completion_wait_cycles << "/100" << endl;
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

			// Reset for second phase
			completion_wait_cycles = 0;
			ready_flag = 0;
			mapping_again = 2;  // Move to SAMOS mapping phase

			// Reset MAC status like CNN does
			for (int i = 0; i < macNum; i++) {
				LLMMAC_list[i]->selfstatus = 0;
				// Also reset the current task ID to avoid assertion failure
			}
			return;
		}
		#endif
		

		// Print MAC completion times
		cout << "[DEBUG-STATS-1] Starting MAC completion time analysis" << endl;
		cout << "\n=== MAC Completion Times ===" << endl;
		int min_complete = INT_MAX, max_complete = 0;
		int min_mac = -1, max_mac = -1;
		for (int i = 0; i < macNum; i++) {
			if (LLMMAC_list[i]->task_timings.size() > 0) {
				// Get the last task's completion time for this MAC
				int last_complete = LLMMAC_list[i]->task_timings.back().compute_end_cycle;
				cout << "MAC " << i << ": completed at cycle " << last_complete 
				     << " (processed " << LLMMAC_list[i]->task_timings.size() << " tasks)" << endl;
				if (last_complete < min_complete) {
					min_complete = last_complete;
					min_mac = i;
				}
				if (last_complete > max_complete) {
					max_complete = last_complete;
					max_mac = i;
				}
			}
		}
		cout << "[DEBUG-STATS-2] min_complete=" << min_complete << ", max_complete=" << max_complete << endl;
		cout << "Earliest finish: MAC " << min_mac << " at cycle " << min_complete << endl;
		cout << "Latest finish: MAC " << max_mac << " at cycle " << max_complete << endl;
		if (min_complete > 0 && min_complete != INT_MAX) {
			cout << "Load balance ratio: " << (float)(max_complete - min_complete) / min_complete * 100 << "%" << endl;
		} else {
			cout << "[WARNING] Cannot calculate load balance ratio: min_complete=" << min_complete << endl;
		}
		cout << "=== End MAC Completion Times ===" << endl;
		
		// Print timing statistics
		cout << "[DEBUG-STATS-3] Calling llmPrintTimingStatistics()" << endl;
		llmPrintTimingStatistics();
		cout << "[DEBUG-STATS-4] llmPrintTimingStatistics() completed" << endl;

		layer_latency.push_back(cycles);
		cout << "[DEBUG-STATS-5] Setting ready_flag=2 to signal completion" << endl;
		ready_flag = 2;
		
		// Adjust packet_id for next layer based on actual tasks processed
		cout << "[DEBUG-STATS-6] Adjusting packet_id. Current: " << packet_id << endl;
		#ifdef YZSAMOSSampleMapping
		int available_macs = macNum - MEM_NODES;
		int total_pixels = 	matrixOutputPixels_inputsequencelength *  matrixOutputPixels_queryoutputdim  ;
		cout << "[DEBUG-STATS-7] SAMOS mode: total_pixels=" << total_pixels << ", available_macs=" << available_macs << endl;
		if (total_pixels / available_macs < samplingTasksPerMAC) {
			// Used normal mapping, add all tasks (pixels * 4)
			packet_id = packet_id + total_pixels * 4;
			cout << "[DEBUG-STATS-8] Normal mapping path. New packet_id=" << packet_id << endl;
		} else {
			// Used SAMOS mapping, already adjusted during mapping
			// No need to adjust here as it was done incrementally
			cout << "[DEBUG-STATS-8] SAMOS mapping path. packet_id unchanged=" << packet_id << endl;
		}
		#else
		// Normal mapping: total_task_slicedPixels now represents pixels, so multiply by 4 for actual tasks
		packet_id = packet_id + total_task_slicedPixels;
		cout << "[DEBUG-STATS-8] Non-SAMOS mode. Added " << total_task_slicedPixels << ", new packet_id=" << packet_id << endl;
		#endif
		last_layer_packet_id = packet_id;
		cout << "[DEBUG-STATS-9] Returning from llmCheckStatus() with ready_flag=2" << endl;
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
		double lat = double(samplingWindowDelay[id]) / std::max(1, samplingTasksPerMAC);
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
		double lat = double(samplingWindowDelay[id]) / std::max(1, samplingTasksPerMAC);
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
				//if (task_id == 34880 || task_id == 43840) {
				//	cout << "Line1181: Creating task_id=" << task_id << " for MAC " << n.id << " (current_pixel_id=" << current_pixel_id << ")" << endl;
				//}
				this->llmTaskMappingTable[n.id].push_back(task_id);
			}
			current_pixel_id++;
		}
	}
	
	// Print SAMOS task distribution
	cout << "\n=== SAMOS Task Distribution (Phase 2) ===" << endl;
	for (auto &n : nodes) {
		int task_count = n.alloc * this->tasks_per_pixel;
		double avgLat = double(samplingWindowDelay[n.id]) / std::max(1, samplingTasksPerMAC);
		cout << "MAC " << n.id << ": " << task_count << " tasks (pixels=" << n.alloc 
		     << ", avgLat=" << avgLat << ", weight=" << n.w << ")" << endl;
	}
	cout << "=== End SAMOS Distribution ===" << endl;

	return 0;
}


void LLMMACnet::llmPrintTimingStatistics() {
	std::cout << "[DEBUG-TIMING-1] Entering llmPrintTimingStatistics()" << std::endl;
	std::cout << "\n=== Task Timing Statistics ===" << std::endl;

	// Collect timing data from all MACs
	std::cout << "[DEBUG-TIMING-2] Collecting timing data from all MACs" << std::endl;
	std::vector<int> all_request_travel_times;
	std::vector<int> all_response_travel_times;
	std::vector<int> all_compute_times;
	std::vector<int> all_result_travel_times;
	std::vector<int> all_total_times;
	std::vector<int> all_request_hops;
	std::vector<int> all_response_hops;
	std::vector<int> all_result_hops;
	
	// Per-MAC statistics with tracking of maximum
	std::cout << "[DEBUG-TIMING-3] Starting per-MAC statistics" << std::endl;
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
		std::cout << "[DEBUG-TIMING-4] Processing MAC " << i << " with " << LLMMAC_list[i]->task_timings.size() << " tasks" << std::endl;
		
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
	std::cout << "[DEBUG-TIMING-5] Printing MAC with maximum average total time" << std::endl;
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
	std::cout << "[DEBUG-TIMING-6] Starting network-wide statistics" << std::endl;
	std::cout << "\n--- Network-wide Timing Statistics ---" << std::endl;

	if (all_request_travel_times.size() > 0) {
		std::cout << "[DEBUG-TIMING-7] Processing " << all_request_travel_times.size() << " total tasks" << std::endl;
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
	std::cout << "[DEBUG-TIMING-8] Exiting llmPrintTimingStatistics()" << std::endl;
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
