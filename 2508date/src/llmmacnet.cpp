// 小矩阵版本的 llmmacnet.cpp - 4x4可调试
#include "llmmacnet.hpp"
#include "llmmac.hpp"
#include <cassert>
#include <ctime>
#include <iomanip>

// Debug macro switch - 强制开启详细调试
#define LLM_DEBUG_PRINT_ENABLED
#define LLM_DEBUG_SMALL_MATRIX
#ifdef LLM_DEBUG_PRINT_ENABLED
    #define LLM_DEBUG(x) do { \
        std::cout << "[DEBUG] " << x << std::endl; \
    } while(0)
#else
    #define LLM_DEBUG(x) do {} while(0)
#endif

// Helper function
template<class C, typename T>
bool contains(C &&c, T e) {
	return find(begin(c), end(c), e) != end(c);
}

LLMMACnet::LLMMACnet(int mac_num, int t_pe_x, int t_pe_y, VCNetwork *t_Network) {
	macNum = mac_num;
	LLMMAC_list.reserve(mac_num);
	pe_x = t_pe_x;
	pe_y = t_pe_y;
	vcNetwork = t_Network;

	current_layer = 0;
	total_layers = 1;

	// 小矩阵配置 - 修改为4x4 tile和1个时间片
	matrix_size = 4;    // 4x4矩阵
	tile_size = 4;      // 改为4x4 tile（整个矩阵作为一个tile）
	time_slices = 1;    // 改为1个时间片

	tiles_per_dim = matrix_size / tile_size;  // 1
	total_tiles = tiles_per_dim * tiles_per_dim;  // 1
	total_tasks = matrix_size * matrix_size * time_slices;  // 4*4*1 = 16

	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;

	LLM_DEBUG("=== 小矩阵LLM调试版本 ===");
	LLM_DEBUG("Creating LLMMACnet with " << macNum << " MAC units");
	LLM_DEBUG("Matrix size: " << matrix_size << "x" << matrix_size);
	LLM_DEBUG("Tile size: " << tile_size << "x" << tile_size);
	LLM_DEBUG("Time slices: " << time_slices);
	LLM_DEBUG("Total tiles: " << total_tiles);
	LLM_DEBUG("Total tasks: " << total_tasks);

	for (int i = 0; i < macNum; i++) {
		int temp_ni_id = i % TOT_NUM;
		LLMMAC *newLLMMAC = new LLMMAC(i, this, temp_ni_id);
		LLMMAC_list.push_back(newLLMMAC);
	}

	LLM_DEBUG("Created " << LLMMAC_list.size() << " LLMMAC units");

	// Initialize matrices
	llmInitializeMatrices();

	// Export matrices for verification
	llmExportMatricesToFile();

	// Generate all tasks
	llmGenerateAllTasks();

	// Export tasks for verification
	llmExportTasksToFile();

	layer_latency.clear();
	LLM_DEBUG("Small matrix LLMMACnet initialized successfully!");
}

void LLMMACnet::llmInitializeMatrices() {
	LLM_DEBUG("Initializing " << matrix_size << "x" << matrix_size << " attention matrices...");

	// 使用固定种子确保可重现性
	srand(42);

	attention_query_table.resize(matrix_size);
	attention_key_table.resize(matrix_size);
	attention_value_table.resize(matrix_size);
	attention_output_table.resize(matrix_size);

	for (int i = 0; i < matrix_size; i++) {
		attention_query_table[i].resize(matrix_size);
		attention_key_table[i].resize(matrix_size);
		attention_value_table[i].resize(matrix_size);
		attention_output_table[i].resize(matrix_size);

		for (int j = 0; j < matrix_size; j++) {
			attention_query_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_key_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_value_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_output_table[i][j] = 0.0f;
		}
	}

	// 打印完整的小矩阵用于调试
	LLM_DEBUG("\n=== Query Matrix (" << matrix_size << "x" << matrix_size << ") ===");
	for (int i = 0; i < matrix_size; i++) {
		std::cout << "Row " << i << ": ";
		for (int j = 0; j < matrix_size; j++) {
			std::cout << std::fixed << std::setprecision(6) << attention_query_table[i][j] << "\t";
		}
		std::cout << std::endl;
	}

	LLM_DEBUG("\n=== Key Matrix (" << matrix_size << "x" << matrix_size << ") ===");
	for (int i = 0; i < matrix_size; i++) {
		std::cout << "Row " << i << ": ";
		for (int j = 0; j < matrix_size; j++) {
			std::cout << std::fixed << std::setprecision(6) << attention_key_table[i][j] << "\t";
		}
		std::cout << std::endl;
	}

	LLM_DEBUG("Matrices initialized successfully");
}

void LLMMACnet::llmExportMatricesToFile() {
	LLM_DEBUG("Exporting small matrices to output/ for verification...");

	// Export Query matrix
	std::ofstream query_file("output/cpp_query_matrix.txt");
	if (query_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				query_file << std::fixed << std::setprecision(10) << attention_query_table[i][j];
				if (j < matrix_size - 1) query_file << ",";
			}
			query_file << "\n";
		}
		query_file.close();
		LLM_DEBUG("Query matrix exported to output/cpp_query_matrix.txt");
	}

	// Export Key matrix
	std::ofstream key_file("output/cpp_key_matrix.txt");
	if (key_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				key_file << std::fixed << std::setprecision(10) << attention_key_table[i][j];
				if (j < matrix_size - 1) key_file << ",";
			}
			key_file << "\n";
		}
		key_file.close();
		LLM_DEBUG("Key matrix exported to output/cpp_key_matrix.txt");
	}

	// Export Value matrix
	std::ofstream value_file("output/cpp_value_matrix.txt");
	if (value_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				value_file << std::fixed << std::setprecision(10) << attention_value_table[i][j];
				if (j < matrix_size - 1) value_file << ",";
			}
			value_file << "\n";
		}
		value_file.close();
		LLM_DEBUG("Value matrix exported to output/cpp_value_matrix.txt");
	}
}

void LLMMACnet::llmGenerateAllTasks() {
	LLM_DEBUG("Generating " << total_tasks << " LLM tasks...");
	all_tasks.clear();
	all_tasks.reserve(total_tasks);

	int task_id = 0;
	// 只有1个时间片
	for (int ts = 0; ts < time_slices; ts++) {
		for (int pixel_y = 0; pixel_y < matrix_size; pixel_y++) {
			for (int pixel_x = 0; pixel_x < matrix_size; pixel_x++) {
				LLMTask task;
				task.task_id = task_id;
				task.pixel_x = pixel_x;
				task.pixel_y = pixel_y;
				task.time_slice = ts;
				task.tile_id = 0;  // 只有一个tile

				// Generate query and key data (4 elements for small matrix)
				int data_elements = matrix_size; // 4 elements for 4x4 matrix
				task.query_data.resize(data_elements);
				task.key_data.resize(data_elements);
				task.value_data.resize(data_elements);

				for (int i = 0; i < data_elements; i++) {
					int data_idx = (ts * data_elements + i) % matrix_size;
					task.query_data[i] = attention_query_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.key_data[i] = attention_key_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.value_data[i] = attention_value_table[pixel_y][(pixel_x + data_idx) % matrix_size];
				}

				// 打印每个任务的详细信息
				LLM_DEBUG("Task " << task_id << " [pixel(" << pixel_x << "," << pixel_y << "), ts=" << ts << "]:");
				std::cout << "  Query data: ";
				for (int i = 0; i < data_elements; i++) {
					std::cout << std::fixed << std::setprecision(6) << task.query_data[i];
					if (i < data_elements - 1) std::cout << ",";
				}
				std::cout << std::endl;

				std::cout << "  Key data: ";
				for (int i = 0; i < data_elements; i++) {
					std::cout << std::fixed << std::setprecision(6) << task.key_data[i];
					if (i < data_elements - 1) std::cout << ",";
				}
				std::cout << std::endl;

				all_tasks.push_back(task);
				task_id++;
			}
		}
	}

	LLM_DEBUG("Generated " << all_tasks.size() << " tasks successfully");
}

void LLMMACnet::llmExportTasksToFile() {
	LLM_DEBUG("Exporting all tasks to output/cpp_tasks.txt...");

	std::ofstream tasks_file("output/cpp_tasks.txt");
	if (tasks_file.is_open()) {
		// 导出所有任务的详细信息
		for (int i = 0; i < all_tasks.size(); i++) {
			const LLMTask& task = all_tasks[i];
			tasks_file << "Task " << task.task_id << ":\n";
			tasks_file << "  Position: (" << task.pixel_x << "," << task.pixel_y << ")\n";
			tasks_file << "  Time slice: " << task.time_slice << "\n";
			tasks_file << "  Tile ID: " << task.tile_id << "\n";

			tasks_file << "  Query data: ";
			for (int j = 0; j < task.query_data.size(); j++) {
				tasks_file << std::fixed << std::setprecision(10) << task.query_data[j];
				if (j < task.query_data.size() - 1) tasks_file << ",";
			}
			tasks_file << "\n";

			tasks_file << "  Key data: ";
			for (int j = 0; j < task.key_data.size(); j++) {
				tasks_file << std::fixed << std::setprecision(10) << task.key_data[j];
				if (j < task.key_data.size() - 1) tasks_file << ",";
			}
			tasks_file << "\n\n";
		}

		tasks_file.close();
		LLM_DEBUG("All " << all_tasks.size() << " tasks exported to output/cpp_tasks.txt");
	}
}

void LLMMACnet::llmExportVerificationResults() {
	LLM_DEBUG("Exporting verification results...");

	std::ofstream verify_file("output/cpp_verification.txt");
	if (verify_file.is_open()) {
		verify_file << "=== C++ Small Matrix Verification Results ===\n";
		verify_file << "Matrix size: " << matrix_size << "x" << matrix_size << "\n";
		verify_file << "Data elements per vector: " << matrix_size << "\n";
		verify_file << "Time slices: " << time_slices << "\n\n";

		// 导出所有任务的详细计算过程
		for (int i = 0; i < all_tasks.size(); i++) {
			const LLMTask& task = all_tasks[i];

			// 手动计算注意力值
			float dot_product = 0.0f;
			for (int j = 0; j < task.query_data.size(); j++) {
				dot_product += task.query_data[j] * task.key_data[j];
			}

			float scaled = dot_product / sqrt((float)task.query_data.size());
			float attention_output = tanh(scaled);

			verify_file << "Task " << i << " [(" << task.pixel_x << "," << task.pixel_y << "), ts=" << task.time_slice << "]:\n";

			verify_file << "  Query data: ";
			for (int j = 0; j < task.query_data.size(); j++) {
				verify_file << std::fixed << std::setprecision(10) << task.query_data[j];
				if (j < task.query_data.size() - 1) verify_file << ",";
			}
			verify_file << "\n";

			verify_file << "  Key data: ";
			for (int j = 0; j < task.key_data.size(); j++) {
				verify_file << std::fixed << std::setprecision(10) << task.key_data[j];
				if (j < task.key_data.size() - 1) verify_file << ",";
			}
			verify_file << "\n";

			verify_file << "  Dot product: " << std::fixed << std::setprecision(10) << dot_product << "\n";
			verify_file << "  Scaled: " << std::fixed << std::setprecision(10) << scaled << "\n";
			verify_file << "  Attention output: " << std::fixed << std::setprecision(10) << attention_output << "\n\n";
		}

		// 导出最终输出矩阵
		verify_file << "\n=== Final Output Matrix ===\n";
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				verify_file << std::fixed << std::setprecision(10) << attention_output_table[i][j];
				if (j < matrix_size - 1) verify_file << ",";
			}
			verify_file << "\n";
		}

		verify_file.close();
		LLM_DEBUG("Verification results exported to output/cpp_verification.txt");
	}

	// 导出最终输出矩阵到单独文件
	std::ofstream output_file("output/llm_attention_output.txt");
	if (output_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				output_file << std::fixed << std::setprecision(10) << attention_output_table[i][j];
				if (j < matrix_size - 1) output_file << ",";
			}
			output_file << "\n";
		}
		output_file.close();
		LLM_DEBUG("Output matrix exported to output/llm_attention_output.txt");
	}
}

// 辅助函数
int LLMMACnet::llmGetTileId(int pixel_x, int pixel_y) {
	int tile_x = pixel_x / tile_size;
	int tile_y = pixel_y / tile_size;
	return tile_y * tiles_per_dim + tile_x;
}

int LLMMACnet::llmGetMACIdForTile(int tile_id) {
	int mac_id = tile_id % macNum;
	while (llmIsMemoryNode(mac_id % TOT_NUM)) {
		mac_id = (mac_id + 1) % macNum;
	}
	return mac_id;
}

bool LLMMACnet::llmIsMemoryNode(int node_id) {
	for (int i = 0; i < MEM_NODES; i++) {
		if (dest_list[i] == node_id) {
			return true;
		}
	}
	return false;
}

void LLMMACnet::llmXMapping(int total_tasks) {
	LLM_DEBUG("Starting small matrix mapping for " << total_tasks << " tasks...");

	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	vector<int> available_macs;
	for (int i = 0; i < macNum; i++) {
		int ni_id = i % TOT_NUM;
		if (!llmIsMemoryNode(ni_id)) {
			available_macs.push_back(i);
		}
	}

	LLM_DEBUG("Available MAC units: " << available_macs.size() << " out of " << macNum);

	int j = 0;
	while (j < total_tasks) {
		for (int mac_id : available_macs) {
			this->mapping_table[mac_id].push_back(j);
			LLM_DEBUG("Assigning task " << j << " to MAC " << mac_id);
			j = j + 1;
			if (j >= total_tasks)
				break;
		}
	}

	LLM_DEBUG("Task mapping completed");

	// 打印详细的任务分配
	for (int i = 0; i < macNum; i++) {
		if (mapping_table[i].size() > 0) {
			std::cout << "MAC " << i << " tasks: ";
			for (int task_id : mapping_table[i]) {
				std::cout << task_id << " ";
			}
			std::cout << std::endl;
		}
	}
}

void LLMMACnet::llmLoadBalanceMapping(int total_tasks) {
	LLM_DEBUG("Starting load-balanced mapping...");
	llmXMapping(total_tasks);  // 对于小矩阵，使用简单映射即可
}

void LLMMACnet::llmCheckStatus() {
	static int status_check_count = 0;
	status_check_count++;

	if (status_check_count % 10000 == 0) {  // 更频繁的状态检查
		LLM_DEBUG("Status check #" << status_check_count << " at cycle " << cycles);
		LLM_DEBUG("Ready flag: " << ready_flag << ", Mapping again: " << mapping_again);
	}

	if (ready_flag == 0) {
		LLM_DEBUG("Initializing small matrix layer at cycle " << cycles);

		if (mapping_again == 0) {
			this->vcNetwork->resetVNRoundRobin();
		}

		#ifdef rowmapping
		this->llmXMapping(total_tasks);
		#else
		this->llmLoadBalanceMapping(total_tasks);
		#endif

		int active_macs = 0;
		for (int i = 0; i < macNum; i++) {
			if (mapping_table[i].size() == 0) {
				this->LLMMAC_list[i]->selfstatus = 5;
				this->LLMMAC_list[i]->send = 3;
			} else {
				this->LLMMAC_list[i]->routing_table.assign(
					mapping_table[i].begin(), mapping_table[i].end());
				active_macs++;
			}
		}

		LLM_DEBUG("Activated " << active_macs << " MAC units with tasks");
		ready_flag = 1;
		return;
	}

	int finished_count = 0;
	int active_count = 0;
	int assigned_macs = 0;
	
	for (int i = 0; i < macNum; i++) {
		// Only count MACs that actually have tasks assigned
		if (mapping_table[i].size() > 0) {
			assigned_macs++;
			if (LLMMAC_list[i]->selfstatus == 5 && LLMMAC_list[i]->send == 3) {
				finished_count++;
			} else if (LLMMAC_list[i]->selfstatus != 5) {
				active_count++;
			}
		}
	}

	if (status_check_count % 10000 == 0) {
		LLM_DEBUG("Status: " << finished_count << " finished, " << active_count << " active out of " << assigned_macs << " assigned MACs");
	}

	// Complete when no MACs are active (all have finished their tasks)
	// Add a delay to ensure messages are delivered through the network
	static int completion_wait_cycles = 0;
	
	if (assigned_macs > 0 && active_count == 0) {
		completion_wait_cycles++;
		
		// Wait for 10000 cycles after all MACs finish to ensure messages are delivered
		// (Network latency can be high in 32x32 NoC)
		if (completion_wait_cycles < 10000) {
			return;
		}
		LLM_DEBUG("\n=== All small matrix tasks completed at cycle " << cycles << " ===");
		LLM_DEBUG("Assigned MACs: " << assigned_macs << ", Finished: " << finished_count);

		// 打印最终结果矩阵 - 增强调试
		std::cout << "\n=== Final Output Matrix (" << matrix_size << "x" << matrix_size << ") ===" << std::endl;
		int zero_count = 0;
		int non_zero_count = 0;

		for (int i = 0; i < matrix_size; i++) {
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrix_size; j++) {
				float value = attention_output_table[i][j];
				std::cout << std::fixed << std::setprecision(6) << value << "\t";

				if (value == 0.0) {
					zero_count++;
				} else {
					non_zero_count++;
				}
			}
			std::cout << std::endl;
		}

		std::cout << "\n统计: " << non_zero_count << " 个非零值, " << zero_count << " 个零值" << std::endl;

		// 导出验证结果
		llmExportVerificationResults();

		layer_latency.push_back(cycles);
		ready_flag = 2;
		last_layer_packet_id = packet_id;
		return;
	}

	ready_flag = 1;
}
void LLMMACnet::llmRunOneStep() {
	static int run_step_count = 0;
	run_step_count++;

	for (int i = 0; i < macNum; i++) {
		LLMMAC_list[i]->llmRunOneStep();
	}

	if (run_step_count % 10000 == 0) {
		int active_macs = 0;
		for (int i = 0; i < macNum; i++) {
			if (LLMMAC_list[i]->selfstatus != 5) {
				active_macs++;
			}
		}
		LLM_DEBUG("Run step #" << run_step_count << ": " << active_macs << " active MACs");
	}

	// Memory operations handling
	int pbuffer_size;
	int src, pid_signal_id, mem_id, src_mac;
	LLMMAC *tmpLLMMAC;
	Packet *tmpPacket;
	NI *tmpNI;

	for (int memidx = 0; memidx < MEM_NODES; memidx++) {
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];

		// Process ALL message types in buffer[0]
		pbuffer_size = tmpNI->packet_buffer_out[0].size();
		
		// Debug: check if we have any packets
		static int debug_count = 0;
		if (pbuffer_size > 0 && debug_count++ % 1000 == 0) {
			LLM_DEBUG("Memory " << mem_id << " has " << pbuffer_size << " packets in buffer[0]");
		}
		
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			
			// Debug: log all packet types
			static int pkt_debug_count = 0;
			if (pkt_debug_count++ % 100 == 0) {
				LLM_DEBUG("Memory " << mem_id << " processing packet type " << tmpPacket->message.msgtype 
				          << " from MAC " << tmpPacket->message.mac_id);
			}
			if (tmpPacket->message.msgtype == 3) {
				std::cout << "[TYPE3-FOUND] Memory " << mem_id << " found type 3 packet from MAC " 
				          << tmpPacket->message.mac_id << std::endl;
			}
			
			// Handle type 0 requests
			if (tmpPacket->message.msgtype == 0) {
				if (tmpPacket->message.out_cycle >= cycles) {
					tmpNI->packet_buffer_out[0].pop_front();
					tmpNI->packet_buffer_out[0].push_back(tmpPacket);
					continue;
				}

			src = tmpPacket->message.source_id;
			pid_signal_id = tmpPacket->message.signal_id;
			src_mac = tmpPacket->message.mac_id;
			tmpLLMMAC = LLMMAC_list[src_mac];

			if (tmpLLMMAC->selfstatus == 2) {
				int task_id = tmpPacket->message.data[0];

				if (task_id >= 0 && task_id < all_tasks.size()) {
					LLMTask& task = all_tasks[task_id];

					tmpLLMMAC->input_buffer.clear();
					tmpLLMMAC->input_buffer.push_back(1.0f);
					tmpLLMMAC->input_buffer.push_back(task.query_data.size());
					tmpLLMMAC->input_buffer.push_back(task.time_slice);
					tmpLLMMAC->input_buffer.push_back(task.pixel_x * matrix_size + task.pixel_y);

					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.query_data.begin(), task.query_data.end());
					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.key_data.begin(), task.key_data.end());

					LLM_DEBUG("Memory " << mem_id << " sending data to MAC " << src_mac
					          << " for task " << task_id
					          << " [pixel(" << task.pixel_x << "," << task.pixel_y
					          << "), ts=" << task.time_slice << "]");

					int mem_delay = static_cast<int>(ceil((task.query_data.size() * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
					LLMMAC_list[mem_id]->pecycle = cycles + mem_delay;
					LLMMAC_list[mem_id]->input_buffer = tmpLLMMAC->input_buffer;
					LLMMAC_list[mem_id]->llmInject(1, src, tmpLLMMAC->input_buffer.size(),
						1.0f, vcNetwork->NI_list[mem_id], pid_signal_id, src_mac);
				}
				tmpNI->packet_buffer_out[0].pop_front();
			}
			// Handle type 2 and 3 results in the same loop iteration
			else if (tmpPacket->message.msgtype == 2 || tmpPacket->message.msgtype == 3) {
				// Check out_cycle for result packets too
				if (tmpPacket->message.out_cycle >= cycles) {
					tmpNI->packet_buffer_out[0].pop_front();
					tmpNI->packet_buffer_out[0].push_back(tmpPacket);
					continue;
				}
				int msg_type = tmpPacket->message.msgtype;
				src = tmpPacket->message.source_id;
				src_mac = tmpPacket->message.mac_id;
				tmpLLMMAC = LLMMAC_list[src_mac];

				if (tmpPacket->message.data.size() >= 4) {
					float result_value = tmpPacket->message.data[0];
					int pixel_x = tmpPacket->message.data[1];
					int pixel_y = tmpPacket->message.data[2];
					int time_slice = tmpPacket->message.data[3];

					if (msg_type == 2) {
						// Type 2: 中间结果，仅用于调试
						LLM_DEBUG("[INTERMEDIATE] Memory " << mem_id
						          << " received intermediate result from MAC " << src_mac
						          << " for pixel(" << pixel_x << "," << pixel_y << ") ts=" << time_slice
						          << " value=" << std::fixed << std::setprecision(6) << result_value);
					}
					else if (msg_type == 3) {
						// Type 3: 最终聚合结果，更新输出表！
						std::cout << "[FINAL-UPDATE] Memory " << mem_id
						          << " received FINAL result from MAC " << src_mac << std::endl;
						std::cout << "  Pixel: (" << pixel_x << "," << pixel_y << ")" << std::endl;
						std::cout << "  Final value: " << std::fixed << std::setprecision(10) << result_value << std::endl;

						if (pixel_x >= 0 && pixel_x < matrix_size &&
						    pixel_y >= 0 && pixel_y < matrix_size) {

							// 保存旧值用于比较
							float old_value = attention_output_table[pixel_y][pixel_x];

							// 更新输出表
							attention_output_table[pixel_y][pixel_x] = result_value;

							std::cout << "[TABLE-UPDATE] Updated output table:" << std::endl;
							std::cout << "  Position [" << pixel_y << "][" << pixel_x << "]" << std::endl;
							std::cout << "  Old value: " << std::fixed << std::setprecision(10) << old_value << std::endl;
							std::cout << "  New value: " << std::fixed << std::setprecision(10) << result_value << std::endl;
							std::cout << "  Verification: " << attention_output_table[pixel_y][pixel_x] << std::endl;

							// 统计非零元素
							int non_zero_count = 0;
							for (int i = 0; i < matrix_size; i++) {
								for (int j = 0; j < matrix_size; j++) {
									if (attention_output_table[i][j] != 0.0) {
										non_zero_count++;
									}
								}
							}
							std::cout << "  Total non-zero elements in table: " << non_zero_count
							          << "/" << (matrix_size * matrix_size) << std::endl;
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

	// Handle responses (type 1 messages)
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
			tmpLLMMAC->llmReceive(&tmpPacket->message);
			tmpNI->packet_buffer_out[0].pop_front();
		}
	}

	// 定期打印输出矩阵状态
	if (run_step_count % 50000 == 0) {
		std::cout << "\n[MATRIX-STATUS] Current output matrix at step " << run_step_count << ":" << std::endl;
		int non_zero_count = 0;
		for (int i = 0; i < matrix_size; i++) {
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrix_size; j++) {
				float val = attention_output_table[i][j];
				if (val != 0.0) {
					non_zero_count++;
					std::cout << std::fixed << std::setprecision(6) << val << " ";
				} else {
					std::cout << "0.000000 ";
				}
			}
			std::cout << std::endl;
		}
		std::cout << "Non-zero elements: " << non_zero_count << "/" << (matrix_size * matrix_size) << std::endl;
	}
}





}  // End of llmRunOneStep()

// Destructor
LLMMACnet::~LLMMACnet() {
	LLMMAC *llmmac;
	while (LLMMAC_list.size() != 0) {
		llmmac = LLMMAC_list.back();
		LLMMAC_list.pop_back();
		delete llmmac;
	}
}
