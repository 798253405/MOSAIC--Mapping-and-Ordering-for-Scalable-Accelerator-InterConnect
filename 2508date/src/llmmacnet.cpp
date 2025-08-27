// 修复的 llmmacnet.cpp
#include "llmmacnet.hpp"
#include "llmmac.hpp"  // 在 .cpp 文件中包含完整定义
#include <cassert>
#include <ctime>
#include <iomanip>

// Debug macro switch - 检查全局开关，添加仿真周期数
#ifdef LLM_DEBUG_PRINT_ENABLED
    #define LLM_DEBUG(x) do { \
        time_t now = time(0); \
        struct tm* timeinfo = localtime(&now); \
        std::cout << "[" << std::setfill('0') << std::setw(2) << timeinfo->tm_hour << ":" \
                  << std::setw(2) << timeinfo->tm_min << ":" << std::setw(2) << timeinfo->tm_sec \
                  << "][Cycle:" << cycles << "] " << x << std::endl; \
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

	// LLM-specific initialization
	current_layer = 0;
	total_layers = 1;

	// 快速测试版本：32x32 矩阵，2x2 tile，2个时间片
	matrix_size = 32;  // 512x512 太大，改为 32x32
	tile_size = 4;     // 16x16 太大，改为 4x4 tile
	time_slices = 2;   // 4个时间片太多，改为 2个

	// 保留512x512版本（注释掉）
	// matrix_size = 512;
	// tile_size = 16;
	// time_slices = 4;

	tiles_per_dim = matrix_size / tile_size; // 8 (32/4)
	total_tiles = tiles_per_dim * tiles_per_dim; // 64
	total_tasks = matrix_size * matrix_size * time_slices; // 32*32*2 = 2,048

	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;

	LLM_DEBUG("Creating LLMMACnet with " << macNum << " MAC units");
	LLM_DEBUG("Matrix size: " << matrix_size << "x" << matrix_size);
	LLM_DEBUG("Tile size: " << tile_size << "x" << tile_size);
	LLM_DEBUG("Total tiles: " << total_tiles);
	LLM_DEBUG("Total tasks: " << total_tasks);

	// 修复：只创建一次 LLMMAC 单元
	for (int i = 0; i < macNum; i++) {
		int temp_ni_id = i % TOT_NUM;
		LLMMAC *newLLMMAC = new LLMMAC(i, this, temp_ni_id);
		LLMMAC_list.push_back(newLLMMAC);
	}

	LLM_DEBUG("Created " << LLMMAC_list.size() << " LLMMAC units");

	// Initialize matrices
	llmInitializeMatrices();

	// Generate all tasks
	llmGenerateAllTasks();

	layer_latency.clear();
	LLM_DEBUG("LLMMACnet initialized successfully!");
}

void LLMMACnet::llmInitializeMatrices() {
	LLM_DEBUG("Initializing attention matrices...");

	// Initialize attention matrices (512x512)
	attention_query_table.resize(matrix_size);
	attention_key_table.resize(matrix_size);
	attention_value_table.resize(matrix_size);
	attention_output_table.resize(matrix_size);

	// Fill with random data for testing
	srand(42); // For reproducible results
	for (int i = 0; i < matrix_size; i++) {
		attention_query_table[i].resize(matrix_size);
		attention_key_table[i].resize(matrix_size);
		attention_value_table[i].resize(matrix_size);
		attention_output_table[i].resize(matrix_size);

		for (int j = 0; j < matrix_size; j++) {
			attention_query_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_key_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_value_table[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
			attention_output_table[i][j] = 0.0f; // Initialize output to zero
		}
	}
	LLM_DEBUG("Attention matrices initialized successfully");
}

void LLMMACnet::llmGenerateAllTasks() {
	LLM_DEBUG("Generating LLM tasks...");
	all_tasks.clear();
	all_tasks.reserve(total_tasks);

	int task_id = 0;
	for (int pixel_y = 0; pixel_y < matrix_size; pixel_y++) {
		for (int pixel_x = 0; pixel_x < matrix_size; pixel_x++) {
			for (int ts = 0; ts < time_slices; ts++) {
				LLMTask task;
				task.task_id = task_id++;
				task.pixel_x = pixel_x;
				task.pixel_y = pixel_y;
				task.time_slice = ts;
				task.tile_id = llmGetTileId(pixel_x, pixel_y);

				// Generate query and key data (16 elements each for this time slice - 缩小数据量)
				task.query_data.resize(16);  // 从64减少到16
				task.key_data.resize(16);    // 从64减少到16
				task.value_data.resize(16);  // 从64减少到16

				for (int i = 0; i < 16; i++) {
					// Extract relevant data based on time slice and pixel position
					int data_idx = (ts * 16 + i) % matrix_size;
					task.query_data[i] = attention_query_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.key_data[i] = attention_key_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.value_data[i] = attention_value_table[pixel_y][(pixel_x + data_idx) % matrix_size];
				}

				// Debug: Print first few tasks' data
				if (task_id <= 5) {
					LLM_DEBUG("Task " << task_id - 1 << " [(" << pixel_x << "," << pixel_y << "), ts=" << ts << "]:");
					LLM_DEBUG("  Query[0-3]: " << task.query_data[0] << ", " << task.query_data[1] << ", " << task.query_data[2] << ", " << task.query_data[3]);
					LLM_DEBUG("  Key[0-3]: " << task.key_data[0] << ", " << task.key_data[1] << ", " << task.key_data[2] << ", " << task.key_data[3]);
				}

				all_tasks.push_back(task);
			}
		}
	}

	LLM_DEBUG("Generated " << all_tasks.size() << " tasks successfully");
}

int LLMMACnet::llmGetTileId(int pixel_x, int pixel_y) {
	int tile_x = pixel_x / tile_size;
	int tile_y = pixel_y / tile_size;
	return tile_y * tiles_per_dim + tile_x;
}

int LLMMACnet::llmGetMACIdForTile(int tile_id) {
	// Map tile to MAC unit (simple modulo mapping, excluding memory nodes)
	int mac_id = tile_id % macNum;
	while (llmIsMemoryNode(mac_id % TOT_NUM)) {  // 修复：使用 mac_id % TOT_NUM
		mac_id = (mac_id + 1) % macNum;
	}
	return mac_id;
}

bool LLMMACnet::llmIsMemoryNode(int node_id) {
	// Check if node_id is in dest_list (memory nodes)
	for (int i = 0; i < MEM_NODES; i++) {
		if (dest_list[i] == node_id) {
			return true;
		}
	}
	return false;
}

void LLMMACnet::llmXMapping(int total_tasks) {
	LLM_DEBUG("Starting LLM X-mapping for " << total_tasks << " tasks...");

	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	// Count available MAC units (excluding memory nodes)
	vector<int> available_macs;
	for (int i = 0; i < macNum; i++) {
		int ni_id = i % TOT_NUM;
		if (!llmIsMemoryNode(ni_id)) {
			available_macs.push_back(i);
		}
	}

	LLM_DEBUG("Available MAC units: " << available_macs.size() << " out of " << macNum);

	if (available_macs.empty()) {
		LLM_DEBUG("ERROR: No available MAC units found!");
		return;
	}

	// Distribute tasks among available MAC units
	int j = 0;
	while (j < total_tasks) {
		for (int mac_id : available_macs) {
			this->mapping_table[mac_id].push_back(j);
			j = j + 1;
			if (j >= total_tasks)
				break;
		}
	}

	LLM_DEBUG("LLM X-mapping completed, mapped " << j << " tasks");

	// Print task distribution
	int total_mapped = 0;
	for (int i = 0; i < macNum; i++) {
		if (mapping_table[i].size() > 0) {
			LLM_DEBUG("MAC " << i << ": " << mapping_table[i].size() << " tasks");
			total_mapped += mapping_table[i].size();
		}
	}
	LLM_DEBUG("Total mapped tasks: " << total_mapped);
}

void LLMMACnet::llmLoadBalanceMapping(int total_tasks) {
	LLM_DEBUG("Starting load-balanced mapping...");

	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	// Count non-memory nodes
	vector<int> pe_ids;
	for (int i = 0; i < macNum; i++) {
		if (!llmIsMemoryNode(i % TOT_NUM)) {
			pe_ids.push_back(i);
		}
	}

	LLM_DEBUG("Available PEs: " << pe_ids.size());

	if (pe_ids.empty()) {
		LLM_DEBUG("ERROR: No available PEs found!");
		return;
	}

	int tasks_per_pe = total_tasks / pe_ids.size();
	int remainder = total_tasks % pe_ids.size();

	LLM_DEBUG("Tasks per PE: " << tasks_per_pe << ", Remainder: " << remainder);

	int task_id = 0;
	for (int i = 0; i < pe_ids.size(); i++) {
		int pe_id = pe_ids[i];
		int tasks_for_this_pe = tasks_per_pe + (i < remainder ? 1 : 0);

		for (int j = 0; j < tasks_for_this_pe; j++) {
			this->mapping_table[pe_id].push_back(task_id++);
		}
	}

	LLM_DEBUG("Load-balance mapping completed for " << pe_ids.size() << " PEs");
}

void LLMMACnet::llmCheckStatus() {
	static int status_check_count = 0;
	status_check_count++;

	if (status_check_count % 100000 == 0) {
		LLM_DEBUG("Status check #" << status_check_count << " at cycle " << cycles);
		LLM_DEBUG("Ready flag: " << ready_flag << ", Mapping again: " << mapping_again);
	}

	if (ready_flag == 0) { // New layer initialization
		LLM_DEBUG("Initializing new layer at cycle " << cycles);

		if (mapping_again == 0) {
			this->vcNetwork->resetVNRoundRobin();
		}

		// Perform task mapping
		#ifdef rowmapping
		this->llmXMapping(total_tasks);
		#else
		this->llmLoadBalanceMapping(total_tasks);
		#endif

		// Assign tasks to MAC units
		int active_macs = 0;
		for (int i = 0; i < macNum; i++) {
			if (mapping_table[i].size() == 0) {
				this->LLMMAC_list[i]->selfstatus = 5; // No tasks assigned
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

	// Check if all MAC units are finished
	int finished_count = 0;
	int active_count = 0;
	for (int i = 0; i < macNum; i++) {
		if (LLMMAC_list[i]->selfstatus == 5 && LLMMAC_list[i]->send == 3) {
			finished_count++;
		} else if (LLMMAC_list[i]->selfstatus != 5) {
			active_count++;
		}
	}

	if (status_check_count % 100000 == 0) {
		LLM_DEBUG("Status: " << finished_count << " finished, " << active_count << " active out of " << macNum << " total MACs");
	}

	// 计算实际分配了任务的MAC数量
	int assigned_macs = 0;
	for (int i = 0; i < macNum; i++) {
		if (mapping_table[i].size() > 0) {
			assigned_macs++;
		}
	}

	if (finished_count >= assigned_macs) {
		// 所有分配了任务的MAC单元都已完成
		LLM_DEBUG("All LLM attention tasks completed at cycle " << cycles);
		LLM_DEBUG("Assigned MACs: " << assigned_macs << ", Finished: " << finished_count);
		LLM_DEBUG("Total packets sent: " << packet_id);

		// 统计有多少非零结果
		int non_zero_count = 0;
		int total_positions = 0;
		float min_val = 1e10, max_val = -1e10;
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				total_positions++;
				float val = attention_output_table[i][j];
				if (abs(val) > 1e-10) {
					non_zero_count++;
					min_val = std::min(min_val, val);
					max_val = std::max(max_val, val);
				}
			}
		}

		LLM_DEBUG("Output statistics: " << non_zero_count << "/" << total_positions << " non-zero values");
		if (non_zero_count > 0) {
			LLM_DEBUG("Value range: " << min_val << " to " << max_val);
		}

		layer_latency.push_back(cycles);
		ready_flag = 2; // Finished
		last_layer_packet_id = packet_id;
		return;
	}

	ready_flag = 1; // Continue processing
}

void LLMMACnet::llmRunOneStep() {
	static int run_step_count = 0;
	run_step_count++;

	// Run one step for each LLMMAC unit
	for (int i = 0; i < macNum; i++) {
		LLMMAC_list[i]->llmRunOneStep();
	}

	// Debug: Print status every 100k cycles
	if (run_step_count % 100000 == 0) {
		int active_macs = 0;
		for (int i = 0; i < macNum; i++) {
			if (LLMMAC_list[i]->selfstatus != 5) {
				active_macs++;
			}
		}
		LLM_DEBUG("Run step #" << run_step_count << ": " << active_macs << " active MACs");
	}

	// Handle memory operations
	int pbuffer_size;
	int src, pid_signal_id, mem_id, src_mac;
	LLMMAC *tmpLLMMAC;
	Packet *tmpPacket;
	NI *tmpNI;

	// Process memory nodes
	for (int memidx = 0; memidx < MEM_NODES; memidx++) {
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];

		// Handle type 0 requests (MAC to MEM)
		pbuffer_size = tmpNI->packet_buffer_out[0].size();
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();

			if (tmpPacket->message.type != 0 || tmpPacket->message.out_cycle >= cycles) {
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}

			src = tmpPacket->message.source_id;
			pid_signal_id = tmpPacket->message.signal_id;
			src_mac = tmpPacket->message.mac_id;

			tmpLLMMAC = LLMMAC_list[src_mac];

			if (tmpLLMMAC->selfstatus == 2) { // Waiting for data
				// Prepare response data for LLM attention
				int task_id = tmpLLMMAC->request;
				if (task_id < all_tasks.size()) {
					LLMTask& task = all_tasks[task_id];

					// Prepare input buffer with task data
					tmpLLMMAC->input_buffer.clear();
					tmpLLMMAC->input_buffer.push_back(1.0f); // Function type (attention)
					tmpLLMMAC->input_buffer.push_back(32.0f); // Data size (16+16=32)
					tmpLLMMAC->input_buffer.push_back(task.time_slice); // Time slice
					tmpLLMMAC->input_buffer.push_back(task.pixel_x * matrix_size + task.pixel_y); // Pixel ID

					// Add query data (16 elements)
					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.query_data.begin(), task.query_data.end());

					// Add key data (16 elements)
					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.key_data.begin(), task.key_data.end());

					// Debug: Print data being sent
					if (src_mac < 3) {
						LLM_DEBUG("Sending data to MAC " << src_mac << " for task " << task_id);
						LLM_DEBUG("Query data[0-3]: " << task.query_data[0] << ", " << task.query_data[1] << ", " << task.query_data[2] << ", " << task.query_data[3]);
						LLM_DEBUG("Key data[0-3]: " << task.key_data[0] << ", " << task.key_data[1] << ", " << task.key_data[2] << ", " << task.key_data[3]);
						LLM_DEBUG("Total input buffer size: " << tmpLLMMAC->input_buffer.size());
					}

					// Send response
					int mem_delay = static_cast<int>(ceil((32 * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
					LLMMAC_list[mem_id]->pecycle = cycles + mem_delay;
					LLMMAC_list[mem_id]->input_buffer = tmpLLMMAC->input_buffer;
					LLMMAC_list[mem_id]->llmInject(1, src, tmpLLMMAC->input_buffer.size(),
						1.0f, vcNetwork->NI_list[mem_id], pid_signal_id, src_mac);
				}
			}
			tmpNI->packet_buffer_out[0].pop_front();
		}

		// Handle type 2 results (MAC to MEM)
		pbuffer_size = tmpNI->packet_buffer_out[1].size();
		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[1].front();

			if (tmpPacket->message.type != 2) {
				tmpNI->packet_buffer_out[1].pop_front();
				tmpNI->packet_buffer_out[1].push_back(tmpPacket);
				continue;
			}

			src = tmpPacket->message.source_id;
			pid_signal_id = tmpPacket->message.signal_id;
			src_mac = tmpPacket->message.mac_id;

			tmpLLMMAC = LLMMAC_list[src_mac];

			// Store result in output matrix
			int pixel_x = tmpPacket->message.data[1];
			int pixel_y = tmpPacket->message.data[2];
			float result_value = tmpPacket->message.data[0];

			if (pixel_x < matrix_size && pixel_y < matrix_size) {
				attention_output_table[pixel_y][pixel_x] = result_value;
				if (src_mac < 3) {
					LLM_DEBUG("Stored result from MAC " << src_mac << " at position (" << pixel_x << "," << pixel_y << "): " << result_value);
				}
			} else {
				LLM_DEBUG("ERROR: Invalid pixel position (" << pixel_x << "," << pixel_y << ") from MAC " << src_mac);
			}

			if (tmpLLMMAC->selfstatus == 5) {
				tmpLLMMAC->send = 3;
			}
			tmpNI->packet_buffer_out[1].pop_front();
		}
	}

	// Handle responses (MEM to MAC)
	for (int i = 0; i < TOT_NUM; i++) {
		if (llmIsMemoryNode(i)) {
			continue;
		}

		tmpNI = this->vcNetwork->NI_list[i];
		pbuffer_size = tmpNI->packet_buffer_out[0].size();

		for (int j = 0; j < pbuffer_size; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();

			if (tmpPacket->message.type != 1) {
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}

			src_mac = tmpPacket->message.mac_id;
			tmpLLMMAC = LLMMAC_list[src_mac];

			// Process response
			tmpLLMMAC->llmReceive(&tmpPacket->message);

			tmpNI->packet_buffer_out[0].pop_front();
		}
	}
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
