#include "llmmacnet.hpp"
#include "llmmac.hpp"
#include <cassert>

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
	total_layers = 1; // For now, single attention layer
	matrix_size = 512;
	tile_size = 16;
	tiles_per_dim = matrix_size / tile_size; // 32
	total_tiles = tiles_per_dim * tiles_per_dim; // 1024
	time_slices = 4;
	total_tasks = matrix_size * matrix_size * time_slices; // 512*512*4 = 1,048,576

	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;

	int temp_ni_id;
	cout << "Creating LLMMACnet with " << macNum << " MAC units" << endl;
	cout << "Matrix size: " << matrix_size << "x" << matrix_size << endl;
	cout << "Tile size: " << tile_size << "x" << tile_size << endl;
	cout << "Total tiles: " << total_tiles << endl;
	cout << "Total tasks: " << total_tasks << endl;

	// Create LLMMAC units
	for (int i = 0; i < macNum; i++) {
		temp_ni_id = i % TOT_NUM;
		LLMMAC *newLLMMAC = new LLMMAC(i, this, temp_ni_id);
		LLMMAC_list.push_back(newLLMMAC);
	}

	// Initialize matrices
	llmInitializeMatrices();

	// Generate all tasks
	llmGenerateAllTasks();

	layer_latency.clear();
	cout << "LLMMACnet created successfully!" << endl;
}

void LLMMACnet::llmInitializeMatrices() {
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
	cout << "Attention matrices initialized" << endl;
}

void LLMMACnet::llmGenerateAllTasks() {
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

				// Generate query and key data (64 elements each for this time slice)
				task.query_data.resize(64);
				task.key_data.resize(64);
				task.value_data.resize(64);

				for (int i = 0; i < 64; i++) {
					// Extract relevant data based on time slice and pixel position
					int data_idx = (ts * 64 + i) % matrix_size;
					task.query_data[i] = attention_query_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.key_data[i] = attention_key_table[pixel_y][(pixel_x + data_idx) % matrix_size];
					task.value_data[i] = attention_value_table[pixel_y][(pixel_x + data_idx) % matrix_size];
				}

				all_tasks.push_back(task);
			}
		}
	}

	cout << "Generated " << all_tasks.size() << " tasks" << endl;
}

int LLMMACnet::llmGetTileId(int pixel_x, int pixel_y) {
	int tile_x = pixel_x / tile_size;
	int tile_y = pixel_y / tile_size;
	return tile_y * tiles_per_dim + tile_x;
}

int LLMMACnet::llmGetMACIdForTile(int tile_id) {
	// Map tile to MAC unit (simple modulo mapping, excluding memory nodes)
	int mac_id = tile_id % macNum;
	while (llmIsMemoryNode(mac_id)) {
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
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	int j = 0;
	while (j < total_tasks) {
		for (int i = 0; i < macNum; i++) {
			int temp_i = i % TOT_NUM;
			if (llmIsMemoryNode(temp_i)) {
				continue;
			}

			this->mapping_table[i].push_back(j);
			j = j + 1;
			if (j == total_tasks)
				break;
		}
	}

	cout << "LLM X-mapping completed, mapped " << j << " tasks" << endl;
}

void LLMMACnet::llmLoadBalanceMapping(int total_tasks) {
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	// Count non-memory nodes
	vector<int> pe_ids;
	for (int i = 0; i < macNum; i++) {
		if (!llmIsMemoryNode(i % TOT_NUM)) {
			pe_ids.push_back(i);
		}
	}

	int tasks_per_pe = total_tasks / pe_ids.size();
	int remainder = total_tasks % pe_ids.size();

	int task_id = 0;
	for (int i = 0; i < pe_ids.size(); i++) {
		int pe_id = pe_ids[i];
		int tasks_for_this_pe = tasks_per_pe + (i < remainder ? 1 : 0);

		for (int j = 0; j < tasks_for_this_pe; j++) {
			this->mapping_table[pe_id].push_back(task_id++);
		}
	}

	cout << "LLM Load-balance mapping completed for " << pe_ids.size() << " PEs" << endl;
}

void LLMMACnet::llmCheckStatus() {
	if (ready_flag == 0) { // New layer initialization
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
		for (int i = 0; i < macNum; i++) {
			if (mapping_table[i].size() == 0) {
				this->LLMMAC_list[i]->selfstatus = 5; // No tasks assigned
				this->LLMMAC_list[i]->send = 3;
			} else {
				this->LLMMAC_list[i]->routing_table.assign(
					mapping_table[i].begin(), mapping_table[i].end());
			}
		}
		ready_flag = 1;
		return;
	}

	// Check if all MAC units are finished
	for (int i = 0; i < macNum; i++) {
		if (LLMMAC_list[i]->selfstatus != 5) {
			ready_flag = 1;
			return;
		}
		if (LLMMAC_list[i]->send != 3) {
			ready_flag = 1;
			return;
		}
	}

	// All tasks completed
	cout << "All LLM attention tasks completed at cycle " << cycles << endl;
	cout << "Total packets sent: " << packet_id << endl;
	layer_latency.push_back(cycles);
	ready_flag = 2; // Finished

	last_layer_packet_id = packet_id;
}

void LLMMACnet::llmRunOneStep() {
	// Run one step for each LLMMAC unit
	for (int i = 0; i < macNum; i++) {
		LLMMAC_list[i]->llmRunOneStep();
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
					tmpLLMMAC->input_buffer.push_back(128.0f); // Data size
					tmpLLMMAC->input_buffer.push_back(task.time_slice); // Time slice
					tmpLLMMAC->input_buffer.push_back(task.pixel_x * matrix_size + task.pixel_y); // Pixel ID

					// Add query data (64 elements)
					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.query_data.begin(), task.query_data.end());

					// Add key data (64 elements)
					tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
						task.key_data.begin(), task.key_data.end());

					// Send response
					int mem_delay = static_cast<int>(ceil((128 * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
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
			if (pixel_x < matrix_size && pixel_y < matrix_size) {
				attention_output_table[pixel_y][pixel_x] = tmpPacket->message.data[0];
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
