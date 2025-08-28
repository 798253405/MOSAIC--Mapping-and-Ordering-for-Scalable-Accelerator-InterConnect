// 小矩阵版本的 llmmacnet.cpp - 4x4可调试
#include "llmmacnet.hpp"
#include "llmmac.hpp"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <climits>
#include <chrono>

// Helper function to get current time string
static inline std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm* timeinfo = localtime(&time_t);
    char buffer[10];
    strftime(buffer, sizeof(buffer), "%H:%M", timeinfo);
    return std::string(buffer);
}

// Hierarchical debug macros based on LLM_DEBUG_LEVEL from parameters.hpp
// With cycle info and system time (for runtime use)
#define LLM_INFO(x) do { \
    if (LLM_DEBUG_LEVEL >= 1) { \
        std::cout << "[" << getCurrentTimeStr() << "] [INFO @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_DEBUG(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[DEBUG @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[TRACE @" << cycles << "] " << x << std::endl; \
    } \
} while(0)

// Without cycle info (for initialization)
#define LLM_INFO_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 1) { \
        std::cout << "[" << getCurrentTimeStr() << "] [INFO @init] " << x << std::endl; \
    } \
} while(0)

#define LLM_DEBUG_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 2) { \
        std::cout << "[DEBUG @init] " << x << std::endl; \
    } \
} while(0)

#define LLM_TRACE_INIT(x) do { \
    if (LLM_DEBUG_LEVEL >= 3) { \
        std::cout << "[TRACE @init] " << x << std::endl; \
    } \
} while(0)

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

	// Use configuration from parameters.hpp
	#if LLM_TEST_CASE == 1
	// Test Case 1: Small matrix test
	matrix_size = 4;
	tile_size = 4;
	time_slices = 1;
	LLM_INFO_INIT("\n[LLM Config] Test Case 1: Small 4x4 matrix");
	
	#elif LLM_TEST_CASE == 2
	// Test Case 2: Scaled LLaMA attention matrix (128x128 for testing)
	matrix_size = 128;  // Scaled down for comprehensive testing
	tile_size = 16;     // 16x16 tiles
	time_slices = 4;    // Standard 4 time slices for non-debug models
	LLM_INFO_INIT("\n[LLM Config] Test Case 2: Scaled LLaMA 128x128 attention matrix with 4 time slices");
	
	#elif LLM_TEST_CASE == 3
	// Test Case 3: Large configuration for SAMOS testing
	matrix_size = 256;
	tile_size = 16;
	time_slices = 4;
	LLM_INFO_INIT("\n[LLM Config] Test Case 3: Large 256x256 for SAMOS testing");
	
	#else
	// Default to test case 1 if not specified
	matrix_size = 4;
	tile_size = 4;
	time_slices = 1;
	LLM_INFO_INIT("\n[LLM Config] Default: Test Case 1 (4x4 matrix)");
	#endif

	// Calculate derived parameters
	tiles_per_dim = matrix_size / tile_size;
	total_tiles = tiles_per_dim * tiles_per_dim;
	
	// For small matrices where tile_size == matrix_size, we process each pixel as a task
	if (tile_size == matrix_size) {
		// Each pixel is a task
		total_tasks = matrix_size * matrix_size * time_slices;
	} else {
		// Each tile generates tile_size x tile_size tasks
		total_tasks = total_tiles * tile_size * tile_size * time_slices;
	}
	
	LLM_INFO_INIT("  Matrix: " << matrix_size << "x" << matrix_size);
	LLM_INFO_INIT("  Tile size: " << tile_size << "x" << tile_size);
	LLM_INFO_INIT("  Tiles: " << tiles_per_dim << "x" << tiles_per_dim << " = " << total_tiles);
	LLM_INFO_INIT("  Time slices: " << time_slices);
	LLM_INFO_INIT("  Total tasks: " << total_tasks);

	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;

	LLM_DEBUG_INIT("Creating LLMMACnet with " << macNum << " MAC units");
	LLM_TRACE_INIT("Matrix size: " << matrix_size << "x" << matrix_size);
	LLM_TRACE_INIT("Tile size: " << tile_size << "x" << tile_size);
	LLM_TRACE_INIT("Time slices: " << time_slices);
	LLM_TRACE_INIT("Total tiles: " << total_tiles);
	LLM_TRACE_INIT("Total tasks: " << total_tasks);

	for (int i = 0; i < macNum; i++) {
		int temp_ni_id = i % TOT_NUM;
		LLMMAC *newLLMMAC = new LLMMAC(i, this, temp_ni_id);
		LLMMAC_list.push_back(newLLMMAC);
	}

	LLM_DEBUG_INIT("Created " << LLMMAC_list.size() << " LLMMAC units");

	// Initialize matrices
	llmInitializeMatrices();

	// Export matrices for verification
	llmExportMatricesToFile();

	// Generate all tasks
	llmGenerateAllTasks();

	// Export tasks for verification
	llmExportTasksToFile();

	layer_latency.clear();
	LLM_DEBUG_INIT("LLMMACnet initialized successfully!");
}

void LLMMACnet::llmInitializeMatrices() {
	LLM_DEBUG_INIT("Initializing " << matrix_size << "x" << matrix_size << " attention matrices...");

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

	// 仅在TRACE级别打印完整矩阵
	if (LLM_DEBUG_LEVEL >= 3) {
		LLM_TRACE("\n=== Query Matrix (" << matrix_size << "x" << matrix_size << ") ===");
		for (int i = 0; i < matrix_size && i < 4; i++) {  // 限制最多打印4行
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrix_size && j < 4; j++) {
				std::cout << std::fixed << std::setprecision(6) << attention_query_table[i][j] << "\t";
			}
			if (matrix_size > 4) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrix_size > 4) std::cout << "..." << std::endl;

		LLM_TRACE("\n=== Key Matrix (" << matrix_size << "x" << matrix_size << ") ===");
		for (int i = 0; i < matrix_size && i < 4; i++) {
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrix_size && j < 4; j++) {
				std::cout << std::fixed << std::setprecision(6) << attention_key_table[i][j] << "\t";
			}
			if (matrix_size > 4) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrix_size > 4) std::cout << "..." << std::endl;
	}

	LLM_DEBUG_INIT("Matrices initialized successfully");
}

void LLMMACnet::llmExportMatricesToFile() {
	LLM_DEBUG_INIT("Exporting matrices to output/ for verification...");

	// Export Query matrix
	std::ofstream query_file("src/output/cpp_query_matrix.txt");
	if (query_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				query_file << std::fixed << std::setprecision(10) << attention_query_table[i][j];
				if (j < matrix_size - 1) query_file << ",";
			}
			query_file << "\n";
		}
		query_file.close();
		LLM_DEBUG_INIT("Query matrix exported to src/output/cpp_query_matrix.txt");
	}

	// Export Key matrix
	std::ofstream key_file("src/output/cpp_key_matrix.txt");
	if (key_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				key_file << std::fixed << std::setprecision(10) << attention_key_table[i][j];
				if (j < matrix_size - 1) key_file << ",";
			}
			key_file << "\n";
		}
		key_file.close();
		LLM_DEBUG_INIT("Key matrix exported to src/output/cpp_key_matrix.txt");
	}

	// Export Value matrix
	std::ofstream value_file("src/output/cpp_value_matrix.txt");
	if (value_file.is_open()) {
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				value_file << std::fixed << std::setprecision(10) << attention_value_table[i][j];
				if (j < matrix_size - 1) value_file << ",";
			}
			value_file << "\n";
		}
		value_file.close();
		LLM_DEBUG_INIT("Value matrix exported to src/output/cpp_value_matrix.txt");
	}
}

void LLMMACnet::llmGenerateAllTasks() {
	LLM_DEBUG_INIT("Generating " << total_tasks << " LLM tasks...");
	all_tasks.clear();
	all_tasks.reserve(total_tasks);

	int task_id = 0;
	
	// Generate tasks based on tiles and time slices
	for (int ts = 0; ts < time_slices; ts++) {
		for (int tile_y = 0; tile_y < tiles_per_dim; tile_y++) {
			for (int tile_x = 0; tile_x < tiles_per_dim; tile_x++) {
				// Each tile generates tile_size x tile_size tasks (one per pixel in tile)
				for (int in_tile_y = 0; in_tile_y < tile_size; in_tile_y++) {
					for (int in_tile_x = 0; in_tile_x < tile_size; in_tile_x++) {
						int pixel_x = tile_x * tile_size + in_tile_x;
						int pixel_y = tile_y * tile_size + in_tile_y;
						
						// Skip if out of bounds (for non-divisible matrices)
						if (pixel_x >= matrix_size || pixel_y >= matrix_size)
							continue;
						
						LLMTask task;
						task.task_id = task_id;
						task.pixel_x = pixel_x;
						task.pixel_y = pixel_y;
						task.time_slice = ts;
						task.tile_id = tile_y * tiles_per_dim + tile_x;

						// Generate query and key data
						int data_elements = tile_size;  // Process tile_size elements per task
						task.query_data.resize(data_elements);
						task.key_data.resize(data_elements);
						task.value_data.resize(data_elements);

						for (int i = 0; i < data_elements; i++) {
							int data_idx = (ts * data_elements + i) % matrix_size;
							task.query_data[i] = attention_query_table[pixel_y][(pixel_x + data_idx) % matrix_size];
							task.key_data[i] = attention_key_table[pixel_y][(pixel_x + data_idx) % matrix_size];
							task.value_data[i] = attention_value_table[pixel_y][(pixel_x + data_idx) % matrix_size];
						}

						// Print debug info for first few tasks only
						if (task_id < 3 && LLM_DEBUG_LEVEL >= 3) {
							LLM_TRACE("Task " << task_id << " [pixel(" << pixel_x << "," << pixel_y << "), ts=" << ts << ", tile=" << task.tile_id << "]:");
							std::cout << "  Query data: ";
							for (int i = 0; i < data_elements && i < 4; i++) {
								std::cout << std::fixed << std::setprecision(6) << task.query_data[i];
								if (i < data_elements - 1) std::cout << ",";
							}
							if (data_elements > 4) std::cout << "...";
							std::cout << std::endl;
						}

						all_tasks.push_back(task);
						task_id++;
					}
				}
			}
		}
	}

	LLM_DEBUG_INIT("Generated " << all_tasks.size() << " tasks successfully");
}

void LLMMACnet::llmExportTasksToFile() {
	LLM_DEBUG_INIT("Exporting all tasks to output/cpp_tasks.txt...");

	std::ofstream tasks_file("src/output/cpp_tasks.txt");
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
		LLM_DEBUG_INIT("All " << all_tasks.size() << " tasks exported to output/cpp_tasks.txt");
	}
}

void LLMMACnet::llmExportVerificationResults() {
	LLM_DEBUG("Exporting verification results...");

	std::ofstream verify_file("src/output/cpp_verification.txt");
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
		LLM_DEBUG("Verification results exported to src/output/cpp_verification.txt");
	}

	// 导出最终输出矩阵到单独文件
	std::ofstream output_file("src/output/llm_attention_output.txt");
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

	// 打印详细的任务分配 (Level 2+)
	if (LLM_DEBUG_LEVEL >= 2) {
		for (int i = 0; i < macNum; i++) {
			if (mapping_table[i].size() > 0) {
				std::cout << "[DEBUG @" << cycles << "] MAC " << i << " tasks: ";
				for (int task_id : mapping_table[i]) {
					std::cout << task_id << " ";
				}
				std::cout << std::endl;
			}
		}
	}
}

void LLMMACnet::llmLoadBalanceMapping(int total_tasks) {
	LLM_DEBUG("Starting load-balanced mapping...");
	llmXMapping(total_tasks);  // 对于小矩阵，使用简单映射即可
}

int LLMMACnet::llmSAMOSSampleMapping(int task_count, int start_task_id) {
	// Clear and prepare mapping table
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);
	
	// 1) Collect compute nodes (exclude memory nodes)
	std::vector<int> pe_ids;
	pe_ids.reserve(macNum);
	for (int id = 0; id < macNum; ++id) {
		if (!contains(dest_list, id))
			pe_ids.push_back(id);
	}
	if (pe_ids.empty() || task_count <= 0)
		return 0;
	
	// 2) Calculate average latency for each node (from sampling window)
	//    Fallback strategy: use average of all non-zero samples if any node has no samples
	double sum_lat = 0.0;
	int nz = 0;
	for (int id : pe_ids) {
		double lat = double(samplingWindowDelay[id]) / std::max(1, samplingWindowLength);
		if (lat > 0.0) {
			sum_lat += lat;
			++nz;
		}
	}
	const double default_lat = (nz > 0) ? (sum_lat / nz) : 1.0; // Default to 1 if all zeros
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
		int base = task_count / int(nodes.size());
		int rem = task_count - base * int(nodes.size());
		int j = start_task_id;
		for (auto &n : nodes) {
			for (int k = 0; k < base; ++k)
				this->mapping_table[n.id].push_back(j++);
		}
		for (int i = 0; i < rem; ++i)
			this->mapping_table[nodes[i].id].push_back(j++);
		return 0;
	}
	
	// 3) Hamilton's method (largest remainder)
	int allocated = 0;
	for (auto &n : nodes) {
		double exact = task_count * (n.w / sumW);
		n.want = exact;
		n.alloc = int(std::floor(exact));
		n.frac = exact - n.alloc;
		allocated += n.alloc;
	}
	int remainder = task_count - allocated;
	
	// Allocate remaining tasks to nodes with largest fractional parts
	std::sort(nodes.begin(), nodes.end(), [](const NodeW &a, const NodeW &b) {
		return a.frac > b.frac;
	});
	for (int i = 0; i < remainder; ++i)
		nodes[i % nodes.size()].alloc++;
	
	// 4) Generate specific routing mapping (task ids increment continuously)
	int j = start_task_id;
	for (auto &n : nodes) {
		for (int k = 0; k < n.alloc; ++k)
			this->mapping_table[n.id].push_back(j++);
	}
	
	// Debug output for LLM SAMOS mapping
	if (LLM_DEBUG_LEVEL >= 2) {
		std::cout << "[SAMOS] Total tasks=" << task_count << " Total PEs=" 
		          << nodes.size() << " Total allocated=" << j << "\n";
		if (LLM_DEBUG_LEVEL >= 3) {
			for (auto &n : nodes) {
				double avgLat = double(samplingWindowDelay[n.id]) / std::max(1, samplingWindowLength);
				std::cout << "  MAC " << n.id << " lat=" << avgLat << " w=" << n.w
				          << " want=" << n.want << " alloc=" << n.alloc << "\n";
			}
		}
	}
	
	return 0;
}

void LLMMACnet::llmCheckStatus() {
	static int status_check_count = 0;
	status_check_count++;

	// Progress reporting at different levels
	if (cycles % 50000 == 0 && LLM_DEBUG_LEVEL >= 1) {  // Level 1: Basic progress every 50k cycles
		int progress = (executed_tasks * 100) / std::max(1, total_tasks);
		LLM_INFO("[Cycle " << cycles << "] Progress: " << progress << "% (" << executed_tasks << "/" << total_tasks << " tasks)");
	} else if (status_check_count % 10000 == 0 && LLM_DEBUG_LEVEL >= 2) {  // Level 2: More frequent status
		LLM_DEBUG("Status check #" << status_check_count << " at cycle " << cycles);
		LLM_DEBUG("Ready flag: " << ready_flag << ", Mapping again: " << mapping_again);
	}

	if (ready_flag == 0) {
		LLM_INFO("Initializing layer " << current_layer << " at cycle " << cycles);

		if (mapping_again == 0) {
			this->vcNetwork->resetVNRoundRobin();
		}

		// SAMOS mapping logic for LLM
		#ifdef YZSAMOSSampleMapping
		// Calculate how many tasks per MAC for sampling window
		int available_macs = macNum - MEM_NODES;  // Exclude memory nodes
		
		if (total_tasks / available_macs < samplingWindowLength) {
			// If tasks are fewer than sampling window, use normal row mapping
			LLM_DEBUG("[SAMOS] Layer has fewer tasks than sampling window!");
			LLM_DEBUG("  Total tasks: " << total_tasks << ", Available MACs: " << available_macs);
			LLM_DEBUG("  Tasks per MAC: " << (total_tasks / available_macs) << " < " << samplingWindowLength);
			LLM_DEBUG("  Using row mapping instead of SAMOS");
			this->llmXMapping(total_tasks);
		} else {
			if (mapping_again == 0) {
				// First phase: run sampling window
				int sampling_tasks = available_macs * samplingWindowLength;
				LLM_DEBUG("[SAMOS] Starting sampling phase");
				LLM_DEBUG("  Sampling tasks: " << sampling_tasks << " (" << available_macs 
				          << " MACs * " << samplingWindowLength << " window)");
				
				// Reset sampling statistics
				std::fill_n(samplingWindowDelay, TOT_NUM, 0);
				
				// Map sampling window tasks using row mapping
				this->llmXMapping(sampling_tasks);
				mapping_again = 1;  // Mark that sampling is being done
				
			} else if (mapping_again == 2) {
				// Second phase: map remaining tasks based on sampling results
				int sampling_tasks = available_macs * samplingWindowLength;
				int remaining_tasks = total_tasks - sampling_tasks;
				
				std::cout << "[SAMOS DEBUG] Phase 2 mapping:" << std::endl;
				std::cout << "  Total tasks: " << total_tasks << std::endl;
				std::cout << "  Available MACs: " << available_macs << std::endl;
				std::cout << "  Sampling window: " << samplingWindowLength << std::endl;
				std::cout << "  Sampling tasks: " << sampling_tasks << std::endl;
				std::cout << "  Remaining tasks: " << remaining_tasks << std::endl;
				std::cout << "  Current packet_id: " << packet_id << std::endl;
				
				packet_id = packet_id + sampling_tasks;
				
				LLM_DEBUG("[SAMOS] Applying SAMOS mapping for remaining tasks");
				LLM_DEBUG("  Remaining tasks: " << remaining_tasks);
				
				// Use SAMOS mapping based on latency measurements
				// Generate task IDs directly from sampling_tasks to sampling_tasks + remaining_tasks - 1
				int start_task_id = sampling_tasks;
				std::cout << "[SAMOS DEBUG] Task IDs will range from " << start_task_id << " to " << (start_task_id + remaining_tasks - 1) << std::endl;
				
				this->llmSAMOSSampleMapping(remaining_tasks, start_task_id);
				
				LLM_DEBUG("[SAMOS] Second phase mapping complete");
				mapping_again = 0;  // Reset for next layer
			} else {
				LLM_INFO("[SAMOS] ERROR: Invalid mapping_again state: " << mapping_again);
			}
		}
		#else
		// Normal mapping without SAMOS
		#ifdef rowmapping
		this->llmXMapping(total_tasks);
		#else
		this->llmLoadBalanceMapping(total_tasks);
		#endif
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

		LLM_INFO("Activated " << active_macs << " MAC units with tasks");
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

	if (status_check_count % 10000 == 0 && LLM_DEBUG_LEVEL >= 2) {
		LLM_DEBUG("Status: " << finished_count << " finished, " << active_count << " active out of " << assigned_macs << " assigned MACs");
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
		
		LLM_INFO("\n=== All tasks completed at cycle " << cycles << " ===");
		LLM_INFO("Total executed tasks: " << executed_tasks << "/" << total_tasks);

		// 打印最终结果矩阵 - 仅在Level 2+显示
		if (LLM_DEBUG_LEVEL >= 2) {
			std::cout << "\n=== Final Output Matrix (" << matrix_size << "x" << matrix_size << ") ===" << std::endl;
			int zero_count = 0;
			int non_zero_count = 0;

			int max_rows = (LLM_DEBUG_LEVEL >= 3) ? matrix_size : std::min(4, matrix_size);
			for (int i = 0; i < max_rows; i++) {
				std::cout << "Row " << i << ": ";
				int max_cols = (LLM_DEBUG_LEVEL >= 3) ? matrix_size : std::min(4, matrix_size);
				for (int j = 0; j < max_cols; j++) {
					float value = attention_output_table[i][j];
					std::cout << std::fixed << std::setprecision(6) << value << "\t";
				}
				if (matrix_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "...";
				std::cout << std::endl;
			}
			if (matrix_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "..." << std::endl;

			// Count all values
			for (int i = 0; i < matrix_size; i++) {
				for (int j = 0; j < matrix_size; j++) {
					if (attention_output_table[i][j] == 0.0) {
						zero_count++;
					} else {
						non_zero_count++;
					}
				}
			}
			std::cout << "\nStatistics: " << non_zero_count << " non-zero, " << zero_count << " zero values" << std::endl;
		}

		// Print timing statistics
		llmPrintTimingStatistics();

		// 导出验证结果
		llmExportVerificationResults();

		layer_latency.push_back(cycles);
		ready_flag = 2;
		
		// Adjust packet_id for next layer based on actual tasks processed
		#ifdef YZSAMOSSampleMapping
		int available_macs = macNum - MEM_NODES;
		if (total_tasks / available_macs < samplingWindowLength) {
			// Used normal mapping, add all tasks
			packet_id = packet_id + total_tasks;
		} else {
			// Used SAMOS mapping, already adjusted during mapping
			// No need to adjust here as it was done incrementally
		}
		#else
		packet_id = packet_id + total_tasks;
		#endif
		
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

	if (run_step_count % 50000 == 0) {
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
					
					// Track when request arrives at memory
					tmpLLMMAC->current_task_timing.request_arrive_cycle = cycles;

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

						LLM_DEBUG("Memory " << mem_id << " received request from MAC " << src_mac
						          << " for task " << task_id << " at cycle " << cycles
						          << " [pixel(" << task.pixel_x << "," << task.pixel_y
						          << "), ts=" << task.time_slice << "]");

						int mem_delay = static_cast<int>(ceil((task.query_data.size() * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
						LLMMAC_list[mem_id]->pecycle = cycles + mem_delay;
						
						// Track when response will be sent
						tmpLLMMAC->current_task_timing.response_send_cycle = cycles + mem_delay;
						LLMMAC_list[mem_id]->input_buffer = tmpLLMMAC->input_buffer;
						LLMMAC_list[mem_id]->llmInject(1, src, tmpLLMMAC->input_buffer.size(),
							1.0f, vcNetwork->NI_list[mem_id], pid_signal_id, src_mac);
					}
					tmpNI->packet_buffer_out[0].pop_front();
				}
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
							
							// 增加已执行任务计数
							executed_tasks++;

							std::cout << "[TABLE-UPDATE] Updated output table:" << std::endl;
							std::cout << "  Position [" << pixel_y << "][" << pixel_x << "]" << std::endl;
							std::cout << "  Old value: " << std::fixed << std::setprecision(10) << old_value << std::endl;
							std::cout << "  New value: " << std::fixed << std::setprecision(10) << result_value << std::endl;
							std::cout << "  Verification: " << attention_output_table[pixel_y][pixel_x] << std::endl;
							std::cout << "  Tasks completed: " << executed_tasks << "/" << total_tasks << std::endl;

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

	// 定期打印输出矩阵状态（仅在DEBUG级别2或3）
	if (run_step_count % 50000 == 0 && LLM_DEBUG_LEVEL >= 2) {
		LLM_DEBUG("\n[MATRIX-STATUS] Current output matrix at step " << run_step_count << ":");
		int non_zero_count = 0;
		int max_rows = (LLM_DEBUG_LEVEL >= 3) ? matrix_size : std::min(4, matrix_size);
		for (int i = 0; i < max_rows; i++) {
			std::cout << "Row " << i << ": ";
			int max_cols = (LLM_DEBUG_LEVEL >= 3) ? matrix_size : std::min(4, matrix_size);
			for (int j = 0; j < max_cols; j++) {
				float val = attention_output_table[i][j];
				if (val != 0.0) {
					non_zero_count++;
					std::cout << std::fixed << std::setprecision(6) << val << " ";
				} else {
					std::cout << "0.000000 ";
				}
			}
			if (matrix_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrix_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "..." << std::endl;
		
		// 总是显示非零元素统计
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				if (attention_output_table[i][j] != 0.0) non_zero_count++;
			}
		}
		std::cout << "Non-zero elements: " << non_zero_count << "/" << (matrix_size * matrix_size) << std::endl;
	}
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
