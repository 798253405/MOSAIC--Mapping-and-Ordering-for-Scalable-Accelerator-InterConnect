// 小矩阵版本的 llmmacnet.cpp - 4x4可调试

/*
==================================================================
LLM 排序优化7步骤总览
目标：减少NoC传输中的bit翻转，提高能效
==================================================================

Step 1: 数据生成与加载
  函数：LLMMACnet::llmLoadRealMatrices() [llmmacnet.cpp]
  功能：从文件加载真实LLaMA矩阵或生成随机数据
  - 1.1: 初始化矩阵尺寸为512x512
  - 1.2: 分配Query/Key/Value/Output矩阵存储空间
  - 1.3: 尝试打开LLaMA数据文件
  - 1.4: 读取文件维度信息
  - 1.5: 逐行读取矩阵数据
  - 1.6: 循环复制填充不足的行/列
  - 1.7: 对Key矩阵重复加载流程

Step 2: 任务分配与数据提取  
  函数：LLMMACnet::llmGenerateAllTasks() [llmmacnet.cpp]
  功能：将512x512大矩阵分解为多个小任务
  - 2.1: 计算任务参数（262,144像素×4子任务=1,048,576任务）
  - 2.2: 根据像素生成任务
  - 2.3: 遍历所有512×512像素位置
  - 2.4: 每像素生成4个子任务（2×2 subchunks）
  - 2.5: 计算数据偏移量（每子块64×64）
  - 2.6: 提取64个Query和64个Key元素
  - 2.7: 根据模式选择数据提取方式（权重为主/输入为主）

Step 3: 数据打包为Payload
  函数：LLMMACnet::llmRunOneStep() [llmmacnet.cpp]
  功能：将任务数据打包成NoC传输格式
  - 3.1: 创建payload容器[元数据(4)+query(64)+key(64)]
  - 3.2: 添加元数据（标识符、数据大小、子块ID、像素ID）
  - 3.3: 复制Query和Key数据准备排序
  - 3.4: 应用排序优化（调用Step 5函数）
  - 3.5: 添加Query数据到payload
  - 3.6: 添加Key数据到payload

Step 4 & 5: 矩阵重组与排序
  主函数：LLMMAC::llmReshapeFlatToQueryKeyMatrix() [llmmac.cpp]
  功能：将线性数据重组为8×8矩阵并应用排序
  - 4.1: 检查payload格式完整性
  - 4.2: 提取Query和Key数据段
  - 4.3: 调用排序算法（Step 5.1或5.2）
  - 4.4: 将线性数据重组为8×8矩阵

  Step 5.1 分离排序：
    函数：LLMMAC::sortMatrix_LLMSeparated() [llmmac.cpp]
    辅助：countOnesInIEEE754(), compareFloatsByOnes() [yzIEEE754.cpp]
    - Query和Key独立按列排序
    - 每列内按1-bit数量递增排列
    
  Step 5.2 关联排序：
    函数：LLMMAC::sortMatrix_LLMAffiliated() [llmmac.cpp]  
    辅助：countOnesInIEEE754(), compareFloatsByOnes() [yzIEEE754.cpp]
    - Key按bit数排序，Query保持配对关系
    - 保持Query-Key语义关联

Step 6: CNN风格半半重组
  函数：继续在llmReshapeFlatToQueryKeyMatrix()中 [llmmac.cpp]
  功能：按行组合Query和Key数据
  - 6.1: 每行格式[Query第i行(8个)+Key第i行(8个)]=16元素
  - 6.2: 将重组后的数据写回payload
  备注：CNN版本在cnnReshapeFlatToInputWeightMatrix() [yzIEEE754.cpp]

Step 7: NoC传输与注入
  函数：LLMMAC::inject()/llmInject() [llmmac.cpp/llmmacnet.cpp]
  功能：将打包好的数据注入NoC网络
  - 7.1: 创建NoC消息包和数据包
  - 7.2: 将payload复制到消息中
  - 7.3: 注入到网络接口(NI)
  - 7.4: 数据切分为多个flit传输
  - 7.5: 每个flit包含16个float(512位)
  辅助：calculate32BitDiff()评估bit翻转 [yzIEEE754.cpp]

配置选项（parameters.hpp）：
  - case1_default: 基准线（无排序）
  - case3_affiliatedordering: 关联排序
  - case4_seperratedordering: 分离排序
  - YzAffiliatedOrdering: 启用排序
  - YZSeperatedOrdering_reArrangeInput: 选择分离排序
  - LLM_CNN_HALFHALF_RESHAPE: 启用半半重组

优化效果：
  基准线：1620.5 bit flips → 排序后：1595 bit flips (1.57%改善)
==================================================================
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
	matrixOutputPixels_size = 4;
	tile_Pixels_size = 4;
	time_slices = 1;
	LLM_INFO_INIT("\n[LLM Config] Test Case 1: Small 4x4 matrix");
	
	#elif LLM_TEST_CASE == 2
	// Test Case 2: LLaMA-7B Single Head Configuration (Hardware-Adapted)
	matrixOutputPixels_size = 512;  // 512x512 attention matrix
	
	// Calculate tile parameters based on NoC configuration from parameters.hpp  
	int noc_total_nodes = TOT_NUM;  // From parameters.hpp (currently 8x8 = 64)
	int memory_controller_nodes = YZMEMCount;  // From parameters.hpp (currently 4)
	totalTileCount = noc_total_nodes - memory_controller_nodes;  // Processing tiles
	pixelNumPerTile = matrixOutputPixels_size / sqrt(totalTileCount);  // Calculate based on NoC configuration
	pixelNumPerTile = static_cast<int>(ceil(pixelNumPerTile));  // Round up to ensure complete coverage
	// Note: 512/sqrt(60) ≈ 66.1 → 67, so each tile processes 67x67 ≈ 4,489 pixels
	
	time_slices = 4;  // Each pixel has 4 time slices (2x2 subchunks)
	LLM_INFO_INIT("\n[LLM Config] Test Case 2: LLaMA Single Head (" << totalTileCount << " tiles, " << pixelNumPerTile << "x" << pixelNumPerTile << " per tile, 64x64 elements per task)");
	

	#endif

	// Calculate derived parameters based on new naming convention
	#if LLM_TEST_CASE == 2
	tiles_Pixels_per_dim = matrixOutputPixels_size / pixelNumPerTile;  // 512/67 ≈ 7.6
	total_tile_Pixels = totalTileCount;  // 60 tiles (based on NoC-MC)
	
	// Each pixel generates 4 tasks (2x2 subchunks of 64x64 each)
	tasks_per_pixel = 1;  // Override to 1 task per pixel for testing
	int elements_per_task = 128;  // 64 query + 64 key per task
	
	// ==== SINGLE CONFIGURATION POINT FOR TEST SIZE ====
	// Change this ONE variable to control test size:
	int pixels_to_test = 5000;  // 1000 pixels
	
	// Calculate total tasks
	total_task_slicedPixels = pixels_to_test * tasks_per_pixel;  // 1000 * 1 = 1000 tasks total
	
	#else
	// Legacy calculation for other test cases
	tiles_Pixels_per_dim = matrixOutputPixels_size / tile_Pixels_size;
	total_tile_Pixels = tiles_Pixels_per_dim * tiles_Pixels_per_dim;
	
	if (tile_Pixels_size == matrixOutputPixels_size) {
		total_task_slicedPixels = matrixOutputPixels_size * matrixOutputPixels_size * time_slices;
	} else {
		total_task_slicedPixels = total_tile_Pixels * tile_Pixels_size * tile_Pixels_size * time_slices;
	}
	#endif
	
	LLM_INFO_INIT("  MatrixOutputPixels: " << matrixOutputPixels_size << "x" << matrixOutputPixels_size);
	#if LLM_TEST_CASE == 2
	LLM_INFO_INIT("  pixelNumPerTile: " << pixelNumPerTile << "x" << pixelNumPerTile);
	LLM_INFO_INIT("  totalTileCount: " << totalTileCount << " (NoC=" << noc_total_nodes << " - MC=" << memory_controller_nodes << ")");
	LLM_INFO_INIT("  Tasks per pixel: " << tasks_per_pixel);
	LLM_INFO_INIT("  Elements per task: " << elements_per_task << " (64 query + 64 key)");
	LLM_INFO_INIT("  Total pixels: " << pixels_to_test);
	#else
	LLM_INFO_INIT("  tile_Pixels size: " << tile_Pixels_size << "x" << tile_Pixels_size);
	LLM_INFO_INIT("  tile_Pixels: " << tiles_Pixels_per_dim << "x" << tiles_Pixels_per_dim << " = " << total_tile_Pixels);
	#endif
	LLM_INFO_INIT("  Time slices: " << time_slices);
	LLM_INFO_INIT("  Total task_slicedPixels: " << total_task_slicedPixels);

	ready_flag = 0;
	mapping_again = 0;
	last_layer_packet_id = 0;
	executed_tasks = 0;

	LLM_DEBUG_INIT("Creating LLMMACnet with " << macNum << " MAC units");
	LLM_TRACE_INIT("Matrix size: " << matrixOutputPixels_size << "x" << matrixOutputPixels_size);
	LLM_TRACE_INIT("Tile size: " << tile_Pixels_size << "x" << tile_Pixels_size);
	LLM_TRACE_INIT("Time slices: " << time_slices);
	LLM_TRACE_INIT("Total tiles: " << total_tile_Pixels);
	LLM_TRACE_INIT("Total tasks: " << total_task_slicedPixels);

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

	// Test bit representation functions with demo data
	std::cout << "\n=== LLM Initialization: Demo Data Debug ===\n";

	layer_latency.clear();
	LLM_DEBUG_INIT("LLMMACnet initialized successfully!");
}

// ==================================================
// Step 1: 数据生成与加载
// 功能：从文件加载真实LLaMA矩阵或生成随机数据
// ==================================================
bool LLMMACnet::llmLoadRealMatrices(const std::string& input_dir) {
	try {
		// Step 1.1: 初始化矩阵尺寸为512x512
		const int matrix_size = matrixOutputPixels_size;  // 512
		
		// Step 1.2: 分配矩阵存储空间
		attention_query_table.resize(matrix_size);  // Query矩阵
		attention_key_table.resize(matrix_size);    // Key矩阵
		attention_value_table.resize(matrix_size);  // Value矩阵（暂未使用）
		attention_output_table.resize(matrix_size); // 输出矩阵（暂未使用）
		
		// Step 1.3: 尝试打开LLaMA数据文件
		std::ifstream query_file(input_dir + "llama_query_512x128.txt");
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
		
		// Step 1.5: 逐行读取矩阵数据并存储
		for (int i = 0; i < matrix_size && i < file_rows; i++) {
			attention_query_table[i].resize(matrix_size);
			std::string line;
			if (!std::getline(query_file, line)) break;
			
			std::istringstream iss(line);
			for (int j = 0; j < matrix_size; j++) {
				float value;
				if (j < file_cols && (iss >> value)) {
					attention_query_table[i][j] = value;  // 存储实际数据
				} else if (file_cols > 0) {
					// 如果列数不足512，循环复制已有数据
					attention_query_table[i][j] = attention_query_table[i][j % file_cols];
				} else {
					attention_query_table[i][j] = 0.0f;
				}
			}
		}
		
		// Step 1.6: 如果行数不足512，循环复制已有行
		for (int i = file_rows; i < matrix_size; i++) {
			attention_query_table[i].resize(matrix_size);
			for (int j = 0; j < matrix_size; j++) {
				if (file_rows > 0) {
					// 使用模运算循环复制数据
					attention_query_table[i][j] = attention_query_table[i % file_rows][j];
				} else {
					attention_query_table[i][j] = 0.0f;
				}
			}
		}
		query_file.close();
		
		// Step 1.7: 加载Key矩阵（流程与Query相同）
		std::ifstream key_file(input_dir + "llama_key_512x128.txt");
		if (!key_file.is_open()) {
			// 如果512x128文件不存在，尝试2048x128版本
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
		
		for (int i = 0; i < matrix_size && i < file_rows; i++) {
			attention_key_table[i].resize(matrix_size);
			std::string line;
			if (!std::getline(key_file, line)) break;
			
			std::istringstream iss(line);
			for (int j = 0; j < matrix_size; j++) {
				float value;
				if (j < file_cols && (iss >> value)) {
					attention_key_table[i][j] = value;
				} else if (file_cols > 0) {
					// Replicate data if needed
					attention_key_table[i][j] = attention_key_table[i][j % file_cols];
				} else {
					attention_key_table[i][j] = 0.0f;
				}
			}
		}
		
		// Fill remaining rows if needed
		for (int i = file_rows; i < matrix_size; i++) {
			attention_key_table[i].resize(matrix_size);
			for (int j = 0; j < matrix_size; j++) {
				if (file_rows > 0) {
					attention_key_table[i][j] = attention_key_table[i % file_rows][j];
				} else {
					attention_key_table[i][j] = 0.0f;
				}
			}
		}
		key_file.close();
		
		// Initialize value and output tables to zero
		for (int i = 0; i < matrix_size; i++) {
			attention_value_table[i].resize(matrix_size, 0.0f);
			attention_output_table[i].resize(matrix_size, 0.0f);
		}
		
		std::cerr << "Successfully loaded real LLaMA matrices!" << std::endl;
		return true;
		
	} catch (const std::exception& e) {
		std::cerr << "Error loading LLaMA matrices: " << e.what() << std::endl;
		return false;
	}
}
void LLMMACnet::llmInitializeMatrices() {
	LLM_DEBUG_INIT("Initializing " << matrixOutputPixels_size << "x" << matrixOutputPixels_size << " attention matrices...");

#ifdef LLM_USE_REAL_MATRICES
	// 使用真实LLaMA矩阵模式
	std::string input_dir = "../llama_matrices/";
	LLM_DEBUG_INIT("LLM_USE_REAL_MATRICES mode: Loading real LLaMA matrices from " << input_dir);
	
	bool load_success = llmLoadRealMatrices(input_dir);
	if (load_success) {
		LLM_DEBUG_INIT("Successfully loaded real LLaMA matrices!");
		return;
	} else {
		LLM_DEBUG_INIT("ERROR: Failed to load LLaMA matrices! Check if files exist in " << input_dir);
		LLM_DEBUG_INIT("Expected files: llama_query_512x128.txt, llama_key_512x128.txt");
		// 可以选择退出或回退到随机生成
		#ifdef LLM_STRICT_REAL_MATRIX_MODE
			std::cerr << "FATAL: Cannot proceed without real matrix data in strict mode!" << std::endl;
			exit(1);
		#else
			LLM_DEBUG_INIT("WARNING: Falling back to random matrix generation...");
		#endif
	}
#endif

#ifdef LLM_USE_RANDOM_MATRICES
	// 使用随机生成矩阵模式
	LLM_DEBUG_INIT("LLM_USE_RANDOM_MATRICES mode: Generating random matrices...");
	// 使用固定种子确保可重现性（与CNN相同）
	srand(0);
	
	// Create a demo deque to test our bit representation functions
	std::deque<float> demo_data;
	demo_data.push_back(1.5f);
	demo_data.push_back(-0.25f);
	demo_data.push_back(3.14159f);
	demo_data.push_back(0.0f);
	demo_data.push_back(-1.0f);
	demo_data.push_back(2.718f);
	
	attention_query_table.resize(matrixOutputPixels_size);
	attention_key_table.resize(matrixOutputPixels_size);
	attention_value_table.resize(matrixOutputPixels_size);
	attention_output_table.resize(matrixOutputPixels_size);

	for (int i = 0; i < matrixOutputPixels_size; i++) {
		attention_query_table[i].resize(matrixOutputPixels_size);
		attention_key_table[i].resize(matrixOutputPixels_size);
		attention_value_table[i].resize(matrixOutputPixels_size);
		attention_output_table[i].resize(matrixOutputPixels_size);

		for (int j = 0; j < matrixOutputPixels_size; j++) {
			attention_query_table[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
			attention_key_table[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
			attention_value_table[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
			attention_output_table[i][j] = 0.0f;
		}
	}

	// 仅在TRACE级别打印完整矩阵
	if (LLM_DEBUG_LEVEL >= 3) {
		LLM_TRACE("\n=== Query Matrix (" << matrixOutputPixels_size << "x" << matrixOutputPixels_size << ") ===");
		for (int i = 0; i < matrixOutputPixels_size && i < 4; i++) {  // 限制最多打印4行
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrixOutputPixels_size && j < 4; j++) {
				std::cout << std::fixed << std::setprecision(6) << attention_query_table[i][j] << "\t";
			}
			if (matrixOutputPixels_size > 4) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrixOutputPixels_size > 4) std::cout << "..." << std::endl;

		LLM_TRACE("\n=== Key Matrix (" << matrixOutputPixels_size << "x" << matrixOutputPixels_size << ") ===");
		for (int i = 0; i < matrixOutputPixels_size && i < 4; i++) {
			std::cout << "Row " << i << ": ";
			for (int j = 0; j < matrixOutputPixels_size && j < 4; j++) {
				std::cout << std::fixed << std::setprecision(6) << attention_key_table[i][j] << "\t";
			}
			if (matrixOutputPixels_size > 4) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrixOutputPixels_size > 4) std::cout << "..." << std::endl;
	}
#endif  // LLM_USE_RANDOM_MATRICES

	LLM_DEBUG_INIT("Matrices initialized successfully");
}

void LLMMACnet::llmExportMatricesToFile() {
	LLM_DEBUG_INIT("Exporting matrices to output/ for verification...");

	// Export Query matrix
	std::ofstream query_file("src/output/cpp_query_matrix.txt");
	if (query_file.is_open()) {
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				query_file << std::fixed << std::setprecision(10) << attention_query_table[i][j];
				if (j < matrixOutputPixels_size - 1) query_file << ",";
			}
			query_file << "\n";
		}
		query_file.close();
		LLM_DEBUG_INIT("Query matrix exported to src/output/cpp_query_matrix.txt");
	}

	// Export Key matrix
	std::ofstream key_file("src/output/cpp_key_matrix.txt");
	if (key_file.is_open()) {
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				key_file << std::fixed << std::setprecision(10) << attention_key_table[i][j];
				if (j < matrixOutputPixels_size - 1) key_file << ",";
			}
			key_file << "\n";
		}
		key_file.close();
		LLM_DEBUG_INIT("Key matrix exported to src/output/cpp_key_matrix.txt");
	}

	// Export Value matrix
	std::ofstream value_file("src/output/cpp_value_matrix.txt");
	if (value_file.is_open()) {
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				value_file << std::fixed << std::setprecision(10) << attention_value_table[i][j];
				if (j < matrixOutputPixels_size - 1) value_file << ",";
			}
			value_file << "\n";
		}
		value_file.close();
		LLM_DEBUG_INIT("Value matrix exported to src/output/cpp_value_matrix.txt");
	}
}

void LLMMACnet::llmGenerateAllTasks() {
	// === Step 2: 任务分配与数据提取 ===
	// 功能：将512x512的大矩阵分解为多个小任务
	LLM_DEBUG_INIT("Generating LLM tasks for " << matrixOutputPixels_size << "x" << matrixOutputPixels_size << " matrix...");
	all_tasks.clear();
	
	// Step 2.1: 计算任务参数
	int task_id = 0;
	int total_pixels = matrixOutputPixels_size * matrixOutputPixels_size;  // 512*512 = 262,144个像素
	// int tasks_per_pixel = 4;  // This is now a member variable: this->tasks_per_pixel
	int total_tasks = total_pixels * this->tasks_per_pixel;                     // 总共 1,048,576个任务
	
	all_tasks.reserve(total_tasks);
	LLM_DEBUG_INIT("Total pixels: " << total_pixels << ", Tasks per pixel: " << this->tasks_per_pixel << ", Total tasks: " << total_tasks);
	
	// Step 2.2: 根据像素生成任务
	#if LLM_TEST_CASE == 2
	// 测试配置：512x512矩阵，128元素向量，分为64x64子块
	// 使用构造函数中设置的pixels_to_test值
	int pixels_to_generate = total_task_slicedPixels / this->tasks_per_pixel;  // 从总任务数反算像素数
	int pixel_count = 0;
	// Step 2.3: 遍历所有像素位置
	for (int pixel_y = 0; pixel_y < matrixOutputPixels_size && pixel_count < pixels_to_generate; pixel_y++) {
		for (int pixel_x = 0; pixel_x < matrixOutputPixels_size && pixel_count < pixels_to_generate; pixel_x++) {
			int pixel_id = pixel_y * matrixOutputPixels_size + pixel_x;  // 计算像素的线性ID
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
				task.tile_id = (pixel_y / pixelNumPerTile) * (matrixOutputPixels_size / pixelNumPerTile) + (pixel_x / pixelNumPerTile);  // 所属瓦片ID
				
				// Step 2.5: 计算子块的数据偏移量
				// 每个像素计算128x128的点积，分为4个64x64的子块：
				// subchunk 0: query[0:63] x key[0:63]      - 左上角
				// subchunk 1: query[0:63] x key[64:127]    - 右上角
				// subchunk 2: query[64:127] x key[0:63]    - 左下角
				// subchunk 3: query[64:127] x key[64:127]  - 右下角
				// 根据子块ID计算偏移量
				task.query_offset = (subchunk_id / 2) * 64;  // 0(子块0,1) 或 64(子块2,3)
				task.key_offset = (subchunk_id % 2) * 64;    // 0(子块0,2) 或 64(子块1,3)
				
				// Step 2.6: 从矩阵中提取64个Query和64个Key元素
				task.query_data.clear();
				task.key_data.clear();
				task.query_data.reserve(64);  // 预分配空间
				task.key_data.reserve(64);
				
				// Step 2.7: 提取具体数据（支持两种模式）
				for (int i = 0; i < 64; i++) {
#ifdef LLM_INPUT_BASED
					// 输入为主模式：pixel_x变化（不同输入），pixel_y固定（相同权重）
					// 同一行的不同像素获得不同的input_query和input_key
					int input_query_idx = (pixel_y + task.query_offset + i) % matrixOutputPixels_size;
					task.query_data.push_back(attention_query_table[pixel_x][input_query_idx]);
					
					int input_key_idx = (pixel_y + task.key_offset + i) % matrixOutputPixels_size;
					task.key_data.push_back(attention_key_table[pixel_x][input_key_idx]);
#else
					// 权重为主模式：pixel_y变化（不同权重），pixel_x固定（相同输入）
					// 同一行的不同像素使用相同权重但加上偏移
					// 注意：加入pixel_x偏移以增加数据多样性
					int weight_query_idx = (pixel_x + task.query_offset + i) % matrixOutputPixels_size;
					task.query_data.push_back(attention_query_table[pixel_y][weight_query_idx]);
					
					int weight_key_idx = (pixel_x + task.key_offset + i) % matrixOutputPixels_size;
					task.key_data.push_back(attention_key_table[pixel_y][weight_key_idx]);
#endif
				}
				
#ifdef LLM_RANDOM_DATA_REPLACE_TEST
				// Step 2.8: 随机数据替换测试
				// 用随机数据替换真实数据，测试bit翻转优化的潜力
				for (int i = 0; i < 64; i++) {
					task.query_data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;  // [-0.5, 0.5]随机数
					task.key_data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
				}
#endif
				
				task.partial_sum = 0.0f;  // Initialize
				
				// Debug: Verify data diversity for first few tasks in row 0
				if (pixel_y == 0 && pixel_x < 4 && subchunk_id == 0) {
					LLM_DEBUG_INIT("Task " << task_id << " (pixel " << pixel_x << "," << pixel_y << "):");
					LLM_DEBUG_INIT("  First 3 query values: " 
						<< task.query_data[0] << ", " 
						<< task.query_data[1] << ", " 
						<< task.query_data[2]);
					LLM_DEBUG_INIT("  First 3 key values: " 
						<< task.key_data[0] << ", " 
						<< task.key_data[1] << ", " 
						<< task.key_data[2]);
				}
				
				all_tasks.push_back(task);
				task_id++;
			}
		}
	}
	
	// Update total_task_slicedPixels to reflect actual number of tasks
	total_task_slicedPixels = all_tasks.size();
	LLM_DEBUG_INIT("Generated " << all_tasks.size() << " tasks successfully");
	LLM_DEBUG_INIT("First few tasks:");
	for (int i = 0; i < std::min(4, (int)all_tasks.size()); i++) {
		LLM_DEBUG_INIT("  Task " << all_tasks[i].task_id 
		              << ": pixel(" << all_tasks[i].pixel_x << "," << all_tasks[i].pixel_y << ")"
		              << " subchunk=" << all_tasks[i].subchunk_id
		              << " query_offset=" << all_tasks[i].query_offset
		              << " key_offset=" << all_tasks[i].key_offset);
	#else
	// Legacy approach for TEST_CASE 1: simple 4x4 matrix
	for (int ts = 0; ts < time_slices; ts++) {
		for (int tile_y = 0; tile_y < tiles_Pixels_per_dim; tile_y++) {
			for (int tile_x = 0; tile_x < tiles_Pixels_per_dim; tile_x++) {
				// Each tile generates tile_Pixels_size x tile_Pixels_size tasks (one per pixel in tile)
				for (int in_tile_y = 0; in_tile_y < tile_Pixels_size; in_tile_y++) {
					for (int in_tile_x = 0; in_tile_x < tile_Pixels_size; in_tile_x++) {
						int pixel_x = tile_x * tile_Pixels_size + in_tile_x;
						int pixel_y = tile_y * tile_Pixels_size + in_tile_y;
						
						// Skip if out of bounds (for non-divisible matrices)
						if (pixel_x >= matrixOutputPixels_size || pixel_y >= matrixOutputPixels_size)
							continue;
						
						LLMTask task;
						task.task_id = task_id;
						task.pixel_id = pixel_y * matrixOutputPixels_size + pixel_x;
						task.pixel_x = pixel_x;
						task.pixel_y = pixel_y;
						task.time_slice = ts;
						task.subchunk_id = ts;
						task.tile_id = tile_y * tiles_Pixels_per_dim + tile_x;
						
						int d_head = tile_Pixels_size;  // Legacy: Process tile_Pixels_size elements per task
						
						task.query_data.resize(d_head);
						task.key_data.resize(d_head);
						task.value_data.resize(d_head);

						for (int i = 0; i < d_head; i++) {
							int data_idx = (ts * d_head + i) % matrixOutputPixels_size;
							task.query_data[i] = attention_query_table[pixel_y][(pixel_x + data_idx) % matrixOutputPixels_size];
							task.key_data[i] = attention_key_table[pixel_y][(pixel_x + data_idx) % matrixOutputPixels_size];
							task.value_data[i] = attention_value_table[pixel_y][(pixel_x + data_idx) % matrixOutputPixels_size];
						}

						// Print debug info for first few tasks only
						if (task_id < 3 && LLM_DEBUG_LEVEL >= 3) {
							LLM_TRACE("Task " << task_id << " [pixel(" << pixel_x << "," << pixel_y << "), ts=" << ts << ", tile=" << task.tile_id << "]:");
							std::cout << "  Query data: ";
							for (int i = 0; i < d_head && i < 4; i++) {
								std::cout << std::fixed << std::setprecision(6) << task.query_data[i];
								if (i < d_head - 1) std::cout << ",";
							}
							if (d_head > 4) std::cout << "...";
							std::cout << std::endl;
						}

						all_tasks.push_back(task);
						task_id++;
					}
				}
			}
		}
	}
	#endif
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
		verify_file << "Matrix size: " << matrixOutputPixels_size << "x" << matrixOutputPixels_size << "\n";
		verify_file << "Data elements per vector: " << matrixOutputPixels_size << "\n";
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
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				verify_file << std::fixed << std::setprecision(10) << attention_output_table[i][j];
				if (j < matrixOutputPixels_size - 1) verify_file << ",";
			}
			verify_file << "\n";
		}

		verify_file.close();
		LLM_DEBUG("Verification results exported to src/output/cpp_verification.txt");
	}

	// 导出最终输出矩阵到单独文件
	std::ofstream output_file("src/output/llm_attention_output.txt");
	if (output_file.is_open()) {
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				output_file << std::fixed << std::setprecision(10) << attention_output_table[i][j];
				if (j < matrixOutputPixels_size - 1) output_file << ",";
			}
			output_file << "\n";
		}
		output_file.close();
		LLM_DEBUG("Output matrix exported to output/llm_attention_output.txt");
	}
}

// 辅助函数
int LLMMACnet::llmGetTileId(int pixel_x, int pixel_y) {
	int tile_x = pixel_x / tile_Pixels_size;
	int tile_y = pixel_y / tile_Pixels_size;
	return tile_y * tiles_Pixels_per_dim + tile_x;
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

void LLMMACnet::llmXMapping(int total_pixels) {
	// 计算总任务数（像素数 * N）
	int total_tasks = total_pixels * this->tasks_per_pixel;
	
	LLM_DEBUG("Starting pixel-to-task mapping for " << total_pixels << " pixels (" << total_tasks << " tasks)...");

	this->llmOutputPixelMappingTable.clear();
	this->llmOutputPixelMappingTable.resize(macNum);
	this->llmTaskMappingTable.clear();
	this->llmTaskMappingTable.resize(macNum);
	// Also sync with base class mapping_table for compatibility
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
		
		if (LLM_DEBUG_LEVEL >= 3) {
			LLM_TRACE("Pixel " << pixel_id << " → MAC " << mac_id << " (tasks " 
			          << (pixel_id * this->tasks_per_pixel) << "-" << (pixel_id * this->tasks_per_pixel + (this->tasks_per_pixel - 1)) << ")");
		}
	}

	LLM_DEBUG("Pixel-to-task mapping completed: " << total_pixels << " pixels, " 
	          << total_tasks << " tasks distributed to " << available_macs.size() << " nodes");

	// 打印详细的分配统计 (Level 2+)
	if (LLM_DEBUG_LEVEL >= 2) {
		for (int i = 0; i < macNum; i++) {
			if (llmTaskMappingTable[i].size() > 0) {
				std::cout << "[DEBUG @" << cycles << "] MAC " << i 
				          << " pixels: " << llmOutputPixelMappingTable[i].size()
				          << ", tasks: " << llmTaskMappingTable[i].size() << std::endl;
			}
		}
	}
}

void LLMMACnet::llmLoadBalanceMapping(int total_pixels) {
	LLM_DEBUG("Starting load-balanced mapping...");
	llmXMapping(total_pixels);  // 对于小矩阵，使用简单映射即可
}

int LLMMACnet::llmSAMOSTaskMapping(int pixel_count, int start_pixel_id) {
	// Clear and prepare mapping tables
	this->llmOutputPixelMappingTable.clear();
	this->llmOutputPixelMappingTable.resize(macNum);
	this->llmTaskMappingTable.clear();
	this->llmTaskMappingTable.resize(macNum);
	// Also sync with base class mapping_table for compatibility
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);
	
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
			
			// 每个像素生成4个任务，都分配给同一个节点（便于聚合）
			for (int chunk_id = 0; chunk_id < this->tasks_per_pixel; chunk_id++) {
				int task_id = current_pixel_id * this->tasks_per_pixel + chunk_id;
				this->llmTaskMappingTable[n.id].push_back(task_id);
			}
			current_pixel_id++;
		}
	}
	
	// Debug output for LLM SAMOS mapping
	if (LLM_DEBUG_LEVEL >= 2) {
		int total_tasks = pixel_count * this->tasks_per_pixel;
		std::cout << "[SAMOS] Total pixels=" << pixel_count 
		          << " Total tasks=" << total_tasks
		          << " Total PEs=" << nodes.size() << "\n";
		if (LLM_DEBUG_LEVEL >= 3) {
			for (auto &n : nodes) {
				double avgLat = double(samplingWindowDelay[n.id]) / std::max(1, samplingWindowLength);
				std::cout << "  MAC " << n.id << " lat=" << avgLat 
                  << " pixels=" << n.alloc << " tasks=" << (n.alloc * this->tasks_per_pixel) << "\n";
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
		int progress = (executed_tasks * 100) / std::max(1, total_task_slicedPixels);
		LLM_INFO("[Cycle " << cycles << "] Progress: " << progress << "% (" << executed_tasks << "/" << total_task_slicedPixels << " tasks)");
	} else if (status_check_count % 10000 == 0 && LLM_DEBUG_LEVEL >= 2) {  // Level 2: More frequent status
		LLM_DEBUG("Status check #" << status_check_count << " at cycle " << cycles);
		LLM_DEBUG("Ready flag: " << ready_flag << ", Mapping again: " << mapping_again);
	}

	if (ready_flag == 0) {
		LLM_INFO("Initializing layer " << current_layer << " at cycle " << cycles);

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
				this->LLMMAC_list[i]->llmtasktable.assign(
					llmTaskMappingTable[i].begin(), llmTaskMappingTable[i].end());
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
		if (llmTaskMappingTable[i].size() > 0) {
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
		LLM_INFO("Total executed tasks: " << executed_tasks << "/" << total_task_slicedPixels);

		// 打印最终结果矩阵 - 仅在Level 2+显示
		if (LLM_DEBUG_LEVEL >= 2) {
			std::cout << "\n=== Final Output Matrix (" << matrixOutputPixels_size << "x" << matrixOutputPixels_size << ") ===" << std::endl;
			int zero_count = 0;
			int non_zero_count = 0;

			int max_rows = (LLM_DEBUG_LEVEL >= 3) ? matrixOutputPixels_size : std::min(4, matrixOutputPixels_size);
			for (int i = 0; i < max_rows; i++) {
				std::cout << "Row " << i << ": ";
				int max_cols = (LLM_DEBUG_LEVEL >= 3) ? matrixOutputPixels_size : std::min(4, matrixOutputPixels_size);
				for (int j = 0; j < max_cols; j++) {
					float value = attention_output_table[i][j];
					std::cout << std::fixed << std::setprecision(6) << value << "\t";
				}
				if (matrixOutputPixels_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "...";
				std::cout << std::endl;
			}
			if (matrixOutputPixels_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "..." << std::endl;

			// Count all values
			for (int i = 0; i < matrixOutputPixels_size; i++) {
				for (int j = 0; j < matrixOutputPixels_size; j++) {
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
		packet_id = packet_id + total_task_slicedPixels * 4;
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
					// === Step 3: 数据打包为Payload ===
					int task_id = tmpPacket->message.data[0];  // 从请求中获取任务ID
					
					// 记录请求到达内存的时间
					tmpLLMMAC->current_task_timing.request_arrive_cycle = cycles;

					if (task_id >= 0 && task_id < all_tasks.size()) {
						LLMTask& task = all_tasks[task_id];  // 获取对应任务

						// Step 3.1: 构建payload（总共132个float）
						tmpLLMMAC->input_buffer.clear();
						
						// Step 3.2: 添加元数据（4个float）
						tmpLLMMAC->input_buffer.push_back(1.0f);              // [0] 函数标志
						tmpLLMMAC->input_buffer.push_back(64);                // [1] 数据大小（每个64元素）
						tmpLLMMAC->input_buffer.push_back(task.subchunk_id);  // [2] 子块ID
						tmpLLMMAC->input_buffer.push_back(task.pixel_id);     // [3] 像素ID（用于跟踪）

						// Copy data for potential ordering
						std::deque<float> query_data_copy(task.query_data.begin(), task.query_data.end());
						std::deque<float> key_data_copy(task.key_data.begin(), task.key_data.end());
						
						// Debug: Print data BEFORE sorting (only for first few tasks)
						static int debug_print_count = 0;
						bool should_print = (debug_print_count < 10);  // Print first 10 tasks total
						
						if (should_print) {
							debug_print_count++;  // Increment here to count actual prints
							std::cout << "Task " << task_id << " - Memory " << mem_id 
							          << " (Pixel " << task.pixel_id << ", Subchunk " << task.subchunk_id << ")" << std::endl;
							

						}
						
						// Apply ordering BEFORE transmission to reduce bit flips
						#ifdef YzAffiliatedOrdering
							if (should_print) {
								std::cout << "\n=== APPLYING DATA ORDERING ===" << std::endl;
							}
							#ifdef YZSeperatedOrdering_reArrangeInput
								// Separated ordering: sort both independently
								if (should_print) {
									std::cout << "Using SEPARATED ordering (Query and Key sorted independently)" << std::endl;
								}
								LLMMAC_list[mem_id]->sortMatrix_LLMSeparated(query_data_copy, 8, 8);
								LLMMAC_list[mem_id]->sortMatrix_LLMSeparated(key_data_copy, 8, 8);
							#else
								// Affiliated ordering: sort together
								if (should_print) {
									std::cout << "Using AFFILIATED ordering (Query and Key sorted together by Key's bit count)" << std::endl;
								}
								LLMMAC_list[mem_id]->sortMatrix_LLMAffiliated(query_data_copy, key_data_copy, 8, 8);
							#endif
							
							// Debug: Print data AFTER sorting
							if (should_print) {

								// Show row-wise bit count to understand transmission order
								std::cout << "\n=== ROW-WISE BIT COUNT (AFTER SORTING) ===" << std::endl;
								std::cout << "Query Matrix - Bit counts by row:" << std::endl;
								for (int row = 0; row < 8; row++) {
									std::cout << "Row " << row << ": ";
									for (int col = 0; col < 8; col++) {
										int idx = row * 8 + col;
										if (idx < query_data_copy.size()) {
											int ones = countOnesInIEEE754(query_data_copy[idx]);
											std::cout << std::setw(2) << ones << " ";
										}
									}
									std::cout << std::endl;
								}
								
								std::cout << "\nKey Matrix - Bit counts by row:" << std::endl;
								for (int row = 0; row < 8; row++) {
									std::cout << "Row " << row << ": ";
									for (int col = 0; col < 8; col++) {
										int idx = row * 8 + col;
										if (idx < key_data_copy.size()) {
											int ones = countOnesInIEEE754(key_data_copy[idx]);
											std::cout << std::setw(2) << ones << " ";
										}
									}
									std::cout << std::endl;
								}
								
								// Show concatenated Query+Key matrix with both values and bit counts
								// NOTE: At this point, query_data_copy and key_data_copy have been reorganized by sorting
								// They are now in row-major order ready for transmission
								std::cout << "\n=== 拼接后的矩阵 (Query | Key) - 按Flit传输顺序 ===" << std::endl;
								std::cout << "NOTE: This shows the actual data that will be transmitted in each flit" << std::endl;
								
								// Print bit counts (按行组织的实际传输顺序)
								std::cout << "\nBit counts:" << std::endl;
								// 重建最终的payload以正确显示
							}
						#else
							// Baseline - no ordering
							// Only print details for first few tasks
							if (should_print) {
								std::cout << "  [Baseline - No ordering applied]" << std::endl;
							}
						#endif
						
						// Step 3.5: 添加Query数据（64个float）
						tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
							query_data_copy.begin(), query_data_copy.end());
						// Step 3.6: 添加Key数据（64个float）  
						tmpLLMMAC->input_buffer.insert(tmpLLMMAC->input_buffer.end(),
							key_data_copy.begin(), key_data_copy.end());



						// Step 7: NoC传输与注入
						// 功能：将打包好的数据注入NoC网络进行传输
						int mem_delay = static_cast<int>(ceil((task.query_data.size() * 2 + 1) * MEM_read_delay)) + CACHE_DELAY;
						LLMMAC_list[mem_id]->pecycle = cycles + mem_delay;
						
						// 记录响应发送时间
						tmpLLMMAC->current_task_timing.response_send_cycle = cycles + mem_delay;
						LLMMAC_list[mem_id]->input_buffer = tmpLLMMAC->input_buffer;
						// Step 7.1: 注入到网络接口，数据将被切分为多个flit进行传输
						// 每个flit包含16个float元素（512位）
						// 注意：虽然input_buffer有132个元素，但我们只传输128个（跳过前4个元数据）
						int actual_payload_size = tmpLLMMAC->input_buffer.size() - 4;  // 132 - 4 = 128
						LLMMAC_list[mem_id]->llmInject(1, src, actual_payload_size,
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
						if (LLM_DEBUG_LEVEL >= 2) {
							std::cout << "[DEBUG @" << cycles << "] [INTERMEDIATE] Memory " << mem_id
							          << " received intermediate result from MAC " << src_mac
							          << " for pixel(" << pixel_x << "," << pixel_y << ") ts=" << time_slice
							          << " value=" << std::fixed << std::setprecision(6) << result_value << std::endl;
						}
					}
					else if (msg_type == 3) {
						// Type 3: 最终聚合结果，更新输出矩阵
						std::cout << "[FINAL-UPDATE] Memory " << mem_id
						          << " received FINAL result from MAC " << src_mac << std::endl;
						std::cout << "  Pixel: (" << pixel_x << "," << pixel_y << ")" << std::endl;
						std::cout << "  Final value: " << std::fixed << std::setprecision(10) << result_value << std::endl;

						if (pixel_x >= 0 && pixel_x < matrixOutputPixels_size &&
						    pixel_y >= 0 && pixel_y < matrixOutputPixels_size) {

							// 保存旧值用于比较
							float old_value = attention_output_table[pixel_y][pixel_x];

							// 更新输出矩阵
							attention_output_table[pixel_y][pixel_x] = result_value;
							
							// 增加已执行任务计数
							executed_tasks++;

							std::cout << "[TABLE-UPDATE] Updated output table:" << std::endl;
							std::cout << "  Position [" << pixel_y << "][" << pixel_x << "]" << std::endl;
							std::cout << "  Old value: " << std::fixed << std::setprecision(10) << old_value << std::endl;
							std::cout << "  New value: " << std::fixed << std::setprecision(10) << result_value << std::endl;
							std::cout << "  Verification: " << attention_output_table[pixel_y][pixel_x] << std::endl;
							std::cout << "  Tasks completed: " << executed_tasks << "/" << total_task_slicedPixels << std::endl;

							// 统计非零元素
							int non_zero_count = 0;
							for (int i = 0; i < matrixOutputPixels_size; i++) {
								for (int j = 0; j < matrixOutputPixels_size; j++) {
									if (attention_output_table[i][j] != 0.0) {
										non_zero_count++;
									}
								}
							}
							std::cout << "  Total non-zero elements in table: " << non_zero_count
							          << "/" << (matrixOutputPixels_size * matrixOutputPixels_size) << std::endl;
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
		std::cout << "[DEBUG @" << cycles << "] \n[MATRIX-STATUS] Current output matrix at step " << run_step_count << ":" << std::endl;
		int non_zero_count = 0;
		int max_rows = (LLM_DEBUG_LEVEL >= 3) ? matrixOutputPixels_size : std::min(4, matrixOutputPixels_size);
		for (int i = 0; i < max_rows; i++) {
			std::cout << "Row " << i << ": ";
			int max_cols = (LLM_DEBUG_LEVEL >= 3) ? matrixOutputPixels_size : std::min(4, matrixOutputPixels_size);
			for (int j = 0; j < max_cols; j++) {
				float val = attention_output_table[i][j];
				if (val != 0.0) {
					non_zero_count++;
					std::cout << std::fixed << std::setprecision(6) << val << " ";
				} else {
					std::cout << "0.000000 ";
				}
			}
			if (matrixOutputPixels_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "...";
			std::cout << std::endl;
		}
		if (matrixOutputPixels_size > 4 && LLM_DEBUG_LEVEL < 3) std::cout << "..." << std::endl;
		
		// 总是显示非零元素统计
		for (int i = 0; i < matrixOutputPixels_size; i++) {
			for (int j = 0; j < matrixOutputPixels_size; j++) {
				if (attention_output_table[i][j] != 0.0) non_zero_count++;
			}
		}
		std::cout << "Non-zero elements: " << non_zero_count << "/" << (matrixOutputPixels_size * matrixOutputPixels_size) << std::endl;
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
