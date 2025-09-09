#ifndef LLMMACNET_HPP_
#define LLMMACNET_HPP_

#include <cmath>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>

#include "Model.hpp"
#include "NoC/VCNetwork.hpp"

using namespace std;

extern long long packet_id;
extern unsigned int cycles;
extern vector<vector<int>> DNN_latency;
extern double samplingWindowDelay[TOT_NUM];
extern int samplingAccumlatedCounter;

// Forward declarations
class VCNetwork;
class LLMMAC;

class LLMMACnet
{
public:
	LLMMACnet(int mac_num, int t_pe_x, int t_pe_y, VCNetwork* t_Network);

	std::vector<LLMMAC*> LLMMAC_list;
	VCNetwork* vcNetwork;

	// LLM-specific data structures
	vector<vector<float>> attention_query_table;
	vector<vector<float>> attention_key_table;
	vector<vector<float>> attention_value_table;
	vector<vector<float>> attention_output_table;

	// Task mapping and scheduling
	deque<deque<int>> mapping_table;  // Keep for base class compatibility
	deque<deque<int>> llmOutputPixelMappingTable;  // LLM-specific: maps output pixels to MAC units
	                                               // llmOutputPixelMappingTable[MAC_ID] = {pixel_ids...}
	deque<deque<int>> llmTaskMappingTable;         // LLM-specific: maps tasks to MAC units
	                                               // llmTaskMappingTable[MAC_ID] = {task_ids...}
	void llmXMapping(int total_pixels);

	void llmLoadBalanceMapping(int total_pixels);
	int llmSAMOSTaskMapping(int pixel_count, int start_pixel_id = 0);  // SAMOS mapping function

	// LLM attention layer management
	void llmCreateAttentionData();
	void llmInitializeMatrices();
	bool llmLoadRealMatrices(const std::string& input_dir);
	void llmTaskPartitioning();

	// 新增：数据导出函数
	void llmExportMatricesToFile();
	void llmExportTasksToFile();
	void llmExportVerificationResults();
	void llmPrintTimingStatistics();

	// Network and execution management
	void llmRunOneStep();
	void llmCheckStatus();

	// LLM parameters
	int macNum;
	int pe_x;
	int pe_y;
	int current_layer;
	int total_layers;

	// Matrix dimensions
	int matrixOutputPixels_size;  // Size of output pixel matrix (e.g., 128x128)
	int totalTileCount;           // Total number of tiles (NoC nodes - MC nodes)
	int tile_Pixels_size;
	int pixelNumPerTile;              // Size of each tile (calculated from NoC)

	int tiles_Pixels_per_dim;     // Number of tiles per dimension
	int total_tile_Pixels;        // Total number of tiles
	int time_slices;
	int tasks_per_pixel;          // Number of sub-tasks each pixel is divided into (e.g., 4)
	int total_task_slicedPixels;  // Total number of tasks (sliced pixels)

	/*	matrixOutputPixels_size = 512, 512*512
	totalTileCount = noc_total_nodes - memory_controller_nodes;  //64- 4= 60
	pixelNumPerTile = matrixOutputPixels_size / sqrt(totalTileCount);  // Calculate based on NoC configuration
	pixelNumPerTile = static_cast<int>(ceil(pixelNumPerTile));  // Round up to ensure complete coverage
	// Note: 128/sqrt(60) ≈ 16.5 → 17, so each tile processes 17x17 ≈ 289 pixels
	 */


	// Execution state
	int ready_flag;
	int mapping_again;
	int last_layer_packet_id;
	int executed_tasks;

	// Performance tracking
	vector<int> layer_latency;
	int breakdown_time[TOT_NUM][4][11];

	// Task generation and distribution
	struct LLMTask {
		int task_id;              // 全局任务ID
		int pixel_id;             // 所属pixel ID
		int pixel_x, pixel_y;     // pixel坐标 (在512x512矩阵中)
		int time_slice;           // 时间片ID，与subchunk_id相同
		int subchunk_id;          // 子块ID (0-3)
		int tile_id;              // tile ID（保留用于兼容）
		
		// 数据内容
		vector<float> query_data; // 64个query元素
		vector<float> key_data;   // 64个key元素
		vector<float> value_data; // 保留用于future扩展
		
		// 数据范围信息
		int query_offset;         // query数据起始偏移 (0或64)
		int key_offset;           // key数据起始偏移 (0或64)
		float partial_sum;        // 该子块的计算结果
	};

	vector<LLMTask> all_tasks;
	void llmGenerateAllTasks();
	void llmDistributeTasks();

	// Helper functions
	int llmGetTileId(int pixel_x, int pixel_y);
	int llmGetMACIdForTile(int tile_id);
	bool llmIsMemoryNode(int node_id);

	~LLMMACnet();
};

#endif /* LLMMACNET_HPP_ */
