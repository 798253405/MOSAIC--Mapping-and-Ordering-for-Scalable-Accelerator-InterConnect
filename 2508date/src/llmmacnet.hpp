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
// 注意: llmmac.hpp 会在 .cpp 文件中包含，避免循环依赖

using namespace std;

extern int packet_id;
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
	vector<vector<float>> attention_query_table;    // Query matrix (32x32)
	vector<vector<float>> attention_key_table;      // Key matrix (32x32)
	vector<vector<float>> attention_value_table;    // Value matrix (32x32)
	vector<vector<float>> attention_output_table;   // Output matrix (32x32)

	// Task mapping and scheduling
	deque<deque<int>> mapping_table;
	void llmXMapping(int total_tasks);
	void llmYMapping(int total_tasks);
	void llmRandomMapping(int total_tasks);
	void llmDistanceMapping(int total_tasks);
	void llmLoadBalanceMapping(int total_tasks);

	// LLM attention layer management
	void llmCreateAttentionData();
	void llmInitializeMatrices();
	void llmTaskPartitioning();

	// Network and execution management
	void llmRunOneStep();
	void llmCheckStatus();

	// LLM parameters
	int macNum;
	int pe_x;
	int pe_y;
	int current_layer;
	int total_layers;

	// Matrix dimensions (32x32 for fast testing)
	int matrix_size;
	int tile_size;        // 4x4 per tile
	int tiles_per_dim;    // 8 tiles per dimension (32/4)
	int total_tiles;      // 64 total tiles
	int time_slices;      // 2 time slices per pixel
	int total_tasks;      // Total tasks = 32*32*2

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
		int task_id;
		int pixel_x, pixel_y;
		int time_slice;
		int tile_id;
		vector<float> query_data;
		vector<float> key_data;
		vector<float> value_data;
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
