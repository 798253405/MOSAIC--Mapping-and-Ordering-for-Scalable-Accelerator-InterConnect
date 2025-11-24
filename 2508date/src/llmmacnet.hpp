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

	// LLM-specific data structures - InputQuery，Key
	vector<vector<float>> input_matrix;         //  (8×4096)
	vector<vector<float>> query_weight_matrix;  // Query (128×4096)
	// Key
	vector<vector<float>> Q_resOutput_matrix;   //  (8×128) -
	vector<vector<float>> attention_output_table;   // Attention (8×128)

	// Task mapping and scheduling
	deque<deque<int>> llmOutputPixelMappingTable;  // LLM-specific: maps output pixels to MAC units
	                                               // llmOutputPixelMappingTable[MAC_ID] = {pixel_ids...}
	deque<deque<int>> llmTaskMappingTable;         // LLM-specific: maps tasks to MAC units
	                                               // llmTaskMappingTable[MAC_ID] = {task_ids...}
	void llmXMapping(int total_pixels);

	int llmSAMOSTaskMapping(int pixel_count, int start_pixel_id = 0);  // SAMOS mapping function

	// LLM attention layer management
	void llmCreateAttentionData();
	bool llmReadSavedMatrix();  // Read matrices from files, returns true if successful
	void llmInitializeRandomMatrices();  // Initialize with random data
	bool llmLoadRealMatrices(const std::string& input_dir);
	void llmTaskPartitioning();

	void llmExportTasksToFile();
	void llmExportVerificationResults();
	void llmPrintTimingStatistics();

	// Network and execution management
	void llmNetRunStep();
	void llmCheckStatus();

	// LLM parameters
	int macNum;
	int pe_x;
	int pe_y;
	int current_layer;
	int total_layers;

	// New dimensions for 8×4096 input and 128×4096 Query
	int input_sequence_length;    //  (8)
	int input_hidden_dim;         //  (4096)
	int query_output_dim;         // Query (128)
	int matrixOutputPixels_inputsequencelength;
	int matrixOutputPixels_queryoutputdim;


	int tasks_per_pixel;          // Number of sub-tasks each pixel is divided into (e.g., 4)
	int total_task_slicedPixels;  // Total number of tasks (sliced pixels)
	int total_output_pixels;      // Total number of output pixels (without slicing)



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
		int task_id;              // ID
		int pixel_id;             // pixel ID
		int pixel_x, pixel_y;     // pixel (8x128)
		int time_slice;           // ID，subchunk_id
		int subchunk_id;          // ID (0-3)

		//  - InputQuery，Key
		vector<float> input_data;  // Input
		vector<float> query_data;  // Query
		// Key，InputQuery

		int input_offset;         // input
		int query_offset;         // query
	};

	vector<LLMTask> all_tasks;

	void llmGenerateAllTasks();
	void llmDistributeTasks();

	// Helper functions
	bool llmIsMemoryNode(int node_id);

	~LLMMACnet();
};

#endif /* LLMMACNET_HPP_ */
