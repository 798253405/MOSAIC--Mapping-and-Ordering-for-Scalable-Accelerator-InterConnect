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

	// LLM-specific data structures - 只有Input和Query，没有Key
	vector<vector<float>> input_matrix;         // 输入矩阵 (8×4096)
	vector<vector<float>> query_weight_matrix;  // Query权重矩阵 (128×4096)
	// Key已移除
	vector<vector<float>> Q_resOutput_matrix;   // 输出结果矩阵 (8×128)

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

	// Matrix dimensions
	int matrixOutputPixels_size;  // Size of output pixel matrix (e.g., 128x128)

	// New dimensions for 8×4096 input and 128×4096 Query
	int input_sequence_length;    // 输入序列长度 (8)
	int input_hidden_dim;         // 输入隐藏维度 (4096)
	int query_output_dim;         // Query输出维度 (128)
	int matrixOutputPixels_inputsequencelength;
	int matrixOutputPixels_queryoutputdim;


	int time_slices;
	int tasks_per_pixel;          // Number of sub-tasks each pixel is divided into (e.g., 4)
	int total_task_slicedPixels;  // Total number of tasks (sliced pixels)



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
		int pixel_x, pixel_y;     // pixel坐标 (在8x128矩阵中)
		int time_slice;           // 时间片ID，与subchunk_id相同
		int subchunk_id;          // 子块ID (0-3)
		
		// 数据内容 - 只有Input和Query，没有Key
		vector<float> input_data;  // Input数据元素
		vector<float> query_data;  // Query权重元素
		// Key已移除，因为只需要Input和Query
		
		// 数据范围信息
		int input_offset;         // input数据起始偏移
		int query_offset;         // query数据起始偏移
		float partial_sum;        // 该子块的计算结果
	};

	vector<LLMTask> all_tasks;
	
	/**
	 * @brief LLM状态管理与包处理流程
	 * 
	 * 状态转换与包交互：
	 * ==================
	 * 
	 * MAC状态机循环：
	 * ----------------
	 * State 0 (IDLE) → State 1 (REQUEST)
	 *   触发：llmtasktable有待处理任务
	 *   动作：取出task_id，准备发送请求
	 * 
	 * State 1 (REQUEST) → State 2 (WAIT)
	 *   触发：发送Type 0请求包
	 *   包格式：msgtype=0, data[0]=task_id
	 *   目标：内存节点
	 * 
	 * State 2 (WAIT) → State 3 (COMPUTE)
	 *   触发：收到Type 1响应包
	 *   包格式：msgtype=1, payload=[header(4)+query(64)+key(64)]
	 *   来源：内存节点
	 * 
	 * State 3 (COMPUTE) → State 4 (COMPLETE)
	 *   触发：Attention计算完成
	 *   动作：发送结果包(Type 2或3)
	 * 
	 * State 4 (COMPLETE) → State 0/5
	 *   分支：有任务→State 0，无任务→State 5(FINISHED)
	 * 
	 * 包类型详解：
	 * -----------
	 * Type 0 (REQUEST): MAC请求数据
	 *   - 方向：MAC → Memory
	 *   - 内容：task_id
	 *   - 处理：Memory查找all_tasks[task_id]
	 * 
	 * Type 1 (RESPONSE): Memory返回数据
	 *   - 方向：Memory → MAC
	 *   - 内容：132个float (header+payload)
	 *   - 处理：MAC开始计算
	 * 
	 * Type 2 (INTERMEDIATE): 中间结果
	 *   - 方向：MAC → Memory
	 *   - 内容：[value, x, y, slice]
	 *   - 用途：调试验证
	 * 
	 * Type 3 (FINAL): 最终结果
	 *   - 方向：MAC → Memory
	 *   - 内容：[value, x, y, slice]
	 *   - 处理：更新output_table
	 */
	void llmGenerateAllTasks();
	void llmDistributeTasks();
	
	// Set external input matrices for real data processing
	void setInputMatrices(const vector<vector<float>>& X_input, 
	                     const vector<vector<float>>& Wq);

	// Helper functions
	bool llmIsMemoryNode(int node_id);

	~LLMMACnet();
};

#endif /* LLMMACNET_HPP_ */
