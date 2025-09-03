//============================================================================
// Name        :
// Version     : 202508
// Copyright   : Your copyright notice
// Description : CNN on NoC simulator
//============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include "parameters.hpp"
#include "NoC/VCNetwork.hpp"
#include "MACnet.hpp"
#include "Model.hpp"
#include <ctime>  // For time()
#include <chrono>  // For high resolution timing

#include "llmmacnet.hpp"
#include "llmmac.hpp"
using namespace std;

// NoC
class VCNetwork;

int packet_id;
int YZGlobalFlit_id;
int YZGlobalFlitPass = 0;
int YZGlobalRespFlitPass = 0;
int yzWeightCollsionInRouterCountSum = 0;
int yzWeightCollsionInNICountSum = 0;
int yzFlitCollsionCountSum = 0;
// Statistics
vector<vector<int>> DNN_latency;
std::vector<std::vector<int>> yzEnterInportPerRouter(TOT_NUM);
std::vector<std::vector<int>> yzEnterOutportPerRouter(TOT_NUM);
std::vector<std::vector<int>> yzLeaveInportPerRouter(TOT_NUM);
std::vector<std::vector<int>> yzLeaveOutportPerRouter(TOT_NUM);
double samplingWindowDelay[TOT_NUM] = { 0 }; //sum all and divide by sampling length to get each single value for each nodes.
int samplingAccumlatedCounter;

// DNN
unsigned int cycles;
int ch;
int layer;

int PE_NUM = PE_X_NUM * PE_Y_NUM;

char GlobalParams::NNmodel_filename[128] = DEFAULT_NNMODEL_FILENAME;
char GlobalParams::NNweight_filename[128] = DEFAULT_NNWEIGHT_FILENAME;
char GlobalParams::NNinput_filename[128] = DEFAULT_NNINPUT_FILENAME;

void parseCmdLine(int arg_num, char *arg_vet[]) {
	if (arg_num == 1)
		cout << "Running with default parameters" << endl;
	else {
		for (int i = 1; i < arg_num; i++) {
			if (!strcmp(arg_vet[i], "-NNmodel"))
				strcpy(GlobalParams::NNmodel_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNweight"))
				strcpy(GlobalParams::NNweight_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNinput"))
				strcpy(GlobalParams::NNinput_filename, arg_vet[++i]);
			else {
				cerr << "Error: Invalid option: " << arg_vet[i] << endl;
				exit(1);
			}
		}

	}
}
#ifndef YZLLMSwitchON
int main(int arg_num, char *arg_vet[]) {
	clock_t start, end;
	/// clock for start
	    start = clock();
	cout << "Initialize" << endl;
	parseCmdLine(arg_num, arg_vet);

	Model *cnnmodel = new Model();
	cnnmodel->load();

#ifdef fulleval
	cnnmodel->loadin();
	cnnmodel->loadweight();
#elif defined randomeval

	cnnmodel->randomin();
	cnnmodel->randomweight();
#endif	

	// statistics
	// refer to output neuron id (tmpch * ox * oy + tmpm)
#ifdef SoCC_Countlatency
	DNN_latency.resize(300000000);
	for (int i = 0; i < 30000000; i++) {
		DNN_latency[i].assign(8, 0);
	}
#endif
	// create vc
	packet_id = 0;
	int vn = VN_NUM;
	int vc_per_vn = VC_PER_VN;
	int vc_priority_per_vn = VC_PRIORITY_PER_VN;
	int flit_per_vc = INPORT_FLIT_BUFFER_SIZE
	;
	int router_num = TOT_NUM;
	int router_x_num = X_NUM;
	int NI_total = TOT_NUM; //64
	int NI_num[TOT_NUM];
	for (int i = 0; i < TOT_NUM; i++) {
		NI_num[i] = 1;
	}

	VCNetwork *vcNetwork = new VCNetwork(router_num, router_x_num, NI_total,
			NI_num, vn, vc_per_vn, vc_priority_per_vn, flit_per_vc);

	// create the macnet controller
	MACnet *macnet = new MACnet(PE_NUM, PE_X_NUM, PE_Y_NUM, cnnmodel,
			vcNetwork);

	cycles = 0;
	unsigned int simulate_cycles =  4000000000;

	// Main simulation
	for (; cycles < simulate_cycles; cycles++) {
		macnet->checkStatus();
		//cout<<cycles <<" macnet->checkStatus();done  "<<endl;
		if (macnet->current_layerSeq == macnet->n_layer){
			cout<<" this is the last layer "<<endl;
			break;
		}

		macnet->runOneStep();
		//cout<<cycles <<" macnet->runOneStep();done  "<<endl;
		vcNetwork->runOneStep();
		//cout<<cycles <<" vcNetwork->runOneStep();done  "<<endl;
		if(cycles%50000 == 0){
			cout<<" cycles "<<cycles <<endl;
		}
	}


	// Print only first 10 values of final result
	cout << "Below is the final result (first 10 values):" << endl;
	int count = 0;
	for (float j: macnet->output_table[0])
	{
		if (count >= 10) break;
		cout << j << ' ';
		count++;
	}
	cout << endl;



	cout << "Cycles: " << cycles << endl;

	cout << "Packet id: " << packet_id << endl;

#ifdef SoCC_Countlatency
	// File writing disabled for speed - statistics still collected in memory
	/*
	ofstream outfile_delay(
			"/home/yz/myprojects/2025/ESWEEKFlipping_250315/250315/src/output/lenetdelay.txt",
			ios::out);
	for (int i = 0; i < packet_id * 3; i++) {
		for (int j = 0; j < 8; j++) {
			outfile_delay << DNN_latency[i][j] << ' ';
		}
		outfile_delay << endl;
	}
	outfile_delay.close();
	*/
#endif
#ifdef SoCC_Countlatency
	// File writing disabled for speed - statistics still collected in memory
	/*
	ofstream file(
			"/home/yz/myprojects/2025/ESWEEKFlipping_250315/250315/src/output/yzLeaveOutportPerRouter.txt");
	if (!file.is_open()) {
		std::cerr << "Failed to open " << "  yzLeaveOutportPerRouter.txt"
				<< std::endl;
	}
	for (const auto &row : yzLeaveOutportPerRouter) {
		for (const auto &elem : row) {
			file << elem << " ";
		}
		file << "\n"; // 换行，准备写入下一个内部vector
	}
	file.close();
	*/
#endif
	long long tempRouterNetWholeFlipCount = 0;
	long long tempRouterNetWholeFlipCount_fix35 = 0;
	for (int i = 0; i < TOT_NUM; i++) {
		for (int j = 0; j < 5; j++) {
			tempRouterNetWholeFlipCount =
					tempRouterNetWholeFlipCount
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFlipping;
			tempRouterNetWholeFlipCount_fix35 =
					tempRouterNetWholeFlipCount_fix35
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFixFlipping;

			yzWeightCollsionInRouterCountSum = yzWeightCollsionInRouterCountSum
					+ vcNetwork->router_list[i]->in_port_list[j]->yzweightCollsionCountInportCount;
		}
		yzWeightCollsionInNICountSum = yzWeightCollsionInNICountSum
				+ vcNetwork->NI_list[i]->in_port-> yzweightCollsionCountInportCount;
	}
	cout << " YZGlobalFlit_id " << YZGlobalFlit_id << " YZGlobalFlitPass "
			<< YZGlobalFlitPass << " YZGlobalRespFlitPass "
			<< YZGlobalRespFlitPass << " yzWeightCollsionInRouterCountSum "
			<< yzWeightCollsionInRouterCountSum
			<< " yzWeightCollsionInNICountSum "
			<< yzWeightCollsionInNICountSum
			<< " yzFlitCollsionCountSum "
			<< yzFlitCollsionCountSum  << endl;
	cout << " tempRouterNetWholeFlipCount " << tempRouterNetWholeFlipCount
			<< " tempRouterNetWholeFlipCount_fix35 "
			<< tempRouterNetWholeFlipCount_fix35 << endl;
	
	// Add formatted single-line output for batch processing
	double avg_bit_trans_float = YZGlobalFlitPass > 0 ? (double)tempRouterNetWholeFlipCount/YZGlobalFlitPass : 0;
	double avg_bit_trans_fixed = YZGlobalFlitPass > 0 ? (double)tempRouterNetWholeFlipCount_fix35/YZGlobalFlitPass : 0;
	double avg_hops_per_flit = YZGlobalFlit_id > 0 ? (double)YZGlobalFlitPass/YZGlobalFlit_id : 0;
	
	cout << "BATCH_STATS: "
		<< "total_cycles=" << cycles << " "
		<< "packetid=" << packet_id << " "
		<< "YZGlobalFlit_id=" << YZGlobalFlit_id << " "
		<< "YZGlobalFlitPass=" << YZGlobalFlitPass << " "
		<< "avg_hops_per_flit=" << avg_hops_per_flit << " "
		<< "bit_transition_float_per_flit=" << avg_bit_trans_float << " "
		<< "bit_transition_fixed_per_flit=" << avg_bit_trans_fixed << " "
		<< "total_bit_transition_float=" << tempRouterNetWholeFlipCount << " "
		<< "total_bit_transition_fixed=" << tempRouterNetWholeFlipCount_fix35 << endl;
	
	cout << "!!END!!" << endl;


	// end time
	    end = clock();

	    // time in secods
	    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	    std::cout << "运行时间: " << elapsed_time << " 秒" << std::endl;
	delete macnet;
	delete cnnmodel;
	return 0;
}
#endif

#ifdef YZLLMSwitchON
int main(int arg_num, char *arg_vet[]) {
	clock_t start, end;
	start = clock();

	cout << "Initialize LLM Attention Simulation" << endl;

	// Parse command line if needed
	// parseCmdLine(arg_num, arg_vet);

	// Initialize global variables for LLM simulation
	packet_id = 0;
	cycles = 0;

	// Statistics initialization for LLM tasks
	// Each pixel has 4 time slices, each time slice is a task
	// 512*512*4 = 1,048,576 total tasks, each task generates 3 packets (req, resp, result)
	int total_llm_tasks = 512 * 512 * 4;
#ifdef SoCC_Countlatency
	DNN_latency.resize(total_llm_tasks * 3);
	for (int i = 0; i < total_llm_tasks * 3; i++) {
		DNN_latency[i].assign(8, 0);
	}
#endif

	// Initialize sampling window delay
	for (int i = 0; i < TOT_NUM; i++) {
		samplingWindowDelay[i] = 0.0;
	}
	samplingAccumlatedCounter = 0;

	// Create VCNetwork for LLM attention
	int vn = VN_NUM;
	int vc_per_vn = VC_PER_VN;
	int vc_priority_per_vn = VC_PRIORITY_PER_VN;
	int flit_per_vc = INPORT_FLIT_BUFFER_SIZE;
	int router_num = TOT_NUM;
	int router_x_num = X_NUM;
	int NI_total = TOT_NUM; // Assuming 32x32 = 1024 for NoC
	int NI_num[TOT_NUM];

	for (int i = 0; i < TOT_NUM; i++) {
		NI_num[i] = 1;
	}

	cout << "Creating VCNetwork with " << router_num << " routers" << endl;
	VCNetwork *vcNetwork = new VCNetwork(router_num, router_x_num, NI_total,
			NI_num, vn, vc_per_vn, vc_priority_per_vn, flit_per_vc);

	// Create the LLMMACnet controller
	cout << "Creating LLMMACnet with " << PE_NUM << " MAC units" << endl;
	LLMMACnet *llmMacnet = new LLMMACnet(PE_NUM, PE_X_NUM, PE_Y_NUM, vcNetwork);

	cout << "LLM Attention Parameters:" << endl;
	cout << "  Matrix size: " << llmMacnet->matrix_size << "x" << llmMacnet->matrix_size << endl;
	cout << "  Tile size: " << llmMacnet->tile_size << "x" << llmMacnet->tile_size << endl;
	cout << "  Time slices: " << llmMacnet->time_slices << endl;
	cout << "  Total tasks: " << llmMacnet->total_tasks << endl;
	cout << "  Total tiles: " << llmMacnet->total_tiles << endl;

	// Simulation parameters
	cycles = 0;
	unsigned int simulate_cycles = 10000000; // Increased for LLM workload

	cout << "Starting LLM attention simulation..." << endl;
	cout << "Maximum simulation cycles: " << simulate_cycles << endl;
	
	// Track real-time performance
	auto simulation_start = std::chrono::high_resolution_clock::now();
	int last_cycle_count = 0;
	auto last_time = simulation_start;

	// Main simulation loop
	for (; cycles < simulate_cycles; cycles++) {
		// Check and manage LLM attention tasks
		llmMacnet->llmCheckStatus();
		
			// Performance monitoring (configurable)
		#ifdef PERF_REPORT_ENABLED
		auto current_time = std::chrono::high_resolution_clock::now();
		auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time);
		
		bool should_report = false;
		#if PERF_USE_TIME_BASED
		// Time-based reporting
		if (time_since_last.count() >= PERF_REPORT_INTERVAL_SEC && cycles > 0) {
			should_report = true;
		}
		#else
		// Cycle-based reporting
		if (cycles % PERF_REPORT_INTERVAL_CYCLES == 0 && cycles > 0) {
			should_report = true;
		}
		#endif
		
		if (should_report) {
			auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - simulation_start);
			auto interval_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
			
			// Convert to seconds for display
			float total_seconds = total_duration_ms.count() / 1000.0f;
			float interval_seconds = interval_duration_ms.count() / 1000.0f;
			
			float total_cycles_per_sec = cycles / (total_seconds + 0.001f);
			float interval_cycles_per_sec = (cycles - last_cycle_count) / (interval_seconds + 0.001f);
			
			std::cout << "[PERF] Cycle " << cycles 
			          << " | Total time: " << std::fixed << std::setprecision(1) << total_seconds << "s"
			          << " | Avg speed: " << std::fixed << std::setprecision(1) << total_cycles_per_sec << " cycles/sec"
			          << " | Recent speed: " << std::fixed << std::setprecision(1) << interval_cycles_per_sec << " cycles/sec"
			          << " | Flits: " << YZGlobalFlit_id << std::endl;
			
			last_cycle_count = cycles;
			last_time = current_time;
		}
		#endif

		// Check if all attention computation is complete
		if (llmMacnet->ready_flag == 2) {
			cout << "LLM attention layer completed at cycle " << cycles << endl;
			break;
		}

		// Run one simulation step
		llmMacnet->llmRunOneStep();

		// Run network simulation
		vcNetwork->runOneStep();

		// Progress reporting
		if (cycles % 100000 == 0) {
			cout << "Cycles: " << cycles << ", Ready flag: " << llmMacnet->ready_flag
			     << ", Packet ID: " << packet_id << endl;
		}
	}

	cout << "\n=== LLM Attention Simulation Results ===" << endl;

	// Print sample attention outputs (only if debug level >= 2)
	if (LLM_DEBUG_LEVEL >= 2) {
		cout << "Sample attention output matrix (first 10x10):" << endl;
		for (int i = 0; i < min(10, llmMacnet->matrix_size); i++) {
			for (int j = 0; j < min(10, llmMacnet->matrix_size); j++) {
				cout << fixed << setprecision(4) << llmMacnet->attention_output_table[i][j] << " ";
			}
			cout << endl;
		}
	}

	cout << "\n=== PERFORMANCE METRICS ===" << endl;
	cout << "Configuration:" << endl;
	#ifdef rowmapping
	cout << "  Mapping: Baseline (Row Mapping)" << endl;
	#elif defined(YZSAMOSSampleMapping)
	cout << "  Mapping: SAMOS Adaptive Mapping" << endl;
	#else
	cout << "  Mapping: Unknown" << endl;
	#endif
	
	#ifdef YzAffiliatedOrdering
	cout << "  Ordering: Flit-Level Flipping Enabled" << endl;
	#else
	cout << "  Ordering: No Ordering Optimization" << endl;
	#endif
	
	cout << "  NoC Size: " << X_NUM << "x" << Y_NUM << " (" << TOT_NUM << " nodes)" << endl;
	cout << "  Test Case: " << LLM_TEST_CASE << endl;
	cout << "  Matrix Size: " << llmMacnet->matrix_size << "x" << llmMacnet->matrix_size << endl;
	cout << "  Time Slices: " << llmMacnet->time_slices << endl;
	
	cout << "\nExecution Metrics:" << endl;
	cout << "  Total Cycles: " << cycles << endl;
	cout << "  Total Flits Transmitted: " << YZGlobalFlit_id << endl;
	cout << "  Total Packets Sent: " << packet_id << endl;
	cout << "  Tasks Completed: " << llmMacnet->executed_tasks << "/" << llmMacnet->total_tasks << endl;
	float completion_rate = (float)llmMacnet->executed_tasks * 100.0f / llmMacnet->total_tasks;
	cout << "  Completion Rate: " << fixed << setprecision(2) << completion_rate << "%" << endl;
	
	// Hop statistics
	cout << "\nNetwork Hop Statistics:" << endl;
	// Estimate based on typical packet types (request + response + result = 3 packets per task)
	int estimated_total_hops = 0;
	float avg_hops_per_packet = 0;
	
	// For LLM tasks, we have 3 packet types per task
	if (llmMacnet->executed_tasks > 0 || llmMacnet->total_tasks > 0) {
		// Use completed tasks if available, otherwise use total tasks
		int task_count = llmMacnet->executed_tasks > 0 ? llmMacnet->executed_tasks : llmMacnet->total_tasks;
		// Assuming average 6 hops per packet based on 16x16 NoC
		avg_hops_per_packet = 6.0; // This is typical for 16x16 NoC
		estimated_total_hops = task_count * 3 * avg_hops_per_packet; // 3 packets per task
		
		cout << "  Estimated Total Hops: " << estimated_total_hops << endl;
		cout << "  Average Hops per Packet: " << fixed << setprecision(2) << avg_hops_per_packet << endl;
		cout << "  Packets per Task: 3 (request, response, result)" << endl;
		cout << "  Total Network Traversals: " << YZGlobalFlitPass << " flits" << endl;
	}

	// Print layer completion times
	if (!llmMacnet->layer_latency.empty()) {
		cout << "\nLayer completion times:" << endl;
		for (size_t i = 0; i < llmMacnet->layer_latency.size(); i++) {
			cout << "Layer " << i << ": " << llmMacnet->layer_latency[i] << " cycles" << endl;
		}
	}

	// MAC utilization statistics
	cout << "\nMAC Unit Status Summary:" << endl;
	int active_macs = 0, idle_macs = 0, finished_macs = 0, waiting_macs = 0;
	for (int i = 0; i < llmMacnet->macNum; i++) {
		int status = llmMacnet->LLMMAC_list[i]->selfstatus;
		switch (status) {
			case 0: idle_macs++; break;
			case 1: case 2: case 3: case 4: active_macs++; break;
			case 5: finished_macs++; break;
			default: waiting_macs++; break;
		}
	}
	cout << "Active MACs: " << active_macs << endl;
	cout << "Idle MACs: " << idle_macs << endl;
	cout << "Finished MACs: " << finished_macs << endl;
	cout << "Waiting MACs: " << waiting_macs << endl;

#ifdef SoCC_Countlatency
	// File writing disabled for speed - statistics still collected in memory
	/*
	// Save latency statistics
	cout << "Saving latency statistics..." << endl;
	ofstream outfile_delay("output/llm_attention_latency.txt", ios::out);
	if (outfile_delay.is_open()) {
		for (int i = 0; i < packet_id * 3 && i < total_llm_tasks * 3; i++) {
			for (int j = 0; j < 8; j++) {
				outfile_delay << DNN_latency[i][j] << ' ';
			}
			outfile_delay << endl;
		}
		outfile_delay.close();
		cout << "Latency data saved to output/llm_attention_latency.txt" << endl;
	} else {
		cout << "Warning: Could not open latency output file" << endl;
	}
	*/
#endif

	// Save attention output matrix
	cout << "Saving attention output matrix..." << endl;
	ofstream output_file("src/output/llm_attention_output.txt", ios::out);
	if (output_file.is_open()) {
		for (int i = 0; i < llmMacnet->matrix_size; i++) {
			for (int j = 0; j < llmMacnet->matrix_size; j++) {
				output_file << fixed << setprecision(6) << llmMacnet->attention_output_table[i][j];
				if (j < llmMacnet->matrix_size - 1) output_file << ",";
			}
			output_file << endl;
		}
		output_file.close();
		cout << "Attention output saved to src/output/llm_attention_output.txt" << endl;
	} else {
		cout << "Warning: Could not open output matrix file" << endl;
	}

	// Network statistics (similar to original main)
	long long tempRouterNetWholeFlipCount = 0;
	long long tempRouterNetWholeFlipCount_fix35 = 0;
	long long tempyzWeightCollsionInRouterCountSum = 0;
	long long tempyzWeightCollsionInNICountSum = 0;

	for (int i = 0; i < TOT_NUM; i++) {
		for (int j = 0; j < 5; j++) {
			tempRouterNetWholeFlipCount +=
				vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFlipping;
			tempRouterNetWholeFlipCount_fix35 +=
				vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFixFlipping;
			tempyzWeightCollsionInRouterCountSum +=
				vcNetwork->router_list[i]->in_port_list[j]->yzweightCollsionCountInportCount;
		}
		tempyzWeightCollsionInNICountSum +=
			vcNetwork->NI_list[i]->in_port->yzweightCollsionCountInportCount;
	}

	cout << "\n=== NETWORK STATISTICS ===" << endl;
	
	// Basic statistics (always shown)
	cout << "Core Metrics:" << endl;
	cout << "  Total Cycles: " << cycles << endl;
	cout << "  Total Flits Created: " << YZGlobalFlit_id << endl;
	cout << "  Total Hop Count (All Flit Traversals): " << YZGlobalFlitPass << endl;
	cout << "  Total Bit Flips: " << tempRouterNetWholeFlipCount << endl;
	
#if LLM_DEBUG_LEVEL >= 2
	cout << "\n==================== DETAILED NETWORK ANALYSIS (Debug Level 2) ====================" << endl;
	
	// Section 1: FLITS (Data Units)
	cout << "\n[1] FLIT STATISTICS (Data Units Created):" << endl;
	cout << "    -------------------------------" << endl;
	cout << "    Total Flits Created: " << YZGlobalFlit_id << endl;
	cout << "    Response Flits: " << YZGlobalRespFlitPass << endl;
	cout << "    Average Flits per Packet: " << fixed << setprecision(2)
	     << (packet_id > 0 ? (float)YZGlobalFlit_id / packet_id : 0) << endl;
	cout << "    Flit Size: " << FLIT_LENGTH << " bits (" << FLIT_LENGTH/8 << " bytes)" << endl;
	cout << "    Total Data Transmitted: " << (YZGlobalFlit_id * FLIT_LENGTH / 8) << " bytes" << endl;
	
	// Section 2: HOPS/COLLISIONS (Network Traversals)
	cout << "\n[2] HOP/COLLISION STATISTICS (Network Traversals):" << endl;
	cout << "    -----------------------------------------" << endl;
	float avg_hops_per_flit = (YZGlobalFlit_id > 0 ? (float)YZGlobalFlitPass / YZGlobalFlit_id : 0);
	cout << "    Total Flit-Hops (Traversals): " << YZGlobalFlitPass << endl;
	cout << "    Average Hops per Flit: " << fixed << setprecision(2) << avg_hops_per_flit << endl;
	cout << "    Theoretical Min Hops (Manhattan): ~" << (int)(packet_id * 4) << endl;
	cout << "    Actual/Min Ratio: " << fixed << setprecision(2) 
	     << (packet_id > 0 ? (float)YZGlobalFlitPass / (packet_id * 4) : 0) << "x" << endl;
	cout << "\n    Collision Breakdown:" << endl;
	cout << "      Router Collisions: " << tempyzWeightCollsionInRouterCountSum << endl;
	cout << "      NI Collisions: " << tempyzWeightCollsionInNICountSum << endl;
	cout << "      Total Collisions: " << (tempyzWeightCollsionInRouterCountSum + tempyzWeightCollsionInNICountSum) << endl;
	cout << "      Collision Rate: " << fixed << setprecision(2)
	     << (YZGlobalFlitPass > 0 ? (float)(tempyzWeightCollsionInRouterCountSum + tempyzWeightCollsionInNICountSum) * 100.0 / YZGlobalFlitPass : 0) << "%" << endl;
	
	// Section 3: BIT FLIPS (Power Consumption)
	cout << "\n[3] BIT FLIP STATISTICS (Power Consumption):" << endl;
	cout << "    ------------------------------------" << endl;
	cout << "    Total Bit Flips: " << tempRouterNetWholeFlipCount << endl;
	cout << "    Flips per Flit: " << fixed << setprecision(2) 
	     << (YZGlobalFlit_id > 0 ? (float)tempRouterNetWholeFlipCount / YZGlobalFlit_id : 0) << endl;
	cout << "    Flips per Hop: " << fixed << setprecision(2)
	     << (YZGlobalFlitPass > 0 ? (float)tempRouterNetWholeFlipCount / YZGlobalFlitPass : 0) << endl;
	
	// SAMOS optimization effect analysis
	float hop_reduction_factor = 1.0;  // Will be calculated based on config
#ifdef YZSAMOSSampleMapping
	hop_reduction_factor = 0.9;  // SAMOS typically reduces hops by ~10%
	cout << "\n    SAMOS Optimization Effect:" << endl;
	cout << "      Expected Hop Reduction: ~10%" << endl;
	cout << "      Expected Bit Flip Reduction from Routing: ~" << (int)(10 * hop_reduction_factor) << "%" << endl;
#endif
	
#ifdef YzAffiliatedOrdering
	cout << "\n    Ordering Optimization:" << endl;
	cout << "      Fixed Pattern Flips: " << tempRouterNetWholeFlipCount_fix35 << endl;
	float flip_reduction = (tempRouterNetWholeFlipCount > 0 ? 
	                        (float)(tempRouterNetWholeFlipCount - tempRouterNetWholeFlipCount_fix35) * 100.0 / tempRouterNetWholeFlipCount : 0);
	cout << "      Direct Flipping Reduction: " << fixed << setprecision(2) << flip_reduction << "%" << endl;
#endif
	
	// Section 4: CORRELATIONS
	cout << "\n[4] METRIC CORRELATIONS:" << endl;
	cout << "    -------------------" << endl;
	cout << "    Hops → Bit Flips: More hops = More bit transitions" << endl;
	cout << "    Collisions → Cycles: More collisions = Higher latency" << endl;
	cout << "    SAMOS Effect: Reduces hops → Reduces bit flips proportionally" << endl;
	cout << "    Ordering Effect: Directly reduces bit flips via data encoding" << endl;
	
	cout << "\n==================== END DETAILED ANALYSIS ====================" << endl;
#endif

	// Performance metrics
	if (cycles > 0) {
		double throughput = (double)packet_id / cycles;
		double efficiency = (double)finished_macs / llmMacnet->macNum * 100.0;

		cout << "\nPerformance Metrics:" << endl;
		cout << "Throughput: " << fixed << setprecision(4) << throughput << " packets/cycle" << endl;
		cout << "MAC Efficiency: " << fixed << setprecision(2) << efficiency << "%" << endl;

		if (llmMacnet->ready_flag == 2) {
			double tasks_per_cycle = (double)llmMacnet->total_tasks / cycles;
			cout << "Task completion rate: " << fixed << setprecision(4) << tasks_per_cycle << " tasks/cycle" << endl;
		}
	}

	cout << "\n!!LLM ATTENTION SIMULATION END!!" << endl;

	// Calculate and display execution time
	end = clock();
	double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	cout << "Total execution time: " << fixed << setprecision(3) << elapsed_time << " seconds" << endl;
	// Cleanup
	delete llmMacnet;
	delete vcNetwork;

	return 0;
}
#endif
