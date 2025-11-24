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
#include <algorithm>  // For sort
#include <climits>    // For INT_MAX
#include "parameters.hpp"
#include "NoC/VCNetwork.hpp"
#include "NoC/VCRouter.hpp"
#include "MACnet.hpp"
#include "Model.hpp"
#include <ctime>  // For time()
#include <chrono>  // For high resolution timing
#include <cmath>  // For sqrt()

#include "llmmacnet.hpp"
#include "llmmac.hpp"
using namespace std;

// NoC
class VCNetwork;

long long  packet_id;
long long  authorGlobalFlit_id;
long long  authorGlobalFlitPass = 0;  // Total hop count (router + NI)
long long authorGlobalRouterHopCount = 0;  // Router-only hop count
long long authorGlobalNIHopCount = 0;  // NI-only hop count
long long authorGlobalRespFlitPass = 0;
long long authorFlitCollsionCountSum = 0;

// Statistics
vector<vector<int>> DNN_latency;
std::vector<std::vector<int>> authorEnterInportPerRouter(TOT_NUM);
std::vector<std::vector<int>> authorEnterOutportPerRouter(TOT_NUM);
std::vector<std::vector<int>> authorLeaveInportPerRouter(TOT_NUM);
std::vector<std::vector<int>> authorLeaveOutportPerRouter(TOT_NUM);
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
#ifndef AUTHORLLMSwitchON
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
	DNN_latency.resize(9000000);
	for (int i = 0; i < 9000000; i++) {
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
			"src/output/lenetdelay.txt",
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
			"src/output/authorLeaveOutportPerRouter.txt");
	if (!file.is_open()) {
		std::cerr << "Failed to open " << "  authorLeaveOutportPerRouter.txt"
				<< std::endl;
	}
	for (const auto &row : authorLeaveOutportPerRouter) {
		for (const auto &elem : row) {
			file << elem << " ";
		}
		file << "\n"; // ，vector
	}
	file.close();
	*/
#endif


	// Network statistics (similar to original main)

	long long tempauthorWeightCollsionInRouterCountSum = 0;
	long long tempauthorWeightCollsionInNICountSum = 0;
	long long mainauthorRouterZeroBTHopTotalCount = 0;
	long long authorWeightCollsionInRouterCountSum = 0;
	long long authorWeightCollsionInNICountSum = 0;
	long long tempRouterNetWholeFlipCount = 0;
	long long tempRouterNetWholeFlipCount_fix35 = 0;
	long long reqRouterFlip = 0;
	long long respRouterFlip = 0;
	long long resRouterFlip = 0;
	long long reqRouterHop = 0;
	long long respRouterHop = 0;
	long long resRouterHop = 0;
	for (int i = 0; i < TOT_NUM; i++) {
		for (int j = 0; j < 5; j++) {
			tempRouterNetWholeFlipCount =
					tempRouterNetWholeFlipCount
							+ vcNetwork->router_list[i]->in_port_list[j]->totalauthorInportFlipping;
			tempRouterNetWholeFlipCount_fix35 =
					tempRouterNetWholeFlipCount_fix35
							+ vcNetwork->router_list[i]->in_port_list[j]->totalauthorInportFixFlipping;

			authorWeightCollsionInRouterCountSum = authorWeightCollsionInRouterCountSum
					+ vcNetwork->router_list[i]->in_port_list[j]->authorweightCollsionCountInportCount;
			mainauthorRouterZeroBTHopTotalCount  = mainauthorRouterZeroBTHopTotalCount  +vcNetwork->router_list[i]->in_port_list[j]->zeroBTHopCount;

			reqRouterFlip = reqRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->reqRouterFlipInport;
			respRouterFlip = respRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->respRouterFlipInport;
			resRouterFlip = resRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->resRouterFlipInport;

			reqRouterHop = reqRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->reqRouterHopInport;
			respRouterHop = respRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->respRouterHopInport;
			resRouterHop = resRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->resRouterHopInport;
		}
		authorWeightCollsionInNICountSum = authorWeightCollsionInNICountSum
				+ vcNetwork->NI_list[i]->in_port-> authorweightCollsionCountInportCount;
	}
	cout << " authorGlobalFlit_id " << authorGlobalFlit_id
			<< " authorGlobalFlitPass(total) " << authorGlobalFlitPass
			<< " authorGlobalRouterHopCount " << authorGlobalRouterHopCount
			<< " authorGlobalNIHopCount " << authorGlobalNIHopCount
			<< " authorGlobalRespFlitPass " << authorGlobalRespFlitPass
			<< " authorWeightCollsionInRouterCountSum "
			<< authorWeightCollsionInRouterCountSum
			<< " authorWeightCollsionInNICountSum "
			<< authorWeightCollsionInNICountSum
			<< " authorFlitCollsionCountSum "
			<< authorFlitCollsionCountSum  << endl;
	cout << " tempRouterNetWholeFlipCount " << tempRouterNetWholeFlipCount
			<< " tempRouterNetWholeFlipCount_fix35 "
			<< tempRouterNetWholeFlipCount_fix35 << endl;

	// Message type-specific bit flip statistics
	cout << " reqRouterFlip " << reqRouterFlip
		 << " respRouterFlip " << respRouterFlip
		 << " resRouterFlip " << resRouterFlip << endl;

	// Message type-specific hop count statistics
	cout << " reqRouterHop " << reqRouterHop
		 << " respRouterHop " << respRouterHop
		 << " resRouterHop " << resRouterHop << endl;

	// Add formatted single-line output for batch processing
	// Use authorGlobalRouterHopCount for router-only statistics
	double avg_bit_trans_float = authorGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount/authorGlobalRouterHopCount : 0;
	double avg_bit_trans_fixed = authorGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount_fix35/authorGlobalRouterHopCount : 0;
	double avg_hops_per_flit = authorGlobalFlit_id > 0 ? (double)authorGlobalFlitPass/authorGlobalFlit_id : 0;
	double avg_flips_per_flit_total = authorGlobalFlit_id > 0 ? (double)tempRouterNetWholeFlipCount/authorGlobalFlit_id : 0;
	double avg_flips_per_flit_per_router_hop = authorGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount/authorGlobalRouterHopCount : 0;

	cout << "BATCH_STATS: "
		<< "total_cycles=" << cycles << " "
		<< "packetid=" << packet_id << " "
		<< "authorGlobalFlit_id=" << authorGlobalFlit_id << " "
		<< "authorGlobalFlitPass=" << authorGlobalFlitPass << " "
		<< "avg_hops_per_flit=" << avg_hops_per_flit << " "
		<< "avg_flips_per_flit_total=" << avg_flips_per_flit_total << " "
		<< "avg_flips_per_flit_per_router_hop=" << avg_flips_per_flit_per_router_hop << " "
		<< "bit_transition_float_per_hop=" << avg_bit_trans_float << " "
		<< "bit_transition_fixed_per_hop=" << avg_bit_trans_fixed << " "
		<< "total_bit_transition_float=" << tempRouterNetWholeFlipCount << " "
		<< "total_bit_transition_fixed=" << tempRouterNetWholeFlipCount_fix35 << endl;



	// Basic statistics (always shown)
	cout << "Core Metrics:" << endl;
	cout << "  Total Cycles: " << cycles << endl;
	cout << "  Total Flits Created: " << authorGlobalFlit_id << endl;
	cout << "  Total Hop Count (Router+NI): " << authorGlobalFlitPass << endl;
	cout << "  Router Hop Count: " << authorGlobalRouterHopCount << endl;
	cout<<" mainauthorRouterZeroBTHopTotalCount  " <<mainauthorRouterZeroBTHopTotalCount <<endl;
	cout << "  NI Hop Count: " << authorGlobalNIHopCount << endl;
	cout << "  Total Bit Flips (Router-only): " << tempRouterNetWholeFlipCount << endl;
	// Calculate per-flit averages

	float avg_router_hops_per_flit = 0.0;
	float avg_ni_hops_per_flit = 0.0;
	float avg_flips_per_flit = 0.0;
	float avg_flips_per_router_hop = 0.0;
	if (authorGlobalFlit_id > 0) {
		avg_hops_per_flit = (float)authorGlobalFlitPass / authorGlobalFlit_id;
		avg_router_hops_per_flit = (float)authorGlobalRouterHopCount / authorGlobalFlit_id;
		avg_ni_hops_per_flit = (float)authorGlobalNIHopCount / authorGlobalFlit_id;
		avg_flips_per_flit = (float)tempRouterNetWholeFlipCount / authorGlobalFlit_id;
	}
	if (authorGlobalRouterHopCount > 0) {
		avg_flips_per_router_hop = (float)tempRouterNetWholeFlipCount / authorGlobalRouterHopCount;
	}
	cout << "  Average Hops per Flit (total): " << fixed << setprecision(2) << avg_hops_per_flit << endl;
	cout << "  Average Router Hops per Flit: " << fixed << setprecision(2) << avg_router_hops_per_flit << endl;
	cout << "  Average NI Hops per Flit: " << fixed << setprecision(2) << avg_ni_hops_per_flit << endl;
	cout << "  Average Bit Flips per Flit（totalhops）: " << fixed << setprecision(2) << avg_flips_per_flit << endl;
	cout << "  Average Bit Flips per Router Hop: " << fixed << setprecision(2) << avg_flips_per_router_hop << endl;
	cout << "  Average Bit Flips per respRouter Hop: " << fixed << setprecision(2) <<respRouterFlip/respRouterHop  << endl;

	cout << "!!END!!" << endl;


	// end time
	    end = clock();

	    // time in secods
	    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	    std::cout << ": " << elapsed_time << " " << std::endl;
	delete macnet;
	delete cnnmodel;
	return 0;
}
#endif









#ifdef AUTHORLLMSwitchON
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
	// LLM uses its own timing mechanism (task_timings in each LLMMAC)
	// DNN_latency is only needed when SoCC_Countlatency is enabled
	#ifdef SoCC_Countlatency
	// Allocate based on expected packet count (conservative estimate)
	int estimated_packets = 10000000;  // 10M packets should be enough
	DNN_latency.resize(estimated_packets * 3);
	for (int i = 0; i < estimated_packets * 3; i++) {
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

	// Matrix loading is now handled inside LLMMACnet initialization
	// No need to load here anymore

	cout << "LLM Attention Parameters:" << endl;
	cout << "  Output matrix size: " << llmMacnet->input_sequence_length << "x" << llmMacnet->query_output_dim << endl;
	cout << "  Total tasks (quicktest): " << llmMacnet->total_task_slicedPixels << endl;

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
			          << " | Flits: " << authorGlobalFlit_id << std::endl;

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
		llmMacnet->llmNetRunStep();

		// Run network simulation
		vcNetwork->runOneStep();

		// Progress reporting
		if (cycles % 100000 == 0) {
			cout << "Cycles: " << cycles << ", Ready flag: " << llmMacnet->ready_flag
			     << ", Packet ID: " << packet_id << endl;
		}
	}

	cout << "\n=== LLM Attention Simulation Results ===" << endl;

	cout << "\n=== PERFORMANCE METRICS ===" << endl;
	cout << "Configuration:" << endl;

	// Mapping strategy
	#ifdef rowmapping
	cout << "  Mapping: Baseline (Row Mapping)" << endl;
	#elif defined(AUTHORSAMOSSampleMapping)
	cout << "  Mapping: SAMOS Adaptive Mapping" << endl;
	#else
	cout << "  Mapping: Unknown" << endl;
	#endif

	// Ordering optimization
	#ifdef AuthorAffiliatedOrdering
	cout << "  Ordering: Flit-Level Flipping Enabled" << endl;
	#else
	cout << "  Ordering: No Ordering Optimization" << endl;
	#endif

	// Separated ordering
	#ifdef AUTHORSeperatedOrdering_reArrangeInput
	cout << "  Separated Ordering: Enabled" << endl;
	#endif

	// Fire Advance
	#ifdef fireAdvance
	cout << "  Fire Advance: Enabled (Delay=" << FIRE_ADVANCE_DELAY << " cycles)" << endl;
	#endif

	// Binary Routing Switch
	#ifdef binaryroutingSwitch
	cout << "  Binary Routing: Enabled (Request=XY, Response=YX)" << endl;
	#endif

	cout << "  NoC Size: " << X_NUM << "x" << Y_NUM << " (" << TOT_NUM << " nodes)" << endl;
	cout << "  LLM Test Case: " << LLM_TOKEN_SIZE << endl;

	cout << "\nExecution Metrics:" << endl;
	cout << "  Total Cycles: " << cycles << endl;
	cout << "  Total Flits Transmitted: " << authorGlobalFlit_id << endl;
	cout << "  Total Packets Sent: " << packet_id << endl;
	cout << "  Pixels Completed: " << llmMacnet->executed_tasks << "/" << llmMacnet->total_output_pixels << endl;
	cout << "  Total Sub-tasks: " << llmMacnet->total_task_slicedPixels << " (" << llmMacnet->tasks_per_pixel << " per pixel)" << endl;
	float completion_rate = (float)llmMacnet->executed_tasks * 100.0f / llmMacnet->total_output_pixels;
	cout << "  Completion Rate: " << fixed << setprecision(2) << completion_rate << "%" << endl;

	// Hop statistics
	cout << "\nNetwork Hop Statistics:" << endl;
	// Estimate based on typical packet types (request + response + result = 3 packets per task)
	int estimated_total_hops = 0;
	float avg_hops_per_packet = 0;

	// For LLM tasks, we have 3 packet types per task
	if (llmMacnet->executed_tasks > 0 || llmMacnet->total_task_slicedPixels > 0) {
		// Use completed tasks if available, otherwise use total tasks
		int task_count = llmMacnet->executed_tasks > 0 ? llmMacnet->executed_tasks : llmMacnet->total_task_slicedPixels;
		// Assuming average 6 hops per packet based on 16x16 NoC
		avg_hops_per_packet = 6.0; // This is typical for 16x16 NoC
		estimated_total_hops = task_count * 3 * avg_hops_per_packet; // 3 packets per task

		cout << "  Estimated Total Hops: " << estimated_total_hops << endl;
		cout << "  Average Hops per Packet: " << fixed << setprecision(2) << avg_hops_per_packet << endl;
		cout << "  Packets per Task: 3 (request, response, result)" << endl;
		cout << "  Total Network Traversals: " << authorGlobalFlitPass << " flits" << endl;
	}

	// Print layer completion times
	cout << "[DEBUG-MAIN-7] Printing layer completion times" << endl;
	if (!llmMacnet->layer_latency.empty()) {
		cout << "\nLayer completion times:" << endl;
		for (size_t i = 0; i < llmMacnet->layer_latency.size(); i++) {
			cout << "Layer " << i << ": " << llmMacnet->layer_latency[i] << " cycles" << endl;
		}
	}

	// MAC utilization statistics
	cout << "[DEBUG-MAIN-8] Computing MAC utilization statistics" << endl;
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
	cout << "[DEBUG-MAIN-9] MAC status summary completed" << endl;

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

	// Q computation is now handled inside LLMMACnet during initialization
	// The matrices are loaded and Q is computed in llmInitializeMatrices()

	// Network statistics (similar to original main)
	cout << "[DEBUG-MAIN-10] Starting network statistics collection" << endl;
	long long tempRouterNetWholeFlipCount = 0;
	long long tempRouterNetWholeFlipCount_fix35 = 0;
	long long tempauthorWeightCollsionInRouterCountSum = 0;
	long long tempauthorWeightCollsionInNICountSum = 0;
	long long mainauthorRouterZeroBTHopTotalCount = 0;
	long long reqRouterFlip = 0;
	long long respRouterFlip = 0;
	long long resRouterFlip = 0;
	long long reqRouterHop = 0;
	long long respRouterHop = 0;
	long long resRouterHop = 0;
	for (int i = 0; i < TOT_NUM; i++) {
		for (int j = 0; j < 5; j++) {
			tempRouterNetWholeFlipCount +=
				vcNetwork->router_list[i]->in_port_list[j]->totalauthorInportFlipping;
			tempRouterNetWholeFlipCount_fix35 +=
				vcNetwork->router_list[i]->in_port_list[j]->totalauthorInportFixFlipping;
			tempauthorWeightCollsionInRouterCountSum +=
				vcNetwork->router_list[i]->in_port_list[j]->authorweightCollsionCountInportCount;
			mainauthorRouterZeroBTHopTotalCount  = mainauthorRouterZeroBTHopTotalCount  +vcNetwork->router_list[i]->in_port_list[j]->zeroBTHopCount;

			reqRouterFlip = reqRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->reqRouterFlipInport;
			respRouterFlip = respRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->respRouterFlipInport;
			resRouterFlip = resRouterFlip
					+ vcNetwork->router_list[i]->in_port_list[j]->resRouterFlipInport;

			reqRouterHop = reqRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->reqRouterHopInport;
			respRouterHop = respRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->respRouterHopInport;
			resRouterHop = resRouterHop
					+ vcNetwork->router_list[i]->in_port_list[j]->resRouterHopInport;

		}
		tempauthorWeightCollsionInNICountSum +=
			vcNetwork->NI_list[i]->in_port->authorweightCollsionCountInportCount;
	}

	// Collect port utilization statistics
	long long total_port_utilization = 0;
	long long total_innet_utilization = 0;
	int max_utilization = 0;
	int max_router_id = -1;
	int min_utilization = INT_MAX;
	int min_router_id = -1;

	struct RouterUtil {
		int router_id;
		int total_util;
		int innet_util;
	};
	vector<RouterUtil> router_utils;

	for (int i = 0; i < TOT_NUM; i++) {
		VCRouter* router = dynamic_cast<VCRouter*>(vcNetwork->router_list[i]);
		if (router != NULL) {
			total_port_utilization += router->port_total_utilization;
			total_innet_utilization += router->port_utilization_innet;

			RouterUtil ru;
			ru.router_id = i;
			ru.total_util = router->port_total_utilization;
			ru.innet_util = router->port_utilization_innet;
			router_utils.push_back(ru);

			if (router->port_total_utilization > max_utilization) {
				max_utilization = router->port_total_utilization;
				max_router_id = i;
			}
			if (router->port_total_utilization < min_utilization) {
				min_utilization = router->port_total_utilization;
				min_router_id = i;
			}
		}
	}

	cout << "[DEBUG-MAIN-11] Network statistics collection completed" << endl;
	cout << "\n=== NETWORK STATISTICS ===" << endl;

	// Basic statistics (always shown)
	cout << "Core Metrics:" << endl;
	cout << "  Total Cycles: " << cycles << endl;
	cout << "  Total Flits Created: " << authorGlobalFlit_id << endl;
	cout << "  Total Hop Count (Router+NI): " << authorGlobalFlitPass << endl;
	cout << "  Router Hop Count: " << authorGlobalRouterHopCount << endl;
	cout<<" mainauthorRouterZeroBTHopTotalCount  " <<mainauthorRouterZeroBTHopTotalCount <<endl;
	cout << "  NI Hop Count: " << authorGlobalNIHopCount << endl;
	cout << "  Total Bit Flips (Router-only): " << tempRouterNetWholeFlipCount << endl;

	// Calculate per-flit averages
	float avg_hops_per_flit = 0.0;
	float avg_router_hops_per_flit = 0.0;
	float avg_ni_hops_per_flit = 0.0;
	float avg_flips_per_flit = 0.0;
	float avg_flips_per_router_hop = 0.0;
	if (authorGlobalFlit_id > 0) {
		avg_hops_per_flit = (float)authorGlobalFlitPass / authorGlobalFlit_id;
		avg_router_hops_per_flit = (float)authorGlobalRouterHopCount / authorGlobalFlit_id;
		avg_ni_hops_per_flit = (float)authorGlobalNIHopCount / authorGlobalFlit_id;
		avg_flips_per_flit = (float)tempRouterNetWholeFlipCount / authorGlobalFlit_id;
	}
	if (authorGlobalRouterHopCount > 0) {
		avg_flips_per_router_hop = (float)tempRouterNetWholeFlipCount / authorGlobalRouterHopCount;
	}

	// Message type-specific bit flip statistics
	cout << " reqRouterFlip " << reqRouterFlip
		 << " respRouterFlip " << respRouterFlip
		 << " resRouterFlip " << resRouterFlip << endl;

	// Message type-specific hop count statistics
	cout << " reqRouterHop " << reqRouterHop
		 << " respRouterHop " << respRouterHop
		 << " resRouterHop " << resRouterHop << endl;
	cout << "  Average Hops per Flit (total): " << fixed << setprecision(2) << avg_hops_per_flit << endl;
	cout << "  Average Router Hops per Flit: " << fixed << setprecision(2) << avg_router_hops_per_flit << endl;
	cout << "  Average NI Hops per Flit: " << fixed << setprecision(2) << avg_ni_hops_per_flit << endl;
	cout << "  Average Bit Flips per Flit（totalhops）: " << fixed << setprecision(2) << avg_flips_per_flit << endl;
	cout << "  Average Bit Flips per Router Hop: " << fixed << setprecision(2) << avg_flips_per_router_hop << endl;
	cout << "  Average Bit Flips per respRouter Hop: " << fixed << setprecision(2) <<respRouterFlip/respRouterHop  << endl;

	// Performance metrics
	if (cycles > 0) {
		double efficiency = (double)finished_macs / llmMacnet->macNum * 100.0;

		cout << "\nPerformance Metrics:" << endl;
		cout << "MAC Efficiency: " << fixed << setprecision(2) << efficiency << "%" << endl;
	}

	// Port Utilization Statistics
	cout << "\n=== PORT UTILIZATION STATISTICS ===" << endl;
	cout << "Total Port Transmissions:" << endl;
	cout << "  All ports (including NI): " << total_port_utilization << " flits" << endl;
	cout << "  Network-internal ports only: " << total_innet_utilization << " flits" << endl;

	if (cycles > 0) {
		double avg_util_per_cycle = (double)total_port_utilization / (TOT_NUM * cycles);
		double avg_innet_per_cycle = (double)total_innet_utilization / (TOT_NUM * cycles);
		cout << "\nAverage Utilization:" << endl;
		cout << "  Per router per cycle: " << fixed << setprecision(4) << avg_util_per_cycle << " flits" << endl;
		cout << "  Network-internal per cycle: " << fixed << setprecision(4) << avg_innet_per_cycle << " flits" << endl;

		// Calculate percentage utilization (assuming 5 ports per router, 1 flit/cycle max per port)
		double total_capacity = TOT_NUM * 5.0 * cycles;  // Total port-cycles available
		double util_percentage = (double)total_port_utilization / total_capacity * 100.0;
		cout << "  Overall network utilization: " << fixed << setprecision(2) << util_percentage << "%" << endl;
	}

	cout << "\nRouter Utilization Range:" << endl;
	if (max_router_id >= 0) {
		int max_x = max_router_id / X_NUM;
		int max_y = max_router_id % X_NUM;
		cout << "  Max: Router " << max_router_id << " (" << max_x << "," << max_y
		     << ") - " << max_utilization << " flits transmitted" << endl;
	}
	if (min_router_id >= 0) {
		int min_x = min_router_id / X_NUM;
		int min_y = min_router_id % X_NUM;
		cout << "  Min: Router " << min_router_id << " (" << min_x << "," << min_y
		     << ") - " << min_utilization << " flits transmitted" << endl;
	}

	// Sort and print top 10 busiest routers
	sort(router_utils.begin(), router_utils.end(),
	     [](const RouterUtil& a, const RouterUtil& b) {
	         return a.total_util > b.total_util;
	     });

	cout << "\nTop 10 Busiest Routers:" << endl;
	cout << "  Rank  Router ID  Position    Total Flits  InNet Flits  Util%" << endl;
	cout << "  ----  ---------  ----------  -----------  -----------  -----" << endl;
	int print_count = min(10, (int)router_utils.size());
	for (int i = 0; i < print_count; i++) {
		int rid = router_utils[i].router_id;
		int rx = rid / X_NUM;
		int ry = rid % X_NUM;
		double util_pct = (cycles > 0) ? (double)router_utils[i].total_util / (5.0 * cycles) * 100.0 : 0.0;
		cout << "  " << setw(4) << (i+1)
		     << "  " << setw(9) << rid
		     << "  (" << rx << "," << ry << ")     "
		     << setw(11) << router_utils[i].total_util
		     << "  " << setw(11) << router_utils[i].innet_util
		     << "  " << fixed << setprecision(1) << setw(5) << util_pct << endl;
	}

	// Check if MC locations are hotspots
	cout << "\nMemory Controller (MC) Router Utilization:" << endl;

	// Define MC locations based on NoC size
	int mc_locations[AUTHORMEMCount];

#if defined NOCSIZEMC2_4X4
	// 4x4: 2 MCs at (2,1) and (2,3)
	mc_locations[0] = 9;   // (2,1)
	mc_locations[1] = 11;  // (2,3)
#elif defined NOCSIZEMC8_8X8
	// 8x8: 8 MCs at tile centers
	mc_locations[0] = 17;  // (2,1)
	mc_locations[1] = 19;  // (2,3)
	mc_locations[2] = 21;  // (2,5)
	mc_locations[3] = 23;  // (2,7)
	mc_locations[4] = 49;  // (6,1)
	mc_locations[5] = 51;  // (6,3)
	mc_locations[6] = 53;  // (6,5)
	mc_locations[7] = 55;  // (6,7)
#elif defined NOCSIZEMC32_16X16
	// 16x16: 32 MCs - pattern: every 4x4 tile has 2 MCs at (tile_base_x+2, tile_base_y+1) and (tile_base_x+2, tile_base_y+3)
	for (int tile_row = 0; tile_row < 4; tile_row++) {
		for (int tile_col = 0; tile_col < 4; tile_col++) {
			int base_x = tile_row * 4 + 2;
			int base_y = tile_col * 4;
			int idx = (tile_row * 4 + tile_col) * 2;
			mc_locations[idx] = base_x * X_NUM + base_y + 1;     // Left MC in tile
			mc_locations[idx + 1] = base_x * X_NUM + base_y + 3; // Right MC in tile
		}
	}
#elif defined NOCSIZEMC128_32X32
	// 32x32: 128 MCs - pattern: every 4x4 tile has 2 MCs
	for (int tile_row = 0; tile_row < 8; tile_row++) {
		for (int tile_col = 0; tile_col < 8; tile_col++) {
			int base_x = tile_row * 4 + 2;
			int base_y = tile_col * 4;
			int idx = (tile_row * 8 + tile_col) * 2;
			mc_locations[idx] = base_x * X_NUM + base_y + 1;     // Left MC in tile
			mc_locations[idx + 1] = base_x * X_NUM + base_y + 3; // Right MC in tile
		}
	}
#endif

	for (int i = 0; i < AUTHORMEMCount; i++) {
		int mc_id = mc_locations[i];
		int mc_x = mc_id / X_NUM;
		int mc_y = mc_id % X_NUM;
		VCRouter* router = dynamic_cast<VCRouter*>(vcNetwork->router_list[mc_id]);
		if (router != NULL) {
			double util_pct = (cycles > 0) ? (double)router->port_total_utilization / (5.0 * cycles) * 100.0 : 0.0;
			cout << "  MC at Router " << mc_id << " (" << mc_x << "," << mc_y << "): "
			     << router->port_total_utilization << " flits ("
			     << fixed << setprecision(1) << util_pct << "%)" << endl;
		}
	}

	cout << "\n!!LLM ATTENTION SIMULATION END!!" << endl;

#ifdef AUTHORLLMSwitchON
#ifdef AUTHORSeperatedOrdering_reArrangeInput
	// Separated Ordering mode: Skip output matrix printing
	// This mode breaks input-query pairing for bit flip optimization
	// Output matrix requires extra ID tracking to reconstruct correct results
	cout << "\n==================== ATTENTION OUTPUT TABLE ====================" << endl;
	cout << "NOTE: Separated Ordering mode enabled." << endl;
	cout << "Output matrix printing skipped - requires extra ID for result reconstruction." << endl;
	cout << "Only latency and bit flip (BT) statistics are meaningful in this mode." << endl;
	cout << "==================== END OUTPUT MATRIX ====================" << endl;
#else
	// Print output matrix (attention_output_table)
	cout << "\n==================== ATTENTION OUTPUT TABLE ====================" << endl;
	cout << "Matrix dimensions: " << llmMacnet->input_sequence_length << " x " << llmMacnet->query_output_dim << endl;
	cout << "\nFirst 5x5 elements of attention_output_table:" << endl;
	cout << "-----------------------------------------------" << endl;

	int rows_to_print = min(5, llmMacnet->input_sequence_length);
	int cols_to_print = min(5, llmMacnet->query_output_dim);

	// Print column headers
	cout << "      ";
	for (int j = 0; j < cols_to_print; j++) {
		cout << "    [" << j << "]     ";
	}
	cout << endl;

	// Print matrix values
	for (int i = 0; i < rows_to_print; i++) {
		cout << "[" << i << "]  ";
		for (int j = 0; j < cols_to_print; j++) {
			cout << fixed << setprecision(6) << setw(12) << llmMacnet->attention_output_table[i][j] << " ";
		}
		if (llmMacnet->query_output_dim > 5) {
			cout << "  ...";
		}
		cout << endl;
	}

	if (llmMacnet->input_sequence_length > 5) {
		cout << "...   (showing first 5 of " << llmMacnet->input_sequence_length << " rows)" << endl;
	}

	// Print last 5x5 elements (bottom-right corner)
	cout << "\nLast 5x5 elements of attention_output_table:" << endl;
	cout << "-----------------------------------------------" << endl;

	int start_row = max(0, llmMacnet->input_sequence_length - 5);
	int start_col = max(0, llmMacnet->query_output_dim - 5);
	int end_row = llmMacnet->input_sequence_length;
	int end_col = llmMacnet->query_output_dim;

	// Print column headers for last columns
	cout << "      ";
	for (int j = start_col; j < end_col; j++) {
		cout << "   [" << j << "]    ";
	}
	cout << endl;

	// Print matrix values for bottom-right corner
	for (int i = start_row; i < end_row; i++) {
		cout << "[" << i << "]  ";
		if (i < 10) cout << " ";  // Extra space for single digit row numbers
		for (int j = start_col; j < end_col; j++) {
			cout << fixed << setprecision(6) << setw(12) << llmMacnet->attention_output_table[i][j] << " ";
		}
		cout << endl;
	}

	// Calculate and print some statistics about the output matrix
	cout << "\nOutput Matrix Statistics:" << endl;
	cout << "-------------------------" << endl;

	float min_val = 1e9, max_val = -1e9, sum = 0;
	int zero_count = 0;

	for (int i = 0; i < llmMacnet->input_sequence_length; i++) {
		for (int j = 0; j < llmMacnet->query_output_dim; j++) {
			float val = llmMacnet->attention_output_table[i][j];
			min_val = min(min_val, val);
			max_val = max(max_val, val);
			sum += val;
			if (abs(val) < 1e-6) zero_count++;
		}
	}

	int total_elements = llmMacnet->input_sequence_length * llmMacnet->query_output_dim;
	float avg = sum / total_elements;

	cout << "  Min value: " << fixed << setprecision(6) << min_val << endl;
	cout << "  Max value: " << fixed << setprecision(6) << max_val << endl;
	cout << "  Average value: " << fixed << setprecision(6) << avg << endl;
	cout << "  Zero elements: " << zero_count << " / " << total_elements
	     << " (" << fixed << setprecision(2) << (100.0 * zero_count / total_elements) << "%)" << endl;

	// Load and compare with golden reference
	cout << "\n==================== GOLDEN REFERENCE COMPARISON ====================" << endl;

	// Select golden reference file based on LLM_TOKEN_SIZE
	string golden_file;
	#if LLM_TOKEN_SIZE == 1
		// Test Case 1: 8-sequence version
		golden_file = "src/Input/llminput/Q_result_python.txt";
		cout << "Using golden reference: Q_result_python.txt (8 sequences)" << endl;
	#elif LLM_TOKEN_SIZE == 2
		// Test Case 2: 128-sequence version
		golden_file = "src/Input/llminput/Q_result_python_128seq.txt";
		cout << "Using golden reference: Q_result_python_128seq.txt (128 sequences)" << endl;
	#else
		#error "Unknown LLM_TOKEN_SIZE value"
	#endif

	ifstream golden_in(golden_file.c_str());

	if (golden_in.is_open()) {
		// Allocate golden reference matrix
		float** golden_matrix = new float*[llmMacnet->input_sequence_length];
		for (int i = 0; i < llmMacnet->input_sequence_length; i++) {
			golden_matrix[i] = new float[llmMacnet->query_output_dim];
		}

		// Read golden reference
		bool read_success = true;
		for (int i = 0; i < llmMacnet->input_sequence_length; i++) {
			for (int j = 0; j < llmMacnet->query_output_dim; j++) {
				if (!(golden_in >> golden_matrix[i][j])) {
					cout << "Error: Failed to read golden reference at [" << i << "][" << j << "]" << endl;
					read_success = false;
					break;
				}
			}
			if (!read_success) break;
		}
		golden_in.close();

		if (read_success) {
			// Calculate error metrics
			float mae = 0.0f;  // Mean Absolute Error
			float mse = 0.0f;  // Mean Squared Error
			float max_error = 0.0f;
			int max_error_i = 0, max_error_j = 0;
			float golden_min = 1e9, golden_max = -1e9, golden_sum = 0;

			for (int i = 0; i < llmMacnet->input_sequence_length; i++) {
				for (int j = 0; j < llmMacnet->query_output_dim; j++) {
					float actual = llmMacnet->attention_output_table[i][j];
					float expected = golden_matrix[i][j];
					float error = abs(actual - expected);

					mae += error;
					mse += error * error;

					if (error > max_error) {
						max_error = error;
						max_error_i = i;
						max_error_j = j;
					}

					golden_min = min(golden_min, expected);
					golden_max = max(golden_max, expected);
					golden_sum += expected;
				}
			}

			mae /= total_elements;
			mse /= total_elements;
			float rmse = sqrt(mse);
			float golden_avg = golden_sum / total_elements;

			// Print golden reference statistics
			cout << "\nGolden Reference Statistics:" << endl;
			cout << "  Min value: " << fixed << setprecision(6) << golden_min << endl;
			cout << "  Max value: " << fixed << setprecision(6) << golden_max << endl;
			cout << "  Average value: " << fixed << setprecision(6) << golden_avg << endl;


		// Print first 5x5 elements of golden reference
		cout << "\nGolden Reference - First 5x5 elements:" << endl;
		cout << "-----------------------------------------------" << endl;
		cout << "      ";
		for (int j = 0; j < min(5, llmMacnet->query_output_dim); j++) {
			cout << "    [" << j << "]     ";
		}
		cout << endl;
		for (int i = 0; i < min(5, llmMacnet->input_sequence_length); i++) {
			cout << "[" << i << "]  ";
			if (i < 10) cout << " ";
			for (int j = 0; j < min(5, llmMacnet->query_output_dim); j++) {
				cout << fixed << setprecision(6) << setw(12) << golden_matrix[i][j] << " ";
			}
			cout << endl;
		}

		// Print element-by-element comparison for [0][0:5]
		cout << "\nElement-by-Element Comparison [Row 0, Cols 0-4]:" << endl;
		cout << "Col    Actual         Golden         Diff          Error%" << endl;
		cout << "---    ------         ------         ----          ------" << endl;
		for (int j = 0; j < min(5, llmMacnet->query_output_dim); j++) {
			float actual = llmMacnet->attention_output_table[0][j];
			float golden = golden_matrix[0][j];
			float diff = actual - golden;
			float error_pct = (golden != 0) ? (diff / golden) * 100 : 0;
			cout << " " << j << "   "
			     << fixed << setprecision(6) << setw(12) << actual << "  "
			     << setw(12) << golden << "  "
			     << setw(12) << diff << "  "
			     << setw(8) << setprecision(2) << error_pct << "%" << endl;
		}
			// Print error metrics
			cout << "\nError Metrics (Actual vs Golden):" << endl;
			cout << "  Mean Absolute Error (MAE): " << scientific << setprecision(6) << mae << endl;
			cout << "  Root Mean Squared Error (RMSE): " << scientific << setprecision(6) << rmse << endl;
			cout << "  Max Absolute Error: " << scientific << setprecision(6) << max_error << endl;
			cout << "    at position [" << max_error_i << "][" << max_error_j << "]" << endl;
			cout << "    Actual: " << fixed << setprecision(6) << llmMacnet->attention_output_table[max_error_i][max_error_j] << endl;
			cout << "    Golden: " << fixed << setprecision(6) << golden_matrix[max_error_i][max_error_j] << endl;

			// Relative error
			float relative_mae = (golden_avg != 0) ? (mae / abs(golden_avg)) * 100 : 0;
			cout << "  Relative MAE: " << fixed << setprecision(2) << relative_mae << "%" << endl;

			// Correctness assessment
			cout << "\nCorrectness Assessment:" << endl;
			if (mae < 1e-5) {
				cout << "  ✓ EXCELLENT: Results match golden reference within floating-point precision" << endl;
			} else if (mae < 1e-3) {
				cout << "  ✓ GOOD: Results are very close to golden reference" << endl;
			} else if (mae < 0.01) {
				cout << "  ⚠ ACCEPTABLE: Results have small differences from golden reference" << endl;
			} else {
				cout << "  ✗ WARNING: Results differ significantly from golden reference" << endl;
			}
		}

		// Cleanup
		for (int i = 0; i < llmMacnet->input_sequence_length; i++) {
			delete[] golden_matrix[i];
		}
		delete[] golden_matrix;
	} else {
		cout << "Warning: Could not open golden reference file: " << golden_file << endl;
	}

	cout << "==================== END OUTPUT MATRIX ====================" << endl;
#endif // AUTHORSeperatedOrdering_reArrangeInput
#endif // AUTHORLLMSwitchON

	// Calculate and display execution time
	end = clock();
	double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	cout << "Total execution time: " << fixed << setprecision(3) << elapsed_time << " seconds" << endl;
	// Cleanup
	// cout << "[DEBUG-MAIN-17] Starting cleanup" << endl;
	delete llmMacnet;
	// cout << "[DEBUG-MAIN-18] Deleted llmMacnet" << endl;
	delete vcNetwork;
	// cout << "[DEBUG-MAIN-19] Deleted vcNetwork" << endl;
	// cout << "[DEBUG-MAIN-20] Cleanup completed, returning from main()" << endl;

	return 0;
}
#endif
