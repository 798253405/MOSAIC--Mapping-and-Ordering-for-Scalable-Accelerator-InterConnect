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


	cout << " below is the final result "<<endl;
	for (float j: macnet->output_table[0])
	{
		cout << j << ' ';
	}



	cout << "Cycles: " << cycles << endl;

	cout << "Packet id: " << packet_id << endl;

#ifdef SoCC_Countlatency
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
#endif
#ifdef SoCC_Countlatency
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
#endif
	long long tempRouterNetWholeFlipCount = 0;
	long long tempRouterNetWholeFlipCount_fix35 = 0;
	for (int i = 0; i < TOT_NUM; i++) {
		for (int j = 0; j < 5; j++) {
			tempRouterNetWholeFlipCount =
					tempRouterNetWholeFlipCount
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFlipping;
			yzWeightCollsionInRouterCountSum =
			tempRouterNetWholeFlipCount_fix35 =
					tempRouterNetWholeFlipCount_fix35
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFixFlipping;

			yzWeightCollsionInRouterCountSum
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

	// Main simulation loop
	for (; cycles < simulate_cycles; cycles++) {
		// Check and manage LLM attention tasks
		llmMacnet->llmCheckStatus();

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

	// Print sample attention outputs
	cout << "Sample attention output matrix (first 10x10):" << endl;
	for (int i = 0; i < min(10, llmMacnet->matrix_size); i++) {
		for (int j = 0; j < min(10, llmMacnet->matrix_size); j++) {
			cout << fixed << setprecision(4) << llmMacnet->attention_output_table[i][j] << " ";
		}
		cout << endl;
	}

	cout << "\nSimulation Statistics:" << endl;
	cout << "Total cycles: " << cycles << endl;
	cout << "Total packets sent: " << packet_id << endl;
	cout << "Ready flag: " << llmMacnet->ready_flag << endl;

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
		cout << "Attention output saved to output/llm_attention_output.txt" << endl;
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

	cout << "\nNetwork Statistics:" << endl;
	cout << "YZGlobalFlit_id: " << YZGlobalFlit_id << endl;
	cout << "YZGlobalFlitPass: " << YZGlobalFlitPass << endl;
	cout << "YZGlobalRespFlitPass: " << YZGlobalRespFlitPass << endl;
	cout << "Router collision count: " << tempyzWeightCollsionInRouterCountSum << endl;
	cout << "NI collision count: " << tempyzWeightCollsionInNICountSum << endl;
	cout << "Total router flipping: " << tempRouterNetWholeFlipCount << endl;
	cout << "Fixed router flipping: " << tempRouterNetWholeFlipCount_fix35 << endl;

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
