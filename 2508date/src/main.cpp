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

using namespace std;

// NoC
class VCNetwork;

long long  packet_id;
long long  YZGlobalFlit_id;
long long  YZGlobalFlitPass = 0;  // Total hop count (router + NI)
long long YZGlobalRouterHopCount = 0;  // Router-only hop count
long long YZGlobalNIHopCount = 0;  // NI-only hop count
long long YZGlobalRespFlitPass = 0;
long long yzFlitCollsionCountSum = 0;

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
int main(int arg_num, char *arg_vet[]) {
	clock_t start, end;
	/// clock for start
	    start = clock();
	// Print current date and time
	auto now = std::chrono::system_clock::now();
	std::time_t now_time = std::chrono::system_clock::to_time_t(now);
	cout << "=== Test Started: " << std::ctime(&now_time);
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


	// Network statistics (similar to original main)

	long long tempyzWeightCollsionInRouterCountSum = 0;
	long long tempyzWeightCollsionInNICountSum = 0;
	long long mainyzRouterZeroBTHopTotalCount = 0;
	long long yzWeightCollsionInRouterCountSum = 0;
	long long yzWeightCollsionInNICountSum = 0;
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
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFlipping;
			tempRouterNetWholeFlipCount_fix35 =
					tempRouterNetWholeFlipCount_fix35
							+ vcNetwork->router_list[i]->in_port_list[j]->totalyzInportFixFlipping;

			yzWeightCollsionInRouterCountSum = yzWeightCollsionInRouterCountSum
					+ vcNetwork->router_list[i]->in_port_list[j]->yzweightCollsionCountInportCount;
			mainyzRouterZeroBTHopTotalCount  = mainyzRouterZeroBTHopTotalCount  +vcNetwork->router_list[i]->in_port_list[j]->zeroBTHopCount;
			
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
		yzWeightCollsionInNICountSum = yzWeightCollsionInNICountSum
				+ vcNetwork->NI_list[i]->in_port-> yzweightCollsionCountInportCount;
	}
	cout << " YZGlobalFlit_id " << YZGlobalFlit_id 
			<< " YZGlobalFlitPass(total) " << YZGlobalFlitPass 
			<< " YZGlobalRouterHopCount " << YZGlobalRouterHopCount
			<< " YZGlobalNIHopCount " << YZGlobalNIHopCount
			<< " YZGlobalRespFlitPass " << YZGlobalRespFlitPass 
			<< " yzWeightCollsionInRouterCountSum "
			<< yzWeightCollsionInRouterCountSum
			<< " yzWeightCollsionInNICountSum "
			<< yzWeightCollsionInNICountSum
			<< " yzFlitCollsionCountSum "
			<< yzFlitCollsionCountSum  << endl;
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
	// Use YZGlobalRouterHopCount for router-only statistics
	double avg_bit_trans_float = YZGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount/YZGlobalRouterHopCount : 0;
	double avg_bit_trans_fixed = YZGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount_fix35/YZGlobalRouterHopCount : 0;
	double avg_hops_per_flit = YZGlobalFlit_id > 0 ? (double)YZGlobalFlitPass/YZGlobalFlit_id : 0;
	double avg_flips_per_flit_total = YZGlobalFlit_id > 0 ? (double)tempRouterNetWholeFlipCount/YZGlobalFlit_id : 0;
	double avg_flips_per_flit_per_router_hop = YZGlobalRouterHopCount > 0 ? (double)tempRouterNetWholeFlipCount/YZGlobalRouterHopCount : 0;
	
	cout << "BATCH_STATS: "
		<< "total_cycles=" << cycles << " "
		<< "packetid=" << packet_id << " "
		<< "YZGlobalFlit_id=" << YZGlobalFlit_id << " "
		<< "YZGlobalFlitPass=" << YZGlobalFlitPass << " "
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
	cout << "  Total Flits Created: " << YZGlobalFlit_id << endl;
	cout << "  Total Hop Count (Router+NI): " << YZGlobalFlitPass << endl;
	cout << "  Router Hop Count: " << YZGlobalRouterHopCount << endl;
	cout<<" mainyzRouterZeroBTHopTotalCount  " <<mainyzRouterZeroBTHopTotalCount <<endl;
	cout << "  NI Hop Count: " << YZGlobalNIHopCount << endl;
	cout << "  Total Bit Flips (Router-only): " << tempRouterNetWholeFlipCount << endl;
	// Calculate per-flit averages

	float avg_router_hops_per_flit = 0.0;
	float avg_ni_hops_per_flit = 0.0;
	float avg_flips_per_flit = 0.0;
	float avg_flips_per_router_hop = 0.0;
	if (YZGlobalFlit_id > 0) {
		avg_hops_per_flit = (float)YZGlobalFlitPass / YZGlobalFlit_id;
		avg_router_hops_per_flit = (float)YZGlobalRouterHopCount / YZGlobalFlit_id;
		avg_ni_hops_per_flit = (float)YZGlobalNIHopCount / YZGlobalFlit_id;
		avg_flips_per_flit = (float)tempRouterNetWholeFlipCount / YZGlobalFlit_id;
	}
	if (YZGlobalRouterHopCount > 0) {
		avg_flips_per_router_hop = (float)tempRouterNetWholeFlipCount / YZGlobalRouterHopCount;
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
	    std::cout << "运行时间: " << elapsed_time << " 秒" << std::endl;
	delete macnet;
	delete cnnmodel;
	return 0;
}
