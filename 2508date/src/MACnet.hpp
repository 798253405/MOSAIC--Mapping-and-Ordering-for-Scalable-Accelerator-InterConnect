
/**
 * @file MACnet.hpp
 * @brief CNN MAC
 *
 * CNNMAC，MACCNN。
 *
 * ：
 * - MACnet: ，MAC
 *   - MAC_list: MAC
 *   - VCNetwork: NoC
 *   - Model: CNN
 *
 * ：
 * - input_table:
 * - output_table:
 * - weight_table:
 * - pe_allocation: PE
 *
 * @see MAC.hpp - MAC
 * @see VCNetwork.hpp - NoC
 * @see Model.hpp - CNN
 */

#ifndef MACNET_HPP_
#define MACNET_HPP_
#include <cmath>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <algorithm>    //std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include "Model.hpp"
#include "NoC/VCNetwork.hpp"



#include <map>        // std::map //20250826

#include "IEEE754.hpp"

using namespace std;

extern long long packet_id;

extern unsigned int cycles;

extern vector<vector<int>> DNN_latency;
extern double samplingWindowDelay[TOT_NUM];
extern int samplingAccumlatedCounter;
// NoC
class VCNetwork;

class MAC;

class MACnet
{
public:


	MACnet (int mac_num, int t_pe_x, int t_pe_y, Model *m, VCNetwork* t_Network);
	std::vector<MAC*> MAC_list;
	VCNetwork* vcNetwork;

	void create_input();
	vector<vector<float>> weight_table;
	vector<vector<float>> input_table;
	vector<vector<float>> output_table;

	deque< deque< int > > mapping_table;
	void xmapping(int neuronnum);
	void ymapping(int neuronnum);
	void rmapping(int neuronnum);
	void authorrmapping(int neuronnum);
	int authorDistancemapping(int neuronnum);
	int authorFuncSAMOSSampleMapping(int neuronnum);
	int authorPostSimTravelMapping(int neuronnum);



	int breakDownTime[TOT_NUM][4][11];//all nodes ->sum, travel1 travel2 create3 travel3 -> 1 average + 10 recorded values
	int lastLayerPacketID;
	int mappingagain;
	int authorLastSeenPid = 0;

	void runOneStep();
	void checkStatus();









	Model* cnnmodel;
	int macNum;
	int pe_x;
	int pe_y;
	int used_pe;

	int current_layerSeq; // current layer
	int n_layer; // total layer

	int in_ch; // for input channel
	int in_x;
	int in_y;

	// for new pooling
	int no_x;
	int no_y;
	int nw_x;
	int nw_y;
	int no_ch; // next o_ch in pooling layer
	int npad; // padding
	int nstride; // stride
	//vector<vector<float>> pooling_table;


	int w_ch; // for filter
	int w_x;
	int w_y;
	int st_w;
	int pad;
	int stride;

	int o_ch; // for output
	int o_x;
	int o_y;

	int o_fnReluOrPool; // for function

	int readyflag;

	// for print
	vector<int> Layer_latency;


	vector<float>  totalNetTaskInput;





	int executedTask;
	~MACnet ();
};



#endif /* MACNET_HPP_ */
