/*
 * MAC.hpp
 *
 *  Created on: Dec 19, 2022
 *      Author: wenyao
 */

#ifndef MAC_HPP_
#define MAC_HPP_

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <deque>
#include <cmath>
#include <cassert>
#include "parameters.hpp"
#include "NoC/Packet.hpp"
#include "NoC/NI.hpp"
#include "MACnet.hpp"




#if defined MemNode2_4x4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // 4*4

#elif defined MemNode4_4X4
#define MEM_NODES 4
	// 4×4：TL(1,1),BL(3,1), TR(1,3),  BR(3,3)
	const int dest_list[] = {5, 13, 7, 15}; // 8*8

#elif defined  MemNode4_8X8
	#define MEM_NODES 4
	// 8×8：象限中心 -> (2,2),(6,2),(2,6),(6,6)
	const int dest_list[] = {18, 50, 22, 54}; // 8*8
#elif defined MemNode4_16X16
	#define MEM_NODES 4
	 // 16×16：象限中心 -> (4,4),(12,4),(4,12),(12,12)   // 顺序：TL, BL, TR, BR
	 // 节点ID = xid*16 + yid
	 const int dest_list[] = {68, 196, 76, 204}; // 16*16

#elif defined MemNode4_32X32
	#define MEM_NODES 4
	// 32×32：象限中心 -> (8,8),24,8),(8,24),((24,24)
	const int dest_list[] = {264, 776, 280,  792};
#endif
//

using namespace std;

extern int packet_id;

extern unsigned int cycles;

extern vector<vector<int>> DNN_latency;
extern double samplingWindowDelay[TOT_NUM];

class MACnet;
class Packet;









class MAC
{
	public:
  /** @brief MAC
   *
   */
	MAC (int t_id, MACnet* t_net, int t_NI_id);
	MACnet* net;
	int selfMACid;
	int fn;
	int pecycle;
	int selfstatus;
	int request;
	int tmp_requestID;

	int send;
	int NI_id;
	deque<float> weight;
	deque<float> infeature;
	deque<float> inbuffer;
	int ch_size;
	int m_size;
	int dest_mem_id;
	int tmpch;
	int tmpm;
	int m_count;
	float outfeature{}; //from MRL



	deque <int> routing_table;

	// for new pooling
	int npoolflag;
	int n_tmpch;
	deque<int> n_tmpm;


	MAC* nextMAC;

	bool inject (int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src);
	void receive (Message* re_msg);
	void runOneStep();
	void sigmoid(float& x);
	void tanh(float& x);
	void relu(float& x);


	void runLLMOneStep();

	~MAC ();
};




#endif /* MAC_HPP_ */
