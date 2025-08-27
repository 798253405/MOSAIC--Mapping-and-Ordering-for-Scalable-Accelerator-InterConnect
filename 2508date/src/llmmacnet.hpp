/*
 * LLMMAC.hpp
 *
 *  Created on: Dec 19, 2024
 *      Author: LLM Version
 */

#ifndef LLMMAC_HPP_
#define LLMMAC_HPP_

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
#include "llmmacnet.hpp"

#if defined MemNode2_4x4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // 4*4

#elif defined MemNode4_4X4
#define MEM_NODES 4
	// 4×4：TL(1,1),BL(3,1), TR(1,3),  BR(3,3)
	const int dest_list[] = {5, 13, 7, 15}; // 4*4

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
	const int dest_list[] = {264, 776, 280, 792};
#endif

using namespace std;

extern int packet_id;
extern unsigned int cycles;
extern vector<vector<int>> DNN_latency;
extern double samplingWindowDelay[TOT_NUM];

class LLMMACnet;
class Packet;

class LLMMAC
{
	public:
		/** @brief LLMMAC - LLM-specific MAC unit
		 */
		LLMMAC(int t_id, LLMMACnet* t_net, int t_NI_id);

		LLMMACnet* net;
		int selfMACid;
		int fn;
		int pecycle;
		int selfstatus;
		int request;
		int tmp_requestID;

		int send;
		int NI_id;

		// LLM-specific data structures
		deque<float> query_data;     // Query vectors
		deque<float> key_data;       // Key vectors
		deque<float> value_data;     // Value vectors
		deque<float> input_buffer;   // Input buffer for received data

		// LLM attention parameters
		int tile_x_start, tile_y_start;  // Starting position of this tile
		int tile_size;                   // Size of tile (16x16)
		int time_slice;                  // Current time slice (0-3)
		int dest_mem_id;                 // Memory node ID

		float attention_output;          // Computed attention output

		deque<int> routing_table;

		LLMMAC* nextLLMMAC;

		// Core functions
		bool llmInject(int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src);
		void llmReceive(Message* re_msg);
		void llmRunOneStep();

		// LLM-specific attention computation
		void llmComputeAttention();
		void llmComputeQueryKeyDot();
		void llmApplySoftmax();
		void llmComputeValueWeightedSum();

		// State management
		bool llmIsWaitingForData();
		void llmResetForNextTask();

		~LLMMAC();
};

#endif /* LLMMAC_HPP_ */
