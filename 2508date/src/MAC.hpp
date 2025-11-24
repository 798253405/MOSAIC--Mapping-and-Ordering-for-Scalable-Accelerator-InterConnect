/**
 * @file MAC.hpp
 * @brief CNN MAC
 *
 * CNNMAC (Multiply-Accumulate) 。
 * MACCNN，、。
 *
 * ：
 * - MemNode2_4X4: 2，{9, 11}
 * - MemNode4_4X4: 4，{5, 6, 9, 10}
 * - MemNode8_4X4: 8，
 * - MemNode16_4X4: 16，
 *
 * MAC：
 * - selfstatus: 0(), 1(), 2(), 3()
 * - pecycle: PE
 * - send:
 *
 * ：
 * - weight: ，
 * - infeature:
 * - inbuffer:
 * - outfeature:
 *
 * ：
 * - : weight * input
 * - : 、
 * - : NoC
 *
 * @date 2022-12-19 (original), 2025 (updated)
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




#if defined NOCSIZEMC2_4X4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // (2,1) and (2,3) in 4x4 grid

#elif defined NOCSIZEMC8_8X8
	#define MEM_NODES 8
	// 2x2 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		17, 19,   // Tile(0,0): (2,1), (2,3)
		21, 23,   // Tile(0,1): (2,5), (2,7)
		49, 51,   // Tile(1,0): (6,1), (6,3)
		53, 55    // Tile(1,1): (6,5), (6,7)
	};

#elif defined NOCSIZEMC32_16X16
	#define MEM_NODES 32
	// 4x4 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		// Row 0 tiles (y=2)
		33, 35,   37, 39,   41, 43,   45, 47,
		// Row 1 tiles (y=6)
		97, 99,   101, 103, 105, 107, 109, 111,
		// Row 2 tiles (y=10)
		161, 163, 165, 167, 169, 171, 173, 175,
		// Row 3 tiles (y=14)
		225, 227, 229, 231, 233, 235, 237, 239
	};

#elif defined NOCSIZEMC128_32X32
	#define MEM_NODES 128
	// 8x8 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		// Tile row 0
		65,  67,  69,  71,  73,  75,  77,  79,
		81,  83,  85,  87,  89,  91,  93,  95,
		// Tile row 1
		193, 195, 197, 199, 201, 203, 205, 207,
		209, 211, 213, 215, 217, 219, 221, 223,
		// Tile row 2
		321, 323, 325, 327, 329, 331, 333, 335,
		337, 339, 341, 343, 345, 347, 349, 351,
		// Tile row 3
		449, 451, 453, 455, 457, 459, 461, 463,
		465, 467, 469, 471, 473, 475, 477, 479,
		// Tile row 4
		577, 579, 581, 583, 585, 587, 589, 591,
		593, 595, 597, 599, 601, 603, 605, 607,
		// Tile row 5
		705, 707, 709, 711, 713, 715, 717, 719,
		721, 723, 725, 727, 729, 731, 733, 735,
		// Tile row 6
		833, 835, 837, 839, 841, 843, 845, 847,
		849, 851, 853, 855, 857, 859, 861, 863,
		// Tile row 7
		961, 963, 965, 967, 969, 971, 973, 975,
		977, 979, 981, 983, 985, 987, 989, 991
	};
#endif
//

using namespace std;

extern long long packet_id;

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

	/**
	 * @brief CNNID（request）
	 *
	 * CNNID：
	 * - /
	 * - routing_table
	 * - ：0  -1
	 * - -1MAC，
	 *
	 * LLM：
	 * - CNN: ID = ，
	 * - LLM: ID = pixel_id*4 + subchunk_id，
	 */
	int cnn_current_layer_task_id;  // CNNID（request）

	/**
	 * @brief CNNID（tmp_requestID）
	 *
	 * ：
	 * - : tmp_requestID = request
	 * - : DNN_latency[packet_id + tmp_requestID]
	 * - packet ID
	 */
	int cnn_saved_task_id;  // CNNID（tmp_requestID）

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

	/**
	 * @brief CNN（routing_table）
	 *
	 * ：64，[0,1,2,...,63]
	 * MACID，
	 */
	deque <int> cnn_task_queue;  // CNN（routing_table）

	// for new pooling
	int npoolflag;
	int n_tmpch;
	deque<int> n_tmpm;

#ifdef binaryroutingSwitch
	int lastResponseRouting;  // response packetrouting (1 or 2)
#endif

#ifdef fireAdvance
	// Fire advance：request (CNN)
	int total_tasks;              // （）
	int requests_sent;            // request
	int responses_received;       // response
	int tasks_completed;

	int fire_advance_counter;     // ：responserequest
	bool fire_advance_armed;      // fire advance

	int computing_task_id;        // task ID（-1）
#endif

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
