/**
 * @file MAC.hpp
 * @brief CNN MAC计算单元头文件
 * 
 * 定义了CNN模式下的单个MAC (Multiply-Accumulate) 计算单元。
 * MAC是CNN硬件加速器的基础计算单元，负责执行卷积、池化等操作。
 * 
 * 内存节点配置：
 * - MemNode2_4X4: 2个内存节点，位于{9, 11}
 * - MemNode4_4X4: 4个内存节点，位于{5, 6, 9, 10}
 * - MemNode8_4X4: 8个内存节点，分布在网格中
 * - MemNode16_4X4: 16个内存节点，每个节点都是内存
 * 
 * MAC单元状态：
 * - selfstatus: 0(空闲), 1(请求数据), 2(计算中), 3(发送结果)
 * - pecycle: PE计算周期计数
 * - send: 发送控制标志
 * 
 * 数据缓存：
 * - weight: 权重缓存，支持权重复用
 * - infeature: 输入特征缓存
 * - inbuffer: 输入数据缓冲区
 * - outfeature: 输出特征值
 * 
 * 支持的操作：
 * - 卷积计算: weight * input的累加
 * - 池化操作: 最大池化、平均池化
 * - 数据传输: 通过NoC请求和发送数据
 * 
 * @author wenyao (original), YZ (comments)
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




#if defined DATEMC2_4X4
	#define MEM_NODES 2
	const int dest_list[] = {9, 11}; // (2,1) and (2,3) in 4x4 grid

#elif defined DATEMC8_8X8
	#define MEM_NODES 8
	// 2x2 tiles, each tile has MCs at local (2,1) and (2,3)
	const int dest_list[] = {
		17, 19,   // Tile(0,0): (2,1), (2,3)
		21, 23,   // Tile(0,1): (2,5), (2,7)
		49, 51,   // Tile(1,0): (6,1), (6,3)
		53, 55    // Tile(1,1): (6,5), (6,7)
	};

#elif defined DATEMC32_16X16
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

#elif defined DATEMC128_32X32
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
	 * @brief CNN当前处理的层计算任务ID（原名request）
	 * 
	 * CNN任务ID含义：
	 * - 表示当前层的第几个输出通道/特征图
	 * - 从routing_table队列中取出
	 * - 范围：0 到 该层输出通道数-1
	 * - -1表示MAC空闲，无任务处理
	 * 
	 * 与LLM的区别：
	 * - CNN: 任务ID = 输出通道索引，用于索引权重
	 * - LLM: 任务ID = pixel_id*4 + subchunk_id，用于索引矩阵块
	 */
	int cnn_current_layer_task_id;  // CNN当前处理的层任务ID（原名request）
	
	/**
	 * @brief CNN保存的任务ID副本（原名tmp_requestID）
	 * 
	 * 作用：
	 * - 在发送请求时保存: tmp_requestID = request
	 * - 用于统计延迟: DNN_latency[packet_id + tmp_requestID]
	 * - 用于发送结果包时的packet ID计算
	 */
	int cnn_saved_task_id;  // CNN保存的任务ID副本（原名tmp_requestID）

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
	 * @brief CNN任务队列（原名routing_table）
	 * 
	 * 存储待处理的输出通道索引
	 * 例如：对于有64个输出通道的卷积层，队列包含[0,1,2,...,63]
	 * MAC从队列取出任务ID，计算对应的输出特征图
	 */
	deque <int> cnn_task_queue;  // CNN待处理任务队列（原名routing_table）

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
