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




#if defined MemNode2_4X4
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
