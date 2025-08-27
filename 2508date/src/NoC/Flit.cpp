/*
 * flit.cpp
 *
 *  Created on: 2019年8月15日
 *      Author: wr
 */

#include "Flit.hpp"
#include "../parameters.hpp"

const int Flit::length = FLIT_LENGTH;

Flit::Flit(int t_seqidInFlit, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid){
  seqid = t_seqidInFlit;
  type = t_type;
  vnet = t_vnet;
  vc = t_vc;
  out_port = -1;
  packet = t_packet;
  sched_time = t_cycles;
  signalid=t_pid;
  yzFlitPayload.clear();
}




