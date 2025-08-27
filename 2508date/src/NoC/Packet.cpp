/*
 * Packet.cpp
 *
 *  Created on: 2019年8月15日
 *      Author: wr
 */

#include "Packet.hpp"
#include <iostream>
//type convert: add change type and vnet

Packet::Packet(Message t_message, int router_num_x, int* NI_num){
  this->message = t_message;
  int t_type = message.type; //0-> read request (93 bits); 1-> read response (6+y bits); 2-> write request (93+y bits); 3-> write response (6 bits);
  int elementNumPerPacket = message.msgdata_length; //  element num
  dest_convert(message.destination, router_num_x, NI_num);
  switch (t_type){
    case 0: lengthInBits = 29;// requst address assuming 16 bits //29
            type = 0;
            vnet = 0;
            break;
    case 1: lengthInBits =  elementNumPerPacket*bitsPerElement ; //length = n float (fp16)
            type = 0;	// changed case 1 and 2
            vnet = 0;
            break;
    case 2: lengthInBits =  bitsPerElement;// one  element result
            type = 1;
            vnet = 0;
            break;

  }
  send_out_time = 0;
  // added
  in_net_time = 0;
}

/*
  * @brief convert destination to be x/y/output port of the router
   */
void Packet::dest_convert(int dest, int router_num_x, int* NI_num){
  int hist_num = 0, cur_num = 0, router = 0;
  while (cur_num <= dest){
      hist_num = cur_num;
      cur_num += NI_num[router];
      router++;
  }
  router--;
  destination[0] = router/router_num_x;
  destination[1] = router%router_num_x;
  destination[2] = dest-hist_num;
  //std::cout << destination[0] <<destination[1] << destination[2]<<std::endl;
}


