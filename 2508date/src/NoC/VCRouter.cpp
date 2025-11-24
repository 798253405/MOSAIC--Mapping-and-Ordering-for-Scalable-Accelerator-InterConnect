/*
 * Router.cpp
 *
 */

#include "VCRouter.hpp"
#include "../parameters.hpp"
#include <iostream>


VCRouter::VCRouter(int* t_id, int in_out_port_num, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth)
{
  // Router position in two-dimension site
  id[0] = t_id[0];  // x-axis
  id[1] = t_id[1];  // y-axis
  authorRouterID=id[0] *X_NUM  +  id[1];
  //cout<<authorRouterID <<" authorRouterID routerline17" << endl;
  vcNetwork = t_vcNetwork;

  in_port_list.reserve(in_out_port_num);
  out_port_list.reserve(in_out_port_num);

  for (int i=0; i< in_out_port_num; i++){
      //Establish IN_PORT component
      Link * link = new Link((RInPort*)NULL);
      RInPort * t_rInPort = new RInPort(i, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_in_depth, this, link); // i, 2, 1, 1 or i, 2, 2, 1, in depth = 4
      // added
      t_rInPort->rid[0] = t_id[0];
      t_rInPort->rid[1] = t_id[1];

      link->rInPort = t_rInPort;
      in_port_list.push_back(t_rInPort);

      //Establish OUT_PORT component
#ifdef outPortNoInfinite
      //added oct20
      ROutPort * t_rOutPort = new ROutPort(i, 1, 1, 0, 1); //last element: INFINITE
      if(i==4) // Mem Outport
      {
    	  t_rOutPort = new ROutPort(i, 1, 1, 0, 2);
      }
#else
      ROutPort * t_rOutPort = new ROutPort(i, 1, 1, 0, INFINITE);
#endif
      out_port_list.push_back(t_rOutPort);
  }
  rr_port = 0;
  port_num = in_out_port_num;
  port_total_utilization = 0;
  port_utilization_innet = 0;
}
//  - packetrouting
// xyroutingBool: 0=(old), 1=old(XY routing), 2=new(YX routing)
int VCRouter::getRoute(Flit* t_flit){
    int routingMode = t_flit->packet->xyroutingBool;

    // Debug: Count routing mode usage (only for first 1000 flits)
    static int routing_count[3] = {0, 0, 0};
    static int total_routed = 0;
    if(total_routed < 1000 && routingMode >= 0 && routingMode <= 2){
        routing_count[routingMode]++;
        total_routed++;
        if(total_routed == 1000){
            cout << "[ROUTING-DEBUG] First 1000 flits routed: mode0(default)=" << routing_count[0]
                 << " mode1(XY)=" << routing_count[1] << " mode2(YX)=" << routing_count[2] << endl;
        }
    }

    if(routingMode == 0 || routingMode == 1){
        return getRouteOld(t_flit);  // XY routing（）
    } else if(routingMode == 2){
        return getRouteNew(t_flit);  // YX routing（）
    } else {
        // routing
        assert(false && "ERROR: Invalid xyroutingBool value, must be 0, 1, or 2");
        return -1;
    }
}

// routing: XY routing（，）
int VCRouter::getRouteOld(Flit* t_flit){
    int x = t_flit->packet->destination[0];
    int y = t_flit->packet->destination[1];
    int z = t_flit->packet->destination[2];

    if(y < id[1]){ // turn left
        return 3;
    } else if(y > id[1]){ // turn right
        return 1;
    } else { // y direction matches
        if(x < id[0]){ // turn up
            return 0;
        } else if(x > id[0]){ // turn down
            return 2;
        } else { // arrival
            return (z+4); // 0->y+(up); 1->x+(right); 2->y-(down); 3->x-(left); 4/4+-> controller
        }
    }
    assert(1==0);
    return -1;
}

// routing: YX routing（，）
int VCRouter::getRouteNew(Flit* t_flit){
    int x = t_flit->packet->destination[0];
    int y = t_flit->packet->destination[1];
    int z = t_flit->packet->destination[2];

    // X（）
    if(x < id[0]){ // turn up
        return 0;
    } else if(x > id[0]){ // turn down
        return 2;
    } else { // X，Y（）
        if(y < id[1]){ // turn left
            return 3;
        } else if(y > id[1]){ // turn right
            return 1;
        } else { // arrival
            return (z+4); // 0->y+(up); 1->x+(right); 2->y-(down); 3->x-(left); 4/4+-> controller
        }
    }
    assert(1==0);
    return -1;
}



void VCRouter::vcRequest(){  //for each vc in each port which is in state v(2), do vc request
  for(int i=0; i<port_num; i++){ //port round robin
      in_port_list[(i+rr_port)%port_num]->vc_request();
  }
}


void VCRouter::getSwitch(){
  for(int i=0; i<port_num; i++){ //port round robin
      in_port_list[(i+rr_port)%port_num]->getSwitch(authorRouterID);
  }
  rr_port = (rr_port+1)%port_num;
}


void VCRouter::outPortDequeue(){
  for(int i=0; i<port_num; i++){ //port round robin
      if(out_port_list[i]->buffer_list[0]->cur_flit_num != 0 && out_port_list[i]->buffer_list[0]->read()->sched_time < cycles){
	  Flit* flit = out_port_list[i]->buffer_list[0]->dequeue();

	    //  if( out_port_list[i]->out_link->rInPort->buffer_list[flit->vc]->isFull())
	     	 //    cout<< "router dequeue" << endl;
	  out_port_list[i]->out_link->rInPort->buffer_list[flit->vc]->enqueue(flit);

	  authorGlobalFlitPass ++;

	  // Count router-only hops (ports 0-3 are router-to-router)
	  if (i <= 3) {
		  authorGlobalRouterHopCount++;
	  } else {
		  // Port 4 is NI connection
		  authorGlobalNIHopCount++;
	  }

	  // Debug: Print first 31 flit passes
	  /*
	  if (authorGlobalFlitPass <= 31) {
		  cout << "[Flit Pass #" << authorGlobalFlitPass << "] "
		       << "SeqID: " << flit->seqid
		       << ", Router(" << this->id[0] << "," << this->id[1] << ")"
		       << ", Type: " << flit->type
		       << ", MsgType: " << flit->packet->message.msgtype
		       << endl;
	  }
	  */

	  if(flit->packet->message.msgtype == 1) // only type 1 is recorded
		  {
		  authorGlobalRespFlitPass ++;
		  }
	   //if(flit->packet->message.msgtype == 1) // only type 1 is recorded
	  {

		  // cout <<" authorGlobalRespFlitPassinvcRouterCPP " <<  authorGlobalRespFlitPass <<endl;

		   //cout<<"aaaa "<<cycles<<" authorflippinginrouter "<< authorRouterID <<" port "<<i<<" "<<flit->id <<" reqresprestype "<<flit->packet->message.msgtype<<endl;

		#ifdef  all128BitInvert //convert，linklink，128bit + 1bit invert line
		  out_port_list[i]->out_link->rInPort->authorInportall128BitInvertFlippingCounts(flit,  authorRouterID ,/* i is the portseqID*/ i  );
		#else //，128 bit link，
		  out_port_list[i]->out_link->rInPort->authorInportFlippingCounts(flit,  authorRouterID ,/* i is the portseqID*/ i  );
		#endif
	  }
	  flit->sched_time = cycles + LINK_TIME - 1;



	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port_list[i]->out_link->rInPort->router_owner);
	  	      if (vcRouter != NULL){
	  int tempNextRouterID= dynamic_cast<VCRouter*>(out_port_list[i]->out_link->rInPort->router_owner)->authorRouterID;
	  //authorEnterInportPerRouter[tempNextRouterID].push_back(cycles);
	  //authorEnterInportPerRouter[tempNextRouterID].push_back(i);
	 // cout<<"tempNextRouterID" <<tempNextRouterID<<endl;
	  	      }

#ifdef flitTraceNodeTime
	  // added
	  flit->trace_node.push_back(out_port_list[i]->out_link->rInPort->rid[0]*X_NUM + out_port_list[i]->out_link->rInPort->rid[1]);
	  flit->trace_time.push_back(cycles + LINK_TIME - 1);
#endif
	  port_total_utilization++;

	  if (i<=3) port_utilization_innet++;

	  //cout<<"outport of router:"<< id[0]<<","<<id[1]<<", dequeue at cycles:" << cycles << "; packet length:" << flit->packet->length <<endl;

	  if(flit->type == 0 || flit->type == 10){
	      VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port_list[i]->out_link->rInPort->router_owner);
	      if (vcRouter != NULL){
#ifdef SHARED_VC //for QoS USING shared VC
		  if(flit->packet->signal->QoS == 1){
		      out_port_list[i]->out_link->rInPort->priority_vc.push_back(flit->vc);
		      out_port_list[i]->out_link->rInPort->priority_switch.push_back(flit->vc);
		  }
#endif
		  int route_result = vcRouter->getRoute(flit);
		  out_port_list[i]->out_link->rInPort->out_port[flit->vc] = route_result;
		  assert(out_port_list[i]->out_link->rInPort->state[flit->vc] == 1);
	      }
	      out_port_list[i]->out_link->rInPort->state[flit->vc] = 2; //wait for vc allocation
	  }
      }
  }
}


// To run Routing, VC_allocation, Switching
void VCRouter::runOneStep(){
  vcRequest();
  getSwitch();
  outPortDequeue();
  //if((cycles%100000)==0) {cout << "running cycles: " << cycles << ' ' << id[0] << ' ' << id[1] << endl;}
//  if( id[0]==0 &&  id[1]==0 && cycles%10000==0)
//  cout << cycles <<" debugauthorRouterOutput[0] " << id[0] << ' ' << id[1] <<" "<< out_port_list[0]->buffer_list[0]->cur_flit_num  <<" "<< out_port_list[0]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[1]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[2]->buffer_list[0]->cur_flit_num
//  <<" "<< out_port_list[3]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[4]->buffer_list[0]->cur_flit_num<<endl; //<<" " << out_port_list[0]->buffer_list[0]->used_credit <<endl;
//
//  if( id[0]==10 &&  id[1]==10 && cycles%10000==0)
//  {
//   cout << cycles <<" debugauthorRouterOutput[0] " << id[0] << ' ' << id[1] <<" "<< out_port_list[0]->buffer_list[0]->cur_flit_num  <<" "<< out_port_list[0]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[1]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[2]->buffer_list[0]->cur_flit_num
//   <<" "<< out_port_list[3]->buffer_list[0]->cur_flit_num <<" "<< out_port_list[4]->buffer_list[0]->cur_flit_num<<endl; //<<" " << out_port_list[0]->buffer_list[0]->used_credit <<endl;
//  cout << cycles <<" debugauthorRouterInput[0] " << id[0] << " " << id[1]<<" "<<in_port_list[0]->buffer_list[0]->used_credit<<" "<<in_port_list[1]->buffer_list[0]->used_credit<<" "
//		  <<in_port_list[2]->buffer_list[0]->used_credit<<" "<<in_port_list[3]->buffer_list[0]->used_credit<<" "<<in_port_list[4]->buffer_list[0]->used_credit<<" "<<endl;
//  }
}
void VCRouter::resetRouterRoundRobin(){
	rr_port = 0;
	for(int i=0; i<port_num; i++){
		in_port_list[i]->rr_record = 0;
		in_port_list[i]->rr_priority_record = 0 ;
	}
}

VCRouter::~VCRouter ()
{
  RInPort* inPort;
  while(in_port_list.size()!=0){
      inPort = in_port_list.back();//Anonymous: vector.back Returns a reference to the last element in the vector. Unlike member vector::end, which returns an iterator just past this element, this function returns a direct reference.
      in_port_list.pop_back();//Anonymous: popback Removes the last element in the vector, effectively reducing the container size by one.
      delete inPort;
  }

  ROutPort* outPort;
  while(out_port_list.size()!=0){
      outPort = out_port_list.back();
      out_port_list.pop_back();
      delete outPort;
  }
}

