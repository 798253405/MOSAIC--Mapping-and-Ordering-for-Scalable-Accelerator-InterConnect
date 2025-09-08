

#include "RInPort.hpp"

RInPort::RInPort(int t_id, int t_vn_num, int t_vc_per_vn,
		int t_vc_priority_per_vn, int t_depth, NRBase *t_owner, Link *t_in_link) :
		Port(t_id, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_depth) {
	for (int i = 0; i < vn_num * (vc_per_vn + vc_priority_per_vn); i++) {
		state.push_back(0);
		out_port.push_back(-1);
		out_vc.push_back(-1);
	}
	rr_record = 0;
	rr_priority_record = 0;
	router_owner = t_owner;
	in_link = t_in_link;
	count_vc = 0;
	count_switch = 0;
	starvation = 0;
	//added oct20
	flitOperNuminOneCycle = 0;
	firstFlitorNot = 0;
	yzweightCollsionCountInportCount = 0;
	totalyzInportFixFlipping = 0;
	totalyzInportFlipping = 0;

	int preExtraInvertBitline; //250322
	int currentExtraInvertBitline;
}

int RInPort::vc_allocate(Flit *t_flit) {
	int vn = t_flit->vnet;

	for (int i = 0; i < vc_per_vn; i++) {
		if (state[vn * vc_per_vn + i] == 0) { //idle
			state[vn * vc_per_vn + i] = 1;  //wait for routing
			return vn * vc_per_vn + i;  //vc id
		}
	}
	return -1; // no vc available
}

int RInPort::vc_allocate_normal(Flit *t_flit) {
	int vn = t_flit->vnet;

	for (int i = 0; i < vc_per_vn - vc_priority_per_vn; i++) {
		if (state[vn * vc_per_vn + i] == 0) { //idle
			state[vn * vc_per_vn + i] = 1;  //wait for routing
			return vn * vc_per_vn + i;  //vc id
		}
	}
	return -1; // no vc available
}

int RInPort::vc_allocate_priority(int vn_rank) {

	for (int i = 0; i < vc_priority_per_vn; i++) {
		int tag = vn_num * vc_per_vn + vc_priority_per_vn * vn_rank + i;
		if (state[tag] == 0) {
			state[tag] = 1;
			return tag;
		}
	}
	return -1;
}

void RInPort::vc_request() {
	flitOperNuminOneCycle = 0;
	// for priority packet (shared VCs) QoS = 1
#ifdef SHARED_VC
   std::vector<int>::iterator iter;
   for(iter=priority_vc.begin(); iter<priority_vc.end();){
       if(count_vc == STARVATION_LIMIT)
 	  break;
       int tag = (*iter);
       if(state[tag] == 2){
 	  Flit* flit = buffer_list[tag]->flit_queue.front();
 	  assert(flit->type == 0 || flit->type == 10);
 	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
 	  assert(vcRouter != NULL);
 	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(flit);
 	  if (vc_result != -1){
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_vc.push_back(vc_result);
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_switch.push_back(vc_result);
 	      state[tag] = 3; //active
 	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
 	      count_vc++;
 	      iter = priority_vc.erase(iter);
 	  }else
    	    iter++;
       }else
	 iter++;
   }
   count_vc = 0;
#endif

	// for priority packet (individual VCs) QoS = 3
	for (int i = vn_num * vc_per_vn;
			i < vn_num * (vc_per_vn + vc_priority_per_vn); i++) {
		int tag = (i + rr_priority_record) % (vn_num * vc_priority_per_vn)
				+ vn_num * vc_per_vn;
		if (state[tag] == 2) {
			Flit *flit = buffer_list[tag]->flit_queue.front();
			assert(flit->type == 0 || flit->type == 10);
			VCRouter *vcRouter = dynamic_cast<VCRouter*>(router_owner);
			assert(vcRouter != NULL);
			int vn_rank = (tag - vn_num * vc_per_vn) / vc_priority_per_vn;
			int vc_result =
					vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate_priority(
							vn_rank);
			if (vc_result != -1) {
				state[tag] = 3; //active
				out_vc[tag] = vc_result; //record the output vc in the streaming down node
			}
		}
	}

	// for URS packet
	for (int i = 0; i < vn_num * vc_per_vn; i++) {
		int tag = (i + rr_record) % (vn_num * vc_per_vn);
		if (state[tag] == 2) {
			Flit *flit = buffer_list[tag]->flit_queue.front();
			assert(flit->type == 0 || flit->type == 10);
			VCRouter *vcRouter = dynamic_cast<VCRouter*>(router_owner);
			assert(vcRouter != NULL);
#ifdef SHARED_PRI
	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate_normal(flit);
#else
			int vc_result =
					vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(
							flit);
#endif
			if (vc_result != -1) {
				state[tag] = 3; //active
				out_vc[tag] = vc_result; //record the output vc in the streaming down node
			}
		}
	}

	// in case of no normal packet
#ifdef SHARED_VC

   for(; iter<priority_vc.end();){
       int tag = (*iter);
       if(state[tag] == 2){
 	  Flit* flit = buffer_list[tag]->flit_queue.front();
 	  assert(flit->type == 0 || flit->type == 10);
 	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
 	  assert(vcRouter != NULL); // only vc router will call this methed
 	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(flit);
 	  if (vc_result != -1){
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_vc.push_back(vc_result);
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_switch.push_back(vc_result);
 	      state[tag] = 3; //active
 	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
 	      iter = priority_vc.erase(iter);
 	  }else
    	    iter++;
       }else
	 iter++;
   }
#endif

}

void RInPort::getSwitch(int t_RouterIDOweThisPort) {
	//cout<<t_RouterIDOweThisPort <<" t_RouterIDOweThisPort line 166" <<endl;
	// for priority packet
#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_switch.begin(); iter<priority_switch.end();iter++){
      if(count_switch == STARVATION_LIMIT)
	  break;
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL); // only VC router will call this method
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()){
	      buffer_list[tag]->dequeue();

	      //if(vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull())
	     	    // cout<< "router switch" << endl;
#ifdef flitTraceNodeTime
	      // added
	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	      flit->trace_time.push_back(cycles);
#endif
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      count_switch++;
	      if(flit->type == 1 || flit->type == 10 ){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
  count_switch = 0;
#endif

	// for LCS packet (individual VC)
	for (int i = vn_num * vc_per_vn;
			i < vn_num * (vc_per_vn + vc_priority_per_vn); i++) { //vc round robin; pick up non-empty buffer with state A (3)
		if (starvation == STARVATION_LIMIT) {
			starvation = 0;
			break;
		}
		int tag = (i + rr_priority_record) % (vn_num * vc_priority_per_vn)
				+ vn_num * vc_per_vn;
		if (buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3
				&& buffer_list[tag]->read()->sched_time < cycles) {
			Flit *flit = buffer_list[tag]->read();
			flit->vc = out_vc[tag];
			VCRouter *vcRouter = dynamic_cast<VCRouter*>(router_owner);
			assert(vcRouter != NULL); // only vc router will call this methed
			//added oct20
			if (!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()
					&& flitOperNuminOneCycle == 0) {
				buffer_list[tag]->dequeue();
				// added
//	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
//	      flit->trace_time.push_back(cycles);
				vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(
						flit);

				// added oct20
				flitOperNuminOneCycle = flitOperNuminOneCycle + 1; //do one enqueue
				vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
				flit->sched_time = cycles;
				if (flit->type == 1 || flit->type == 10) {
					state[tag] = 0; //idle
				}
				rr_priority_record = (rr_priority_record + 1)
						% (vn_num * vc_priority_per_vn);
				starvation++;
				return;
			}
		}
	}

	// for normal packet
	for (int i = 0; i < vn_num * vc_per_vn; i++) { //vc round robin; pick up non-empty buffer with state A (3)
		int tag = (i + rr_record) % (vn_num * vc_per_vn);
		if (buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3
				&& buffer_list[tag]->read()->sched_time < cycles) {
			Flit *flit = buffer_list[tag]->read();
			flit->vc = out_vc[tag];
			VCRouter *vcRouter = dynamic_cast<VCRouter*>(router_owner);
			assert(vcRouter != NULL); // only vc router will call this methed
#ifdef outPortNoInfinite
			if (!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()
					&& !vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull()
					&& flitOperNuminOneCycle
							== 0)
#else
		  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull())
#endif
							{
				buffer_list[tag]->dequeue();
				// added
				//	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
				//	      flit->trace_time.push_back(cycles);
				// check wrong switch allocation
				vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(
						flit);
				//cout<<" line221t_id "<<this->id<<endl;
				/*
				 yzEnterOutportPerRouter[t_RouterIDOweThisPort].push_back(
				 cycles);
				 yzEnterOutportPerRouter[t_RouterIDOweThisPort].push_back(
				 out_port[tag]);
				 yzLeaveInportPerRouter[t_RouterIDOweThisPort].push_back(cycles);
				 yzLeaveInportPerRouter[t_RouterIDOweThisPort].push_back(id);
				 */
				//added oct20
				flitOperNuminOneCycle = flitOperNuminOneCycle + 1; //do one enqueue
				vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->get_credit();
				//
				vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
				flit->sched_time = cycles;
				if (flit->type == 1 || flit->type == 10) {
					state[tag] = 0; //idle
				}
				rr_record = (rr_record + 1) % (vn_num * vc_per_vn);
				return;
			}
		}
	}

	// in case of no normal packet
#ifdef SHARED_VC

  for(; iter<priority_switch.end();iter++){
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL); // only vc router will call this methed
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()){
	      buffer_list[tag]->dequeue();

	      //if(vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull())
	     	    // cout<< "router switch" << endl;
#ifdef flitTraceNodeTime
	      // added
	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	      flit->trace_time.push_back(cycles);
#endif
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      if(flit->type == 1 || flit->type == 10 ){
			  state[tag] = 0; //idle
			  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
#endif

	// for LCS packet (individual VC)
	for (int i = vn_num * vc_per_vn;
			i < vn_num * (vc_per_vn + vc_priority_per_vn); i++) { //vc round robin; pick up non-empty buffer with state A (3)
		int tag = (i + rr_priority_record) % (vn_num * vc_priority_per_vn)
				+ vn_num * vc_per_vn;
		if (buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3
				&& buffer_list[tag]->read()->sched_time < cycles) {
			Flit *flit = buffer_list[tag]->read();
			flit->vc = out_vc[tag];
			VCRouter *vcRouter = dynamic_cast<VCRouter*>(router_owner);
			assert(vcRouter != NULL);
			if (!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()) {
				buffer_list[tag]->dequeue();
				// added
#ifdef flitTraceNodeTime
				flit->trace_node.push_back(
						vcRouter->id[0] * X_NUM + vcRouter->id[1]);
				flit->trace_time.push_back(cycles);
#endif
				vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(
						flit);
				vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
				flit->sched_time = cycles;
				if (flit->type == 1 || flit->type == 10) {
					state[tag] = 0; //idle
				}
				rr_priority_record = (rr_priority_record + 1)
						% (vn_num * vc_priority_per_vn);
				return;
			}
		}
	}
}

int RInPort::yzInportFlippingCounts(Flit *t_yztempFlit,
		int t_routerIDIntoInport, int t_inportSeqID) {
	int tempDataCount = payloadElementNum; //FLIT_LENGTH/valueBytes-1 ; //how many floating point values in one flit?
	this->currentFlitInLink = t_yztempFlit; //be careful about shallow copy!
	currentFlitInLink->yzFlitPayload.clear();

	if ((currentFlitInLink->seqid + 1) * tempDataCount
			> t_yztempFlit->packet->message.yzMSGPayload.size()) {

		std::cout << "seqid: " << currentFlitInLink->seqid;
		std::cout << " tempDataCount: " << tempDataCount;
		std::cout << " Payload size: "
				<< t_yztempFlit->packet->message.yzMSGPayload.size();
		std::cerr << " Error: Attempting to access beyond the vector limits!"
				<< std::endl;
		// Handle error or adjust indices
		assert(
				false
						&& "Rinports Attempting to access beyond the vector limits!");

	}

#ifdef CoutDebugAll0
		 for (const auto& element :  /* currentFlitInLink->packet->message.yzMSGPayload*/ t_yztempFlit->packet->message.yzMSGPayload) {
		         std::cout << element << " ";
		     }
		     std::cout<< "  line350currentFlitInLink->packet->message.yzMSGPayloadsize =  "<<currentFlitInLink->packet->message.yzMSGPayload.size() <<" "<<t_yztempFlit->packet->message.yzMSGPayload[0] <<" ifflitze, flitnum= "<<(currentFlitInLink->packet->message.yzMSGPayload.size() ) / (16) + 1<< "  msg.yzMSGPayloadBeforePaddingForFlits  " << std::endl;
#endif
//	cout<<" currentFlitInLink->yzFlitPayload.size()line344 " <<currentFlitInLink->yzFlitPayload.size()<<endl;
	//inbuffer.end() //inbuffer.begin()+18

	//	currentFlitInLink->yzFlitPayload.insert(t_yztempFlit->yzFlitPayload.end(), t_yztempFlit->packet->message.yzMSGPayload.begin()+3 + currentFlitInLink->id*( tempDataCount) , t_yztempFlit->packet->message.yzMSGPayload.begin()+3+(t_yztempFlit->id+1)*( tempDataCount) );// if contains three channel info, +3
	//cout<<cycles<<" currentFlitInLink->yzFlitPayload.size()line353 " <<currentFlitInLink->yzFlitPayload.size()<<"   "<< currentFlitInLink->id<<"   "<<t_yztempFlit->packet->message.yzMSGPayload.size() <<endl;
	currentFlitInLink->yzFlitPayload.insert(t_yztempFlit->yzFlitPayload.end(),
			t_yztempFlit->packet->message.yzMSGPayload.begin()
					+ currentFlitInLink->seqid * (tempDataCount),
			t_yztempFlit->packet->message.yzMSGPayload.begin()
					+ (t_yztempFlit->seqid + 1) * (tempDataCount));

#ifdef CoutDebugAll0
        if(currentFlitInLink->yzFlitPayload[i] != 0)
        {
        	cout<<" cycles "<<cycles <<" ith_element  "<<i <<" currentFlitInLink->yzFlitPayload[i]=  "<< currentFlitInLink->yzFlitPayload[i] << " "<<*(t_yztempFlit->packet->message.yzMSGPayload.begin()  + currentFlitInLink->id*( 16) +i)<<" "<<t_yztempFlit->packet->message.yzMSGPayload.size()<<" flitid "<<id <<endl;
        	 cout<<"ieee1Preload[i] " << ieee1 <<" " <<  yzPreviousFlitPayload[i] <<" ieee2CurlLoad[i] "<<ieee2 <<" "<<currentFlitInLink->yzFlitPayload[i]<<endl;
        }
#endif

	if (this->firstFlitorNot == 0) // first flit has no previous flit，so we need to avoid accessing null adress
			{
		yzPreviousFlitPayload.clear();
		yzPreviousFlitPayload.assign(currentFlitInLink->yzFlitPayload.size(),
				0); //flit level comparison. Initial to be all 0.
		//	yzPreviousFlitPayload.insert(yzPreviousFlitPayload.end(),
		//			t_yztempFlit->yzFlitPayload.begin(),
		//			t_yztempFlit->yzFlitPayload.end());	// just for debug。 should rest 0, but debug to be the same as the first flit

		yzPreviousMSGPayload.insert(yzPreviousMSGPayload.end(),
				t_yztempFlit->packet->message.yzMSGPayload.begin(),
				t_yztempFlit->packet->message.yzMSGPayload.end());// // just for debug。 should rest 0, but debug to be the same as the first msg
		this->firstFlitorNot = 1;
	}
	int oneTimeFlipping = 0;
	int oneTimeFlippingFix35 = 0;
//	print_floats_binary

	//cout<<" below is currentt " <<t_yztempFlit->seqid <<" t_yztempFlit->packet->message.yzMSGPayload "<<t_yztempFlit->packet->message.yzMSGPayload.size();
	//print_FlitPayload(t_yztempFlit->yzFlitPayload) ;
	//cout<<"  currenttMSGPayload " ;
	//print_FlitPayload(t_yztempFlit->packet->message.yzMSGPayload);
	//cout<<" below is previous " ;
	// print_FlitPayload(yzPreviousFlitPayload) ;

#ifdef msgcomparison
	    // Assuming  msg1 and msg2 have the same size
	    size_t size =  t_yztempFlit->packet->message.yzMSGPayload.size();
	    size_t sizePreviou =  yzPreviousMSGPayload.size();
	    if(size>sizePreviou ){
	    	// Resize yzPreviousMSGPayload to match the new size
	    	    yzPreviousMSGPayload.resize(size);

	    	    // Optionally fill the new elements with zeros
	    	    std::fill(yzPreviousMSGPayload.begin() + sizePreviou, yzPreviousMSGPayload.end(), 0);

	    }
	    cout <<" rinportCPP_397line_currentFlitInLink->packet->message.yzMSGPayload.size();  "<< t_yztempFlit->packet->message.yzMSGPayload.size() <<" PreMsgSize "<<yzPreviousMSGPayload.size()<<" preload[0] "<<yzPreviousMSGPayload[0]<<" curload[0] "<<t_yztempFlit->packet->message.yzMSGPayload[0] <<" flipping "<<this->totalyzInportFlipping<<endl;
	    for (size_t i = 0; i < size; ++i) {
	    	//msg vs msg //msg level comparison
	    	std::string ieee1 =  float_to_ieee754(yzPreviousMSGPayload[i]);
	    	std::string ieee2 = float_to_ieee754(t_yztempFlit->packet->message.yzMSGPayload[i]);
	    	//cout<<"ieee1 " << ieee1<<" ieee2 "<<ieee2<<endl;
	        int flips = 0;
	            // Assuming binary1 and binary2 are same length
	            for (size_t i = 0; i < ieee1.length(); ++i) {
	                if (ieee1[i] != ieee2[i]) {
	                    flips++;
	                }
	            }
	            this->totalyzInportFlipping += flips;
	           // cout<<"        flips of one floating point "<< flips<<" "<<yzPreviousFlitPayload[i]<<" "<<currentFlitInLink->packet->message.yzMSGPayload[i]<<" "<<  ieee1 <<" "<< ieee2<<endl;
	    }
	    //if(t_yztempFlit->packet->message.msgtype  == 1)
		 cout<<cycles<<" type "<< t_yztempFlit->packet->message.msgtype <<" value: "<< t_yztempFlit->packet->message.yzMSGPayload.front()<<" size: "<<t_yztempFlit->packet->message.yzMSGPayload.size()<< " this->totalyzInportFlippingflipping " <<this->totalyzInportFlipping <<" line374"<<endl;//packet->message.msgtype:  0=req  1=response  2=result

	yzPreviousFlitPayload.clear();
	    yzPreviousMSGPayload.clear();
	    yzPreviousMSGPayload.insert(yzPreviousMSGPayload.end(), t_yztempFlit->packet->message.yzMSGPayload.begin(), t_yztempFlit->packet->message.yzMSGPayload.end());

#endif

#ifdef flitcomparison
	// Function to compare flips
	if (currentFlitInLink->yzFlitPayload.size() != payloadElementNum) { // assert if the flit size has error
		assert(false && "Payload size does not match expected element number.");
	}

	// Debug output: Show flit payload details with bit counts
	cout << "Flit " << currentFlitInLink->seqid 
	     << " (elements " << currentFlitInLink->seqid * 16 
	     << "-" << (currentFlitInLink->seqid * 16 + 15) << "):" << endl;
	cout << "  Values: ";
	for (size_t i = 0; i < payloadElementNum; ++i) {
		cout << currentFlitInLink->yzFlitPayload[i];
		if (i < payloadElementNum - 1) cout << " ";
	}
	cout << endl;
	cout << "  Bit counts: ";
	for (size_t i = 0; i < payloadElementNum; ++i) {
		cout << countOnesInIEEE754(currentFlitInLink->yzFlitPayload[i]);
		if (i < payloadElementNum - 1) cout << " ";
	}
	cout << endl;

	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string ieee1 = float_to_ieee754(yzPreviousFlitPayload[i]);
		std::string ieee2 = float_to_ieee754(
				currentFlitInLink->yzFlitPayload[i]);
#ifdef CoutDebugAll0
        if(currentFlitInLink->yzFlitPayload[i] != 0)
        {
        	cout<<" cycles "<<cycles <<" ith_element  "<<i <<" currentFlitInLink->yzFlitPayload[i]=  "<< currentFlitInLink->yzFlitPayload[i] << " "<<*(t_yztempFlit->packet->message.yzMSGPayload.begin()  + currentFlitInLink->id*( 16) +i)<<" "<<t_yztempFlit->packet->message.yzMSGPayload.size()<<" flitid "<<id <<endl;
        	 cout<<"ieee1Preload[i] " << ieee1 <<" " <<  yzPreviousFlitPayload[i] <<" ieee2CurlLoad[i] "<<ieee2 <<" "<<currentFlitInLink->yzFlitPayload[i]<<endl;
        }
#endif

		int flips = 0;
		// Assuming binary1 and binary2 are same length
		for (size_t i = 0; i < ieee1.length(); ++i) {
			if (ieee1[i] != ieee2[i]) {
				flips++;
			}
		}
		this->totalyzInportFlipping += flips;
		oneTimeFlipping += flips;
	}
// fixe 35 comparison
	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string fixpoint1 = singleFloat_to_fixed17(
				yzPreviousFlitPayload[i]);
		std::string fixpoint2 = singleFloat_to_fixed17(
				currentFlitInLink->yzFlitPayload[i]);
		int flips = 0;
		for (size_t i = 0; i < fixpoint1.length(); ++i) {
			if (fixpoint1[i] != fixpoint2[i]) {
				flips++;
			}
		}
		this->totalyzInportFixFlipping += flips;
		oneTimeFlippingFix35 += flips;
	}

#ifdef CoutDebugAll0
	if (true) { //oneTimeFlipping <120  && //t_routerIDIntoInport == 0
		cout << " thisisline416 current flitpayloadfront "
				<< currentFlitInLink->yzFlitPayload[0] << " previouse"
				<< yzPreviousFlitPayload[0] << " t_routerIDIntoInport  "
				<< t_routerIDIntoInport << " t_inportSeqID " << t_inportSeqID
				<< " yzPreFlitGlobalID " << yzPreFlitGlobalID
				<< " currentglobalflitID "
				<< t_yztempFlit->YZGlobalFlit_idInFlit << " "
				<< " yzPreFlitSeqID " << yzPreFlitSeqID
				<< " currentt_yztempFlitseqid " << t_yztempFlit->seqid
				<< " oneWholeFlitFlipping " << oneTimeFlipping << endl;

// this is 587 / 51072 . Small ratio.
	}
#endif

	yzweightCollsionCountInportCount = yzweightCollsionCountInportCount + 1;
	yzFlitCollsionCountSum = yzFlitCollsionCountSum + 1;
	//cout <<"  yzweightCollsionCountInportCountinRinport " <<  yzweightCollsionCountInportCount  <<endl;

	if (yzPreFlitSeqID == currentFlitInLink->seqid) { //if( weight same(msg same ) && seqID same(flit same)   )  // now weight same temp not implementd
		//yzweightCollsionCountInportCount = yzweightCollsionCountInportCount + 1;
		yzweightCollsionCountInportCount = yzweightCollsionCountInportCount + 0; // do nothing. For debugging.
	}

	yzPreviousFlitPayload.clear();
	yzPreFlitGlobalID = currentFlitInLink->YZGlobalFlit_idInFlit;
	yzPreFlitSeqID = currentFlitInLink->seqid;
	yzPreviousFlitPayload.insert(yzPreviousFlitPayload.end(),
			currentFlitInLink->yzFlitPayload.begin(),
			currentFlitInLink->yzFlitPayload.end()); // flit level
#endif
	return this->totalyzInportFlipping;
}





int RInPort::yzInportall128BitInvertFlippingCounts(Flit *t_yztempFlit,
		int t_routerIDIntoInport, int t_inportSeqID) {
	int tempDataCount = payloadElementNum; //FLIT_LENGTH/valueBytes-1 ; //how many floating point values in one flit?
	this->currentFlitInLink = t_yztempFlit; //be careful about shallow copy!
	currentFlitInLink->yzFlitPayload.clear();

	if ((currentFlitInLink->seqid + 1) * tempDataCount
			> t_yztempFlit->packet->message.yzMSGPayload.size()) {

		std::cout << "seqid: " << currentFlitInLink->seqid;
		std::cout << " tempDataCount: " << tempDataCount;
		std::cout << " Payload size: "
				<< t_yztempFlit->packet->message.yzMSGPayload.size();
		std::cerr << " Error: Attempting to access beyond the vector limits!"
				<< std::endl;
		// Handle error or adjust indices
		assert(
				false
						&& "Rinports Attempting to access beyond the vector limits!");

	}
	currentFlitInLink->yzFlitPayload.insert(t_yztempFlit->yzFlitPayload.end(),
			t_yztempFlit->packet->message.yzMSGPayload.begin()
					+ currentFlitInLink->seqid * (tempDataCount),
			t_yztempFlit->packet->message.yzMSGPayload.begin()
					+ (t_yztempFlit->seqid + 1) * (tempDataCount));

	if (this->firstFlitorNot == 0) // first flit has no previous flit，so we need to avoid accessing null adress
			{
		yzPreviousFlitPayload.clear();
		yzPreviousFlitPayload.assign(currentFlitInLink->yzFlitPayload.size(),
				0); //flit level comparison. Initial to be all 0.

		yzPreviousMSGPayload.insert(yzPreviousMSGPayload.end(),
				t_yztempFlit->packet->message.yzMSGPayload.begin(),
				t_yztempFlit->packet->message.yzMSGPayload.end()); // // just for debug。 should rest 0, but debug to be the same as the first msg
		this->firstFlitorNot = 1;
	}
	int oneTimeFlipping = 0;
	int oneTimeFlippingFix35 = 0;

#ifdef flitcomparison
	// Function to compare flips
	if (currentFlitInLink->yzFlitPayload.size() != payloadElementNum) { // assert if the flit size has error
		assert(false && "Payload size does not match expected element number.");
	}

	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string ieee1 = float_to_ieee754(yzPreviousFlitPayload[i]);
		std::string ieee2 = float_to_ieee754(
				currentFlitInLink->yzFlitPayload[i]);

		int flips = 0;
		// Assuming binary1 and binary2 are same length
		for (size_t i = 0; i < ieee1.length(); ++i) {
			if (ieee1[i] != ieee2[i]) {
				flips++;
			}
		}
#ifdef partionedInvert
		if(flips > 16){
			flips = 32-flips;
		}
#endif
		oneTimeFlipping += flips;
	}
// fixe 35 comparison
	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string fixpoint1 = singleFloat_to_fixed17(
				yzPreviousFlitPayload[i]);
		std::string fixpoint2 = singleFloat_to_fixed17(
				currentFlitInLink->yzFlitPayload[i]);
		int flips = 0;
		for (size_t i = 0; i < fixpoint1.length(); ++i) {
			if (fixpoint1[i] != fixpoint2[i]) {
				flips++;
			}
		}
#ifdef partionedInvert
		if(flips > 4 ){
			flips = 8-flips;
		}
#endif
		oneTimeFlippingFix35 += flips;
	}

	// globalbit
# ifdef all128BitInvert
#ifndef partionedInvert
	if (oneTimeFlipping > 16 * 32/2) {
		oneTimeFlipping = 16 * 32 - oneTimeFlipping;
	}

	if (oneTimeFlippingFix35 > 16 * 8/2) {
		oneTimeFlippingFix35 = 16 * 8 - oneTimeFlippingFix35;
	}
#endif
#endif


	this->totalyzInportFlipping += oneTimeFlipping;
	this->totalyzInportFixFlipping += oneTimeFlippingFix35;

	yzweightCollsionCountInportCount = yzweightCollsionCountInportCount + 1;
	yzFlitCollsionCountSum = yzFlitCollsionCountSum + 1;

	if (yzPreFlitSeqID == currentFlitInLink->seqid) { //if( weight same(msg same ) && seqID same(flit same)   )  // now weight same temp not implementd
		yzweightCollsionCountInportCount = yzweightCollsionCountInportCount + 0; // do nothing. For debugging.
	}

	yzPreviousFlitPayload.clear();
	yzPreFlitGlobalID = currentFlitInLink->YZGlobalFlit_idInFlit;
	yzPreFlitSeqID = currentFlitInLink->seqid;
	yzPreviousFlitPayload.insert(yzPreviousFlitPayload.end(),
			currentFlitInLink->yzFlitPayload.begin(),
			currentFlitInLink->yzFlitPayload.end()); // flit level
#endif
	return this->totalyzInportFlipping;
}

RInPort::~RInPort() {

}

