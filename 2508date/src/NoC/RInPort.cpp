

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
	authorweightCollsionCountInportCount = 0;
	totalauthorInportFixFlipping = 0;
	totalauthorInportFlipping = 0;
	reqRouterFlipInport = 0;
	respRouterFlipInport = 0;
	resRouterFlipInport = 0;
	reqRouterHopInport = 0;
	respRouterHopInport = 0;
	resRouterHopInport = 0;

	int preExtraInvertBitline; //250322
	int currentExtraInvertBitline;

	zeroBTHopCount = 0;
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
				 authorEnterOutportPerRouter[t_RouterIDOweThisPort].push_back(
				 cycles);
				 authorEnterOutportPerRouter[t_RouterIDOweThisPort].push_back(
				 out_port[tag]);
				 authorLeaveInportPerRouter[t_RouterIDOweThisPort].push_back(cycles);
				 authorLeaveInportPerRouter[t_RouterIDOweThisPort].push_back(id);
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

int RInPort::authorInportFlippingCounts(Flit *t_authortempFlit,
		int t_routerIDIntoInport, int t_inportSeqID) {
	int tempDataCount = payloadElementNum; //FLIT_LENGTH/valueBytes-1 ; //how many floating point values in one flit?
	this->currentFlitInLink = t_authortempFlit; //be careful about shallow copy!
	currentFlitInLink->authorFlitPayload.clear();

	if ((currentFlitInLink->seqid + 1) * tempDataCount
			> t_authortempFlit->packet->message.authorMSGPayload.size()) {

		std::cout << "seqid: " << currentFlitInLink->seqid;
		std::cout << " tempDataCount: " << tempDataCount;
		std::cout << " Payload size: "
				<< t_authortempFlit->packet->message.authorMSGPayload.size();
		std::cerr << " Error: Attempting to access beyond the vector limits!"
				<< std::endl;
		// Handle error or adjust indices
		assert(
				false
						&& "Rinports Attempting to access beyond the vector limits!");

	}

#ifdef CoutDebugAll0
		 for (const auto& element :  /* currentFlitInLink->packet->message.authorMSGPayload*/ t_authortempFlit->packet->message.authorMSGPayload) {
		         std::cout << element << " ";
		     }
		     std::cout<< "  line350currentFlitInLink->packet->message.authorMSGPayloadsize =  "<<currentFlitInLink->packet->message.authorMSGPayload.size() <<" "<<t_authortempFlit->packet->message.authorMSGPayload[0] <<" ifflitze, flitnum= "<<(currentFlitInLink->packet->message.authorMSGPayload.size() ) / (16) + 1<< "  msg.authorMSGPayloadBeforePaddingForFlits  " << std::endl;
#endif
//	cout<<" currentFlitInLink->authorFlitPayload.size()line344 " <<currentFlitInLink->authorFlitPayload.size()<<endl;
	//inbuffer.end() //inbuffer.begin()+18

	//	currentFlitInLink->authorFlitPayload.insert(t_authortempFlit->authorFlitPayload.end(), t_authortempFlit->packet->message.authorMSGPayload.begin()+3 + currentFlitInLink->id*( tempDataCount) , t_authortempFlit->packet->message.authorMSGPayload.begin()+3+(t_authortempFlit->id+1)*( tempDataCount) );// if contains three channel info, +3
	//cout<<cycles<<" currentFlitInLink->authorFlitPayload.size()line353 " <<currentFlitInLink->authorFlitPayload.size()<<"   "<< currentFlitInLink->id<<"   "<<t_authortempFlit->packet->message.authorMSGPayload.size() <<endl;
	currentFlitInLink->authorFlitPayload.insert(t_authortempFlit->authorFlitPayload.end(),
			t_authortempFlit->packet->message.authorMSGPayload.begin()
					+ t_authortempFlit->seqid * (tempDataCount),
			t_authortempFlit->packet->message.authorMSGPayload.begin()
					+ (t_authortempFlit->seqid + 1) * (tempDataCount));

#ifdef CoutDebugAll0
        if(currentFlitInLink->authorFlitPayload[i] != 0)
        {
        	cout<<" cycles "<<cycles <<" ith_element  "<<i <<" currentFlitInLink->authorFlitPayload[i]=  "<< currentFlitInLink->authorFlitPayload[i] << " "<<*(t_authortempFlit->packet->message.authorMSGPayload.begin()  + currentFlitInLink->id*( 16) +i)<<" "<<t_authortempFlit->packet->message.authorMSGPayload.size()<<" flitid "<<id <<endl;
        	 cout<<"ieee1Preload[i] " << ieee1 <<" " <<  authorPreviousFlitPayload[i] <<" ieee2CurlLoad[i] "<<ieee2 <<" "<<currentFlitInLink->authorFlitPayload[i]<<endl;
        }
#endif

	if (this->firstFlitorNot == 0) // first flit has no previous flit，so we need to avoid accessing null adress
			{
		authorPreviousFlitPayload.clear();
		authorPreviousFlitPayload.assign(currentFlitInLink->authorFlitPayload.size(),
				0); //flit level comparison. Initial to be all 0.
		//	authorPreviousFlitPayload.insert(authorPreviousFlitPayload.end(),
		//			t_authortempFlit->authorFlitPayload.begin(),
		//			t_authortempFlit->authorFlitPayload.end());	// just for debug。 should rest 0, but debug to be the same as the first flit

		authorPreviousMSGPayload.clear();
		authorPreviousMSGPayload.insert(authorPreviousMSGPayload.end(),
				t_authortempFlit->packet->message.authorMSGPayload.begin(),
				t_authortempFlit->packet->message.authorMSGPayload.end());// // just for debug。 should rest 0, but debug to be the same as the first msg
		this->firstFlitorNot = 1;
	}
	int oneTimeFlipping = 0;
	int oneTimeFlippingFix35 = 0;
//	print_floats_binary

	//cout<<" below is currentt " <<t_authortempFlit->seqid <<" t_authortempFlit->packet->message.authorMSGPayload "<<t_authortempFlit->packet->message.authorMSGPayload.size();
	//print_FlitPayload(t_authortempFlit->authorFlitPayload) ;
	//cout<<"  currenttMSGPayload " ;
	//print_FlitPayload(t_authortempFlit->packet->message.authorMSGPayload);
	//cout<<" below is previous " ;
	// print_FlitPayload(authorPreviousFlitPayload) ;


#ifdef flitcomparison
	// Function to compare flips
	if (currentFlitInLink->authorFlitPayload.size() != payloadElementNum) { // assert if the flit size has error
		assert(false && "Payload size does not match expected element number.");
	}
/*
	// Debug output: Show flit payload details with bit counts
	cout << "Flit " << currentFlitInLink->seqid 
	     << " (elements " << currentFlitInLink->seqid * 16 
	     << "-" << (currentFlitInLink->seqid * 16 + 15) << "):" << endl;
	cout << "  Values: ";
	for (size_t i = 0; i < payloadElementNum; ++i) {
		cout << currentFlitInLink->authorFlitPayload[i];
		if (i < payloadElementNum - 1) cout << " ";
	}
	cout << endl;
	cout << "  Bit counts: ";
	for (size_t i = 0; i < payloadElementNum; ++i) {
		cout << countOnesInIEEE754(currentFlitInLink->authorFlitPayload[i]);
		if (i < payloadElementNum - 1) cout << " ";
	}
	cout << endl;
*/
	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string ieee1 = float_to_ieee754(authorPreviousFlitPayload[i]);
		std::string ieee2 = float_to_ieee754(
				currentFlitInLink->authorFlitPayload[i]);
#ifdef CoutDebugAll0
        if(currentFlitInLink->authorFlitPayload[i] != 0)
        {
        	cout<<" cycles "<<cycles <<" ith_element  "<<i <<" currentFlitInLink->authorFlitPayload[i]=  "<< currentFlitInLink->authorFlitPayload[i] << " "<<*(t_authortempFlit->packet->message.authorMSGPayload.begin()  + currentFlitInLink->id*( 16) +i)<<" "<<t_authortempFlit->packet->message.authorMSGPayload.size()<<" flitid "<<id <<endl;
        	 cout<<"ieee1Preload[i] " << ieee1 <<" " <<  authorPreviousFlitPayload[i] <<" ieee2CurlLoad[i] "<<ieee2 <<" "<<currentFlitInLink->authorFlitPayload[i]<<endl;
        }
#endif

		int flips = 0;
		// Assuming binary1 and binary2 are same length
		for (size_t i = 0; i < ieee1.length(); ++i) {
			if (ieee1[i] != ieee2[i]) {
				flips++;
			}
		}
		this->totalauthorInportFlipping += flips;
		oneTimeFlipping += flips;
	}
// fixe 35 comparison
	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string fixpoint1 = singleFloat_to_fixed17(
				authorPreviousFlitPayload[i]);
		std::string fixpoint2 = singleFloat_to_fixed17(
				currentFlitInLink->authorFlitPayload[i]);
		int flips = 0;
		for (size_t i = 0; i < fixpoint1.length(); ++i) {
			if (fixpoint1[i] != fixpoint2[i]) {
				flips++;
			}
		}
		this->totalauthorInportFixFlipping += flips;
		oneTimeFlippingFix35 += flips;
	}
	if ( oneTimeFlipping == 0)
	{
		zeroBTHopCount = zeroBTHopCount+1;
	}
	// Add to per-inport statistics by message type
	if (t_authortempFlit->packet->message.msgtype == 0) {
		this->reqRouterFlipInport += oneTimeFlipping;
		this->reqRouterHopInport += 1;  // Count one hop per flit
	} else if (t_authortempFlit->packet->message.msgtype == 1) {
		this->respRouterFlipInport += oneTimeFlipping;
		this->respRouterHopInport += 1;  // Count one hop per flit
	} else {
		this->resRouterFlipInport += oneTimeFlipping;  // type 2 or 3
		this->resRouterHopInport += 1;  // Count one hop per flit
	}


	  if ( oneTimeFlipping<0 )
		//  if (cycles<10)
		  {
	      cout << " thisisline416 current flitpayloadfront "
	              << "current: " << currentFlitInLink->authorFlitPayload[0] <<" "<< currentFlitInLink->authorFlitPayload[1] <<" "<<
	  currentFlitInLink->authorFlitPayload[2] <<" "<< currentFlitInLink->authorFlitPayload[3] <<" "<< currentFlitInLink->authorFlitPayload[4] <<" "<<
	  currentFlitInLink->authorFlitPayload[5] <<" "<< currentFlitInLink->authorFlitPayload[6] <<" "<< currentFlitInLink->authorFlitPayload[7] <<" | "<<
	  currentFlitInLink->authorFlitPayload[8] <<" "<< currentFlitInLink->authorFlitPayload[9] <<" "<< currentFlitInLink->authorFlitPayload[10] <<" "<<
	  currentFlitInLink->authorFlitPayload[11] <<" "<< currentFlitInLink->authorFlitPayload[12] <<" "<< currentFlitInLink->authorFlitPayload[13] <<" "<<
	  currentFlitInLink->authorFlitPayload[14] <<" "<< currentFlitInLink->authorFlitPayload[15] <<"          current_Onebits: " << countOnesInIEEE754(currentFlitInLink->authorFlitPayload[0]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[1]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[2]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[3]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[4]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[5]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[6]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[7]) <<" | "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[8]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[9]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[10]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[11]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[12]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[13]) <<" "<< countOnesInIEEE754(currentFlitInLink->authorFlitPayload[14]) <<" "<<
	  countOnesInIEEE754(currentFlitInLink->authorFlitPayload[15]) <<" "<<endl
	              << "                       previous: " << authorPreviousFlitPayload[0] <<" "<< authorPreviousFlitPayload[1] <<" "<<
	  authorPreviousFlitPayload[2] <<" "<< authorPreviousFlitPayload[3] <<" "<< authorPreviousFlitPayload[4] <<" "<< authorPreviousFlitPayload[5] <<" "<<
	  authorPreviousFlitPayload[6] <<" "<< authorPreviousFlitPayload[7] <<" | "<< authorPreviousFlitPayload[8] <<" "<< authorPreviousFlitPayload[9] <<" "<<
	  authorPreviousFlitPayload[10] <<" "<< authorPreviousFlitPayload[11] <<" "<< authorPreviousFlitPayload[12] <<" "<< authorPreviousFlitPayload[13] <<" "<<
	  authorPreviousFlitPayload[14] <<" "<< authorPreviousFlitPayload[15] <<" "
	              << "                       previous_Onebits: " << countOnesInIEEE754(authorPreviousFlitPayload[0]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[1]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[2]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[3]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[4]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[5]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[6]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[7]) <<" | "<< countOnesInIEEE754(authorPreviousFlitPayload[8]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[9]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[10]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[11]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[12]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[13]) <<" "<< countOnesInIEEE754(authorPreviousFlitPayload[14]) <<" "<<
	  countOnesInIEEE754(authorPreviousFlitPayload[15]) <<" " <<endl
	              << "                      seqinfo: " << t_routerIDIntoInport << " t_inportSeqID " << t_inportSeqID
	              << " authorPreFlitGlobalID " << authorPreFlitGlobalID
	              << " currentglobalflitID " << t_authortempFlit->authorGlobalFlit_idInFlit
	              << " authorPreFlitSeqID " << authorPreFlitSeqID
	              << " currentt_authortempFlitseqid " << t_authortempFlit->seqid
	              << " oneWholeFlitFlipping " << oneTimeFlipping << endl;
	              
	      // Print bit counts for all 128 floats in the msg payload
	      cout << "                      AllMsgPayloadBitCounts(128floats): " << endl;


		  int payload_size = std::min(200, (int)t_authortempFlit->packet->message.authorMSGPayload.size());
		 {
			  cout<<"            565 cycles "<<cycles<<"   " <<t_authortempFlit->packet->message.msgtype<<"            "  <<endl;
			  for (int i = 0; i < payload_size; i++) {


				  int bits_current = countOnesInIEEE754(t_authortempFlit->packet->message.authorMSGPayload[i]);
				  float float_value = t_authortempFlit->packet->message.authorMSGPayload[i];
				  cout<< bits_current << "(" << std::fixed << std::setprecision(2) << float_value << ")";

				  // Add separator every 8 elements within a line
				  if ((i + 1) % 8 == 0 && (i + 1) % 16 != 0) {
					  cout << " | ";
				  } else if (i < payload_size - 1 && (i + 1) % 16 != 0) {
					  cout << " ";
				  }

				  // Add newline every 16 elements (flit boundary)
				  if ((i + 1) % 16 == 0) {
					  cout << endl;
				  }
			  }
			  if (payload_size % 16 != 0) cout << endl; // Final newline if not complete flit
		  }
	  }




	authorweightCollsionCountInportCount = authorweightCollsionCountInportCount + 1;
	authorFlitCollsionCountSum = authorFlitCollsionCountSum + 1;


	authorPreviousFlitPayload.clear();
	authorPreFlitGlobalID = currentFlitInLink->authorGlobalFlit_idInFlit;
	authorPreFlitSeqID = currentFlitInLink->seqid;
	authorPreviousFlitPayload.insert(authorPreviousFlitPayload.end(),
			currentFlitInLink->authorFlitPayload.begin(),
			currentFlitInLink->authorFlitPayload.end()); // flit level
	// Update previous MSG payload for all-flits bit count analysis
	authorPreviousMSGPayload.clear();
	authorPreviousMSGPayload.insert(authorPreviousMSGPayload.end(),
			t_authortempFlit->packet->message.authorMSGPayload.begin(),
			t_authortempFlit->packet->message.authorMSGPayload.end()); // msg level
#endif
	return this->totalauthorInportFlipping;
}





int RInPort::authorInportall128BitInvertFlippingCounts(Flit *t_authortempFlit,
		int t_routerIDIntoInport, int t_inportSeqID) {
	int tempDataCount = payloadElementNum; //FLIT_LENGTH/valueBytes-1 ; //how many floating point values in one flit?
	this->currentFlitInLink = t_authortempFlit; //be careful about shallow copy!
	currentFlitInLink->authorFlitPayload.clear();

	if ((currentFlitInLink->seqid + 1) * tempDataCount
			> t_authortempFlit->packet->message.authorMSGPayload.size()) {

		std::cout << "seqid: " << currentFlitInLink->seqid;
		std::cout << " tempDataCount: " << tempDataCount;
		std::cout << " Payload size: "
				<< t_authortempFlit->packet->message.authorMSGPayload.size();
		std::cerr << " Error: Attempting to access beyond the vector limits!"
				<< std::endl;
		// Handle error or adjust indices
		assert(
				false
						&& "Rinports Attempting to access beyond the vector limits!");

	}
	currentFlitInLink->authorFlitPayload.insert(t_authortempFlit->authorFlitPayload.end(),
			t_authortempFlit->packet->message.authorMSGPayload.begin()
					+ t_authortempFlit->seqid * (tempDataCount),
			t_authortempFlit->packet->message.authorMSGPayload.begin()
					+ (t_authortempFlit->seqid + 1) * (tempDataCount));

	if (this->firstFlitorNot == 0) // first flit has no previous flit，so we need to avoid accessing null adress
			{
		authorPreviousFlitPayload.clear();
		authorPreviousFlitPayload.assign(currentFlitInLink->authorFlitPayload.size(),
				0); //flit level comparison. Initial to be all 0.

		authorPreviousMSGPayload.insert(authorPreviousMSGPayload.end(),
				t_authortempFlit->packet->message.authorMSGPayload.begin(),
				t_authortempFlit->packet->message.authorMSGPayload.end()); // // just for debug。 should rest 0, but debug to be the same as the first msg
		this->firstFlitorNot = 1;
	}
	int oneTimeFlipping = 0;
	int oneTimeFlippingFix35 = 0;

#ifdef flitcomparison
	// Function to compare flips
	if (currentFlitInLink->authorFlitPayload.size() != payloadElementNum) { // assert if the flit size has error
		assert(false && "Payload size does not match expected element number.");
	}

	for (size_t i = 0; i < payloadElementNum; ++i) {
		// flit vs flit //flit level comparison
		std::string ieee1 = float_to_ieee754(authorPreviousFlitPayload[i]);
		std::string ieee2 = float_to_ieee754(
				currentFlitInLink->authorFlitPayload[i]);

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
				authorPreviousFlitPayload[i]);
		std::string fixpoint2 = singleFloat_to_fixed17(
				currentFlitInLink->authorFlitPayload[i]);
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


	this->totalauthorInportFlipping += oneTimeFlipping;
	this->totalauthorInportFixFlipping += oneTimeFlippingFix35;


	authorweightCollsionCountInportCount = authorweightCollsionCountInportCount + 1;
	authorFlitCollsionCountSum = authorFlitCollsionCountSum + 1;

	if (authorPreFlitSeqID == currentFlitInLink->seqid) { //if( weight same(msg same ) && seqID same(flit same)   )  // now weight same temp not implementd
		authorweightCollsionCountInportCount = authorweightCollsionCountInportCount + 0; // do nothing. For debugging.
	}

	authorPreviousFlitPayload.clear();
	authorPreFlitGlobalID = currentFlitInLink->authorGlobalFlit_idInFlit;
	authorPreFlitSeqID = currentFlitInLink->seqid;
	authorPreviousFlitPayload.insert(authorPreviousFlitPayload.end(),
			currentFlitInLink->authorFlitPayload.begin(),
			currentFlitInLink->authorFlitPayload.end()); // flit level
	// Update previous MSG payload for all-flits bit count analysis
	authorPreviousMSGPayload.clear();
	authorPreviousMSGPayload.insert(authorPreviousMSGPayload.end(),
			t_authortempFlit->packet->message.authorMSGPayload.begin(),
			t_authortempFlit->packet->message.authorMSGPayload.end()); // msg level
#endif
	return this->totalauthorInportFlipping;
}

RInPort::~RInPort() {

}

