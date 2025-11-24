
#include "MAC.hpp"
#include "mc_mapping.hpp"

MAC::MAC(int t_id, MACnet *t_net, int t_NI_id) {
	selfMACid = t_id;
	net = t_net;
	NI_id = t_NI_id;
	weight.clear();
	infeature.clear();
	inbuffer.clear();
	ch_size = 0;
	m_size = 0;
	fn = -1;
	tmpch = -1;
	tmpm = 0;
	cnn_current_layer_task_id = -1;  // CNNID
	cnn_saved_task_id = -1;           // ID

	outfeature = 0.0;
	nextMAC = NULL;
	pecycle = 0;
	selfstatus = 0;
	send = 0;
	m_count = 0;

	// for new pooling
	npoolflag = 0;
	n_tmpch = 0;
	n_tmpm.clear();

#ifdef binaryroutingSwitch
	lastResponseRouting = 1;  // 1，response packetrouting 1
#endif

#ifdef fireAdvance
	//  Fire Advance
	total_tasks = 0;
	requests_sent = 0;
	responses_received = 0;
	tasks_completed = 0;
	fire_advance_counter = 0;
	fire_advance_armed = false;
	computing_task_id = -1;
#endif

	// find dest id
	// xid = row（），yid = col（）
	dest_mem_id = get_mc_for_pe(NI_id, X_NUM, Y_NUM);

	// Debug output for first few MACs
	if (selfMACid < 4) {
		int xid = NI_id / X_NUM;
		int yid = NI_id % X_NUM;
		std::cout << "MAC[" << selfMACid << "] at (" << xid << "," << yid
		          << ") -> MC " << dest_mem_id << std::endl;
	}
	cnn_task_queue.clear();
}

bool MAC::inject(int type, int d_id, int t_eleNum, float t_output, NI *t_NI,
		int p_id, int mac_src) {

	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src; //MAC
	msg.msgdata_length = t_eleNum ; // element num only for resp and results
	//
	//cout<<" 	msg.msgdata_length "<<	msg.msgdata_length <<"   "<<inbuffer[0] <<" "<<inbuffer[1] <<" "<<inbuffer[2]<<endl;
	int selector = rand() % 90;

	msg.QoS = 0;

	msg.data.assign(1, t_output);
	msg.data.push_back(tmpch);
	msg.data.push_back(tmpm);

	msg.destination = d_id;
	msg.out_cycle = pecycle;
	msg.sequence_id = 0;
	msg.signal_id = p_id;
	msg.slave_id = d_id; //NI
	msg.source_id = NI_id; // NI
	msg.msgtype = type; // 0 1 2 3

	msg.authorMSGPayload.clear();

	//int tempDataCount = FLIT_LENGTH/valueBytes; //32 bytes /2 bytes per data
	//msg.authorMSGPayload.insert(msg.authorMSGPayload.end(), inbuffer.begin(), inbuffer.end());
	if (msg.msgtype == 0) {
		// Request message padding
		msg.authorMSGPayload.assign(payloadElementNum, 0);

#ifdef  PADDING_RANDOM
		// Use random padding instead of zeros
		static bool warning_printed = false;
		if (!warning_printed) {
			cout << "WARNING: PADDING_RANDOM enabled - using random values for padding instead of zeros" << endl;
			warning_printed = true;
		}
		for (int i = 0; i < payloadElementNum; i++) {
			msg.authorMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
	} else if (msg.msgtype == 2){
		// Response message padding
		msg.authorMSGPayload.assign(payloadElementNum, 0);
#ifdef PADDING_RANDOM
		// Use random padding instead of zeros
		for (int i = 1; i < payloadElementNum; i++) { // i1，[0]t_output
			msg.authorMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
		msg.authorMSGPayload[0] = t_output;
	}
	else if (msg.msgtype == 1) { //response
		//msg.authorMSGPayload.assign(FLIT_LENGTH/valueBytes-1, 1); //  15  1    256bit（32byte）/16bit（2byte）-1 = 16 -1 =15 71 ：256/32 - 1=8-1=7
		msg.authorMSGPayload.insert(msg.authorMSGPayload.end(), inbuffer.begin() + 3,
				inbuffer.end());		 //inbuffer.end() //inbuffer.begin()+18
		//cout<<" maccpp check msg.authorMSGPayload.size before grid "<< msg.authorMSGPayload.size()<<endl;
		//int flitNumSinglePacket = (msg.authorMSGPayload.size()) 		/ (payloadElementNum) + 1;
		  int flitNumSinglePacket = (msg.authorMSGPayload.size() -1 + payloadElementNum) / (payloadElementNum) ;

		  /*
		// DEBUG: Fill padding with random values instead of zeros
		int padding_size = flitNumSinglePacket * payloadElementNum - msg.authorMSGPayload.size();
		for (int i = 0; i < padding_size; i++) {
			float tempRandom = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
		//	msg.authorMSGPayload.push_back(tempRandom);
			msg.authorMSGPayload.push_back(0.0f);
		}
		*/
		// cout<<" msg.authorMSGPayload.size() "<<msg.authorMSGPayload.size()<<" padding_size "<< padding_size <<" msg.msgdata_length "<<msg.msgdata_length<<endl;
		// Original code - fill with zeros (commented out for debug)

		std::fill_n(std::back_inserter(msg.authorMSGPayload),
				(flitNumSinglePacket * payloadElementNum
						 - msg.authorMSGPayload.size()), 0.0f);

		//cout<<" maccpp check msg.authorMSGPayload.size after grid "<< msg.authorMSGPayload.size()<<endl;



#ifdef AuthorAffiliatedOrdering
		if (inbuffer[0] != 8)		 //if(not pooling  )
				{
			cnnReshapeFlatToInputWeightMatrix(msg.authorMSGPayload,
					inbuffer[2] * inbuffer[1]   /* t_inputCount */,
					inbuffer[2]
							* inbuffer[1]+1    /*weights used,inbuffer[2]= 5x5=25, inbuffer[1]= inputchannel=3, for example*/,
					8/*input in one row*/, 8/*weight in one row*/,
					16 /*total in one row*/,
					flitNumSinglePacket/*how many rows/flits*/); /// four flits for 32/2-1=15, 15*4>51
		}
		if (inbuffer[0] == 8)		 // pooling, no weights  // pool is only 2x2 so just put in input part (8 floating point value) is ok.
				{
			cnnReshapeFlatToInputWeightMatrix(msg.authorMSGPayload,
					inbuffer[2] * inbuffer[1] /* */, 0 /* 0weight for pooling */,
					8/*input in one row*/, 8/*weight in one row*/,
					16 /*total in one row*/,
					flitNumSinglePacket/*how many rows/flits*/); /// four flits for 32/2-1=15, 15*4>51
			// cout<< " maccpp pool inbuffer[2] * inbuffer[1] " <<  inbuffer[2] * inbuffer[1]<<endl;

		}
		// 32 /4 -1 = 7. 7*8=56>51

		//	cout<<msg.authorMSGPayload.size() << " beforefltordering "<<"msg.authorMSGPayload.front() "<<msg.authorMSGPayload.front()<<" " << msg.authorMSGPayload.back() <<endl;

		//	cout<<msg.authorMSGPayload.size() << " afterfltordering "<<"msg.authorMSGPayload.front() "<<msg.authorMSGPayload.front()<<" " << msg.authorMSGPayload.back() <<endl;
#endif
	} else
		cout << " msg.msgtype wierd " << msg.msgtype << endl;
	//assert (false);

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;

#ifdef binaryroutingSwitch
	// Response packets alternate between routing 1 and 2
	if (packet->message.msgtype == 1) {  // msgtype 1 = response packets
		// Use the next routing mode (toggle between 1 and 2)
		packet->xyroutingBool = (lastResponseRouting == 1) ? 2 : 1;
		// Update for next response packet
		lastResponseRouting = packet->xyroutingBool;
	}
	// Request packets keep default (0)
#endif

	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(
			packet);
	return true;
}


















/******888*/
void MAC::runOneStep() {

	// output stationary (neuron based calculation)
	if (pecycle < cycles) {
		// initial idle state
		int stats1SigID;
		int stats1SigIDplus2;
		if (selfstatus == 0) { // status = 0 is IDLE. routing table is not zero, status = 1, mac is running
			if (cnn_task_queue.size() == 0) {
				selfstatus = 0;
				pecycle = cycles;
			} else {
#ifdef fireAdvance
				// （）
				if (total_tasks == 0) {
					total_tasks = cnn_task_queue.size();
					requests_sent = 0;
					responses_received = 0;
					tasks_completed = 0;
				}
#endif
				pecycle = cycles;
				selfstatus = 1;
			}
		}
		// request data state
		else if (selfstatus == 1) {	// now is 1, we need to send request and wait for response. After sending requst and before recv response is status2.
			cnn_current_layer_task_id = cnn_task_queue.front();
			cnn_saved_task_id = cnn_current_layer_task_id;       // ID //taskid
			cnn_task_queue.pop_front();
			//send_request(), fill inbuffer type 0
			// Debug: Track Pooling requests
			if (net->cnnmodel->all_layer_type[net->current_layerSeq] == 'p') {
				cout << "[DEBUG] MAC " << selfMACid << " sending POOLING request"
				     << " signalid=" << (packet_id + cnn_current_layer_task_id)
				     << " Layer=" << net->current_layerSeq
				     << " cycle=" << cycles << endl;
			}
			// Debug: Track signalid allocation
			int signal_id = packet_id + cnn_current_layer_task_id;
			if (signal_id == 4704 || (signal_id >= 4700 && signal_id <= 4710)) {
				cout << "[SIGNALID] MAC " << selfMACid
				     << " using signalid=" << signal_id
				     << " (packet_id=" << packet_id
				     << " + task_id=" << cnn_current_layer_task_id << ")"
				     << " Layer=" << net->current_layerSeq
				     << " at cycle " << cycles << endl;
			}
			inject(0, dest_mem_id, 1, cnn_current_layer_task_id, net->vcNetwork->NI_list[NI_id],
					signal_id, selfMACid); //taskid

#ifdef fireAdvance
			// Fire Advance:
			requests_sent++;
#endif

			selfstatus = 2;
			pecycle = cycles;
#ifdef SoCC_Countlatency
			//statistics
			stats1SigID = (packet_id + cnn_saved_task_id) * 3;

			DNN_latency[stats1SigID][0] = net->current_layerSeq; //DNN_authorlatency[x][0]	//net->current_layerSeq+1000;
			DNN_latency[stats1SigID][1] = 0; //DNN_authorlatency[x][1] type 0 req
			DNN_latency[stats1SigID][2] = selfMACid; //DNN_authorlatency[x][2] macsrcID
			DNN_latency[stats1SigID][3] = pecycle; //DNN_authorlatency[x][3]	// request(packet1)

#endif
		} else if (selfstatus == 2) {
			if (cnn_current_layer_task_id >= 0) { // if request is still active/waitting after sending request, just return (continue waitting and do nothing)
				pecycle = cycles;
				selfstatus = 2;
				return;
			}
			assert(
					(inbuffer.size() >= 4)
							&& "Inbuffer not correct after cnn_current_layer_task_id is set to 0");

			// inbuffer: [fn]
			fn = inbuffer[0];
			//cout << cycles << " authordebug inbuffer.size  line 218  " << selfMACid
			//		<< " " << inbuffer.size() << endl;
			if (fn >= 0 && fn <= 3) { // Conv [fn] [ch size] [map size] [inputActivation] [w + b]
				ch_size = inbuffer[1]; //in_ch
				m_size = inbuffer[2]; //w_x * w_y
				infeature.assign(inbuffer.begin() + 3,
						inbuffer.begin() + 3 + ch_size * m_size); //inputforConv
				weight.assign(inbuffer.begin() + 3 + ch_size * m_size,
						inbuffer.end()); // w matrix + b (ch_size * m_size + 1)
				assert(
						(weight.size() == ch_size * m_size + 1)
								&& "Weight not correct after cnn_current_layer_task_id (Conv)");
			} else if (fn >= 4 && fn <= 7) // fcDense [fn] [map size] [i] [w + b]
					{
				ch_size = 1;
				m_size = inbuffer[2]; //w_x * w_y
				infeature.assign(inbuffer.begin() + 3,
						inbuffer.begin() + 3 + m_size); //Anonymous: inputforDense
				weight.assign(inbuffer.begin() + 3 + m_size, inbuffer.end()); //Anonymous: weighforDense  //w + b
			} else if (fn == 8) // max pooling [fn] [map size] [i]
					{
				ch_size = 1; // also  inbuffer[1]
				m_size = inbuffer[2]; //w_x * w_y
				infeature.assign(inbuffer.begin() + 3, inbuffer.end());
				assert(
						(infeature.size() == m_size)
								&& "Inbuffer not correct after cnn_current_layer_task_id (pooling)");
			}
			outfeature = 0.0;
			selfstatus = 3;

			pecycle = cycles;
			return;
		} else if (selfstatus == 3) {
			// normal MAC op
			if (fn >= 0 && fn <= 3) // Conv
					{
				for (int i = 0; i < ch_size; i++) {
					for (int j = 0; j < m_size; j++) {
						//outfeature += infeature[i*m_size + j] * weight[i*(m_size+1) + j];
						outfeature += infeature[i * m_size + j]
								* weight[i * m_size + j];
					}
					//outfeature += weight[i*m_size + m_size];
				}
				outfeature += weight[ch_size * m_size]; //bias only added once per output channel
			} else if (fn >= 4 && fn <= 7) // FC
					{
				for (int j = 0; j < m_size; j++) {
					outfeature += infeature[j] * weight[j];
				}
				outfeature += weight[m_size];
			} else if (fn == 8) // max pooling
					{
				outfeature = infeature[0];
				for (int j = 1; j < m_size; j++) {
					if (infeature[j] > outfeature) {
						outfeature = infeature[j];
					}
				}
				selfstatus = 4; // ready for this computation
				pecycle = cycles + 1; // sync cycles

				inject(2, dest_mem_id, 1, outfeature,
						net->vcNetwork->NI_list[NI_id],
						packet_id + cnn_saved_task_id, selfMACid);
#ifdef SoCC_Countlatency
				//statistics
				stats1SigIDplus2 = (packet_id + cnn_saved_task_id) * 3 + 2;
				//  here is pooling
				DNN_latency[stats1SigIDplus2][0] = net->current_layerSeq;//DNN_authorlatency[x+2][2]			// current_layerSeq+3000;
				DNN_latency[stats1SigIDplus2][1] = 2;
				DNN_latency[stats1SigIDplus2][2] = selfMACid;
				DNN_latency[stats1SigIDplus2][3] = pecycle;
				int mac_id_pool = DNN_latency[stats1SigIDplus2][2];
				int delay_add_pool = DNN_latency[stats1SigIDplus2][3] - DNN_latency[stats1SigIDplus2 - 1][7];
				samplingWindowDelay[mac_id_pool] += delay_add_pool;
				// Debug print
				cout << "[LAT_ADD] MAC.cpp:406 Pooling MAC " << mac_id_pool
				     << " += " << delay_add_pool
				     << " (total=" << samplingWindowDelay[mac_id_pool] << ")" << endl;
				samplingAccumlatedCounter += 1;

#endif
				//packet_id++;
				return;
			}

			int calctime = (ch_size * m_size / PE_NUM_OP + 1) * 10;
			/*
			if (net->current_layerSeq == 1 || net->current_layerSeq == 3) {  //

			          cout << "\n=== Calctime Debug ===" << endl;
			          cout << "Layer " << net->current_layerSeq
			               << " (type: " <<
			  net->cnnmodel->all_layer_type[net->current_layerSeq] << ")" << endl;
			          cout << "MAC " << selfMACid << " Task " <<
			  cnn_current_layer_task_id << ":" << endl;
			          cout << "  ch_size (in_ch) = " << ch_size << endl;
			          cout << "  m_size (w_x*w_y) = " << m_size << endl;
			          cout << "  PE_NUM_OP = " << PE_NUM_OP << endl;
			          cout << "  Formula: (" << ch_size << " * " << m_size << " / " <<
			        		  PE_NUM_OP << " + 1) * 10" << endl;

			          cout << "  calctime = " << calctime << " cycles" << endl;
			          cout << "  Current cycle: " << cycles << endl;
			          cout << "  Will complete at: " << (cycles + calctime) << endl;
			          cout << "===================" << endl;
			      }
			 	 */

			// activation
			if ((fn % 4) == 0) //linear
					{
				selfstatus = 4; // ready for this computation
				pecycle = cycles + calctime; // sync cycles
			} else if ((fn % 4) == 1) {
				// activation (relu)
				// cout << "from mac " << id << " output " << outfeature << endl;
				relu(outfeature);
				selfstatus = 4; // ready for output
				pecycle = cycles + calctime; // sync cycles
			} else if ((fn % 4) == 2) {
				// activation (tanh)
				tanh(outfeature);
				selfstatus = 4; // ready for output
				pecycle = cycles + calctime; // sync cycles
			} else if ((fn % 4) == 3) {
				// activation (sigmoid)
				sigmoid(outfeature);
				selfstatus = 4; // ready for output
				pecycle = cycles + calctime; // sync cycles
			} else {
				outfeature = 0.0;
				selfstatus = 0; // back to initial state
				pecycle = cycles + 2; // sync cycles
				assert((0 < 1) && "Wrong function (fn)");
				return;
			}

			// inject
#ifndef newpooling
			inject(2, dest_mem_id, 1, outfeature,
					net->vcNetwork->NI_list[NI_id], packet_id + cnn_saved_task_id,
					selfMACid); // inject type 2
			//cout<<" injectdest "<<dest_mem_id <<" "<<id<<endl;
#ifdef SoCC_Countlatency
			//statistics
			stats1SigIDplus2 = (packet_id + cnn_saved_task_id) * 3 + 2;		//result
			DNN_latency[stats1SigIDplus2][0] = net->current_layerSeq;//DNN_authorlatency[x+2][0]			// current_layerSeq+3000;
			DNN_latency[stats1SigIDplus2][1] = 2; //DNN_authorlatency[x+2][1]
			DNN_latency[stats1SigIDplus2][2] = selfMACid; //DNN_authorlatency[x+2][2]
			DNN_latency[stats1SigIDplus2][3] = pecycle; //DNN_authorlatency[x+2][3]

			int mac_id_comp = DNN_latency[stats1SigIDplus2][2];
			int delay_add_comp = DNN_latency[stats1SigIDplus2][3] - DNN_latency[stats1SigIDplus2 - 1][7];
			samplingWindowDelay[mac_id_comp] += delay_add_comp;
			// Debug print
			cout << "[LAT_ADD] MAC.cpp:482 Compute MAC " << mac_id_comp
			     << " += " << delay_add_comp
			     << " (total=" << samplingWindowDelay[mac_id_comp] << ")" << endl;
			samplingAccumlatedCounter += 1;
#endif
			//packet_id++;
#endif
			return;
		} else if (selfstatus == 4) {
#ifdef only3type
			this->send = 0;

#ifdef fireAdvance
			// Fire Advance:
			tasks_completed++;
			computing_task_id = -1;
#endif

			if (this->cnn_task_queue.size() == 0) {
				this->selfstatus = 5;
				//cout << cycles << " status=5currentPEis " << selfMACid << endl;
			} else {
				this->selfstatus = 0; 				// back to initial state
			}
			//cout << "from mac " << this->id << " output " << this->outfeature << " " << selfstatus << endl;
			this->weight.clear();
			this->infeature.clear();
			this->inbuffer.clear();
			this->outfeature = 0.0;
			this->pecycle = cycles + 1; //cycles + 1
			return;
#endif
		}
	}

#ifdef fireAdvance
	// ===== Fire Advance Logic =====
	// cyclefire advance（MAC）
	if (fire_advance_armed && fire_advance_counter > 0) {
		fire_advance_counter--;

		if (fire_advance_counter == 0) {
			// ，request
			fire_advance_armed = false;

			// ：、、REQUEST
			if (requests_sent < total_tasks &&
			    cnn_task_queue.size() > 0 &&
			    selfstatus != 1) {

				// REQUEST，MAC
				selfstatus = 1;
				pecycle = cycles;
			}
		}
	}
#endif

}

void MAC::sigmoid(float &x) // 3
		{
	x = 1.0 / (1.0 + std::exp(-x));
}

void MAC::tanh(float &x)  // 2
		{
	x = 2.0 / (1.0 + std::exp(-2 * x)) - 1;
}

void MAC::relu(float &x)  // 1
		{
	if (x < 0)
		x = 0.0;
}

// Destructor
MAC::~MAC() {

}
