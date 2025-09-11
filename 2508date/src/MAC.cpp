
/**
 * @file MAC.cpp
 * @brief CNN模式下的MAC (Multiply-Accumulate) 计算单元实现
 * 
 * 本文件实现了CNN加速器中的单个MAC计算单元。
 * MAC是CNN硬件加速的基本计算单元，执行乘累加运算：
 * output = Σ(weight[i] * input[i]) + bias
 * 
 * 主要功能：
 * 1. 权重(weight)和输入特征(input feature)的缓存管理
 * 2. 卷积、池化、全连接层的计算执行
 * 3. 与NoC的数据交互（发送请求、接收数据、输出结果）
 * 4. 支持不同的池化策略（最大池化、平均池化）
 * 
 * 数据处理流程：
 * - requestData(): 向内存节点请求weight/input数据
 * - receiveData(): 接收并缓存数据
 * - compute(): 执行MAC运算或池化操作
 * - sendOutput(): 将结果发送到下一层或输出
 * 
 * 内存节点映射：
 * - 根据NI_id自动计算对应的内存节点dest_mem_id
 * - 支持多种内存配置（MemNode2_4X4, MemNode4_4X4, MemNode8_4X4等）
 * - 采用就近原则减少NoC传输距离
 * 
 * 优化特性：
 * - 数据复用：缓存weight减少重复传输
 * - 流水线处理：计算与数据传输重叠
 * - 批量处理：一次处理多个channel提高效率
 * 
 * 与LLMMAC的区别：
 * - MAC处理规则的CNN层数据，LLMMAC处理不规则的attention矩阵
 * - MAC的数据访问模式固定，LLMMAC更加动态
 * - MAC支持多种层类型，LLMMAC专注于transformer计算
 * 
 * @note 本实现中使用了PADDING_RANDOM宏来控制padding策略
 * @note dest_list定义了内存节点的物理位置映射
 * 
 * @author YZ
 * @date 2025
 */

#include "MAC.hpp"

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
	cnn_current_layer_task_id = -1;  // 初始化CNN任务ID为空闲
	cnn_saved_task_id = -1;           // 初始化保存的任务ID

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

	// find dest id
	//这里 xid = row（行），yid = col（列）
	int xid = NI_id / X_NUM;
	int yid = NI_id % X_NUM;
	// MC nodes

#if defined MemNode2_4X4
	dest_mem_id = dest_list[(yid / 2)];
#elif defined MemNode4_4X4
	if (xid <= 1 && yid <= 1) {
		dest_mem_id = dest_list[0]; // 上左 left and upper
	} else if (xid >= 2 && yid <= 1) {// 下左
		dest_mem_id = dest_list[1];
	} else if (xid <= 1 && yid >= 2) { //上右
		dest_mem_id = dest_list[2];
	} else if ((xid >= 2 && yid >= 2)) { //下右
		dest_mem_id = dest_list[3];
	} else {
		cout << "error!line66";
	}

#elif defined MemNode4_8X8
    const int mid = X_NUM / 2;
    if (xid < mid && yid < mid) {
        dest_mem_id = dest_list[0]; // tl: 上左
    } else if (xid   >= mid && yid < mid) {
        dest_mem_id = dest_list[1]; // bl: 下左
    } else if (xid < mid && yid >= mid) {
        dest_mem_id = dest_list[2]; // tr: 上右
    } else if (xid >= mid && yid >= mid)
	{
        dest_mem_id = dest_list[3]; // br: 下右
    }else {
		cout << "error!line66";
	}
#elif defined MemNode4_16X16
    const int mid = X_NUM / 2;
    if (xid < mid && yid < mid) {
        dest_mem_id = dest_list[0]; // tl: 上左
    } else if (xid   >= mid && yid < mid) {
        dest_mem_id = dest_list[1]; // bl: 下左
    } else if (xid < mid && yid >= mid) {
        dest_mem_id = dest_list[2]; // tr: 上右
    } else if (xid >= mid && yid >= mid)
	{
        dest_mem_id = dest_list[3]; // br: 下右
    }else {
		cout << "error!line66";
	}
#elif defined MemNode4_32X32
    // 32×32 网格按中线分象限：x 是行(0..31)，y 是列(0..31)
    const int mid = X_NUM / 2; // 对 32×32 即 16
    if (xid < mid && yid < mid) {
        dest_mem_id = dest_list[0]; // tl: 上左
    } else if (xid   >= mid && yid < mid) {
        dest_mem_id = dest_list[1]; // bl: 下左
    } else if (xid < mid && yid >= mid) {
        dest_mem_id = dest_list[2]; // tr: 上右
    } else if (xid >= mid && yid >= mid)
	{
        dest_mem_id = dest_list[3]; // br: 下右
    }else {
		cout << "error!line66";
	}



#endif
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

	msg.yzMSGPayload.clear();

	//int tempDataCount = FLIT_LENGTH/valueBytes; //32 bytes /2 bytes per data
	//msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), inbuffer.begin(), inbuffer.end());
	if (msg.msgtype == 0) {
		// Request message padding
		msg.yzMSGPayload.assign(payloadElementNum, 0);

#ifdef  PADDING_RANDOM
		// Use random padding instead of zeros
		static bool warning_printed = false;
		if (!warning_printed) {
			cout << "WARNING: PADDING_RANDOM enabled - using random values for padding instead of zeros" << endl;
			warning_printed = true;
		}
		for (int i = 0; i < payloadElementNum; i++) {
			msg.yzMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
	} else if (msg.msgtype == 2){
		// Response message padding  
		msg.yzMSGPayload.assign(payloadElementNum, 0);
#ifdef PADDING_RANDOM
		// Use random padding instead of zeros
		for (int i = 1; i < payloadElementNum; i++) { // i从1开始，保留[0]位置给t_output
			msg.yzMSGPayload[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Random [-0.5, 0.5]
		}
#endif
		msg.yzMSGPayload[0] = t_output;
	}
	else if (msg.msgtype == 1) { //response
		//msg.yzMSGPayload.assign(FLIT_LENGTH/valueBytes-1, 1); // 替换为 15 个 1    256bit（32byte）/16bit（2byte）-1 = 16 -1 =15 或者7个1 ：256/32 - 1=8-1=7
		msg.yzMSGPayload.insert(msg.yzMSGPayload.end(), inbuffer.begin() + 3,
				inbuffer.end());		 //inbuffer.end() //inbuffer.begin()+18
		//cout<<" maccpp check msg.yzMSGPayload.size before grid "<< msg.yzMSGPayload.size()<<endl;
		//int flitNumSinglePacket = (msg.yzMSGPayload.size()) 		/ (payloadElementNum) + 1;
		  int flitNumSinglePacket = (msg.yzMSGPayload.size() -1 + payloadElementNum) / (payloadElementNum) ;
		
		  /*
		// DEBUG: Fill padding with random values instead of zeros
		int padding_size = flitNumSinglePacket * payloadElementNum - msg.yzMSGPayload.size();
		for (int i = 0; i < padding_size; i++) {
			float tempRandom = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
		//	msg.yzMSGPayload.push_back(tempRandom);
			msg.yzMSGPayload.push_back(0.0f);
		}
		*/
		// cout<<" msg.yzMSGPayload.size() "<<msg.yzMSGPayload.size()<<" padding_size "<< padding_size <<" msg.msgdata_length "<<msg.msgdata_length<<endl;
		// Original code - fill with zeros (commented out for debug)

		std::fill_n(std::back_inserter(msg.yzMSGPayload),
				(flitNumSinglePacket * payloadElementNum
						 - msg.yzMSGPayload.size()), 0.0f);

		//cout<<" maccpp check msg.yzMSGPayload.size after grid "<< msg.yzMSGPayload.size()<<endl;



#ifdef YzAffiliatedOrdering
		if (inbuffer[0] != 8)		 //if(not pooling  )
				{
			cnnReshapeFlatToInputWeightMatrix(msg.yzMSGPayload,
					inbuffer[2] * inbuffer[1]   /* t_inputCount */,
					inbuffer[2]
							* inbuffer[1]+1    /*weights used,inbuffer[2]= 5x5=25, inbuffer[1]= inputchannel=3, for example*/,
					8/*input in one row*/, 8/*weight in one row*/,
					16 /*total in one row*/,
					flitNumSinglePacket/*how many rows/flits*/); /// four flits for 32/2-1=15, 15*4>51
		}
		if (inbuffer[0] == 8)		 // pooling, no weights  // pool is only 2x2 so just put in input part (8 floating point value) is ok.
				{
			cnnReshapeFlatToInputWeightMatrix(msg.yzMSGPayload,
					inbuffer[2] * inbuffer[1] /* */, 0 /* 0weight for pooling */,
					8/*input in one row*/, 8/*weight in one row*/,
					16 /*total in one row*/,
					flitNumSinglePacket/*how many rows/flits*/); /// four flits for 32/2-1=15, 15*4>51
			// cout<< " maccpp pool inbuffer[2] * inbuffer[1] " <<  inbuffer[2] * inbuffer[1]<<endl;

		}
		// 32 /4 -1 = 7. 7*8=56>51

		//	cout<<msg.yzMSGPayload.size() << " beforefltordering "<<"msg.yzMSGPayload.front() "<<msg.yzMSGPayload.front()<<" " << msg.yzMSGPayload.back() <<endl;

		//	cout<<msg.yzMSGPayload.size() << " afterfltordering "<<"msg.yzMSGPayload.front() "<<msg.yzMSGPayload.front()<<" " << msg.yzMSGPayload.back() <<endl;
#endif
	} else
		cout << " msg.msgtype wierd " << msg.msgtype << endl;
	//assert (false);

	Packet *packet = new Packet(msg, X_NUM, t_NI->NI_num);
	packet->send_out_time = pecycle;
	packet->in_net_time = pecycle;
	net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(
			packet);
	return true;
}


















/******888别误动下面的*/
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
				pecycle = cycles;
				selfstatus = 1;
			}
		}
		// request data state
		else if (selfstatus == 1) {	// now is 1, we need to send request and wait for response. After sending requst and before recv response is status2.
			cnn_current_layer_task_id = cnn_task_queue.front();  // 从队列取出输出通道索引
			cnn_saved_task_id = cnn_current_layer_task_id;       // 保存任务ID副本 //taskid
			cnn_task_queue.pop_front();
			//send_request(), fill inbuffer type 0
			inject(0, dest_mem_id, 1, cnn_current_layer_task_id, net->vcNetwork->NI_list[NI_id],
					packet_id + cnn_current_layer_task_id, selfMACid); //taskid
			selfstatus = 2;
			pecycle = cycles;
#ifdef SoCC_Countlatency
			//statistics
			stats1SigID = (packet_id + cnn_saved_task_id) * 3;

			DNN_latency[stats1SigID][0] = net->current_layerSeq; //DNN_yzlatency[x][0]	//net->current_layerSeq+1000;
			DNN_latency[stats1SigID][1] = 0; //DNN_yzlatency[x][1] type 0 req
			DNN_latency[stats1SigID][2] = selfMACid; //DNN_yzlatency[x][2] macsrcID
			DNN_latency[stats1SigID][3] = pecycle; //DNN_yzlatency[x][3]	// request(packet1)

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
			//cout << cycles << " yzdebug inbuffer.size  line 218  " << selfMACid
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
						inbuffer.begin() + 3 + m_size); //yz:inputforDense
				weight.assign(inbuffer.begin() + 3 + m_size, inbuffer.end()); //yz:weighforDense  //w + b
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
				DNN_latency[stats1SigIDplus2][0] = net->current_layerSeq;//DNN_yzlatency[x+2][2]			// current_layerSeq+3000;
				DNN_latency[stats1SigIDplus2][1] = 2;
				DNN_latency[stats1SigIDplus2][2] = selfMACid;
				DNN_latency[stats1SigIDplus2][3] = pecycle;
				samplingWindowDelay[DNN_latency[stats1SigIDplus2][2]] +=
						DNN_latency[stats1SigIDplus2][3]
								- DNN_latency[stats1SigIDplus2 - 1][7];
				samplingAccumlatedCounter += 1;

#endif
				//packet_id++;
				return;
			}

			int calctime = (ch_size * m_size / PE_NUM_OP + 1) * 10;
			//cout<<calctime  <<" line 287 "<<ch_size   <<" "<< m_size <<endl;

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
			DNN_latency[stats1SigIDplus2][0] = net->current_layerSeq;//DNN_yzlatency[x+2][0]			// current_layerSeq+3000;
			DNN_latency[stats1SigIDplus2][1] = 2; //DNN_yzlatency[x+2][1]
			DNN_latency[stats1SigIDplus2][2] = selfMACid; //DNN_yzlatency[x+2][2]
			DNN_latency[stats1SigIDplus2][3] = pecycle; //DNN_yzlatency[x+2][3]

			samplingWindowDelay[DNN_latency[stats1SigIDplus2][2]] +=
					DNN_latency[stats1SigIDplus2][3]
							- DNN_latency[stats1SigIDplus2 - 1][7];
			samplingAccumlatedCounter += 1;
#endif
			//packet_id++;
#endif
			return;
		} else if (selfstatus == 4) {
#ifdef only3type
			this->send = 0;
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
