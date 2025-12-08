/**
 * @file MACnet.cpp
 * @brief CNN (Convolutional Neural Network) MAC网络实现
 * 
 * 本文件实现了CNN模式下的MAC（Multiply-Accumulate）网络管理器。
 * MACnet是CNN加速器的核心控制组件，负责协调多个MAC单元完成CNN推理。
 * 
 * =============================
 * 主要执行步骤 (Step Functions)
 * =============================
 * 
 * Step() 函数是MACnet的核心调度器，每个时钟周期执行一次：
 * 
 * 1. **数据初始化阶段** (cycles == 1)
 *    - 调用 preparelayer() 准备当前层数据
 *    - 初始化 input_table、weight_table、output_table
 *    again
 *
 *    - 根据层类型（conv/pool/FC）配置参数
 * 
 * 2. **MAC分配阶段** (mapping)
 *    - xmapping(): 按行映射neurons到MAC单元
 *    - ymapping(): 按列映射neurons到MAC单元
 *    - 将neurons均匀分配给各MAC单元
 * 
 * 3. **数据请求阶段** (MAC发送请求)
 *    - MAC单元向内存节点发送type 0消息请求数据
 *    - 请求weight数据（卷积核参数）
 *    - 请求input数据（输入特征图）
 *    - 通过NoC发送Packet到对应内存节点
 * 
 * 4. **数据接收阶段** (处理type 1响应)
 *    - 内存节点返回type 1消息携带数据
 *    - MAC接收并缓存weight和input数据
 *    - 检查数据完整性，准备计算
 * 
 * 5. **计算执行阶段** (MAC compute)
 *    - 执行MAC运算：output += weight * input
 *    - 卷积层：滑动窗口卷积计算
 *    - 池化层：最大/平均池化
 *    - FC层：矩阵乘法运算
 *    - 激活函数：ReLU/Sigmoid等
 * 
 * 6. **结果输出阶段** (发送type 2消息)
 *    - MAC计算完成后生成type 2消息
 *    - 结果写入output_table供下一层使用
 *    - 最后一层输出到指定内存节点
 * 
 * 7. **层切换阶段** (layer transition)
 *    - 检查当前层是否完成（used_pe == 0）
 *    - current_layerSeq++切换到下一层
 *    - output_table变为下一层的input_table
 *    - 重复步骤1-6直到所有层完成
 * 
 * =============================
 * 关键数据结构
 * =============================
 * 
 * - input_table[channel][data]: 输入特征图
 * - weight_table[och*ich + j][kernel]: 卷积核权重
 * - output_table[channel][data]: 输出特征图
 * - mapping_table[mac_id][neuron_ids]: MAC-neuron映射表
 * - MAC_list[]: 所有MAC单元的列表
 * 
 * =============================
 * 消息类型处理
 * =============================
 * 
 * - Type 0 (Request): MAC请求数据
 *   格式：{src_id, dest_mem_id, data_addr, request_type}
 *   
 * - Type 1 (Response): 内存返回数据
 *   格式：{mem_id, dest_mac_id, data_payload[16]}
 *   
 * - Type 2 (Result): MAC输出结果
 *   格式：{mac_id, dest_id, result_data, layer_info}
 * 
 * =============================
 * 优化策略
 * =============================
 * 
 * - Weight复用：同一卷积核在多个位置使用时缓存
 * - 流水线并行：计算与数据传输重叠执行
 * - 负载均衡：根据MAC延迟动态调整任务分配
 * - Padding策略：使用PADDING_RANDOM减少边界效应
 * 
 * =============================
 * 与LLM模式的关键区别
 * =============================
 * 
 * CNN模式特点：
 * - 层级顺序处理，数据流规律可预测
 * - Weight参数固定，可大量复用
 * - 数据访问模式规则（卷积窗口滑动）
 * - Bit flipping较少（12-22 bits随机分布）
 * 
 * LLM模式特点：
 * - 任务级并行处理，数据流动态变化
 * - Attention矩阵每次不同，无法复用
 * - 数据访问模式不规则（attention pattern）
 * - Bit flipping较多（23→13 bits梯度排序）
 * 
 * @author YZ
 * @date 2025
 */

#include "MACnet.hpp"
#include "MAC.hpp" // 確保這一行存在
#include <cassert>
// helper function
template<class C, typename T>
bool contains(C &&c, T e) {
	return find(begin(c), end(c), e) != end(c);
}
;

MACnet::MACnet(int mac_num, int t_pe_x, int t_pe_y, Model *m,
		VCNetwork *t_Network) {
	macNum = mac_num;
	MAC_list.reserve(mac_num);
	pe_x = t_pe_x;
	pe_y = t_pe_y;
	cnnmodel = m;
	vcNetwork = t_Network;

	current_layerSeq = 0;
	n_layer = cnnmodel->all_layer_size.size();
	used_pe = 0;
	o_fnReluOrPool = 0;
	int temp_ni_id;
	cout << "Layer in total " << n_layer << endl;

	for (int i = 0; i < macNum; i++) { // see ppt
		temp_ni_id = i % TOT_NUM;
		//cout << temp_ni_id << " " << i << endl;
		MAC *nMAC = new MAC(i, this, temp_ni_id); // different from NN compute
		MAC_list.push_back(nMAC);

	}

	// layer 0: input
	lastLayerPacketID = 0;

	deque<int> layer_info;
	if (cnnmodel->all_layer_type[current_layerSeq] != 'i') // 0 layer
			{
		cout << "err: first layer is not input" << endl;
		//return 0;
	}
	layer_info = cnnmodel->all_layer_size[current_layerSeq];
	in_x = layer_info[1]; // in_x
	in_y = layer_info[2]; // in_y
	in_ch = layer_info[3]; // in_ch
	current_layerSeq++;

	// for new pooling
	no_x = 0;
	no_y = 0;
	nw_x = 0;
	nw_y = 0;
	no_ch = 0; // next o_ch in pooling layer
	npad = 0; // padding
	nstride = 1; // stride

	// layer 1: 1st conv layer
	layer_info = cnnmodel->all_layer_size[current_layerSeq];
	if (cnnmodel->all_layer_type[current_layerSeq] == 'c') {
		w_x = layer_info[1]; // w_x
		w_y = layer_info[2]; // w_y
		o_ch = layer_info[3]; // o_ch
		w_ch = o_ch * in_ch; // w_ch = o_ch * in_ch
		o_fnReluOrPool = layer_info[5];
		pad = layer_info[6]; // padding
		stride = layer_info[7]; // stride
		cout << " layer_info[4]  " << layer_info[4] << "  layer_info[3] "
				<< layer_info[3] << " " << in_ch << ' ' << w_ch << endl;
		assert((in_ch == layer_info[4]) && "Input channel not correct!");
	}
	st_w = 0;
	// for 1st conv layer output
	o_x = (in_x + 2 * pad - w_x) / stride + 1;
	o_y = (in_y + 2 * pad - w_y) / stride + 1;
	readyflag = 0; //standby
	cout << "!!MACnet created!!" << endl;
	cout << "yzzzzprintlayer " << current_layerSeq << " created "
			<< cnnmodel->all_layer_type[current_layerSeq] << " in_ch " << in_ch
			<< " o_ch  " << o_ch << " outNeruons " << (o_ch * o_x * o_y)
			<< " currentpacket_id " << packet_id << endl;

	Layer_latency.clear();

	executedTask = 0;
}



void MACnet::create_input() {
	input_table.resize(in_ch);
	int outmatsize = o_x * o_y;
	int wmatsize = w_x * w_y;
	int padded_x = in_x + 2 * pad;
	int padded_y = in_y + 2 * pad;
	weight_table.resize(w_ch);

	// for input
	if (this->current_layerSeq == 1) // 1st conv
			{
		// add padding with in_x, in_y
		for (int i = 0; i < in_ch; i++) {
			if (pad == 0) {
				input_table[i].assign(this->cnnmodel->all_data_in[i].begin(),
						this->cnnmodel->all_data_in[i].end());
			} else {
				input_table[i].assign(padded_x * padded_y, 0.0);
				for (int p = 0; p < in_y; ++p) {
					for (int q = 0; q < in_x; ++q) {
						input_table[i][(p + pad) * padded_x + (q + pad)] =
								this->cnnmodel->all_data_in[i][p * in_x + q];
					}
				}
			}
		}
	} else if (this->current_layerSeq >= 2) {
		for (int i = 0; i < in_ch; i++) {

			if (pad == 0) {
				input_table[i].assign(this->output_table[i].begin(),
						this->output_table[i].end());
				//check1
				if (this->cnnmodel->all_layer_type[current_layerSeq] == 'f') {
					//cout << "check1 " << i << " " << input_table[i].size() << " " << this->output_table[i].size() << endl;
				}
			} else {
				input_table[i].assign(padded_x * padded_y, 0.0);
				for (int p = 0; p < in_y; ++p) {
					for (int q = 0; q < in_x; ++q) {
						input_table[i][(p + pad) * padded_x + (q + pad)] =
								this->output_table[i][p * in_x + q];
					}
				}
			}
		}
	}

	// for weight
	if (this->cnnmodel->all_layer_type[current_layerSeq] == 'c') {
		//***************************
		for (int i = 0; i < o_ch; i++) {
			for (int j = 0; j < in_ch; j++) {
				weight_table[i * in_ch + j].assign(
						this->cnnmodel->all_weight_in[st_w + i].begin()
								+ j * wmatsize,
						this->cnnmodel->all_weight_in[st_w + i].begin()
								+ j * wmatsize + wmatsize);	//weight filter
				weight_table[i * in_ch + j].push_back(
						this->cnnmodel->all_weight_in[st_w + i].back());//bias
			}
		}
		st_w += o_ch;
	} else if (this->cnnmodel->all_layer_type[current_layerSeq] == 'f') {
		//check2
		//cout << "check2 " << st_w << " " << w_ch << " " << this->cnnmodel->all_weight_in[st_w].size() << endl;
		for (int i = 0; i < w_ch; i++) {
			weight_table[i].assign(
					this->cnnmodel->all_weight_in[st_w + i].begin(),
					this->cnnmodel->all_weight_in[st_w + i].end());
		}
		st_w += w_ch;
		//check3
		//cout << "check3 " << st_w << " " << weight_table[0].size() << " " << this->cnnmodel->all_weight_in[st_w-1].size() << endl;
	} else if (this->cnnmodel->all_layer_type[current_layerSeq] == 'p') {
		weight_table.clear();
	}

	// for output

	output_table.resize(o_ch);
	for (int i = 0; i < o_ch; i++) {
		output_table[i].assign(outmatsize, 0.0);
	}

	return;
}

// default direct x mapping
void MACnet::xmapping(int neuronnum) {
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	//dir_x, except dest_list

	int j = 0;
	while (j < neuronnum) {
		for (int i = 0; i < macNum; i++) {
			int temp_i = i % TOT_NUM;
			if (contains(dest_list, temp_i)) {
				continue;
			}

			this->mapping_table[i].push_back(j);
			j = j + 1;
			if (j == neuronnum)
				break;
		}
	}
	cout << " line209 xmapping_ Jis " << j << " " << endl;
	for (int i = 0; i < macNum; i++) {
		cout << "xmappingthis->mapping_table[i]size " << i << " "
				<< this->mapping_table[i].size() << endl;
	}
	return;
}






















void MACnet::checkStatus() {

	if (readyflag == 0) // every new layer
			{
		this->vcNetwork->resetVNRoundRobin(); //everylayer,reset vn rr
		this->create_input();
#ifdef rowmapping
		this->xmapping(o_ch * o_x * o_y);
#endif
#ifdef colmapping
			this->ymapping(o_ch * o_x * o_y);
#endif
#ifdef randmapping
			this->rmapping(o_ch * o_x * o_y);
#endif
#ifdef YZrandmapping
			this->yzrmapping(o_ch * o_x * o_y);
#endif
#ifdef YZDistanacemapping
			this->yzDistancemapping(o_ch * o_x * o_y);
#endif
		for (int i = 0; i < macNum; i++) {
			if (mapping_table[i].size() == 0) {
				this->MAC_list[i]->selfstatus = 5;
#ifdef only3type
				this->MAC_list[i]->send = 3;
#endif

			} else {
				this->MAC_list[i]->cnn_task_queue.assign(
						mapping_table[i].begin(), mapping_table[i].end()); //mapping table - 分配输出通道任务
			}
		}
		readyflag = 1; // loading complete
		return;
	}

// previous only happens when new layer. below happens every cycle
// if this layer is not completed, keep ready flag=1, do nothing and return.
	for (int i = 0; i < macNum; i++) {
		if (MAC_list[i]->selfstatus != 5) {
			readyflag = 1;
			return;
		}
#ifdef only3type
		else {

			if (MAC_list[i]->send != 3) {
				//cout <<  MAC_list[i]->send << endl;
				readyflag = 1;
				return;
			}
		}
#endif
	}
// after layer complete, fetch new layer
	deque<int> layer_info;
	in_x = o_x; // in_x
	in_y = o_y; // in_y
	in_ch = o_ch; // in_ch

	current_layerSeq++; // go to next layer normally

	if (current_layerSeq == n_layer) {
		cout << " \n All finished! at cycle " << cycles << " packetid "
				<< packet_id << " yzLastSeenPid " << yzLastSeenPid << endl;
		Layer_latency.push_back(cycles);
		cout << "Latency for all layers: " << endl;

		for (auto element : Layer_latency) {
			cout << element << endl;
		}
		cout << endl;

		cout << "Latency for each layer: " << endl;
		cout << Layer_latency[0] << endl;
		for (int lat = 0; lat < Layer_latency.size() - 1; lat++) {
			cout << Layer_latency[lat + 1] - Layer_latency[lat] << endl;
		}
		cout << endl;

		readyflag = 2;
		cout << "debug packetid1395 " << packet_id << endl;
		packet_id = packet_id + o_ch * o_x * o_y;
		cout << "debug packetid1407 " << packet_id << "  o_ch " << o_ch
				<< " o_x " << o_x << " o_y " << o_y << endl;
		lastLayerPacketID = packet_id;

		cout << "this layer ends  below is all layer ends \n" << endl;
		return;
	} else {
		cout << "intermediate Layer finished " << (current_layerSeq - 1)
				<< " at cycle " << cycles << endl;
		Layer_latency.push_back(cycles);
		packet_id = packet_id + o_ch * o_x * o_y;

		lastLayerPacketID = packet_id;

		cout << "now have sent packetID" << packet_id
				<< " this layer is finished  \n" << endl;
	}

//fetch new layer
	layer_info = cnnmodel->all_layer_size[current_layerSeq];

	if (cnnmodel->all_layer_type[current_layerSeq] == 'c') { // for conv layer output
		w_x = layer_info[1]; // w_x
		w_y = layer_info[2]; // w_y
		o_ch = layer_info[3]; // o_ch
		w_ch = o_ch * in_ch; // w_ch = o_ch * in_ch
		o_fnReluOrPool = layer_info[5]; // 0 to 3
		pad = layer_info[6]; // padding
		stride = layer_info[7]; // stride
		if (in_ch != layer_info[4]) {
			std::cerr << "[Config error] layer " << current_layerSeq
					<< ": in_ch(runtime)=" << in_ch << ", in_ch(config)="
					<< layer_info[4] << " (prev o_ch must equal this in_ch)\n";
		}
		assert((in_ch == layer_info[4]) && "Input channel not correct!");
		o_x = (in_x + 2 * pad - w_x) / stride + 1; //this is the output matrix size
		o_y = (in_y + 2 * pad - w_y) / stride + 1;
		cout << " " << endl;
		cout << " " << endl;
		cout << " " << endl;
		cout << "conv print newlayer atcycle " << cycles << " packetd_id "
				<< packet_id << " yzLastSeenPid " << yzLastSeenPid << " layer"
				<< current_layerSeq << " "
				<< cnnmodel->all_layer_type[current_layerSeq] << " in_ch  "
				<< in_ch << " ofmap " << (o_ch * o_x * o_y) << endl;
	} else if (cnnmodel->all_layer_type[current_layerSeq] == 'f') // for fc layer output
			{
		// in_x = in_x * in_y * in_ch;
		in_x = layer_info[0];
		in_ch = 1;
		in_y = 1;
		w_x = layer_info[0]; // = in_x
		w_y = 1; // w_y
		o_ch = 1; // o_ch
		w_ch = layer_info[1]; // = o_x
		o_fnReluOrPool = layer_info[2] + 4; // 4 to 7
		pad = 0;
		stride = 1;
		assert((in_x == w_x) && "Input channel not correct!");
		o_x = layer_info[1];
		o_y = 1;
		cout << " " << endl;
		cout << " " << endl;
		cout << "cyclesare " << cycles << " packet_id " << packet_id
				<< " yzLastSeenPid " << yzLastSeenPid << " layer"
				<< current_layerSeq << " "
				<< cnnmodel->all_layer_type[current_layerSeq] << ' ' << in_x
				<< ' ' << w_x << ' ' << o_x << endl;
		if (this->output_table.size() > 1) //flatten
				{
			vector<float> temp_out_table;
			for (int z = 0; z < this->output_table.size(); z++) {
				temp_out_table.insert(temp_out_table.end(),
						this->output_table[z].begin(),
						this->output_table[z].end());
			}
			this->output_table.resize(1);
			this->output_table[0].assign(temp_out_table.begin(),
					temp_out_table.end());
			//cout << "tag 2 " << temp_out_table.size() << " " << this->output_table[0].size() << endl;
		}
	} else if (cnnmodel->all_layer_type[current_layerSeq] == 'p') // for max pooling layer output
			{
		w_x = layer_info[1];
		w_y = layer_info[2];
		o_ch = layer_info[3]; // o_ch
		pad = layer_info[4]; // padding
		stride = layer_info[5]; // stride
		w_ch = 0;
		o_fnReluOrPool = 8;
		assert((in_ch == o_ch) && "Input channel not correct!");
		o_x = (in_x + 2 * pad - w_x) / stride + 1;
		o_y = (in_y + 2 * pad - w_y) / stride + 1;
		cout << " " << endl;
		cout << " " << endl;
		cout << "cyclesare " << cycles << " packet_id " << packet_id << " layer"
				<< current_layerSeq << " yzLastSeenPid " << yzLastSeenPid << " "
				<< cnnmodel->all_layer_type[current_layerSeq] << ' ' << in_ch
				<< ' ' << (o_ch * o_x * o_y) << endl;
	}

	readyflag = 0;
// reset Mac status
	for (int i = 0; i < macNum; i++) {
		MAC_list[i]->selfstatus = 0;
		//added hard sync
		MAC_list[i]->pecycle = cycles;
	}

}






void MACnet::runOneStep() {
	MAC *tmpMAC;
	NI *tmpNI;
	Packet *tmpPacket;
	for (int i = 0; i < macNum; i++) { // run one step for each MAC
		// cout <<  "mac:before " << i << ' ' << cycles << endl;
		MAC_list[i]->runOneStep();
		// cout <<  "mac:done " << i << ' ' << cycles << endl;
	}

// check MEM, MEM id is from dest_list

	int pbuffersize;
	int src;
	int pidSignalID;
	int mem_id;
	int src_mac;
	for (int memidx = 0; memidx < MEM_NODES; memidx++) {
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];
		// for message type 0 from MAC to MEM
		pbuffersize = tmpNI->packet_buffer_out[0].size();
		for (int j = 0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			// added check if reached out cycle
			// check received packet at MEM from MAC type 0
			if (tmpPacket->message.msgtype != 0
					|| tmpPacket->message.out_cycle >= cycles) {
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}
			src = tmpPacket->message.source_id;
			pidSignalID = tmpPacket->message.signal_id;
			yzLastSeenPid = pidSignalID;
			src_mac = tmpPacket->message.mac_id;

#ifdef SoCC_Countlatency
			//statistics //this is packet2(response from mem)
			DNN_latency[pidSignalID * 3][4] = tmpPacket->send_out_time;
			DNN_latency[pidSignalID * 3][7] = cycles;			//cycles+2000;

			DNN_latency[pidSignalID * 3 + 1][1] = 1;
			DNN_latency[pidSignalID * 3 + 1][2] = src_mac;
			DNN_latency[pidSignalID * 3 + 1][3] = cycles;
			//cout<<" tmpMAC->cnn_current_layer_task_idline1868 "<<tmpMAC->cnn_current_layer_task_id<<endl;
#endif
			// cout<<cycles << " MEM " << tmpPacket->message.destination << " receive type " << tmpPacket->message.msgtype << " from MAC " << src << endl;
			tmpMAC = MAC_list[src_mac];
			if (this->cnnmodel->all_layer_type[current_layerSeq] == 'c') { // conv layer fetch data
				if (tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[current_layerSeq]=='c'
						{
					tmpMAC->tmpch = tmpMAC->cnn_current_layer_task_id / (o_x * o_y); //current output channel
					tmpMAC->tmpm = tmpMAC->cnn_current_layer_task_id % (o_x * o_y); //current output map id
					tmpMAC->npoolflag = 0;
					int tmpx = tmpMAC->tmpm % o_x;
					int tmpy = tmpMAC->tmpm / o_x;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [ch size] [map size] [i] [w + b]
					tmpMAC->inbuffer.push_back(o_fnReluOrPool);
					tmpMAC->inbuffer.push_back(in_ch);
					tmpMAC->inbuffer.push_back(w_x * w_y);

					// for conv input
					// normal allocate inputs from input table
					// assuming  in_ch=6 input channels, kernel=3x3: 3 input "figures", pick up 3 rows, one row contains 3 data. o
					// overall = 6x3x3 floating point inputs
					for (int k = 0; k < in_ch; k++) {
						for (int p = 0; p < w_y; p++) {
							tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),
									this->input_table[k].begin()
											+ (tmpy * stride + p)
													* (in_x + 2 * pad)
											+ tmpx * stride,
									this->input_table[k].begin()
											+ (tmpy * stride + p)
													* (in_x + 2 * pad)
											+ tmpx * stride + w_x);
						}
					}

					// for conv weight
					for (int k = 0; k < in_ch; k++) {
						//weight// according to current kernelID(output channel), for example, jume every 6 outchannels
						// pick up 3(in_ch) vectors. These 3 vectors is one single 3D-kernel.
						tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),
								this->weight_table[tmpMAC->tmpch * in_ch + k].begin(),
								this->weight_table[tmpMAC->tmpch * in_ch + k].end()
										- 1);
					}
					tmpMAC->inbuffer.push_back(
							this->weight_table[tmpMAC->tmpch * in_ch].back()); //bias

					// 遍历并输出inbuffer中的所有元素 //  先是功能code，1代表relu。然后in—ch，然后
					//for (float value : tmpMAC->inbuffer) {
					//	std::cout << " macnetcppline1608value: " << value;
					//}
					//cout << "  cycles1918 " << cycles << std::endl;

					// added send type 1
					MAC_list[mem_id]->pecycle =
							cycles
									+ std::ceil(
											(in_ch * w_x * w_y * 2 + 1)
													* MEM_read_delay) + CACHE_DELAY;
					//cout<<"debugline1642  "<<std::ceil(
					//		(in_ch * w_x * w_y * 2 + 1)
					//				* MEM_read_delay) + CACHE_DELAY<<endl;
					MAC_list[mem_id]->inbuffer.clear();
					MAC_list[mem_id]->inbuffer = MAC_list[src_mac]->inbuffer;
					MAC_list[mem_id]->inject(1, src, tmpMAC->inbuffer.size(),
							o_fnReluOrPool, vcNetwork->NI_list[mem_id],
							pidSignalID, src_mac);

#ifdef SoCC_Countlatency
					DNN_latency[pidSignalID * 3 + 1][0] =
							(tmpMAC->inbuffer.size() * bitsPerElement
									+ headerPerFlit) / FLIT_LENGTH + 1 + 9000; //DNN_latency[x+1][0] packet size in flits.
#endif
				}
			} else if (this->cnnmodel->all_layer_type[current_layerSeq] == 'p') // pooling
					{
				if (tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[current_layerSeq]=='p'
						{
					tmpMAC->tmpch = tmpMAC->cnn_current_layer_task_id / (o_x * o_y); //current output channel
					tmpMAC->tmpm = tmpMAC->cnn_current_layer_task_id % (o_x * o_y); //current output map id
					int tmpx = tmpMAC->tmpm % o_x;
					int tmpy = tmpMAC->tmpm / o_x;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [map size] [i]

					tmpMAC->inbuffer.push_back(o_fnReluOrPool); // 8
					tmpMAC->inbuffer.push_back(1); // yz added to make sure inbuffer first 3 elements the same.
					tmpMAC->inbuffer.push_back(w_x * w_y);
					for (int p = 0; p < w_y; p++) {
						//tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[tmpMAC->tmpch].begin() + (tmpy*w_y+p)*in_x + tmpx*w_x, this->input_table[tmpMAC->tmpch].begin() + (tmpy*w_y+p)*in_x + tmpx*w_x + w_x);
						tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),
								this->input_table[tmpMAC->tmpch].begin()
										+ (tmpy * stride + p) * (in_x + 2 * pad)
										+ tmpx * stride,
								this->input_table[tmpMAC->tmpch].begin()
										+ (tmpy * stride + p) * (in_x + 2 * pad)
										+ tmpx * stride + w_x);
					}

					// added send type 1
					MAC_list[mem_id]->pecycle = cycles
							+ ceil(w_x * w_y * MEM_read_delay) + CACHE_DELAY;
					MAC_list[mem_id]->inbuffer.clear();
					MAC_list[mem_id]->inbuffer = MAC_list[src_mac]->inbuffer;
					MAC_list[mem_id]->inject(1, src, tmpMAC->inbuffer.size(),
							o_fnReluOrPool, vcNetwork->NI_list[mem_id],
							pidSignalID, src_mac);

#ifdef SoCC_Countlatency
					DNN_latency[pidSignalID * 3 + 1][0] =
							(tmpMAC->inbuffer.size() * bitsPerElement
									+ headerPerFlit) / FLIT_LENGTH + 1 + 9100; //DNN_latency[x+1][0] packet size in flits.
#endif
				}
			} else if (this->cnnmodel->all_layer_type[current_layerSeq]
					== 'f') { // fc layer
				if (tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[current_layerSeq]=='f'
						{
					tmpMAC->tmpch = 0; //current output channel 1*ox*1
					tmpMAC->tmpm = tmpMAC->cnn_current_layer_task_id; //current output vector id (also w_ch)
					tmpMAC->npoolflag = 0;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [map size w_x * w_y] [i] [w + b]
					tmpMAC->inbuffer.push_back(o_fnReluOrPool);
					tmpMAC->inbuffer.push_back(1); // yz added to make sure inbuffer first 3 elements the same.
					tmpMAC->inbuffer.push_back(w_x * w_y); //	for dense	w_x = layer_info[0]; // = in_x  // w_y 	w_y = 1;

					// input table problem
					tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),
							this->input_table[0].begin(),
							this->input_table[0].end());

					tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),
							this->weight_table[tmpMAC->tmpm].begin(),
							this->weight_table[tmpMAC->tmpm].end()); //weight

					//DNN_latency[pid][3] = cycles;
					// added send type 1
					MAC_list[mem_id]->pecycle =
							cycles
									+ ceil(
											(w_x * w_y * 2 + 1) * MEM_read_delay) + CACHE_DELAY;
					MAC_list[mem_id]->inbuffer.clear();
					MAC_list[mem_id]->inbuffer = MAC_list[src_mac]->inbuffer;
					MAC_list[mem_id]->inject(1, src, tmpMAC->inbuffer.size(),
							o_fnReluOrPool, vcNetwork->NI_list[mem_id],
							pidSignalID, src_mac);

#ifdef SoCC_Countlatency
					DNN_latency[pidSignalID * 3 + 1][0] =
							(tmpMAC->inbuffer.size() * bitsPerElement
									+ headerPerFlit) / FLIT_LENGTH + 1 + 9200; //DNN_latency[x+1][0] packet size in flits.
#endif
				}
			}
			tmpNI->packet_buffer_out[0].pop_front();
		}

		// for message type 2 from MAC to MEM, received OFmap
		pbuffersize = tmpNI->packet_buffer_out[1].size();
		for (int j = 0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[1].front();
			if (tmpPacket->message.msgtype != 2) {
				tmpNI->packet_buffer_out[1].pop_front();
				tmpNI->packet_buffer_out[1].push_back(tmpPacket);
				cout << "continue macnet: " << cycles << endl;
				continue;
			}
			src = tmpPacket->message.source_id;
			pidSignalID = tmpPacket->message.signal_id;
			yzLastSeenPid = pidSignalID;
			src_mac = tmpPacket->message.mac_id;
			// cout << "MEM " << tmpPacket->message.destination <<  " receive type " << tmpPacket->message.msgtype << " from MAC " << src << endl;
			tmpMAC = MAC_list[src_mac];

#ifdef SoCC_Countlatency
			//statistics

			DNN_latency[pidSignalID * 3 + 2][4] = tmpPacket->send_out_time;
			DNN_latency[pidSignalID * 3 + 2][7] = cycles;
#endif

			if (this->cnnmodel->all_layer_type[current_layerSeq] == 'c') { // conv

#ifdef only3type
				// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] =
						tmpPacket->message.data[0];
				if (tmpMAC->selfstatus == 5)
					tmpMAC->send = 3;
#endif

			}

			else if (this->cnnmodel->all_layer_type[current_layerSeq] == 'p') {

#ifdef only3type
				// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] =
						tmpPacket->message.data[0];
				if (tmpMAC->selfstatus == 5)
					tmpMAC->send = 3;
#endif
			} else if (this->cnnmodel->all_layer_type[current_layerSeq]
					== 'f') { // fc layer

#ifdef only3type
				// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] =
						tmpPacket->message.data[0];
				if (tmpMAC->selfstatus == 5)
					tmpMAC->send = 3;
#endif
			}
			tmpNI->packet_buffer_out[1].pop_front();
		}
	}

// only check non-mem node recive resp， pick it up(deleted from packetbuffer) and for computation
	for (int i = 0; i < TOT_NUM; i++) {
		// skip mem nodes
		if (contains(dest_list, i)) {
			continue;
		}

		tmpNI = this->vcNetwork->NI_list[i];
		// for message type 1 from MEM to MAC
		pbuffersize = tmpNI->packet_buffer_out[0].size();
		for (int j = 0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			if (tmpPacket->message.msgtype != 1) {
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}
			src_mac = tmpPacket->message.mac_id; //mac
			pidSignalID = tmpPacket->message.signal_id;
			yzLastSeenPid = pidSignalID;
#ifdef SoCC_Countlatency
			DNN_latency[pidSignalID * 3 + 1][4] = tmpPacket->send_out_time; //DNN_yzlatency[x+1][4]
			int mac_id_resp = DNN_latency[pidSignalID * 3 + 1][2];
			int delay_add_resp = DNN_latency[pidSignalID * 3 + 1][4] - DNN_latency[pidSignalID * 3 + 1][3];
			samplingWindowDelay[mac_id_resp] += delay_add_resp;

			DNN_latency[pidSignalID * 3 + 1][7] = cycles; //DNN_yzlatency[x+1][7]
#endif
			tmpMAC = MAC_list[src_mac];
			tmpMAC->cnn_current_layer_task_id = -1;
			tmpNI->packet_buffer_out[0].pop_front();

		}
	}

	return;
}

// Destructor
MACnet::~MACnet() {
	MAC *mac1;
	while (MAC_list.size() != 0) {
		mac1 = MAC_list.back();
		MAC_list.pop_back();
		delete mac1;
	}
}
