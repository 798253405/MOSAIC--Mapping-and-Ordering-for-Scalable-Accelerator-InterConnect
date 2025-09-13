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
 *    - yzFuncSAMOSSampleMapping(): 基于延迟采样的动态映射
 *    - 将neurons均匀或按性能分配给各MAC单元
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


int MACnet::yzFuncSAMOSSampleMapping(int neuronnum) {
	// 清空并按 macNum 大小准备映射表
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	// 1) 收集可计算节点（排除内存节点）
	std::vector<int> pe_ids;
	pe_ids.reserve(macNum);
	for (int id = 0; id < macNum; ++id) {
		if (!contains(dest_list, id))
			pe_ids.push_back(id);
	}
	if (pe_ids.empty() || neuronnum <= 0)
		return 0;

	// 2) 计算每个节点的平均延迟（采样窗口平均），以及缺失/0 值的回退
	//    回退策略：用所有非零样本的均值作为默认延迟，避免除零/未采到样本时的不稳定
	// Debug: Print what SAMOS function actually sees
	cout << "\n[DEBUG] Inside yzFuncSAMOSSampleMapping:" << endl;
	cout << "[DEBUG] Reading samplingWindowDelay values:" << endl;
	double sum_lat = 0.0;
	int nz = 0;
	for (int id : pe_ids) {
		double lat = double(samplingWindowDelay[id])
				/ std::max(1, samplingTasksPerMAC);
		if (samplingWindowDelay[id] > 0) {
			cout << "  MAC " << id << ": raw_delay=" << samplingWindowDelay[id] 
			     << " window_len=" << samplingTasksPerMAC
			     << " avg_lat=" << lat << endl;
		}
		if (lat > 0.0) {
			sum_lat += lat;
			++nz;
		}
	}
	const double default_lat = (nz > 0) ? (sum_lat / nz) : 1.0; // 全 0 时兜底为 1
	const double eps = 1e-12;

	struct NodeW {
		int id;
		double w;     // 权重 = 1/lat
		double want;  // 理想配额
		int alloc;    // 实际整数配额
		double frac;  // 小数余量
	};

	std::vector<NodeW> nodes;
	nodes.reserve(pe_ids.size());

	double sumW = 0.0;
	for (int id : pe_ids) {
		double lat = double(samplingWindowDelay[id])
				/ std::max(1, samplingTasksPerMAC);
		if (lat <= 0.0)
			lat = default_lat;
		double w = 1.0 / (lat + eps);
		nodes.push_back( { id, w, 0.0, 0, 0.0 });
		sumW += w;
	}
	if (sumW <= 0.0) { // 极端兜底：均匀分配
		int base = neuronnum / int(nodes.size());
		int rem = neuronnum - base * int(nodes.size());
		int j = 0;
		for (auto &n : nodes) {
			for (int k = 0; k < base; ++k)
				this->mapping_table[n.id].push_back(j++);
		}
		for (int i = 0; i < rem; ++i)
			this->mapping_table[nodes[i].id].push_back(j++);
		return 0;
	}

	// 3) Hamilton 最大余数配额
	int allocated = 0;
	for (auto &n : nodes) {
		double exact = neuronnum * (n.w / sumW);
		n.want = exact;
		n.alloc = int(std::floor(exact));
		n.frac = exact - n.alloc;
		allocated += n.alloc;
	}
	int remainder = neuronnum - allocated;

	// 把余量大的优先补 1
	std::sort(nodes.begin(), nodes.end(), [](const NodeW &a, const NodeW &b) {
		return a.frac > b.frac;
	});
	for (int i = 0; i < remainder; ++i)
		nodes[i % nodes.size()].alloc++;

	// 4) 生成具体路由映射（任务 id 连续递增）
	int j = 0;
	for (auto &n : nodes) {
		for (int k = 0; k < n.alloc; ++k)
			this->mapping_table[n.id].push_back(j++);
	}

	// 可选：调试输出

	std::cout << "[SAMOS auto]TOTneuronnum=" << neuronnum << " TOTPE="
			<< nodes.size() << " TOTallocated=" << j << "\n";
	for (auto &n : nodes) {
		double avgLat = double(samplingWindowDelay[n.id])
				/ std::max(1, samplingTasksPerMAC);
		std::cout << "  node " << n.id << " lat=" << avgLat << " w=" << n.w
				<< " want=" << n.want << " alloc=" << n.alloc << "\n";
	}

	return 0;
}

int MACnet::yzPostSimTravelMapping(int neuronnum) {

	cout << " yzPostSimTravelMappingneuronnumis " << neuronnum << " atcycles "
			<< cycles << endl;
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);
	int j = 0;
	int countPerPE = 0;
	int tempAllocatedCount = 0;

	double timeHop1 = samplingWindowDelay[13] / samplingTasksPerMAC, timeHop2 =
			samplingWindowDelay[5] / samplingTasksPerMAC, timeHop3 =
			samplingWindowDelay[8] / samplingTasksPerMAC, timeHop4 =
			samplingWindowDelay[1] / samplingTasksPerMAC, timeHop5 =
			samplingWindowDelay[4] / samplingTasksPerMAC, timeHop6 =
			samplingWindowDelay[12] / samplingTasksPerMAC, timeHop7 =
			samplingWindowDelay[0] / samplingTasksPerMAC;
	timeHop1 = 36.80357142857143; // layer1postsimulation
	timeHop2 = 37.729166666666664;
	timeHop3 = 39.839285714285715;
	timeHop4 = 43.666666666666664;
	timeHop5 = 45.520833333333336;
	timeHop6 = 45.67559523809524;
	timeHop7 = 51.11011904761905;

	/*	timeHop1 = 17.511904761904763; // layer2postsimulation
	 timeHop2 = 17.88095238095238;
	 timeHop3 = 17.345238095238095;
	 timeHop4 = 23.80952380952381;
	 timeHop5 = 23.55952380952381;
	 timeHop6 = 23.607142857142854;
	 timeHop7 = 30.19047619047619;*/

	/*	timeHop1 = 116.55263157894737; // layer3postsimulation
	 timeHop2 = 116.66666666666667;
	 timeHop3 = 150.37719298245617;
	 timeHop4 = 119.34782608695653;
	 timeHop5 = 153.7719298245614;
	 timeHop6 = 150.9298245614035;
	 timeHop7 = 159.62608695652173;*/

	/*
	 timeHop1 = 17.607142857142858; // layer4postsimulation
	 timeHop2 = 17.655172413793103;
	 timeHop3 = 17.25;
	 timeHop4 = 24.06896551724138;
	 timeHop5 = 23.448275862068968;
	 timeHop6 = 23.607142857142858;
	 timeHop7 = 29.89655172413793;
	 */

	/*	timeHop1 = 306.0; // layer5postsimulation
	 timeHop2 = 340.6666666666667;
	 timeHop3 = 414.375;
	 timeHop4 = 352.33333333333337;
	 timeHop5 = 384.77777777777777;
	 timeHop6 = 447.375;
	 timeHop7 = 409.44444444444446;*/

	/*	timeHop1 = 93.0; // layer6postsimulation
	 timeHop2 = 97.5;
	 timeHop3 = 113.5;
	 timeHop4 = 104.0;
	 timeHop5 = 119.83333333333333;
	 timeHop6 = 120.83333333333334;
	 timeHop7 = 128.66666666666666;*/

	cout << " timeHop1totimeHop7 " << " " << timeHop1 << " " << timeHop2 << " "
			<< timeHop3 << " " << timeHop4 << " " << timeHop5 << " " << timeHop6
			<< " " << timeHop7 << endl;
	cout << " timeHop1totimeHop7 " << " " << samplingWindowDelay[13] << " "
			<< samplingWindowDelay[5] << " " << samplingWindowDelay[8] << " "
			<< samplingWindowDelay[1] << " " << samplingWindowDelay[4] << " "
			<< samplingWindowDelay[12] << " " << samplingWindowDelay[0] << endl;
	double c1double = neuronnum
			* (timeHop2 * timeHop3 * timeHop4 * timeHop5 * timeHop6 * timeHop7)
			/ (timeHop2 * timeHop3 * timeHop4 * timeHop5 * timeHop6 * timeHop7
					+ timeHop1 * timeHop3 * timeHop4 * timeHop5 * timeHop6
							* timeHop7
					+ timeHop1 * timeHop2 * timeHop4 * timeHop5 * timeHop6
							* timeHop7
					+ timeHop1 * timeHop2 * timeHop3 * timeHop5 * timeHop6
							* timeHop7
					+ timeHop1 * timeHop2 * timeHop3 * timeHop4 * timeHop6
							* timeHop7
					+ timeHop1 * timeHop2 * timeHop3 * timeHop4 * timeHop5
							* timeHop7
					+ timeHop1 * timeHop2 * timeHop3 * timeHop4 * timeHop5
							* timeHop6) / 2;
	// cout << " fenzi " << numeratorfenzi << " " << denominatorfenmu << endl;
	double c2double = c1double * timeHop1 / timeHop2;
	double c3double = c1double * timeHop1 / timeHop3;
	double c4double = c1double * timeHop1 / timeHop4;
	double c5double = c1double * timeHop1 / timeHop5;
	double c6double = c1double * timeHop1 / timeHop6;
	double c7double = c1double * timeHop1 / timeHop7;
	int c1 = int(c1double);
	int c2 = int(c2double);
	int c3 = int(c3double);
	int c4 = int(c4double);
	int c5 = int(c5double);
	int c6 = int(c6double);
	int c7 = int(c7double);
	cout << " c1line436 " << neuronnum << " " << " " << " " << c1 << " " << c2
			<< " " << c3 << " " << endl;
	int numeratorfenzi = timeHop1 * timeHop2;
	int denominatorfenmu = 3 * timeHop3 * timeHop1 + 3 * timeHop3 * timeHop2
			+ timeHop1 * timeHop2;
	//cout << " fenzi " << numeratorfenzi <<" "<< denominatorfenmu<< endl;
	// 9and 11 are memnodes  // or 9and10
	for (int i = 0; i < macNum; i++) {
		int countPerPE;
		if (i == 0 || i == 2) {
			countPerPE = c7;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 13 || i == 15) {
			countPerPE = c1;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 5 || i == 7) {
			countPerPE = c2;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 8 || i == 10) {
			countPerPE = c3;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 1 || i == 3) {
			countPerPE = c4;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 4 || i == 6) {
			countPerPE = c5;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else if (i == 12 || i == 14) {
			countPerPE = c6;
			tempAllocatedCount = tempAllocatedCount + countPerPE;
			cout << "travelTimeMappingcountPerPE " << i << " " << countPerPE
					<< endl;
		} else {
			continue; // 或者 assert(1 == 1);
		}

		for (int k = 0; k < countPerPE; k++) {
			this->mapping_table[i].push_back(j);
			j = j + 1;
			//cout << " mapping  notdone" <<" iis "<<i<< " jis " << j << endl;
		}
	}
	cout << " belowyzPostSimTravelMapping tail " << j << endl;
	// Assuming 'neuronnum' and other relevant variables are defined elsewhere
	int customOrder[] = { 13, 15, 5, 7, 8, 10, 12, 6, 4, 14, 1, 3, 0, 2 }; // Custom order specified
	int orderSize = sizeof(customOrder) / sizeof(customOrder[0]); // Size of the custom order array
	int orderIndex = 0; // Start from the first element in the custom order

	for (int k = tempAllocatedCount; k < neuronnum; k++) {
		int idLoop = customOrder[orderIndex]; // Use the current order from the custom sequence
		this->mapping_table[idLoop].push_back(j);
		j = j + 1;
		orderIndex = (orderIndex + 1) % orderSize; // Move to the next index in the custom order, loop back if at the end
	}
	return 0;
}




















void MACnet::checkStatus() {

	if (readyflag == 0) // every new layer
			{
		if (mappingagain == 0) { // only first time mapping of layer
			this->vcNetwork->resetVNRoundRobin(); //everylayer,reset vn rr
			this->create_input();

		}
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
#ifdef YZSAMOSSampleMapping
		if ((o_ch * o_x * o_y) / (macNum - YZMEMCount) < samplingTasksPerMAC) {
			cout
					<< " thisLayerIsShorterThan15SamplingWindow！！！noSAMOSMapping！JustRowMapping！！！   "
					<< endl;
			this->xmapping(o_ch * o_x * o_y);
		} else {
			if (mappingagain == 0) {
				// Debug: Show values BEFORE reset
				cout << "\n[DEBUG] About to RESET samplingWindowDelay (mappingagain==0), Layer " << current_layerSeq << endl;
				cout << "[DEBUG] Values BEFORE reset:" << endl;
				for(int i = 0; i < TOT_NUM; i++) {
					if(samplingWindowDelay[i] > 0) {
						cout << "  MAC " << i << ": delay=" << samplingWindowDelay[i] << endl;
					}
				}
				samplingAccumlatedCounter = 0;
				std::fill_n(samplingWindowDelay, TOT_NUM, 0); //rest samping window statistic
				cout << " samplingWindowDelay[0] AFTER reset: " << samplingWindowDelay[0]
						<< endl;
				cout << " samplingAccumlatedCounter "
						<< samplingAccumlatedCounter << " macnum " << macNum
						<< endl;
				cout << endl;
				// if sampling window has not been done in this layer, do sampling window
				this->xmapping((macNum - YZMEMCount) * samplingTasksPerMAC);
				mappingagain = 1; // normal = 0, doing sampling and need to do body mapping later = 1, doing body mapping =2 (should be set after complete samping )

			} else if (mappingagain == 2) { // if sampling window has been done in this layer, do left parts
				packet_id = packet_id
						+ (macNum - YZMEMCount) * samplingTasksPerMAC;
				// Debug: Print samplingWindowDelay RIGHT BEFORE calling SAMOS
				cout << "\n[DEBUG] About to call SAMOS (mappingagain==2), Layer " << current_layerSeq << endl;
				cout << "[DEBUG] samplingWindowDelay values BEFORE SAMOS:" << endl;
				for(int i = 0; i < TOT_NUM; i++) {
					if(samplingWindowDelay[i] > 0) {
						cout << "  MAC " << i << ": delay=" << samplingWindowDelay[i] 
						     << " avg=" << (double)samplingWindowDelay[i]/samplingTasksPerMAC << endl;
					}
				}
				cout << " samplingWindowDelay[0] " << samplingWindowDelay[0]
						<< endl;
				cout << " samplingAccumlatedCounter "
						<< samplingAccumlatedCounter << " macnum " << macNum
						<< endl;
				// Debug: Check delay before SAMOS
				cout << "[DEBUG] Right before calling SAMOS: samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
				
				this->yzFuncSAMOSSampleMapping(
						o_ch * o_x
								* o_y- (macNum -YZMEMCount) * samplingTasksPerMAC);
				
				// Debug: Check delay after SAMOS
				cout << "[DEBUG] Right after calling SAMOS: samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
				// 3) ✅ 把局部编号统一转换为“全局编号”（关键修复）
				{
					const int offset = (macNum - YZMEMCount)
							* samplingTasksPerMAC;
					for (int i = 0; i < macNum; ++i) {
						for (int &gid : this->mapping_table[i]) {
							gid += offset;
						}
					}
				}
				cout << " this is second mappi of one layer" << endl;
				
				// Debug: Check delay before resetting mappingagain
				cout << "[DEBUG] Before mappingagain=0: samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
				mappingagain = 0; //reset
				cout << "[DEBUG] After mappingagain=0: samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
			} else {
				cout
						<< " error!line878, mapping again should be 0 or 2 when readyflag=0(new mapping) "
						<< endl;
			}
		}
#endif

#ifdef YZPostSimTravelTime
		this->yzPostSimTravelMapping(o_ch * o_x * o_y);
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
	if (mappingagain == 1) {	// if  =1 , hold up before going to next steps.
		readyflag = 0;
		// Debug: Print samplingWindowDelay BEFORE changing mappingagain
		cout << "\n[DEBUG] End of sampling (mappingagain 1->2), Layer " << current_layerSeq << endl;
		cout << "[DEBUG] samplingWindowDelay values BEFORE transition:" << endl;
		for(int i = 0; i < TOT_NUM; i++) {
			if(samplingWindowDelay[i] > 0) {
				cout << "  MAC " << i << ": delay=" << samplingWindowDelay[i] 
				     << " avg=" << (double)samplingWindowDelay[i]/samplingTasksPerMAC << endl;
			}
		}
		mappingagain = 2;

		// 再次检查
		    cout << "[DEBUG] After mappingagain 1->2 transition:" << endl;
		    for(int i = 0; i < TOT_NUM; i++) {
		        if(samplingWindowDelay[i] > 0) {
		            cout << "  MAC " << i << ": delay=" << samplingWindowDelay[i] << endl;
		        }
		    }

		// 在重置MAC状态前再检查一次
		cout << "[DEBUG] Before resetting MAC status:" << endl;
		cout << "  samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
		
		for (int i = 0; i < macNum; i++) {
			MAC_list[i]->selfstatus = 0;
		}
		
		// 重置MAC状态后再检查
		cout << "[DEBUG] After resetting MAC status:" << endl;
		cout << "  samplingWindowDelay[0]=" << samplingWindowDelay[0] << endl;
		
		return;
	}

	cout << "mappingagain(shouldbe0)  " << mappingagain << " " << cycles
			<< "  current_layerSeq " << current_layerSeq << " " << n_layer
			<< " leftTasksCount "
			<< ((o_ch * o_x * o_y)
					- (TOT_NUM - YZMEMCount) * samplingTasksPerMAC) << endl;
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
#ifdef YZSAMOSSampleMapping  //last layer
		// Always increment packet_id by total tasks to avoid signalid conflicts
		int total_tasks = o_ch * o_x * o_y;
		packet_id = packet_id + total_tasks;  // Use full task count for unique signalids
		cout << "[DEBUG] Last layer, packet_id updated: " << (packet_id - total_tasks) 
		     << " -> " << packet_id 
		     << " (added " << total_tasks << " tasks)" << endl;
#else
		packet_id = packet_id + o_ch * o_x * o_y;
#endif
		cout << "debug packetid1407 " << packet_id << "  o_ch " << o_ch
				<< " o_x " << o_x << " o_y " << o_y << endl;
		lastLayerPacketID = packet_id;

		cout << "this layer ends  below is all layer ends \n" << endl;
		return;
	} else {
		cout << "intermediate Layer finished " << (current_layerSeq - 1)
				<< " at cycle " << cycles << endl;
		Layer_latency.push_back(cycles);
#ifdef YZSAMOSSampleMapping  //pakcetid compensation
		// Always increment packet_id by total tasks to avoid signalid conflicts
		int total_tasks = o_ch * o_x * o_y;
		packet_id = packet_id + total_tasks;  // Use full task count for unique signalids
		cout << "[DEBUG] Layer " << (current_layerSeq - 1) 
		     << " completed, packet_id updated: " << (packet_id - total_tasks) 
		     << " -> " << packet_id 
		     << " (added " << total_tasks << " tasks)" << endl;
#else
		packet_id = packet_id + o_ch * o_x * o_y;
#endif

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
#ifdef CNN_RANDOM_DATA_TEST
					// Generate random input data instead of using real input_table data
					static bool cnn_random_warning_printed = false;
					if (!cnn_random_warning_printed) {
						cout << "WARNING: CNN_RANDOM_DATA_TEST enabled - CNN using random input data [-0.5, 0.5]" << endl;
						cnn_random_warning_printed = true;
					}
					for (int k = 0; k < in_ch; k++) {
						for (int p = 0; p < w_y; p++) {
							for (int q = 0; q < w_x; q++) {
								float random_input = static_cast<float>(rand()) / RAND_MAX - 0.5f;
								tmpMAC->inbuffer.push_back(random_input);
							}
						}
					}
#else
					// Original: Use real input_table data
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
#endif

					// for conv weight
#ifdef CNN_RANDOM_DATA_TEST
					// Generate random weight data instead of using real weight_table data
					for (int k = 0; k < in_ch; k++) {
						for (int i = 0; i < w_x * w_y; i++) {
							float random_weight = static_cast<float>(rand()) / RAND_MAX - 0.5f;
							tmpMAC->inbuffer.push_back(random_weight);
						}
					}
					// Generate random bias
					float random_bias = static_cast<float>(rand()) / RAND_MAX - 0.5f;
					tmpMAC->inbuffer.push_back(random_bias);
#else
					// Original: Use real weight_table data
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
#endif

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
			// Debug print
			cout << "[LAT_ADD] MACnet.cpp:1219 Input_Resp MAC " << mac_id_resp 
			     << " += " << delay_add_resp 
			     << " (total=" << samplingWindowDelay[mac_id_resp] << ")" << endl;
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
