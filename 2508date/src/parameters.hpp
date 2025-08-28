

#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#define DEFAULT_NNWEIGHT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/weight.txt"
#define DEFAULT_NNINPUT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/input2.txt"
#define DEFAULT_NNMODEL_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/newnet2.txt"
//#define DEFAULT_NNMODEL_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/vgg16.txt"



/******************************/
#define fulleval
//#define randomeval



#define YZLLMSwitchON
// --- Llama 2 (7B) 模型及硬件参数设定 ---
constexpr int DIM_MODEL = 4096;
constexpr int NUM_HEAD = 32;
constexpr int SEQUENCE_LENGTH = 512;
constexpr int D_HEAD = DIM_MODEL / NUM_HEAD; // 4096 / 32 = 128
// Note: Debug control is now managed by LLM_TEST_CASE below




// Common output path
#define LLM_OUTPUT_PATH "src/output/"
// ==================== LLM TEST CASE CONFIGURATION ====================
// 选择测试用例：根据实际需求选择合适的测试规模  (实际值在llmmacnet.cpp中定义):
// Test Case 1: 小规模验证 (4x4矩阵) - 用于快速功能验证
// Test Case 2: 中等规模 (128x128矩阵) - 模拟实际LLaMA注意力计算
// Test Case 3: 大规模测试 (256x256矩阵) - 性能压力测试
#define LLM_TEST_CASE 2  // <-- 修改这里切换测试用例

// Debug output control (applies to all test cases)
#ifdef LLM_TEST_CASE
    #if LLM_TEST_CASE == 1
        // Enable verbose debug for small test case
        #define LLM_DEBUG_PRINT_ENABLED
        #define LLM_VERBOSE_DEBUG
        #define MAX_DEBUG_MAC_UNITS 10
    #else
        // Limit debug output for larger test cases
        #define MAX_DEBUG_MAC_UNITS 3
        // #define LLM_DEBUG_PRINT_ENABLED  // Uncomment to enable debug for large cases
    #endif
#endif


// 通用LLM参数
#define LLM_RANDOM_SEED 42  // 固定随机种子确保可重现性


// ==================== NoC SIZE AND MEMORY CONFIGURATION ====================
// 选择NoC规模（根据测试用例选择合适的NoC大小）
// 推荐配置：Test Case 1 -> 4x4, Test Case 2 -> 16x16, Test Case 3 -> 32x32
//#define MemNode2_4x4    // 4x4 NoC, 2 MC cores (适合Test Case 1)
//#define MemNode4_4X4    // 4x4 NoC, 4 MC cores (适合Test Case 1)
//#define MemNode4_8X8    // 8x8 NoC, 4 MC cores
#define MemNode4_16X16   // 16x16 NoC, 4 MC cores (适合Test Case 2)
//#define MemNode4_32X32  // 32x32 NoC, 4 MC cores (适合Test Case 3)


// ==================== ORDERING OPTIMIZATION ====================
// Flit级别翻转优化（降低功耗）
//#define flitLevelFlippingSwitch  // 取消注释以启用Ordering优化

 //#define all128BitInvert
//#define partionedInvert






//#define debugAllInput0
// #define debugAllweight0
//#define CoutDebugAll0


#define flitcomparison  //rinport.cpp


// ==================== TASK MAPPING STRATEGY ====================
// 注意：rowmapping和YZSAMOSSampleMapping互斥，只能选择一个！
#define rowmapping           // 基准行映射（默认）
//#define colmapping         // 列映射
//#define randmapping        // 随机映射
//#define YZrandmapping      // YZ随机映射
//#define YZDistanacemappingw // 距离感知映射
//#define YZSAMOSSampleMapping // SAMOS自适应映射（高级优化）
#define SoCC_Countlatency		// open recording of packet level delay // note 	DNN_latency.resize(3000000); is not enough for large dnn

#define samplingWindowLength 10  // Standard sampling window
//////////////////////////////////
#define VN_NUM 1   //2
#define VC_PER_VN  4  ///<1 A: control URS (control all in other 3 modes)
#define VC_PRIORITY_PER_VN 0 ///< B: only control LCS
#define STARVATION_LIMIT 20 // forbid starvation (no priority packet must go after 20)




#define LCS_URS_TRAFFIC


#define INPORT_FLIT_BUFFER_SIZE 4; // number 4


#define INFINITE 10000    // changed from 10000
#define INFINITE1 10000  //added for flit buffer (10000)



#define CACHE_DELAY 0  // simulate cache memory

#define MASTER_LIST_RECORD_DELAY 1  //
#define MASTER_LIST_REFER_DELAY 0  //
#define SLAVE_LIST_REFER_DEALY 0  //
///< axi4 data to message  message (in NI) to (packer in VC SIGNAL/TDM)
#define DELAY_FROM_P_TO_M  0  // packet/signal to message conversion time; AXI4 to message / message to AXI4 conversion time is 1 by default
#define DELAY_FROM_M_TO_P  0  // message to packet/signal conversion time






// define some functions
#define only3type	//only 3 packets per neuron task

#define outPortNoInfinite // back pressure from VC Router Out Port


#define FREQUENCY 2 // GHz
#define MEM_read_delay  0.0625//0.0625 //0.0625  //0.31255  // 0.1575 / //0.0787 / delay for 2byte / 1 data  // 0.3125 =12.8GB/s pc3-12800  0.0625 =  64GB/S PC5-64000.
#define PE_NUM_OP 64//64 // 25
#define PRINT 100000


#define valueBytes  4// 2 //2 Bytes for one value
#define FLIT_LENGTH 512// 512 bits  for 16 elements *32 bits floating-point  // 256 bits = 32 bytes
#define headerPerFlit   0    // 0 AS WE assume we consider only payload here. Or we can say we assume there are extra lines for header and we ignore this
#define bitsPerElement 32
//below only influence data flipping. (no influcence on flit num?)
#define payloadElementNum 16









/******************************/
#if defined MemNode2_4x4
	#define PE_X_NUM 4
	#define PE_Y_NUM 4
	//NI size
	#define X_NUM 4
	#define Y_NUM 4
	#define TOT_NUM 16
	#define YZMEMCount 2
#elif defined MemNode4_4X4
	#define PE_X_NUM 4
	#define PE_Y_NUM 4
	//NI size
	#define X_NUM 4
	#define Y_NUM 4
	#define TOT_NUM 16
	#define YZMEMCount 4
#elif defined MemNode4_8X8
	#define PE_X_NUM 8
	#define PE_Y_NUM 8
	//NI size
	#define X_NUM 8
	#define Y_NUM 8
	#define TOT_NUM 64
	#define YZMEMCount 4
#elif defined MemNode4_16X16
	#define PE_X_NUM 16
	#define PE_Y_NUM 16
	//NI size
	#define X_NUM 16
	#define Y_NUM 16
	#define TOT_NUM 256
	#define YZMEMCount 4
#elif defined MemNode4_32X32
	#define PE_X_NUM 32
	#define PE_Y_NUM 32
	//NI size
	#define X_NUM 32
	#define Y_NUM 32
	#define TOT_NUM 1024
	#define YZMEMCount 4

#endif








#define LINK_TIME 2//2

#define DISTRIBUTION_NUM 20

struct GlobalParams {
	static char NNmodel_filename[128];
	static char NNweight_filename[128];
	static char NNinput_filename[128];

};

#endif /* PARAMETERS_HPP_ */
