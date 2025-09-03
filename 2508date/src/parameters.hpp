#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_
#define DEFAULT_NNWEIGHT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/weight.txt"
#define DEFAULT_NNINPUT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/input2.txt"
#define DEFAULT_NNMODEL_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/newnet2.txt"
#define randomeval
#define YZLLMSwitchON
constexpr int DIM_MODEL = 4096; 
constexpr int NUM_HEAD = 32; 
constexpr int SEQUENCE_LENGTH = 512; 
constexpr int D_HEAD = DIM_MODEL / NUM_HEAD;
#define LLM_OUTPUT_PATH "src/output/"
#define LLM_TEST_CASE 1
#define LLM_DEBUG_LEVEL 1
#define LLM_RANDOM_SEED 42
#define MemNode2_4X4
#define case3_affiliatedordering
//#define case1_default

#if defined(case3_affiliatedordering)
    #define rowmapping
    #define flitLevelFlippingSwitch
#elif defined(case4_seperratedordering)
    #define rowmapping
    #define flitLevelFlippingSwitch
    #define reArrangeInput
#else
    #define rowmapping
#endif

//#define FIXED_POINT_SORTING

#define only3type
#define outPortNoInfinite
#define FREQUENCY 2
#define MEM_read_delay 0.0625
#define PE_NUM_OP 64
#define PRINT 100000
#define valueBytes 4
#define FLIT_LENGTH 512
#define headerPerFlit 0
#define bitsPerElement 32
#define payloadElementNum 16
#define SoCC_Countlatency
#define samplingWindowLength 100
#define VN_NUM 1
#define VC_PER_VN 4
#define VC_PRIORITY_PER_VN 0
#define STARVATION_LIMIT 20
#define LCS_URS_TRAFFIC
#define INPORT_FLIT_BUFFER_SIZE 4;
#define INFINITE 10000
#define INFINITE1 10000
#define CACHE_DELAY 0
#define flitcomparison

#if defined MemNode2_4X4
	#define PE_X_NUM 4
	#define PE_Y_NUM 4
	#define X_NUM 4
	#define Y_NUM 4
	#define TOT_NUM 16
	#define YZMEMCount 2
#endif

#define LINK_TIME 2
#define DISTRIBUTION_NUM 20
struct GlobalParams { 
    static char NNmodel_filename[128]; 
    static char NNweight_filename[128]; 
    static char NNinput_filename[128]; 
};
#endif
