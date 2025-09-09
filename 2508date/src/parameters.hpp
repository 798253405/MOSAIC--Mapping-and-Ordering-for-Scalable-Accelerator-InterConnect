#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_
#define DEFAULT_NNWEIGHT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/weight.txt"
#define DEFAULT_NNINPUT_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/input2.txt"
#define DEFAULT_NNMODEL_FILENAME	"/home/yz/myprojects/2025/202508/try_uneven+samos+flipping/2508date/src/Input/newnet2.txt"
#define randomeval

//#define PADDING_RANDOM  // THIS IS JUST FOR DEbugging！

// CNN Random Data Test - Replace CNN inbuffer data with random values (same as LLM)
#define CNN_RANDOM_DATA_TEST  // Enable this to make CNN use pure random data like LLM


// NoC Configuration - Choose one
#define MemNode2_4X4
///#define MemNode4_4X4
//#define MemNode4_8X8
//#define MemNode4_16X16
//#define MemNode4_32X32

// Test Case Configuration - Choose one
  //#define case1_default
//#define case2_samos
//#define case3_affiliatedordering
#define case4_seperratedordering
//#define case5_MOSAIC1
//#define case6_MOSAIC2


//#define YZLLMSwitchON
constexpr int DIM_MODEL = 4096; 
constexpr int NUM_HEAD = 32; 
constexpr int SEQUENCE_LENGTH = 512; 
constexpr int D_HEAD = DIM_MODEL / NUM_HEAD;
#define LLM_OUTPUT_PATH "src/output/"
#define LLM_TEST_CASE 2
#define LLM_DEBUG_LEVEL 1
#define LLM_RANDOM_SEED 0

// LLM Data Mode - Toggle between weight-based and input-based
// #define LLM_INPUT_BASED  // Comment this out for weight-based mode

// Random Data Replacement Test - Replace packet data with random values for testing
// #define LLM_RANDOM_DATA_REPLACE_TEST

// LLM Matrix Data Source - Choose one
#define LLM_USE_RANDOM_MATRICES    // Use randomly generated matrices
// #define LLM_USE_REAL_MATRICES   // Use real LLaMA matrices from files


// Test Case Logic
#if defined(case1_default)
    #define rowmapping
#elif defined(case2_samos)
    #define samos
#elif defined(case3_affiliatedordering)
    #define rowmapping
    #define YzAffiliatedOrdering
#elif defined(case4_seperratedordering)
    #define rowmapping
    #define YzAffiliatedOrdering
    #define YZSeperatedOrdering_reArrangeInput
#elif defined(case5_MOSAIC1)
    #define samos
    #define YzAffiliatedOrdering
#elif defined(case6_MOSAIC2)
    #define samos
    #define YzAffiliatedOrdering
    #define YZSeperatedOrdering_reArrangeInput
#else
    #define rowmapping
#endif


#define samplingWindowLength 100
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
#elif defined MemNode4_4X4
	#define PE_X_NUM 4
	#define PE_Y_NUM 4
	#define X_NUM 4
	#define Y_NUM 4
	#define TOT_NUM 16
	#define YZMEMCount 4
#elif defined MemNode4_8X8
	#define PE_X_NUM 8
	#define PE_Y_NUM 8
	#define X_NUM 8
	#define Y_NUM 8
	#define TOT_NUM 64
	#define YZMEMCount 4
#elif defined MemNode4_16X16
	#define PE_X_NUM 16
	#define PE_Y_NUM 16
	#define X_NUM 16
	#define Y_NUM 16
	#define TOT_NUM 256
	#define YZMEMCount 4
#elif defined MemNode4_32X32
	#define PE_X_NUM 32
	#define PE_Y_NUM 32
	#define X_NUM 32
	#define Y_NUM 32
	#define TOT_NUM 1024
	#define YZMEMCount 4
#endif

#define LINK_TIME 2
#define DISTRIBUTION_NUM 20
struct GlobalParams { 
    static char NNmodel_filename[128]; 
    static char NNweight_filename[128]; 
    static char NNinput_filename[128]; 
};
#endif
