/*
 * yzIEEE754.CPP
 *
 *  Created on: Jun 24, 2024
 *      Author: yz
 */

#include "yzIEEE754.hpp"

// Function to convert float to IEEE 754 representation
std::string float_to_ieee754(float float_num) {
	// Check if the float is negative
	bool negative = float_num < 0;

	// Handle special cases for positive zero and negative zero
	if (float_num == 0) {
		if (negative) {
			return "1" + std::string(31, '0');  // Negative zero
		} else {
			return "0" + std::string(31, '0');  // Positive zero
		}
	}

	// Extract sign, exponent, and mantissa from IEEE 754 format
	union {
		float input;
		int output;
	} data;

	data.input = float_num;
	std::bitset<32> bits(data.output);

	// Separate the components
	std::string sign_bit = bits[31] ? "1" : "0";
	std::string exponent = bits.to_string().substr(1, 8);
	std::string mantissa = bits.to_string().substr(9, 23);

	// Construct IEEE 754 representation
	std::string ieee754_representation = sign_bit + exponent + mantissa;
	return ieee754_representation;


//debug
	/*
	 std::string result;
	    for (int i = 0; i < 32; i++) {
	        result += (rand() % 2) ? '1' : '0';  // 每个位独立随机
	    }
	    return result;
	    */
	    //return std::string(32, '0');  // 生成 32 个 '0'
}

// Function to count the number of 1 bits in IEEE 754 representation
int countOnesInIEEE754(float float_num) {
	union {
		float f;
		unsigned int i;
	} u;
	u.f = float_num;
	unsigned int ieee754 = u.i;

	int count = 0;
	while (ieee754 > 0) {
		count += ieee754 & 1;
		ieee754 >>= 1;
	}
	return count;
}

// Comparator function for sorting based on number of 1 bits in IEEE 754 representation
bool compareFloatsByOnes(const float &a, const float &b) {
	return countOnesInIEEE754(a) < countOnesInIEEE754(b);  // Changed to ascending order for column-major organization
}

void cnnReshapeFlatToInputWeightMatrix(std::deque<float> &dq, int t_inputCount,
		int t_weightCount, int inputcolnum_per_row, int weightcolnum_per_row,
		int totalcolnum_per_row, int rownum_per_col) { //weightcout is the kernel size, 25 for 5x5 kernel //for example , one row 8elements =  inputcolnum_per_row=4 input+  weightcolnum_per_row=4 input



	// Size of each section
	int inputSize = t_inputCount;
	int weightSize = t_weightCount;
	//std::cout << " ieee754line71 combined rows ok " << inputcolnum_per_row << std::endl;
	// Check if deque size is sufficient
	if (dq.size() < (inputSize + weightSize)) {
		std::cerr << "Error: Not enough elements in the deque!  cycles= "
				<< cycles << " dq.size()= " << dq.size()
				<< " (inputSize + weightSize)= " << (inputSize + weightSize)
				<< std::endl;
		return;
	}
	//std::cout<<cycles<<" check dequeuesize " <<" "<<dq.size() <<" "<< (inputSize + weightSize)<< std::endl;
	// Separate input and weight data  // flatten format
	std::deque<float> inputData(dq.begin(), dq.begin() + inputSize);
	std::deque<float> weightData(dq.begin() + inputSize,
			dq.begin() + inputSize + weightSize);



//#define printCombinedMatrix
#ifdef printCombinedMatrix
	int tprintnextRowIndex = 0;
	//print matrix
   // print dq
		std::cout << "dq: rows if input and rows of weights not any operations" << std::endl;

		for (const auto &element : dq) {
			std::cout << std::setw(10) << element << " ";
			tprintnextRowIndex++;
			if (tprintnextRowIndex == totalcolnum_per_row) {
				std::cout << std::endl;
				tprintnextRowIndex = 0;
			}
		}
		std::cout << std::endl;

		std::cout << "input data not any operations" << std::endl;
		for (const auto &element : inputData) {
					std::cout << std::setw(10) << element << " ";
					tprintnextRowIndex++;
					if (tprintnextRowIndex == inputcolnum_per_row) {
						std::cout << std::endl;
						tprintnextRowIndex = 0;
					}
				}
				std::cout << std::endl;
				tprintnextRowIndex = 0;
				std::cout << "weight data not any operations" << std::endl;
						for (const auto &element : weightData) {
									std::cout << std::setw(10) << element << " ";
									tprintnextRowIndex++;
									if (tprintnextRowIndex == weightcolnum_per_row) {
										std::cout << std::endl;
										tprintnextRowIndex = 0;
									}
								}
								std::cout << std::endl;
#endif


	//0 bits count based
#ifdef YZSeperatedOrdering_reArrangeInput
								//below two should be enabled at the same time
	 sortMatrix_CNNSeparated(inputData, inputcolnum_per_row, rownum_per_col);
	 sortMatrix_CNNSeparated(weightData, weightcolnum_per_row, rownum_per_col);
#endif
#ifdef YzAffiliatedOrdering
	//this is old and for debug
	 //sortMatrix_CNNSeparated(weightData, weightcolnum_per_row, rownum_per_col); // rearrange weights, and rearrange input accrodingly
	 // this is current version
	 sortMatrix_CNNAffiliated(inputData, weightData,  weightcolnum_per_row, rownum_per_col);

#endif
	// algorithm based
	// rearrangeByAlgorithm(weightData,  weightcolnum_per_row,rownum_per_col);

	std::vector<std::deque<float>> input_rows(rownum_per_col); // one row contains "colnum_per_row" elements //for example， overall 40 = 5row *8 elements。 Inside one row， the left 4 elements are inputs the right 4 elements are weights
	std::vector<std::deque<float>> weight_rows(rownum_per_col); // one row contains "colnum_per_row" elements
	// convert 1 row to matrix. and  padding zero elements from flatten payload to matrix payload
	// Calculate the number of columns required  // to understand, assumming 4 rows, 16 cols
	for (int col_index = 0; col_index < inputcolnum_per_row; col_index++) {	// first col, 4 element, then next col 4elemetn, next col 4 elements...
		for (int row_index = 0; row_index < rownum_per_col; row_index++) {
			if (col_index * rownum_per_col + row_index < inputData.size())// fill elements
					{
				input_rows[row_index].push_back(
						inputData[col_index * rownum_per_col + row_index]);
			}

			else {  //else padding zeros
				input_rows[row_index].push_back(0.0f);
			}
			//std::cout << inputcolnum_per_row << " " << col_index	<< " ieee754line99 combined rows ok " << col_index << std::endl;
		}
	}
	// Calculate the number of columns required  // to understand, assumming 4 rows, 16 cols
	for (int col_index = 0; col_index < weightcolnum_per_row; col_index++) { // first col, 4 element, then next col 4elemetn, next col 4 elements...
		for (int row_index = 0; row_index < rownum_per_col; row_index++) {
			if (col_index * rownum_per_col + row_index < weightData.size()) // fill elements
					{
				weight_rows[row_index].push_back(
						weightData[col_index * rownum_per_col + row_index]);
			}

			else {  //else padding zeros
				weight_rows[row_index].push_back(0.0f);
			}
			//	std::cout << " inputcolnum_per_row " << inputcolnum_per_row << " col_index " << col_index	<< " ieee754line99 combined rows ok " << col_index << std::endl;
		}
	}
	// Combine rows
	std::vector<std::deque<float>> combined_rows(rownum_per_col);

	for (int i = 0; i < rownum_per_col; ++i) { // row

		//std::cout << combined_rows.size() << "  combined_rows.size()line144 " 	<< combined_rows[i].size() << " " << input_rows[i].size() << " "	<< weight_rows[i].size() << std::endl;
		combined_rows[i].insert(combined_rows[i].end(), input_rows[i].begin(),
				input_rows[i].end());
		combined_rows[i].insert(combined_rows[i].end(), weight_rows[i].begin(),
				weight_rows[i].end());
		//std::cout << combined_rows.size() << " " << combined_rows[i].size() 	<< "  combined_rows.size()lineafterinsert" << std::endl;
		if (combined_rows[i].size() != totalcolnum_per_row) { // 4 flits = 4 dequeues.  How many floating points inside dequeue depends on  how many floating point inside one singel flit
			std::cerr
					<< "Error: totalcolnum_per_row not equal toombined_rows.size().size() "
					<< combined_rows.size() << " " << combined_rows[i].size()
					<< " " << inputcolnum_per_row << " " << input_rows[i].size()
					<< " " << weight_rows[i].size() << " " << rownum_per_col
					<< std::endl;
			assert(
					false
							&& "Error: totalcolnum_per_row not equal toombined_rows.size().size() ");
			return;
		}
	}

	// original dq is: k*k inputs, k*k weights, 1 bias. For example, 25 inputs and 25 weights.

#ifdef printCombinedMatrix
	int printnextRowIndex1 = 0;
	std::cout << "dq:before halfinput half weight " << std::endl;
	for (const auto &element : dq) {
				std::cout << std::setw(10) << element << " ";
				printnextRowIndex1++;
				if (printnextRowIndex1 == totalcolnum_per_row) {
					std::cout << std::endl;
					printnextRowIndex1 = 0;
				}
			}
			std::cout << std::endl;
#endif
	//write back to dq
	dq.clear(); // reset
	for (const auto &row : combined_rows) {
		for (const auto &element : row) {
			dq.push_back(element);
		}
	}

#ifdef printCombinedMatrix
	int printnextRowIndex = 0;
	//print matrix
	{
		// print  original Input data. Data is not changed, just change the print format.
		std::cout << "Orded Input Rows nocol-major nopadding zero:" << std::endl;
		/*
		for (const auto &element : inputData) {
			std::cout << std::setw(10) << element << " ";
			printnextRowIndex++;
			if (printnextRowIndex == inputcolnum_per_row) {
				std::cout << std::endl;
				printnextRowIndex = 0;
			}
		}
		std::cout << std::endl;
*/
		/*
		// 打印 Input Rows
		std::cout << "OrdedInput Rows colmajor +zero: " << std::endl;
		for (const auto &row : input_rows) {
			for (const auto &element : row) {
				std::cout << std::setw(10) << element << " "; // 使用 setw 调整输出宽度
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
*/
		/*
		// print  original wegiht  data. Data is not changed, just change the print format.
		std::cout << " Orded  wegiht Rows  nocol-major nopadding zero: (note: input and weight already one to one, as no more ordering will be done)" << std::endl;
		printnextRowIndex = 0;
		for (const auto &element : weightData) {
			std::cout << std::setw(10) << element << " ";
			printnextRowIndex++;
			if (printnextRowIndex == weightcolnum_per_row) {
				std::cout << std::endl;
				printnextRowIndex = 0;
			}
		}
		std::cout << std::endl;
		*/
		/*
		// 打印 Weight Rows
		std::cout << "Weight Rows:" << std::endl;
		for (const auto &row : weight_rows) {
			for (const auto &element : row) {
				std::cout << std::setw(10) << element << " "; // 使用 setw 调整输出宽度
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
*/

		/*// should equla to print dq
		// 打印 Combined Rows
		std::cout << "Combined Rows:" << std::endl;
		for (const auto &row : combined_rows) {
			for (const auto &element : row) {
				std::cout << std::setw(10) << element << " "; // 使用 setw 调整输出宽度
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
*/

		 // print dq
				std::cout << "dqafterAllprocess:" << std::endl;
				printnextRowIndex = 0;
				for (const auto &element : dq) {
					std::cout << std::setw(10) << element << " ";
					printnextRowIndex++;
					if (printnextRowIndex == totalcolnum_per_row) {
						std::cout << std::endl;
						printnextRowIndex = 0;
					}
				}
				std::cout << std::endl;

	}
#endif

}
// task/packet  level all few-zero bits are left, all middel 0-bits are in the center, all many-zero bits are right.
//for example: two rows,  row0:  0000 1111 1110 0011  row1:  1000 1000 1110 0011 should be:  0000 1000    1000 0011   0011 0011     1110  1111
//then convert to two rows: row0  0000 1000 0011  1110   row 1:  1000    0011  0011  1111
void sortMatrix_CNNSeparated(std::deque<float> &dq, int colnum_per_row,
		int rownum_per_col) { // 25: 8 value in one flit colnumperrow= 8   4flits->rownumpercol= 4

	// Sort each row independently based on IEEE 754 bit counts //sort inside weights
	std::sort(dq.begin(), dq.end(), compareFloatsByOnes);
// put the sorted number back to one row. Make sure the order is col-major
//
	std::vector<std::deque<float>> rows(rownum_per_col); //rownum_per_col = the number of flits
	// Fill rows with elements from dq
	int row_index = 0;
	int col_index = 0;
	for (float num : dq) {
		rows[row_index].push_back(num);
		col_index++;
		if (col_index == colnum_per_row) {
			row_index++;
			col_index = 0;
		}
		if (row_index == rownum_per_col) { // reach max row number,reset
			row_index = 0;
		}
	}
	dq.clear();
	for (const auto &row : rows) {
		for (const auto &element : row) {
			dq.push_back(element);
		}
	}

}

// Function to rearrange the deque and an associated weights deque
void sortMatrix_CNNAffiliated(std::deque<float> &inputData,
		std::deque<float> &weightData, int weightcolnum_per_row,
		int weightrownum_per_col) {
	// Check sizes
	/*
	std::cout << " weightData.size() " << weightData.size()
			<< weightcolnum_per_row << " weightcolnum_per_row, "
			<< weightrownum_per_col << std::endl;
*/

	// Create a vector of indices and sort it according to the weights
	std::vector<int> indices(weightData.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}

	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return compareFloatsByOnes(weightData[i], weightData[j]);
	});

	// Rearrange weightData and inputData based on sorted indices
	std::deque<float> sortedWeights;
	std::deque<float> sortedInput;
	for (int idx : indices) {
		sortedWeights.push_back(weightData[idx]);
		sortedInput.push_back(inputData[idx]);

		// below for check whether this func works properly
		//sortedWeights.push_back( idx );
		 //sortedInput.push_back( idx );
		 //sortedWeights.push_back(countOnesInIEEE754(weightData[idx]));
		//sortedInput.push_back(countOnesInIEEE754(inputData[idx]));

	}

	// Assign sorted data back to the original deques
	inputData = sortedInput;
	weightData = sortedWeights;


}

// 计算两个浮点数的位差
int calculate32BitDiff(float a, float b) {
	uint32_t a_bits = *reinterpret_cast<uint32_t*>(&a);
	uint32_t b_bits = *reinterpret_cast<uint32_t*>(&b);
	return std::bitset<32>(a_bits ^ b_bits).count();
}

// 计算矩阵的总位差和
int calculateTotalBitDiffSum(const std::vector<std::vector<float>> &matrix) {
	int totalSum = 0;
	int rows = matrix.size();
	int cols = matrix[0].size();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			for (int k = i + 1; k < rows; ++k) {
				totalSum += calculate32BitDiff(matrix[i][j], matrix[k][j]);
			}
		}
	}
	return totalSum;
}

// 将 std::deque<float> 转换为 m x n 矩阵
std::vector<std::vector<float>> dequeToMatrix(const std::deque<float> &dq,
		int m, int n) {
	std::vector<std::vector<float>> matrix(m, std::vector<float>(n));
	int index = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			matrix[i][j] = dq[index++];
		}
	}
	return matrix;
}

// 将 m x n 矩阵转换回 std::deque<float>
std::deque<float> matrixToDeque(const std::vector<std::vector<float>> &matrix) {
	std::deque<float> dq;
	for (const auto &row : matrix) {
		dq.insert(dq.end(), row.begin(), row.end());
	}
	return dq;
}

// 打印矩阵
void printMatrix(const std::vector<std::vector<float>> &matrix,
		const std::string &label) {
	std::cout << label << std::endl;
	for (const auto &row : matrix) {
		for (float val : row) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}
}

// 优化矩阵
void optimizeMatrix(std::deque<float> &dq, int m, int n) {
	// 将 deque 转换为矩阵
	std::vector<std::vector<float>> matrix(m, std::vector<float>(n));
	int index = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			matrix[i][j] = dq[index++];
		}
	}

	// 扁平化矩阵
	std::vector<float> flatMatrix(dq.begin(), dq.end());
	int totalElements = m * n;
	std::vector<int> indices(totalElements);
	std::iota(indices.begin(), indices.end(), 0);

	// 初始化模拟退火参数
	double temperature = 1000.0;
	double coolingRate = 0.995;
	int iterationCount = 0;
	int minSum = calculateTotalBitDiffSum(matrix);
	std::vector<std::vector<float>> bestMatrix = matrix;
	std::cout << "initial bitdiffSum: " << minSum << std::endl;
	// 模拟退火过程
	while (temperature > 1e-6) {
		// 生成新解
		std::vector<int> newIndices(indices);
		std::random_shuffle(newIndices.begin(), newIndices.end());
		std::vector<std::vector<float>> newMatrix(m, std::vector<float>(n));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				newMatrix[i][j] = flatMatrix[newIndices[i * n + j]];
			}
		}

		int newSum = calculateTotalBitDiffSum(newMatrix);
		int delta = newSum - minSum;
		if (delta < 0
				|| (std::exp(-delta / temperature)
						> static_cast<double>(rand()) / RAND_MAX)) {
			minSum = newSum;
			bestMatrix = newMatrix;
			indices = newIndices;
		}

		// 降低温度
		temperature *= coolingRate;
		iterationCount++;
		if (iterationCount % 100 == 0) {
			//std::cout << "Iteration: " << iterationCount << ", Temperature: " << temperature << std::endl;
		}
	}

	// 打印优化后的矩阵
	std::cout << "Optimized matrix:" << std::endl;
	printMatrix(bestMatrix, "finalmatrix");
	std::cout << "Optimized bitdiffSum: " << minSum << std::endl;

	//wrote  back to dq
	dq.clear(); // 清空 deque 以避免旧数据

	for (const auto &row : bestMatrix) {
		for (float value : row) {
			dq.push_back(value); // 将每个矩阵元素添加到 deque 中
		}
	}
}

void rearrangeByAlgorithm(std::deque<float> &dq, int colnum_per_row,
		int rownum_per_col) { // example 25 elemets : 8 value in one flit colnumperrow= 8   4flits->rownumpercol= 4 other 7 element are padding zeros

	// 计算两个整数之间的位差
	float a = 0, b = 1;
	// 将浮点数转换为二进制表示
	uint32_t a_bits = *reinterpret_cast<uint32_t*>(&a);
	uint32_t b_bits = *reinterpret_cast<uint32_t*>(&b);
	// 计算异或结果
	uint32_t diff_bits = a_bits ^ b_bits;

	std::bitset<32> binaryReps_a(a_bits);
	std::bitset<32> binaryReps_b(b_bits);
	std::bitset<32> binaryReps_diff(diff_bits);

	std::cout << binaryReps_a << " " << binaryReps_b << " diff_bits "
			<< binaryReps_diff << " diff_bitsCount "
			<< std::bitset<32>(diff_bits).count() << std::endl;

	optimizeMatrix(dq, rownum_per_col, colnum_per_row);
}
;
;
;
;

// Function to convert float to fixed-point-8 binary (Q3.5 format)
std::string singleFloat_to_fixed17(float float_num) {
	 // Define the range limits based on the Q1.7 format
		    float min_value = -1.0; // -2^0 (including sign)
		    float max_value = 0.9921875; // 2^0 - 2^(-7)

		    // Clamp the input float to the representable range
		    float clamped = std::max(min_value, std::min(float_num, max_value));

		    // Convert to an integer representation of Q1.7
		    int fixed_point = static_cast<int>(round(clamped * 128)); // Multiply by 128 to shift decimal seven places

		    // Handle negative numbers with two's complement if necessary
		    if (fixed_point < 0) {
		        fixed_point = (1 << 8) + fixed_point; // 1 << 8 is 256, which is 2^8, the bit count for Q1.7
		    }

		    // Convert to binary string
		    std::bitset<8> bits(static_cast<unsigned long>(fixed_point));
		    return bits.to_string();
}
// Function to process a deque of floats and print their binary formats
void print_FlitPayload(const std::deque<float> &floatDeque) {
	for (const auto &num : floatDeque) {
		std::string ieee754 = float_to_ieee754(num);
		std::string fixed35 = singleFloat_to_fixed17(num);
		std::cout << " Float: " << num;
// std::cout<< " IEEE 754: " << ieee754;
		// std::cout << " Fixed3,5: " << fixed35;
	}
	std::cout << " " << std::endl;
}
