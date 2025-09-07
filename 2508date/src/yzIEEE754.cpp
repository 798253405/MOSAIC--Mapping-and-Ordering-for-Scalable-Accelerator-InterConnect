/*
 * yzIEEE754.CPP
 *
 *  Created on: Jun 24, 2024
 *      Author: yz
 */

#include "yzIEEE754.hpp"

// ==================================================
// Step 5 辅助函数: IEEE 754浮点数转换
// 功能：支持LLM排序优化中的bit分析
// ==================================================
// 将浮点数转换为IEEE 754二进制表示（用于Step 5排序算法）
std::string float_to_ieee754(float float_num) {
	// 检查浮点数是否为负数
	bool negative = float_num < 0;

	// Handle special cases for positive zero and negative zero
	if (float_num == 0) {
		if (negative) {
			return "1" + std::string(31, '0');  // Negative zero
		} else {
			return "0" + std::string(31, '0');  // Positive zero
		}
	}

	// 从IEEE 754格式中提取符号位、指数和尾数
	// 用于LLM Step 5中分析数据的bit模式
	union {
		float input;
		int output;
	} data;

	data.input = float_num;
	std::bitset<32> bits(data.output);

	// 分离IEEE 754的三个组成部分
	std::string sign_bit = bits[31] ? "1" : "0";      // 符号位（1位）
	std::string exponent = bits.to_string().substr(1, 8);  // 指数位（8位）
	std::string mantissa = bits.to_string().substr(9, 23); // 尾数位（23位）

	// 构造完整的IEEE 754二进制表示
	std::string ieee754_representation = sign_bit + exponent + mantissa;

	return ieee754_representation;
}

// ==================================================
// Step 5 核心函数: 计算IEEE 754表示中的1-bit数量
// 功能：用于LLM排序优化，统计bit翻转的关键指标
// 应用：Step 5.1分离排序和Step 5.2关联排序都使用此函数
// ==================================================
int countOnesInIEEE754(float float_num) {
	// 使用union将float的bit模式转换为unsigned int
	union {
		float f;
		unsigned int i;
	} u;
	u.f = float_num;
	unsigned int ieee754 = u.i;

	// 统计二进制表示中1的个数
	// 这是LLM排序的核心指标：1-bit数量越接近的数据相邻传输时bit翻转越少
	int count = 0;
	while (ieee754 > 0) {
		count += ieee754 & 1;  // 检查最低位是否为1
		ieee754 >>= 1;          // 右移一位
	}
	return count;
}

// ==================================================
// Step 5 排序比较器: 基于IEEE 754中1-bit数量
// 功能：用于LLM Step 5.1分离排序和Step 5.2关联排序
// 目的：将bit数相近的数据排列在一起，减少NoC传输时的bit翻转
// ==================================================
bool compareFloatsByOnes(const float &a, const float &b) {
	// 降序排列：1-bit数多的在前
	// 在LLM排序中会被重组为列主序，使每列内bit数递增
	return countOnesInIEEE754(a) > countOnesInIEEE754(b);
}

// ==================================================
// Step 5 备选方案: 定点数表示的1-bit计数
// 功能：当启用FIXED_POINT_SORTING时，使用定点数代替浮点数
// 应用：可通过宏FIXED_POINT_SORTING切换排序策略
// ==================================================
int countOnesInFixed17(float float_num) {
	std::string fixed17_str = singleFloat_to_fixed17(float_num);
	int count = 0;
	for (char c : fixed17_str) {
		if (c == '1') count++;
	}
	return count;
}

// 定点数排序比较器（LLM Step 5备选方案）
bool compareFloatsByFixed17Ones(const float &a, const float &b) {
	return countOnesInFixed17(a) > countOnesInFixed17(b);
}

// ==================================================
// Step 6: CNN风格半半重组（也用于LLM）
// 功能：将线性数据重组为矩阵并按行组合
// 应用：LLM中通过LLM_CNN_HALFHALF_RESHAPE宏启用
// ==================================================
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


	// Step 5 应用: 根据配置选择排序方式
#ifdef YZSeperatedOrdering_reArrangeInput
	// 分离排序：input和weight独立按列排序（对应LLM Step 5.1）
	 sortMatrix_CNNSeparated(inputData, inputcolnum_per_row, rownum_per_col);
	 sortMatrix_CNNSeparated(weightData, weightcolnum_per_row, rownum_per_col);
#endif
#ifndef YZSeperatedOrdering_reArrangeInput
	 // 关联排序：input跟随weight的bit数排序（对应LLM Step 5.2）  
	 sortMatrix_CNNAffiliated(inputData, weightData,  weightcolnum_per_row, rownum_per_col);
#endif
	// algorithm based (function removed)

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
	// Step 6.1: 按行组合input和weight数据
	// 每行格式：[input第i行] + [weight第i行]
	// 这与LLM中的Query+Key组合方式相同
	std::vector<std::deque<float>> combined_rows(rownum_per_col);

	for (int i = 0; i < rownum_per_col; ++i) { // 遍历每一行
		// 先添加input行的元素
		combined_rows[i].insert(combined_rows[i].end(), input_rows[i].begin(),
				input_rows[i].end());
		// 再添加weight行的元素
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
	// Step 6.2: 将重组后的数据写回原始容器
	// 这些数据将通过NoC传输（Step 7）
	dq.clear(); // 清空原始数据
	for (const auto &row : combined_rows) {
		for (const auto &element : row) {
			dq.push_back(element);  // 按行展开存储
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
// ==================================================
// CNN版本的Step 5.1: 分离排序实现
// 功能：对单个矩阵按列独立排序，减少列内bit翻转
// 原理：将bit数相近的元素放在同一列，传输时减少翻转
// 示例： row0: 0000 1111 1110 0011
//       row1: 1000 1000 1110 0011
// 排序后：按列重组，使bit数递增
// ==================================================
void sortMatrix_CNNSeparated(std::deque<float> &dq, int colnum_per_row,
		int rownum_per_col) { // 25: 8 value in one flit colnumperrow= 8   4flits->rownumpercol= 4

	// CNN Step 5.1.1: 对整个数据按bit数排序
	// 这个步骤与LLM Step 5.1.1完全相同
#ifdef FIXED_POINT_SORTING
	std::sort(dq.begin(), dq.end(), compareFloatsByFixed17Ones);  // 定点数排序
#else
	std::sort(dq.begin(), dq.end(), compareFloatsByOnes);         // 浮点数按bit数排序
#endif
	// CNN Step 5.1.2: 将排序后的数据按列主序填充
	// 每个flit对应一行，确保同列元素bit数相近
	std::vector<std::deque<float>> rows(rownum_per_col); //rownum_per_col = the number of flits
	// CNN Step 5.1.3: 按列填充矩阵
	// 外循环：遍历列，内循环：填充每列的元素
	int row_index = 0;
	int col_index = 0;
	for (float num : dq) {
		rows[row_index].push_back(num);
		col_index++;
		if (col_index == colnum_per_row) {
			row_index++;
			col_index = 0;
		}
		if (row_index == rownum_per_col) { // 达到最大行数，重置
			row_index = 0;
		}
	}
	// CNN Step 5.1.4: 按行序写回数据
	// 虽然按列填充，但存储仍是行主序
	dq.clear();
	for (const auto &row : rows) {
		for (const auto &element : row) {
			dq.push_back(element);
		}
	}

}

// ==================================================
// CNN版本的Step 5.2: 关联排序实现
// 功能：weight按bit数排序，input保持与weight的对应关系
// 目的：保持input-weight对的相关性，同时优化bit翻转
// 这与LLM中的Query-Key关联排序思想相同
// ==================================================
void sortMatrix_CNNAffiliated(std::deque<float> &inputData,
		std::deque<float> &weightData, int weightcolnum_per_row,
		int weightrownum_per_col) {
	// Check sizes
	/*
	std::cout << " weightData.size() " << weightData.size()
			<< weightcolnum_per_row << " weightcolnum_per_row, "
			<< weightrownum_per_col << std::endl;
	 */

	// CNN Step 5.2.1: 创建索引数组，用于跟踪元素原始位置
	// 这个方法与LLM Step 5.2.1完全相同
	std::vector<int> indices(weightData.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;  // 初始化索引
	}

	// CNN Step 5.2.2: 根据weight的bit数对索引排序
	// 不直接移动数据，而是排序索引，这样input可以跟随相同顺序
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
#ifdef FIXED_POINT_SORTING
		return compareFloatsByFixed17Ones(weightData[i], weightData[j]);  // 定点数比较
#else
		return compareFloatsByOnes(weightData[i], weightData[j]);         // 浮点数bit数比较
#endif
	});

	// CNN Step 5.2.3: 根据排序后的索引重新排列input和weight
	// input和weight保持配对关系，都按weight的bit数顺序排列
	std::deque<float> sortedWeights;
	std::deque<float> sortedInput;
	for (int idx : indices) {
		sortedWeights.push_back(weightData[idx]);  // weight按bit数排序
		sortedInput.push_back(inputData[idx]);     // input跟随weight的顺序

		// below for check whether this func works properly
		//sortedWeights.push_back( idx );
		 //sortedInput.push_back( idx );
		 //sortedWeights.push_back(countOnesInIEEE754(weightData[idx]));
		//sortedInput.push_back(countOnesInIEEE754(inputData[idx]));

	}

	// CNN Step 5.2.4: 将排序后的数据写回原容器
	// 这些数据将在Step 6中进行半半重组
	inputData = sortedInput;
	weightData = sortedWeights;


}

// ==================================================
// Step 7 辅助函数: 计算bit翻转数量
// 功能：计算两个浮点数在NoC传输时的bit翻转数
// 应用：用于评估LLM排序优化的效果
// ==================================================
int calculate32BitDiff(float a, float b) {
	// 将浮点数的bit模式转换为无符号整数
	uint32_t a_bits = *reinterpret_cast<uint32_t*>(&a);
	uint32_t b_bits = *reinterpret_cast<uint32_t*>(&b);
	// 使用XOR计算不同的bit位，然后统计1的数量
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

/* backup
 std::string singleFloat_to_fixed35(float float_num) {
	// Define the range limits based on the Q3.5 format
	float min_value = -8.0; // -2^3
	float max_value = 7.96875; // 2^3 - 2^(-5)

	// Clamp the input float to the representable range
	float clamped = std::max(min_value, std::min(float_num, max_value));

	// Convert to an integer representation of Q3.5
	int fixed_point = static_cast<int>(round(clamped * 32)); // Multiply by 32 to shift decimal five places

	// Handle negative numbers with two's complement if necessary
	if (fixed_point < 0) {
		fixed_point = (1 << 8) + fixed_point; // 1 << 8 is 256, which is 2^8, the bit count for Q3.5
	}

	// Convert to binary string
	std::bitset<8> bits(static_cast<unsigned long>(fixed_point));
	return bits.to_string();
}
 *
 */
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


