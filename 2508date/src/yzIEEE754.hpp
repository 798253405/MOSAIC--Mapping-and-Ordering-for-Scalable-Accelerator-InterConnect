#ifndef YZIEEE754_HPP
#define YZIEEE754_HPP
#include <cassert>
#include <iostream>
#include <bitset>
#include <deque>
#include <algorithm>
#include <vector>
#include <iomanip>  // 包含 std::setw
#include "parameters.hpp"

#include <numeric> // For std::iota
#include <cmath>   // For std::exp
extern unsigned int cycles;
// Function declaration
std::string float_to_ieee754(float float_num);

int countOnesInIEEE754(float float_num);
bool compareFloatsByOnes(const float &a, const float &b);

// Fixed-point bit count functions
int countOnesInFixed17(float float_num);
bool compareFloatsByFixed17Ones(const float &a, const float &b);

void reArrangeHalfInputHalfWeight(std::deque<float> &dq, int t_inputCount, int t_weightCount,
		int inputcolnum_per_row, int weightcolnum_per_row,
		int totalcolnum_per_row, int rownum_per_col);

void rearrangeDeque(std::deque<float> &dq, int num_per_row, int num_per_col);

void rearrangeByAlgorithm(std::deque<float> &dq, int colnum_per_row,
		int rownum_per_col);
std::string singleFloat_to_fixed17(float float_num);
void  print_FlitPayload(const std::deque<float>& floatDeque);
void rearrangeDequeAccordingly(std::deque<float> &inputData, std::deque<float> &weightData, int colnum_per_row, int rownum_per_col);


void codingOnBus2503 ();
#endif // YZIEEE754_HPP

