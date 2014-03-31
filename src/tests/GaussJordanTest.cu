#include <iostream>
#include <thrust/device_vector.h>
#include "../GaussJordan.h"
#include "Util.h"

int main(int argc, char** argv) {
	thrust::device_vector<float> input(9), inverse(9);
	input[0] = 1; input[1] = 2; input[2] = 3;
	input[3] = 4; input[4] = 5; input[5] = 6;
	input[6] = 7; input[7] = 8; input[8] = 10;
	
	gaussJordan(pcast(input), pcast(inverse), 3);
	
	printMat(input, 3, 3);
	std::cout << std::endl;
	printMat(inverse, 3, 3);
	
	return 0;
}
