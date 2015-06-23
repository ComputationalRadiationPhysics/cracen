/*
#include <iostream>
#include <thrust/device_vector.h>
//#include "../Device/UtilKernels.h"
#include "../Device/GaussJordan.hpp"
#include "../Config/FitFunctions/FitFunction.hpp"
#include "Util.hpp"
#include <cstdio>

typedef Polynom<2> Fit;
__global__ void gaussJordanTest(float* mat, float* inverse, const unsigned int dim) {
	MatrixAccess<> Mat(mat, dim, dim), Inverse(inverse, dim,dim);
	gaussJordan<Fit,256>(Mat, Inverse);
}

int main(int argc, char** argv) {
	const unsigned int dim = Fit::numberOfParams;
	
	thrust::device_vector<float> mat(dim*dim), inverse(dim*dim);
	mat[0] = 1; mat[1] = 2; mat[2] = 3;
	mat[3] = 4; mat[4] = 5; mat[5] = 6;
	mat[6] = 7; mat[7] = 8; mat[8] = 10;

	gaussJordanTest<<<1, 256>>>(pcast(mat), pcast(inverse), dim);
	
	printMat(mat, dim, dim);
	printMat(inverse, dim, dim);
	
	
	return 0;
}
*/
