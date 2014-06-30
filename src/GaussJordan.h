#include <stdio.h>
#include "UtilKernels.h"

template <class Fit, unsigned int bs, class MatrixAccess1, class MatrixAccess2>
DEVICE void gaussJordan(MatrixAccess2& result, MatrixAccess1& input) {
	const unsigned int dim = input.getCols();
	__shared__ float dynamicMem[Fit::numberOfParams*Fit::numberOfParams*2];
	MatrixAccess<float> mat(dynamicMem, 2*dim, dim);
	
	#ifdef DEBUG_ENABLED
	if(input.getCols() != Fit::numberOfParams) {
		printf("Some strange error occured. Check GaussJordan.h for additional informations. Abort.'");
		return;
	}
	if(input.getCols() != input.getRows()) {
		printf("Only square matrices can be inverted. Abort");
		return;
	}
	if(input.getCols() != result.getCols() && input.getRows() != result.getRows()) {
		printf("Result and input matrix must have the same dimensions. Abort.");
		return;
	}
	if(dim > 8) {
		printf("GaussJordan does not work for matrices with dimensions bigger than 8x8.\n");	
		return;
	}
	#endif

	//TODO everything below this line
	int x = threadIdx.x%(2*dim);
	int y = threadIdx.x/(2*dim);

	//Copy into shared mem
	if(y < dim) {
		uint2 c = make_uint2(x,y);
		if(x < dim) {
			mat[c] = input[c];
		} else if(x < 2*dim) {
			mat[c] = (x-dim) == y;
		}
	}
	__syncthreads();
	
	for(int i = 0; i < dim; i++) {
		//Normalize line
		float factor = mat[make_uint2(i,i)];
		__syncthreads();
		if(y == 0 && x < 2*dim) {
			uint2 c = make_uint2(x,i);
			mat[c] = mat[c]/factor;
		}
		
		uint2 c1 = make_uint2(i,y), c2 = make_uint2(x,y), c3 = make_uint2(x,i);
		if(y < dim) factor = mat[c1];
		__syncthreads();
		//Reduce left column from input
		if(y != i && x < 2*dim && y < dim) {
			mat[c2] = mat[c2]-factor*mat[c3];
		}
		__syncthreads();
	}
	
	//Copy result back
	//printf("%f\n",mat[x+dim][y]);
	uint2 c1 = make_uint2(x,y), c2 = make_uint2(x+dim, y);
	if(y < dim && x < dim) result[c1] = mat[c2];
}
