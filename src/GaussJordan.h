#include <stdio.h>
#include "UtilKernels.h"

#define MASTER if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

__global__ void gaussJordanKernel(float* _input, float* _result, int dim) {
	extern __shared__ float dynamicMem[];
	MatrixAccess<float> mat(dynamicMem, 2*dim, dim);
	MatrixAccess<float> input(_input, dim, dim);
	MatrixAccess<float> result(_result, dim, dim);
	
	
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	//Copy into shared mem

	if(x < dim) {
		uint2 c = make_uint2(x,y);
		mat[c] = input[c];
	} else {
		uint2 c = make_uint2(x,y);
		if((x-dim) == y ) mat[c] = 1;
		else 		mat[c] = 0;
	}	
	__syncthreads();
	/*
	MASTER {
		for(int j = 0; j < dim; j++) {
			for(int i = 0; i < 2*dim; i++) {
				printf("%f ",mat[i][j]);
			} 
			printf("\n",mat[x+dim][y]);
		}
	}
	*/
	//Do stuff
	for(int i = 0; i < dim; i++) {
		//Normalize line
		if(y == 0) {
			uint2 c = make_uint2(i,i);
			float factor = mat[c];
			c = make_uint2(x,i);
			mat[c] = mat[c]/factor;
		}
		__syncthreads();
		//Reduce left column from input
		if(y != i) {
			uint2 c1 = make_uint2(i,y), c2 = make_uint2(x,y), c3 = make_uint2(x,i);
			float factor = mat[c1];
			mat[c2] = mat[c2]-factor*mat[c3];
		}
		__syncthreads();
	}
	
	//Copy result back
	//printf("%f\n",mat[x+dim][y]);
	uint2 c1 = make_uint2(x,y), c2 = make_uint2(x+dim, y);
	if(x < dim) result[c1] = mat[c2];
}

template <class MatrixAccess1, class MatrixAccess2>
__host__ __device__ void gaussJordan(MatrixAccess1& inverse, MatrixAccess2& mat) {
	unsigned int dim = mat.getCols();
	dim3 gs(1,1);
	dim3 bs(2*dim,dim);
	#ifdef DEBUG_ENABLED
	if(inverse.getRows() != inverse.getCols() || mat.getRows() != mat.getCols() || mat.getCols() != inverse.getCols()) {
		printf("Can't invert non square matrix.");
	}
	#endif
	gaussJordanKernel<<<gs,bs, sizeof(float)*2*dim*dim>>>(mat.getRawPointer(), inverse.getRawPointer(), dim);
}
