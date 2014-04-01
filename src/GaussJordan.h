#include <stdio.h>
#include "UtilKernels.h"

#define MASTER if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

__global__ void gaussJordanKernel(float* _input, float* _result, int dim) {
	extern __shared__ float dynamicMem[];
	MatrixAccess<float> mat(2*dim, dynamicMem);
	MatrixAccess<float> input(dim, _input);
	MatrixAccess<float> result(dim, _result);
	
	
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	//Copy into shared mem
	if(x < dim) {
		mat[x][y] = input[x][y];
	} else {
		if((x-dim) == y ) mat[x][y] = 1;
		else 		mat[x][y] = 0;
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
			float factor = mat[i][i];
			mat[x][i] = mat[x][i]/factor;
		}
		__syncthreads();
		//Reduce left column from input
		if(y != i) {
			float factor = mat[i][y];
			mat[x][y] = mat[x][y]-factor*mat[x][i];
		}
		__syncthreads();
	}
	
	//Copy result back
	//printf("%f\n",mat[x+dim][y]);
	if(x < dim) result[x][y] = mat[x+dim][y];
}

template <class T>
void gaussJordan(T* mat, T* result, int dim) {
	dim3 gs(1,1);
	dim3 bs(2*dim,dim);
	gaussJordanKernel<<<gs,bs, sizeof(float)*2*dim*dim>>>(mat, result, dim);
}
