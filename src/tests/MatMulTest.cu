#include <iostream>
#include <thrust/device_vector.h>
#include "../UtilKernels.h"
#include "Util.h"
#include <cstdio>

template <class T>
void cpuMatMul(thrust::device_vector<T>& A, thrust::device_vector<T>& B, thrust::device_vector<T>& C, int lcols, int lrows, int rcols, int rrows ) {
	for ( uint32_t i = 0; i < lrows; i++ ) {
        for ( uint32_t j = 0; j < rcols; j++ ) {
            for ( uint32_t k = 0; k < rrows; k++ ) {
                // C[i][j] += A[i][k] * B[k][j]
                C[ i * rcols + j ] += A[ i * lcols + k ] * B[ k * rcols + j ];
            }
        }
    }

}

/* __global__ */ void gpuMatProduct(float* a, float* b, float* c, unsigned int lc, unsigned int lr, unsigned int rc, unsigned int rr) {
	MatrixAccess<float> left(a, lc, lr), right(b, rc, rr), result(c,rc,lr);
	MatMul(result, left, right);
}

int main(int argc, char** argv) {
	cudaDeviceReset();
	int w1 = 250000, h1 = 1, w2= 1, h2 = 250000;
	thrust::device_vector<float> A(w1*h1), B(w2*h2), C(h1*w2), D(h1*w2), AT(w1*h1);
	
	random_mat(A,w1,h1);
	
	random_mat(B,w2,h2);
	
	for(int i = 0; i < w1; i++) A[i] = i;
	for(int i = 0; i < w1; i++) B[i] = i;
	
	std::cout << "CPU Mat Prod" << std::endl;
	cpuMatMul(A,B,D,w1,h1,w2,h2);
	
	std::cout << "GPU Mat Prod" << std::endl;
	gpuMatProduct(pcast(A), pcast(B), pcast(C), w1,h1,w2,h2);
	
	cudaDeviceSynchronize();
	handleLastError();
	std::cout << "Test Mat Prod" << std::endl;
	for(int i = 0; i < h1*w2; i++) {
		if(C[i] / D[i] > 1.0001 || C[i] / D[i] < 0.9999) {
			std::cout << "Element " << i << " (" << C[i] << "!=" << D[i] << ") is incorrect!" << std::endl;
			getchar();
		}
	}
	
	/*
	std::cout << "Test Orthogonal Product" << std::endl;
	
	for(int i = 0; i < w; i++) {
		for(int j = 0; j < h; j++) {
			AT[i*h+j] = A[j*w+i];
		}
	}
	
	//cpuMatMul(A,AT,D,50,100,100,50);
	//orthogonalMatProd(pcast(A), pcast(C), 50, 100);
	cpuMatMul(A,AT,D,h,w,w,h);
	
	handleLastError();
	orthogonalMatProd(pcast(A), pcast(C), h, w);
	handleLastError();
	
	std::cout << "Test Mat Prod" << std::endl;
	for(int i = 0; i < h*w; i++) {
		if(C[i]!=D[i]) {
			std::cout << "Element " << i << " (" << C[i] << "!=" << D[i] << ") is incorrect!" << std::endl;
			getchar();
		}
	}
	*/
	//printMat(A, h, w);
	//printMat(AT, w, h);
	//printMat(C, h, h);
	//printMat(D, h, h);
	return 0;
}
