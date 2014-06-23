#include <iostream>
#include <thrust/device_vector.h>
#include "../UtilKernels.h"
#include "Util.h"
#include <cstdio>

typedef int TYPE;
template <class T>
void cpuMatMul(thrust::device_vector<T>& A, thrust::device_vector<T>& B, thrust::device_vector<T>& C, int lcols, int lrows, int rcols, int rrows ) {
	for ( uint32_t i = 0; i < lrows; i++ ) {
        for ( uint32_t k = 0; k < rrows; k++ ) {
		    for ( uint32_t j = 0; j < rcols; j++ ) {
                // C[i][j] += A[i][k] * B[k][j]
                C[ i * rcols + j ] += A[ i * lcols + k ] * B[ k * rcols + j ];
            }
        }
    }

}

template <class T>
__global__ void gpuMatProduct(T* a, T* b, T* c, unsigned int lc, unsigned int lr, unsigned int rc, unsigned int rr) {
	MatrixAccess<T> left(a, lc, lr), right(b, rc, rr), result(c,rc,lr);
	matProdKernel<256>(result, left, right);
}
template <class T>
__global__ void gpuOrthogonalMatProduct(T* a, T* c, unsigned int cols, unsigned int rows) {
	MatrixAccess<TYPE> right(a, cols, rows), result(c,cols,cols);
	MatrixAccess<T, trans> left = right.transpose();
	matProdKernel<256>(result, left, right);
}

int main(int argc, char** argv) {
    srand( ( unsigned ) time( NULL ) );
    int h2;
	int w1 = h2 = 500, h1 = 75, w2= 75;
	cudaDeviceReset();
	thrust::device_vector<TYPE> A(w1*h1), B(w2*h2), C(h1*w2), D(h1*w2), AT(w1*h1);
	
	//random_mat(A,w1,h1);
	//random_mat(B,w2,h2);
	
	for(int i = 0; i < w1*h1; i++) A[i] = i;
	for(int i = 0; i < w2*h2; i++) B[i] = i;
	//printMat(A, h1, w1);
	//printMat(B, h2, w2);	
	
	std::cout << "CPU Mat Prod" << std::endl;
	cpuMatMul(A,B,D,w1,h1,w2,h2);
	
	std::cout << "GPU Mat Prod" << std::endl;
	gpuMatProduct<<<1,256>>>(pcast(A), pcast(B), pcast(C), w1,h1,w2,h2);
	
	
	//printMat(C, h1, w2);
	//printMat(D, h1, w2);
	
	cudaDeviceSynchronize();
	handleLastError();
	std::cout << "Test Mat Prod" << std::endl;
	bool passed = true;
	for(int i = 0; i < h1*w2; i++) {
		if(C[i] != D[i]) {
			std::cout << "Element " << i << " (" << C[i] << "!=" << D[i] << ") is incorrect!" << std::endl;
			getchar();
			passed = false;
		}
	}
	if(passed) std::cout << "TEST PASSED!" << std::endl;
	
	std::cout << "Test Orthogonal Product" << std::endl;
	
	
	thrust::device_vector<TYPE> C2(w1*w1), D2(w1*w1);
	for(int i = 0; i < w1; i++) {
		for(int j = 0; j < h1; j++) {
			AT[i*h1+j] = A[j*w1+i];
		}
	}
	cpuMatMul(AT,A,D2,h1,w1,w1,h1);	
	gpuOrthogonalMatProduct<<<1,256>>>(pcast(A), pcast(C2), w1, h1);
	handleLastError();
	
	std::cout << "Test Orthogonal Mat Prod" << std::endl;
	passed = true;
	for(int i = 0; i < h1*w2; i++) {
		if(C2[i] != D2[i]) {
			std::cout << "Element " << i << " (" << C2[i] << "!=" << D2[i] << ") is incorrect!" << std::endl;
			getchar();
			passed = false;
		}
	}
	if(passed) std::cout << "TEST PASSED!" << std::endl;
	//printMat(A, h1, w1);
	//printMat(AT, w1, h1);
	//printMat(C2, w1, w1);
	//printMat(D2, w1, w1);
	
	return 0;
}
