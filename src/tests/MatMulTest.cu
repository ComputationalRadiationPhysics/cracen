#include <iostream>
#include <thrust/device_vector.h>
#include "UtilKernels.h"
#include <cstdio>

template <class c>
c* pcast(thrust::device_vector<c>& dev) {
	return thrust::raw_pointer_cast(&dev[0]);
}

template <class c>
void printMat(thrust::device_vector<c>& mat, int rows, int cols) {
	for(int j = 0; j < rows; j++) {
		for(int i = 0; i < cols; i++) {
			std::cout << mat[i+cols*j]<< " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template <class T>
static inline void random_mat(thrust::device_vector<T> &matrix, int rows, int cols )
{
    srand( ( unsigned ) time( NULL ) );

    for ( int i = 0; i < rows * cols; ++i)
    {
        matrix[ i ] = ( T )rand();// % 50 - 25;
    }
}

template <class T>
void cpuMatMul(thrust::device_vector<T>& A, thrust::device_vector<T>& B, thrust::device_vector<T>& C, int lrows, int lcols, int rrows, int rcols) {
	for ( uint32_t i = 0; i < lrows; i++ ) {
        for ( uint32_t j = 0; j < rcols; j++ ) {
            for ( uint32_t k = 0; k < rrows; k++ ) {
                // C[i][j] += A[i][k] * B[k][j]
                C[ i * rcols + j ] += A[ i * lcols + k ] * B[ k * rcols + j ];
            }
        }
    }

}

int main(int argc, char** argv) {
	cudaDeviceReset();
	int w = 100, h = 50;
	thrust::device_vector<float> A(100*50), B(100*200), C(50*200), D(50*200), AT(w*h);
	
	random_mat(A,100,50);
	
	random_mat(B,100,200);
	cpuMatMul(A,B,D,50,100,100,200);
	
	matProduct(pcast(A), pcast(B), pcast(C), 50,100,100,200);
	
	
	std::cout << "Test Mat Prod" << std::endl;
	for(int i = 0; i < 50*200; i++) {
		if(C[i]!=D[i]) {
			std::cout << "Element " << i << " (" << C[i] << "!=" << D[i] << ") is incorrect!" << std::endl;
			getchar();
		}
	}
	
	
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
	//printMat(A, h, w);
	//printMat(AT, w, h);
	//printMat(C, h, h);
	//printMat(D, h, h);
	return 0;
}
