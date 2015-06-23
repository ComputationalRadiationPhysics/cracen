/*
#include <thrust/device_vector.h>
#include "../Device/UtilKernels.h"
#include "../Device/GaussJordan.h"
#include "Util.h"

int main(int argc, char** argv) {
	int rows = 5;
	int cols = 2;
	thrust::device_vector<float> A(rows*cols), b(rows), ATb(cols), s(rows), G(cols*cols), G_inverse(cols*cols);
	A[0]=4; A[1]=4;
	A[2]=2; A[3]=13;
	A[4]=0; A[5]=16;
	A[6]=1; A[7]=0;
	A[8]=0; A[9]=1;
	b[0]=0;
	b[1]=3;
	b[2]=4;
	b[3]=0;
	b[4]=0;

	//calc A^T*A => G
	
	transpose(pcast(A), rows, cols);
	handleLastError();

	orthogonalMatProd(pcast(A), pcast(G), cols, rows);
	handleLastError();
	std::cout << "G: " << std::endl;
	printMat(G, cols, cols);

	//calc G^-1
	gaussJordan(pcast(G), pcast(G_inverse), cols);
	handleLastError();
	std::cout << "G^-1: " << std::endl;
	printMat(G_inverse, cols, cols);
	
	//calc AT*b
	matProduct(pcast(A), pcast(b), pcast(b), cols, rows, rows, 1);
	matProduct(pcast(G_inverse), pcast(b), pcast(s), cols, cols, cols, 1);
	//calc G^-1*b => s
	handleLastError();
		
	printMat(s, 1, cols);

	return 0;
}
*/
