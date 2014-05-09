#ifndef UTILKERNELS_H
#define UTILKERNELS_H

#include <cstdio>
#define DEBUG_ENABLED

//#define DEVICE __device__ __forceinline__
#define DEVICE __device__ __host__
#ifdef DEBUG_ENABLED
#define handleLastError() handle_error( cudaGetLastError(),"Kernel Error occured:\"", __LINE__, __FILE__)
#else
#define handleLastError()
#endif
void handle_error(cudaError_t err, const char* error, int line, const char* file) {
	if(err != cudaSuccess) std::cerr << error << cudaGetErrorString(err) << "\" in Line " << line << " in File " << file << std::endl;
}

template<unsigned int blockSize, class T>
__global__ void matProdKernel(T* left, T* right, T* result, int lrows, int lcols, int rrows, int rcols) {
	//BS Blockdim.x maximal, Blockdim.y = 1;
	__shared__ T sright[4*blockSize];
	__shared__ T sleft[2*blockSize];
	T res = 0;
	
	for(int blocks = 0; blockDim.x*4*blocks < lcols; blocks++) {
		int x = threadIdx.x+blockDim.x*4*blocks;
		int y = 1;
		//Spalte der rechten Matrix in shared Mem laden
		sright[threadIdx.x] = right[x];
		sright[threadIdx.x+blockDim.x] = right[x];
		sright[threadIdx.x+2*blockDim.x] = right[x+2*blockDim.x];
		sright[threadIdx.x+3*blockDim.x] = right[x+3*blockDim.x];
		//Block der linken Matrix in shared Mem laden
		if(x+blockDim.x < lcols) sleft[threadIdx.x] 			= (left[x]*sright[threadIdx.x*rcols]+left[x+blockDim.x]*sright[(threadIdx.x+blockDim.x)*rcols]);
		else if(x < lcols) sleft[threadIdx.x] = left[x]*sright[threadIdx.x*rcols]+left[x+blockDim.x];
		else sleft[threadIdx.x] = 0;
		if(x+3*blockDim.x < lcols) sleft[threadIdx.x+blockDim.x]	= (left[x+2*blockDim.x]*sright[(threadIdx.x+2*blockDim.x)*rcols]+left[x+3*blockDim.x]*sright[(threadIdx.x+3*blockDim.x)*rcols]);
		else if(x+2*blockDim.x < lcols) sleft[threadIdx.x+blockDim.x] = left[x+2*blockDim.x]*sright[(threadIdx.x+2*blockDim.x)*rcols];
		else sleft[threadIdx.x+blockDim.x] = 0;
		__syncthreads();
		
		//printf("sright[%u]=%f\n", threadIdx.x, sright[threadIdx.x]);
		//printf("sright[%u]=%f\n", threadIdx.x+blockDim.x, sright[threadIdx.x+blockDim.x]);
		//Vektor folden
		for(int i = blockDim.x; i >= 1; i>>=1) {
			sleft[threadIdx.x] += sleft[threadIdx.x+i];
			__syncthreads();
		}
		//Teilergebniss in Register speichern
		res += sleft[0];
	}
	if(threadIdx.x == 0) result[0] = res;
	//Register in Ergebniss Matrix schreiben
}

DEVICE uint2 identity(uint2& input) {
	return input;
}

DEVICE uint2 trans(uint2& input) {
	return make_uint2(input.y, input.x);
}

template <class T, uint2 (*AccessMode)(uint2&) = identity>
class MatrixAccess {
private:
	unsigned int rows, cols;
	T* mat;
public:
	DEVICE MatrixAccess(T* mat, unsigned int cols, unsigned int rows) : 
		rows(rows),
		cols(cols),
		mat(mat)
	{}
	DEVICE MatrixAccess<T, trans> transpose() {
		return MatrixAccess<T, trans>(mat, cols, rows);
	}
	DEVICE T& operator[](uint2 pos) {
		return mat[pos.y*cols+pos.x];
	}
	DEVICE unsigned int getRows() {
		return rows;
	}
	DEVICE unsigned int getCols() {
		return cols;
	}
	
	DEVICE T* getRawPointer() {
		return mat;
	}
};
/* DEVICE */ unsigned int calcAlignment(unsigned int var) {
	#ifdef __CUDA_CC__
		int msb = 31 - __clz(var);
	#else
		int msb = 31 - __builtin_clz(var);
	#endif
	return (var == (1 << msb) ? var: 1 << msb+1);
}
template <class MatrixAccess1, class MatrixAccess2, class MatrixAccess3>
/* DEVICE */ void MatMul(MatrixAccess1& result, MatrixAccess2& lhs, MatrixAccess3& rhs) {
	#ifdef DEBUG_ENABLED
	if(lhs.getCols() != rhs.getRows() && rhs.getCols() != result.getCols() && lhs.getRows() != result.getRows()) {
		printf("Can not multiply %ux%u with %ux%u to %ux%u matrix.\n", 
			lhs.getCols(),    lhs.getRows(), 
			rhs.getCols(),    rhs.getRows(), 
			result.getCols(), result.getRows());
	}
	#endif
	const int maxBlockSize = 1024;
	unsigned int bsx, bsy, gsx, gsy; 
	bsx = min(maxBlockSize/2, calcAlignment(lhs.getCols())/4);
	bsy = min(maxBlockSize/bsx,calcAlignment(lhs.getRows()));
	gsx = 1;
	gsy = 1;
	std::cout << "bs(" << bsx << ", " << bsy << "), gs(" << gsx << "," << gsy << ")" << std::endl;
	dim3 blockSize(bsx,bsy);
	dim3 gridSize(gsx, gsy);
	matProdKernel<maxBlockSize><<<gridSize, blockSize>>>(
		lhs.getRawPointer(), 
		rhs.getRawPointer(), 
		result.getRawPointer(), 
		lhs.getRows(), lhs.getCols(), 
		rhs.getRows(), rhs.getCols());
}

#endif
