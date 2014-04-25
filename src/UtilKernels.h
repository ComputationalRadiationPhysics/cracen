#ifndef UTILKERNELS_H
#define UTILKERNELS_H

//#include <stdio.h>

/*
template<class T>
__device__ inline T max(T left, T right) {
	if(left > right) return left;
	else 			 return right;
}
*/

template <class T>
class MatrixAccess {	
public:
	class Proxy {
	private:
		int dim;
		int x;
		T* mat;
	public:
		__device__ Proxy(int dim, T* mat) : dim(dim), mat(mat) {}
		__device__ T& operator[](int y)  {return mat[x+y*dim];}
		__device__ Proxy& operator()(int x1) {x=x1; return *this;}	
	};
	__device__ MatrixAccess(int dim, T* mat) : proxy(dim, mat) {}
	__device__ Proxy& operator[](int x) {return proxy(x);}
private:
	Proxy proxy;
};

#define handleLastError() handle_error( cudaGetLastError(),"Kernel Error occured:\"", __LINE__, __FILE__ )

void handle_error(cudaError_t err, const char* error, int line, const char* file) {
	if(err != cudaSuccess) std::cerr << error << cudaGetErrorString(err) << "\" in Line " << line << " in File " << file << std::endl;
}

template<class T>
__global__ void matProdKernel(T* left, T* right, T* result, int lrows, int lcols, int rrows, int rcols) {

	__shared__ T lleft[32][32];
	__shared__ T lright[32][32];
	
	
	T res = 0;
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	for(int i = 0; i*blockDim.x < lcols; i++) {
		if((i*blockDim.x+threadIdx.x) < lcols && y < lrows) lleft[threadIdx.x][threadIdx.y] = left[y*lcols+(i*blockDim.x+threadIdx.x)];
		else lleft[threadIdx.x][threadIdx.y] = 0;
	
		if((i*blockDim.y+threadIdx.y) < rrows && x < rcols) lright[threadIdx.x][threadIdx.y] = right[(i*blockDim.y+threadIdx.y)*rcols+x];
		else lright[threadIdx.x][threadIdx.y] = 0;
		__syncthreads();
	
	
		for(int i = 0; i < 32; i++) {
		//	printf("res[%i,%i]+=%i*%i\n",x,y,lleft[i][threadIdx.y],lright[threadIdx.x][i]);
			res += lleft[i][threadIdx.y]*lright[threadIdx.x][i];
		}
		__syncthreads();
	}
	//__syncthreads();

	if(x < rcols and y < lrows) result[x+y*rcols] = res;

}

template<class T>
void matProduct(T* left, T* right, T* result,  int lrows, int lcols, int rrows, int rcols) {
	dim3 blockSize(32,32);
	dim3 gridSize(ceil((float) rcols/32),ceil((float) lrows/32));
	matProdKernel<<<gridSize, blockSize>>>(left, right, result, lrows, lcols, rrows, rcols);
}
template<class T>
__global__ void orthogonalMatProdKernel(T* left, T* result, int rows, int cols) {

	__shared__ T lleft[32][32];
	__shared__ T lright[32][32];
	
	T res = 0;
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;

	for(int i = 0; i*blockDim.x < cols; i++) {
		if((i*blockDim.x+threadIdx.x) < cols && y < rows) lleft[threadIdx.x][threadIdx.y] = left[y*cols+(i*blockDim.x+threadIdx.x)];
		else lleft[threadIdx.x][threadIdx.y] = 0;
		
		if((i*blockDim.y+threadIdx.y) < cols && x < rows) lright[threadIdx.x][threadIdx.y] = left[x*cols+(i*blockDim.y+threadIdx.y)];
		else lright[threadIdx.x][threadIdx.y] = 0;
	
		__syncthreads();
	
	
		for(int i = 0; i < 32; i++) {
			res += lleft[i][threadIdx.y]*lright[threadIdx.x][i];
			//if(x == 32 && y == 32) printf("res[%i,%i]+=%i*%i = %i\n",x,y,lleft[i][threadIdx.y],lright[i][threadIdx.x],res);
		}
		__syncthreads();
	}
	//__syncthreads();

	if(x < rows and y < rows) result[x+y*rows] = res;

}

template<class T>
void orthogonalMatProd(T* left, T* result, int rows, int cols) {
	dim3 blockSize(32,32);
	dim3 gridSize(ceil((float) rows/32),ceil((float) rows/32));
	orthogonalMatProdKernel<<<gridSize, blockSize>>>(left, result, rows, cols);
	cudaDeviceSynchronize();
}

template <class T>
__global__ void transposeKernel(T* mat, int rows, int cols) {
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	if(x < cols && y < rows) {
		T temp = mat[y*cols+x];
		__syncthreads();
		mat[x*rows+y] = temp;
	}
}

template <class T>
void transpose(T* mat, int rows, int cols) {
	dim3 blockSize(32,32);
	dim3 gridSize(ceil((float) cols/32),ceil((float) rows/32));
	transposeKernel<<<gridSize,blockSize>>>(mat, rows, cols);
}

template <class T>
__global__ void transposeKernel(T* mat, T* res, int rows, int cols) {
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	if(x < cols && y < rows) {
		res[x*rows+y] = mat[y*cols+x];
	}
}

template <class T>
void transpose(T* mat, T* res, int rows, int cols) {
	dim3 blockSize(32,32);
	dim3 gridSize(ceil((float) cols/32),ceil((float) rows/32));
	transposeKernel<<<gridSize,blockSize>>>(mat, res, rows, cols);
}


#endif
