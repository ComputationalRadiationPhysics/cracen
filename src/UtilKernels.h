#ifndef UTILKERNELS_H
#define UTILKERNELS_H

#include <cstdio>

#define DEVICE __device__ __forceinline__
#ifdef DEBUG_ENABLED
#define handleLastError() cudaDeviceSynchronize(); handle_error( cudaGetLastError(),"Kernel Error occured:\"", __LINE__, __FILE__)
#else
#define handleLastError()
#endif
void handle_error(cudaError_t err, const char* error, int line, const char* file) {
	if(err != cudaSuccess) std::cerr << error << cudaGetErrorString(err) << "\" in Line " << line << " in File " << file << std::endl;
}

struct identity {
	DEVICE uint2 operator()(uint2& input) const {
		return input;
	}
};
struct trans {
	DEVICE uint2 operator()(uint2& input) const {
		return make_uint2(input.y, input.x);
	}
};

template <class T = float, class AccessMode = identity>
class MatrixAccess {
private:
	unsigned int rows, cols;
	T* mat;
public:
	typedef T Type;
	DEVICE MatrixAccess(T* mat, unsigned int cols, unsigned int rows) : 
		rows(rows),
		cols(cols),
		mat(mat)
	{}
	DEVICE MatrixAccess(unsigned int cols, unsigned int rows) : 
		rows(rows),
		cols(cols),
		mat(new T[cols*rows])
	{}
	DEVICE MatrixAccess<T, AccessMode> copy() {
		return MatrixAccess<T, AccessMode>(mat, cols, rows);
	}
	DEVICE void finalize() {
		delete mat;
	}
	//TODO: Transposing a matrix two times will not lead to the original matrix
	DEVICE MatrixAccess<T, trans> transpose() {
		return MatrixAccess<T, trans>(mat, cols, rows);
	}
	DEVICE T& operator[](uint2 pos) {
		pos = AccessMode()(pos);		
		return mat[pos.y*cols+pos.x];
	}
	DEVICE unsigned int getRows() const {
		uint2 dim = make_uint2(cols, rows);
		dim = AccessMode()(dim);
		return dim.y;
	}
	DEVICE unsigned int getCols() const {
		uint2 dim = make_uint2(cols, rows);
		dim = AccessMode()(dim);
		return dim.x;
	}
	
	DEVICE T* getRawPointer() {
		return mat;
	}
};

template<unsigned int blockSize, class MatrixAccess1, class MatrixAccess2, class MatrixAccess3>
__global__ void matProdKernel(MatrixAccess1 left, MatrixAccess2 right, MatrixAccess3 result) {
	//BS Blockdim.x maximal, Blockdim.y = 1;
	__shared__ typename MatrixAccess1::Type sright[4*blockSize];
	__shared__ typename MatrixAccess1::Type sleft[2*blockSize][4];
	
	typename MatrixAccess1::Type res = 0;
	
	int y = threadIdx.y+blockDim.y*blockIdx.y;
	
	for(int blocks = 0; blockDim.x*4*blocks < left.getCols(); blocks++) {
		int x = threadIdx.x+blockDim.x*4*blocks;

		//Spalte der rechten Matrix in shared Mem laden
		if(threadIdx.y == 0) {
			sright[threadIdx.x] = right[make_uint2(blockIdx.x,x)];
			sright[threadIdx.x+blockDim.x] = right[make_uint2(blockIdx.x,x+blockDim.x)];
			sright[threadIdx.x+2*blockDim.x] = right[make_uint2(blockIdx.x,2*blockDim.x)];
			sright[threadIdx.x+3*blockDim.x] = right[make_uint2(blockIdx.x,3*blockDim.x)];
		}
		__syncthreads();
		
		//Block der linken Matrix in shared Mem laden
		if(x+blockDim.x < left.getCols()) {
			sleft[threadIdx.x][threadIdx.y]	= (left[make_uint2(x,y)]*sright[threadIdx.x]
											  +left[make_uint2(x+blockDim.x,y)]*sright[threadIdx.x+blockDim.x]);
		} else if(x < left.getCols()) {
			sleft[threadIdx.x][threadIdx.y] = left[make_uint2(x,y)]*sright[threadIdx.x]+left[make_uint2(x+blockDim.x,y)];
		} else {
			sleft[threadIdx.x][threadIdx.y] = 0;
		}
		
		if(x+3*blockDim.x < left.getCols()) {
			sleft[threadIdx.x+blockDim.x][threadIdx.y] = (left[make_uint2(x+2*blockDim.x,y)]*sright[threadIdx.x+2*blockDim.x]
														 +left[make_uint2(x+3*blockDim.x,y)]*sright[threadIdx.x+3*blockDim.x]);
		} else if (x+2*blockDim.x < left.getCols()) {
			sleft[threadIdx.x+blockDim.x][threadIdx.y] = left[make_uint2(x+2*blockDim.x,y)]*sright[threadIdx.x+2*blockDim.x];
		} else {
			sleft[threadIdx.x+blockDim.x][threadIdx.y] = 0;
		}
		__syncthreads();
		
		//Vektor folden
		for(int i = blockDim.x; i >= 1; i/=2) {
			if(threadIdx.x < i) sleft[threadIdx.x][threadIdx.y] += sleft[threadIdx.x+i][threadIdx.y];
			__syncthreads();
		}
		//Teilergebniss in Register speichern
		//if(threadIdx.x +threadIdx.y == 0 && blockIdx.x+blockIdx.y == 0) printf("%i += %i\n", res, sleft[0][threadIdx.y]);
		if(threadIdx.x == 0) res += sleft[0][threadIdx.y];
		__syncthreads();
	}
	if(threadIdx.x == 0) result[make_uint2(blockIdx.x, y)] = res;
	//Register in Ergebniss Matrix schreiben
}

DEVICE unsigned int calcAlignment(unsigned int var) {
	int msb = 31 - __clz(var);
	return (var == (1 << msb) ? var: 1 << msb+1);
}
template <class MatrixAccess1, class MatrixAccess2, class MatrixAccess3>
DEVICE void MatMul(MatrixAccess1& result, const MatrixAccess2& lhs, const MatrixAccess3& rhs) {
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
	bsy = min(bsy, 4);
	gsx = rhs.getCols();
	gsy = ceil(static_cast<float>(lhs.getRows())/bsy);
	//std::cout << "bs(" << bsx << ", " << bsy << "), gs(" << gsx << "," << gsy << ")" << std::endl;
	dim3 blockSize(bsx,bsy);
	dim3 gridSize(gsx, gsy);
	switch(bsx) {
		case 2:
			matProdKernel<2><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 4:
			matProdKernel<4><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 8:
			matProdKernel<8><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 16:
			matProdKernel<16><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 32:
			matProdKernel<32><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 64: 
			matProdKernel<64><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 128:
			matProdKernel<128><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		case 256:
			matProdKernel<256><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
			break;
		default:
			matProdKernel<512><<<gridSize, blockSize>>>(
				lhs,
				rhs,
				result);
	}
}

template <class Mat>
DEVICE void printMat(Mat& mat) {
	for(int j = 0; j < mat.getRows(); j++) {
		for(int i = 0; i < mat.getCols(); i++) {
			printf("%f ",mat[make_uint2(i,j)]);
		}
		printf("\n");
	}
	printf("\n");
}


template <class Mat>
DEVICE void printIntMat(Mat& mat) {
	for(int j = 0; j < mat.getRows(); j++) {
		for(int i = 0; i < mat.getCols(); i++) {
			printf("%i ",mat[make_uint2(i,j)]);
		}
		printf("\n");
	}
	printf("\n");
}
#endif
