/*
#include <iostream>
#include <thrust/device_vector.h>
#include "../LevMarq.h"
#include "../FitFunction.h"
#include "Util.h"

typedef float DATATYPE;
typedef texture<DATATYPE, 2, cudaReadModeElementType> tex_t;

class FitFunction:public FitFunctor<2> {
public:
	static DEVICE float sqr(float a) {
		return a*a;
	}
	static DEVICE float modelFunction(float x, float y, float *param) {
		x += 2;
		return sqr(x-param[0])+expf(param[1]*(sqr(x)+sqr(y)))-5;
	}
	static DEVICE float derivation(int param, float x, float y, float *params) {
		x += 2;
		switch(param) {
			case 0:
				return -2*(x-params[0]);
			case 1:
				float f = (sqr(x)+sqr(y));
				return (sqr(x)+sqr(y))*expf(params[1]*(sqr(x)+sqr(y)));
		}
		return 0;
	}

	static DEVICE Window getWindow(cudaTextureObject_t texObj, int dataset, int sample_count) {
		return Window(0, sample_count);
	}
};

int main(int argc, char** argv) {
	const int sample_count = 3;
	float sample_data[sample_count];
	cudaArray_t texArray;
	sample_data[0] = 0;
	sample_data[1] = 2;
	sample_data[2] = 0;
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&texArray, &channelDesc, sample_count, 1);
	//cudaMalloc((void**)&d_result[i], sizeof(struct fitData) * SAMPLE_COUNT);
	cudaMemcpyToArray(texArray, 0, 0, sample_data, sizeof(DATATYPE) * sample_count, cudaMemcpyHostToDevice);
	
	FitData<2> *fitData;
	cudaMalloc((void**)(&fitData), sizeof(FitData<3>));
	
	// Specify texture
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = texArray;

	// Specify texture object parameters
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	cudaDeviceSynchronize();
	handleLastError();
	
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	dim3 gs(1,1);
	dim3 bs(sample_count+3,1);
	levenbergMarquardt<FitFunction>(stream, texObj, fitData, sample_count, sample_count, 1, 1);
	cudaDeviceSynchronize();
	handleLastError();
	FitData<2> results[1];
	cudaMemcpy(results, fitData, sizeof(results), cudaMemcpyDeviceToHost);
	std::cout << "a=" << results->param[0] << ", b=" << results->param[1] << std::endl;
	
	std::cout << "Test done." << std::endl;
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	cudaFree(fitData);
	cudaStreamDestroy(stream);
	
	return 0;
}
*/
