#include <iostream>
#include <thrust/device_vector.h>
#include "../LevMarq.hpp"
#include "../FitFunction.hpp"
#include "Util.hpp"
#include "Wave.hpp"

typedef float DATATYPE;
typedef texture<DATATYPE, 2, cudaReadModeElementType> tex_t;
const unsigned int ORDER = 6;

template <unsigned int order>
float poly(float x) {
	float res;
	for(int i = 0; i <= order; i++) {
		res += std::pow(x,i);
	}
	return res;
}


int main(int argc, char** argv) {
	const int sample_count = 1000;
	float sample_data[sample_count];
	cudaArray_t texArray;
	for(int i = 0; i < sample_count; i++) {
		const float x = static_cast<float>(i)/sample_count;
		sample_data[i] = firstWave[i]; //poly<ORDER>(x)
		std::cout << "sample_data[" << i << "] = " << sample_data[i] << std::endl;
	}
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&texArray, &channelDesc, sample_count, 1);
	cudaMemcpyToArray(texArray, 0, 0, sample_data, sizeof(DATATYPE) * sample_count, cudaMemcpyHostToDevice);
	
	FitData *fitData;
	cudaMalloc((void**)(&fitData), sizeof(FitData));
	
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
	std::cout << "Texture Object created." << std::endl;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	float* mem;
	const unsigned int window_size = sample_count;
	const size_t SPACE = ((window_size+Polynom<ORDER>::numberOfParams)*2+(window_size+Polynom<ORDER>::numberOfParams)*Polynom<ORDER>::numberOfParams);
	cudaMalloc((void**) &mem, sizeof(float)*SPACE);
	std::cout << "Kernel start." << std::endl;
	//levenbergMarquardt<Polynom<ORDER> >(stream, texObj, fitData, sample_count, sample_count, 1, 1, mem);
	levenbergMarquardt<Gauss>(stream, texObj, fitData, sample_count, sample_count, 1, 1, mem);
	cudaFree(mem);
	cudaDeviceSynchronize();
	handleLastError();
	FitData results[1];
	cudaMemcpy(results, fitData, sizeof(results), cudaMemcpyDeviceToHost);
	std::cout << "status=" << results[0].status << std::endl;
	//std::cout << results[0];
	std::cout << results[0].param[0] << "*exp(-((x-" << results[0].param[1] << ")/" << results[0].param[3] << ")**2) + " <<  results[0].param[2] << std::endl;
	std::cout << "Test done." << std::endl;
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	cudaFree(fitData);
	cudaStreamDestroy(stream);
	
	return 0;
}
