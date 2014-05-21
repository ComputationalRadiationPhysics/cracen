#include <iostream>
#include "../LevMarq.h"
#include "../FitFunction.h"
#include <thrust/device_vector.h>
#include "Util.h"

typedef float DATATYPE;
typedef texture<DATATYPE, 2, cudaReadModeElementType> tex_t;


__global__ void testFetch(cudaTextureObject_t texObj) {
	for(float j = 0; j <= 10; j++) {
		//float i = tex2D<float>(texObj, j, 0.0f);
		float i = getSample(texObj, j, 0);
		printf("Wert = %f", i);
	}
}
__global__ void testKernel(cudaTextureObject_t texObj, float* F, float* param) {
	testFetch<<<1,1>>>(texObj);
	/*
	dim3 gs(1,1);
	dim3 bs(10+3,1);
	unsigned int sample_count = 10;
	calcF<Polynom<2> ><<<gs, bs>>>(0, texObj, param, F, 0, sample_count, 1);
	*/
}
int main(int argc, char** argv) {
	const int sample_count = 1000;
	float sample_data[sample_count];
	
	cudaArray_t texArray;
	for(int i = 0; i < sample_count; i++) sample_data[i] = i*i+i+1;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&texArray, &channelDesc, sample_count, 1);
	//cudaMalloc((void**)&d_result[i], sizeof(struct fitData) * SAMPLE_COUNT);
	cudaMemcpyToArray(texArray, 0, 0, sample_data, sizeof(DATATYPE) * sample_count, cudaMemcpyHostToDevice);
	
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
	thrust::device_vector<float> F(13);
	thrust::device_vector<float> param(3);
	cudaStreamCreate(&stream);
	//levenbergMarquardt<Polynom<2, 0>, 0>(stream, sample_count, sample_count, 1, 1);
	dim3 gs(1,1);
	dim3 bs(10+3,1);
	//testKernel<<<1,1>>>(texObj, pcast(F), pcast(param));
	//calcF<Polynom<2> ><<<gs, bs, 0, stream>>>(texObj, 0, pcast(param), pcast(F), 0, sample_count, 1);
	levMarqIt<Polynom<2> ><<<1,1>>>(texObj, 10, 10, 0, 1);
	cudaDeviceSynchronize();
	handleLastError();
	
	std::cout << "Test done." << std::endl;
	
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	cudaStreamDestroy(stream);
	
	return 0;
}
