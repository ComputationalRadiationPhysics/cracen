#include <iostream>
#include "../LevMarq.h"
#include "../FitFunction.h"
typedef float DATATYPE;
typedef texture<DATATYPE, 2, cudaReadModeElementType> tex_t;

int main(int argc, char** argv) {
	const int sample_count = 1000;
	float sample_data[sample_count];
	for(int i = 0; i < sample_count; i++) sample_data[i] = (i)*(i)+i+1;
	dataTexture0.filterMode=FILTER_MODE;
	dataTexture0.addressMode[0] = cudaAddressModeClamp;
	dataTexture0.addressMode[1] = cudaAddressModeClamp;
	cudaArray* texArray;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATATYPE>();
	cudaMallocArray(&texArray, &channelDesc, sample_count, 1);
	//cudaMalloc((void**)&d_result[i], sizeof(struct fitData) * SAMPLE_COUNT);
	cudaBindTextureToArray(dataTexture0, texArray);
	cudaMemcpyToArray(texArray, 0, 0, sample_data, sizeof(DATATYPE) * sample_count, cudaMemcpyHostToDevice);
	
	cudaStream_t stream;
	
	cudaStreamCreate(&stream);
	levenbergMarquardt<Polynom<2, 0>, 0>(sample_count, 1000, 1, 50);
	
	return 0;
}
