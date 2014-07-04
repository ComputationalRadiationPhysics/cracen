#include "Node.h"
#include "LevMarq.h"
#include <vector>
#include "FitFunction.h"

typedef texture<DATATYPE, 2, cudaReadModeElementType> tex_t;

Node::Node(int deviceIdentifier, InputBuffer* input, OutputBuffer* output) :
	deviceIdentifier(deviceIdentifier),
	finish(false),
	iBuffer(input),
	oBuffer(output)
{
	pthread_create(&thread_tid, NULL, run_helper, this); 
}

void Node::run() {
	
	/* 
	 * Example for Texture usage found here
	 * http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
	*/
	/* Initialise device */
	
	typedef WindowPolynom<2> Fit;
	const unsigned int window_size = 100;//SAMPLE_COUNT/INTERPOLATION_COUNT;
	cudaSetDevice(deviceIdentifier);
	
	cudaTextureObject_t texObj[numberOfTextures];
	cudaStream_t streams[numberOfTextures];
	cudaArray_t texArrays[numberOfTextures];
	bool textureEmpty[numberOfTextures];
	
	// Specify texture object parameters
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	FitData<Fit::numberOfParams> results[CHUNK_COUNT];
	typedef FitData<Fit::numberOfParams> FitDataArray[numberOfTextures][CHUNK_COUNT];
	FitData<Fit::numberOfParams> *fitData;
	cudaMalloc((void**)(&fitData), sizeof(FitDataArray));
	#pragma loop unroll
	for(unsigned int i = 0; i < numberOfTextures; i++) {
		cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
		cudaMallocArray(&texArrays[i], &channelDesc, SAMPLE_COUNT, CHUNK_COUNT);

		// Specify texture
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArrays[i];

		// Create texture object
		cudaCreateTextureObject(&texObj[i], &resDesc, &texDesc, NULL);
		textureEmpty[i] = true;
	}
	std::cout << "Device " << deviceIdentifier << " initialised." << std::endl;
	int tex = 0;
	unsigned int lastTexture = 0;
	while(!iBuffer->isFinished() || !textureEmpty[lastTexture]) {
		/* copy results back */
		if(!textureEmpty[tex]) {
			cudaMemcpyAsync(results, &fitData[tex*CHUNK_COUNT], sizeof(results), cudaMemcpyDeviceToHost, streams[tex]);
			cudaStreamSynchronize(streams[tex]);
			for(int i = 0; i < CHUNK_COUNT; i++) {
					oBuffer->writeFromHost(results[i]);
			}
			textureEmpty[tex] = true;
		}
		//TODO: Racecondition
		if(!iBuffer->isFinished()) {
			/* Take a chunk from ringbuffer and copy to GPU */
			/* Block ringbuffer */
			Chunk *c = iBuffer->reserveTailTry();
			/* Copy to device */
			if(c != NULL) {
				cudaMemcpyToArrayAsync(texArrays[tex], 0, 0, &c->front(), 
		                               sizeof(DATATYPE) * c->size(), 
		                               cudaMemcpyHostToDevice, streams[tex]);
	  			/* Free ringbuffer 
		           This is possible because at the moment we use pageable (non-pinnend)
		           host memory for the ringbuffer.
		           In this case cudaMemcpy...Async will first copy data to a staging 
		           buffer and then return. Only copying from staging buffer to final 
		           destination is asynchronous.
		           Should we switch to pinnend host memory for the ringbuffer we must
		           not call iBuffer->freeTail() directly after cudaMemcpy..Async.
		           See 
				   http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/sync_async.html#MemcpyAsynchronousBehavior
		         */
				iBuffer->freeTail();
				std::cout << "Chunk taken from input buffer (device " << deviceIdentifier << "). " << iBuffer->getSize() << " elements remaining in queue." << std::endl;
				levenbergMarquardt<Fit>(streams[tex], texObj[tex], &fitData[tex*CHUNK_COUNT], SAMPLE_COUNT, window_size, CHUNK_COUNT, INTERPOLATION_COUNT);
				lastTexture = tex;
				tex = (tex+1)%numberOfTextures;
				textureEmpty[tex] = false;
			}
		}
	}
	for(unsigned int i = 0; i < numberOfTextures; i++) {
		cudaDestroyTextureObject(texObj[i]);
		cudaFreeArray(texArrays[i]);
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(fitData);
	oBuffer->producerQuit();
}
