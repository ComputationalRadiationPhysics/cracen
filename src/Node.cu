#include "Node.h"
#include "LevMarq.h"
#include <vector>


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
	cudaSetDevice(deviceIdentifier);
	std::cout << "Device " << deviceIdentifier << " initialised." << std::endl;
	
	/* Set texture parameter */
	dataTexture0.filterMode=FILTER_MODE;
	dataTexture0.addressMode[0] = cudaAddressModeClamp;
	dataTexture0.addressMode[1] = cudaAddressModeClamp;
	dataTexture1.filterMode=FILTER_MODE;
	dataTexture1.addressMode[0] = cudaAddressModeClamp;
	dataTexture1.addressMode[1] = cudaAddressModeClamp;
	dataTexture2.filterMode=FILTER_MODE;
	dataTexture2.addressMode[0] = cudaAddressModeClamp;
	dataTexture2.addressMode[1] = cudaAddressModeClamp;
	dataTexture3.filterMode=FILTER_MODE;
	dataTexture3.addressMode[0] = cudaAddressModeClamp;
	dataTexture3.addressMode[1] = cudaAddressModeClamp;
	dataTexture4.filterMode=FILTER_MODE;
	dataTexture4.addressMode[0] = cudaAddressModeClamp;
	dataTexture4.addressMode[1] = cudaAddressModeClamp;
	dataTexture5.filterMode=FILTER_MODE;
	dataTexture5.addressMode[0] = cudaAddressModeClamp;
	dataTexture5.addressMode[1] = cudaAddressModeClamp;
	
	//TODO: REDUCE MAGIC NUMBERS
	std::vector<cudaArray*> texArrays;
	std::vector<fitData*> d_result;
	cudaStream_t streams[6];
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATATYPE>();
	for(int i = 0; i <= 5; i++) {
		/* Allocate memory */
		texArrays.push_back(NULL);
		d_result.push_back(NULL);
		cudaStreamCreate(&streams[i]);
		cudaMallocArray(&texArrays[i], &channelDesc, SAMPLE_COUNT, CHUNK_COUNT);
		cudaMalloc((void**)&d_result[i], sizeof(struct fitData) * SAMPLE_COUNT);	
	}
	cudaBindTextureToArray(dataTexture0, texArrays[0]);
	cudaBindTextureToArray(dataTexture1, texArrays[1]);
	cudaBindTextureToArray(dataTexture2, texArrays[2]);
	cudaBindTextureToArray(dataTexture3, texArrays[3]);
	cudaBindTextureToArray(dataTexture4, texArrays[4]);
	cudaBindTextureToArray(dataTexture5, texArrays[5]);
	int tex = 0;
	while(!iBuffer->isFinished()) {
		fitData result[6][CHUNK_COUNT];
		
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
			cudaMemcpy(d_result[tex], result[tex], sizeof(struct fitData) * CHUNK_COUNT, cudaMemcpyHostToDevice);
			++tex;
			/* 6 Chucks are copied to the GPU */
			if(tex == 6) {
				tex = 0;
				/* Start kernel */
				kernel<0><<<SAMPLE_COUNT, 1, 0, streams[0]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[0]);
				kernel<1><<<SAMPLE_COUNT, 1, 0, streams[1]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[1]);
				kernel<2><<<SAMPLE_COUNT, 1, 0, streams[2]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[2]);
				kernel<3><<<SAMPLE_COUNT, 1, 0, streams[3]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[3]);
				kernel<4><<<SAMPLE_COUNT, 1, 0, streams[4]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[4]);
				kernel<5><<<SAMPLE_COUNT, 1, 0, streams[5]>>>(SAMPLE_COUNT, INTERPOLATION_COUNT, d_result[5]);
				/* Get result */
				for(int i = 0; i <= 5; i++) {				
					cudaMemcpyAsync(result[i], d_result[i], sizeof(struct fitData) * CHUNK_COUNT, cudaMemcpyDeviceToHost, streams[i]);
				}
				for(int i = 0; i <= 5; i++) {									
					/* Sync */
					cudaStreamSynchronize(streams[i]);
					/* Push result to output buffer */
					
					for(int j = 0; j < CHUNK_COUNT; j++) {
						if(true) { //TODO: Check for fit quality
							oBuffer->writeFromHost(result[i][j]);
						}
					}				
				}
			}	
		}
	}
	cudaUnbindTexture(dataTexture0);
	cudaUnbindTexture(dataTexture1);
	cudaUnbindTexture(dataTexture2);
	cudaUnbindTexture(dataTexture3);
	cudaUnbindTexture(dataTexture4);
	cudaUnbindTexture(dataTexture5);
	
	for(int i = 0; i <= 5; i++) {
		cudaFreeArray(texArrays[i]);
		cudaFree(d_result[i]);
		cudaStreamDestroy(streams[i]);
	}

	oBuffer->producerQuit();
}
