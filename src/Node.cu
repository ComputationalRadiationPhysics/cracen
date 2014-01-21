#include "Node.h"
#include "Constants.h"
#include "Textures.h"

Node::Node(int deviceIdentifier, InputBuffer* input, OutputBuffer* output) :
	deviceIdentifier(deviceIdentifier),
	finish(false),
	iBuffer(input),
	oBuffer(output)
{
	pthread_create(&thread_tid, NULL, run_helper, this);
}

int Node::copyChunk(cudaArray *texArray, fitData* d_result) {
	
	fitData result[CHUNK_COUNT];
		
	/* Take a chunk from ringbuffer and copy to GPU */
	/* Block ringbuffer */
	SampleChunk *c = iBuffer->reserveTail();
	/* Copy to device */
	cudaMemcpyToArray(texArray, 0, 0, c, sizeof(Precision)*SAMPLE_COUNT*CHUNK_COUNT, cudaMemcpyHostToDevice);
	/* Free ringbuffer */
	iBuffer->freeTail();
	std::cout << "Chunk taken from input bufffer" << std::endl;
	cudaMemcpy(d_result, result, sizeof(struct fitData) * CHUNK_COUNT, cudaMemcpyHostToDevice);
	/* Start kernel */

	kernel<<<SAMPLE_COUNT, 1>>>(SAMPLE_COUNT, d_result);
	/* Get result */
	cudaMemcpy(result, d_result, sizeof(struct fitData) * CHUNK_COUNT, cudaMemcpyDeviceToHost);
	/* Push result to output buffer */
	
	for(int i = 0; i < CHUNK_COUNT; i++) {
		if(true) { //TODO: Check for fit quality
			oBuffer->writeFromHost(&result[i]);
		}
	}
	
	
	return 0;
}
void Node::run() {
	
	/* 
	 * Example for Texture usage found here
	 * http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
	 */
	
	/* Initialise device */
	cudaSetDevice(deviceIdentifier);
	
	/* Allocate memory */
	cudaArray *texArray;
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Precision>();
	cudaMallocArray(&texArray, &channelDesc, SAMPLE_COUNT, CHUNK_COUNT); 

	/* Set texture parameter */
	dataTexture.filterMode=FILTER_MODE;
	dataTexture.addressMode[0] = cudaAddressModeClamp;
	dataTexture.addressMode[1] = cudaAddressModeClamp;
	
	/* bind texture to texture reference*/
	cudaBindTextureToArray(dataTexture, texArray);
	
	fitData* d_result;
	cudaMalloc((void**)&d_result, sizeof(struct fitData) * SAMPLE_COUNT);

	while(!finish) {
		copyChunk(texArray,  d_result);		
	}
	
	/* Empty the the iBuffer */
	while(!iBuffer->isEmpty()) {
		copyChunk(texArray,  d_result);	
	}
	
	cudaUnbindTexture(dataTexture);
	cudaFreeArray(texArray);
	cudaFree(d_result);
}

int Node::stop() {
	/* Called by main thread if all sample data is transfered to the devices */
	finish = true;
	return 0;
}
