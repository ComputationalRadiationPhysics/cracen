#include "Node.h"
#include "Constants.h"
#include "Textures.h"

Node::Node(int deviceIdentifier, InputBuffer* input) :
	deviceIdentifier(deviceIdentifier),
	finish(false),
	iBuffer(input)
{
	pthread_create(&thread_tid, NULL, run_helper, this);
}

void Node::run() {
	/* Initialise device */
	cudaSetDevice(deviceIdentifier);
	
	/* Allocate memory */
	cudaArray *texArray;
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Precision>();
	cudaMallocArray(&texArray, &channelDesc, SAMPLE_COUNT, CHUNK_COUNT); 

	/* Set texture parameter */
	sampleData.filterMode=FILTER_MODE;
	sampleData.addressMode[0] = cudaAddressModeBorder;
	//sampleData.addressMode[1] = cudaAddressModeBorder;
	
	/* bind texture to texture reference*/
	cudaBindTextureToArray(sampleData, texArray);

	while(!finish) {
		/* Take a chunk from ringbuffer and copy to GPU */
			/* Block ringbuffer */
			SampleChunk *c = iBuffer->reserveTail();
			/* Copy to device */
			cudaMemcpyToArray(texArray, 0, 0, c, sizeof(Precision)*SAMPLE_COUNT*CHUNK_COUNT, cudaMemcpyHostToDevice);
			/* Free ringbuffer */
			iBuffer->freeTail(c);
		/* Start kernel */
		
		/* Get result */
		//cudaMemcpy(...)
		

		
		/* Push result to output buffer */
		
	}
	
	/* Free GPU Memory*/
	cudaUnbindTexture(sampleData);
	cudaFreeArray(texArray);
}

int Node::stop() {
	/* Called by main thread if all sample data is transfered to the devices */
	finish = true;
	return 0;
}
