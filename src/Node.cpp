#include "Node.h"

Node::Node(int deviceIdentifier, InputBuffer* input) :
	deviceIdentifier(deviceIdentifier),
	finish(false),
	iBuffer(input)
{
	pthread_create(&thread_tid, NULL, run_helper, this);
}

void Node::run() {
	while(!finish) {
		/* Take a chunk from ringbuffer and copy to GPU */
			/* Block ringbuffer */
			/* Copy to device */
			/* Free ringbuffer */
			 
		/* Start kernel */
		
		/* Get result */
		
		/* Push result to output buffer */
		
	}
}

int Node::stop() {
	/* Called by main thread if all sample data is transfered to the devices */
	finish = true;
}
