#ifndef NODE_H
#define NUDE_H

#include <pthread.h>
#include "Types.h"
#include "LevMarq.h"

/*! 
 *  Each installed device should be handled by its own thread. This class provides 
 *  all functions to create a thread, copy data to and from the device and start
 *  the kernel on the device.
 */
class Node {
private:
	int deviceIdentifier;
	bool finish;
	InputBuffer* iBuffer;
	OutputBuffer* oBuffer;
	pthread_t thread_tid;
	void run();
	static void* run_helper(void* This) { 
		static_cast<Node*>(This)->run();
		return NULL;
	};
	
	//! Copy one chunk of data to the GPU and the result back to the output buffer.
	/*!
	 *  \param texArray Location on the GPU, where the raw data will be copied to.
	 *  \param fitData  Location on the GPU, where the result will be written to.
	 */
	int copyChunk(cudaArray *texArray, fitData* d_result);
	
public:
	//! Basic constructor.
	/*!
	 *  Stats a new Thread. The new Thread reads data from the input buffer,
	 *  copies them to the gpu and copy the result back to the output buffer. 
	 *  
	 *  \param deviceIdentifier Number of the Device
	 *  \param input Buffer which provides the raw input data.
	 *  \param output Buffer which will be filled with the result data.
	 */
	Node(int deviceIdentifier, InputBuffer* input, OutputBuffer* output);
	
	//! Signals, that no new data will be written into the buffer.
	/*!
	 * This function will make the Node Thread stop, after all remaining
	 * elements in the buffer are written into the output file
	 */
	int stop();
};

#endif
