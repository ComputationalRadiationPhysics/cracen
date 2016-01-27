#ifndef NODE_H
#define NODE_H

#include <pthread.h>
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"

/*! 
 *  Each installed device should be handled by its own thread. This class provides 
 *  all functions to create a thread, copy data to and from the device and start
 *  the kernel on the device.
 */
class Node {
private:
	const static unsigned int numberOfTextures = pipelineDepth;
	int deviceIdentifier;
	bool finish;
	InputBuffer* iBuffer;
	OutputBuffer* oBuffer;
	size_t* fits;
	
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
	Node(int deviceIdentifier, InputBuffer* input, OutputBuffer* output, size_t* fits = NULL);
};

#endif
