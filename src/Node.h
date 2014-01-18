#ifndef NODE_H
#define NUDE_H

#include <pthread.h>
#include "Types.h"

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
	
	int copyChunk(cudaArray *texArray, fitData* d_result);
	
public:
	Node(int deviceIdentifier, InputBuffer* input, OutputBuffer* output);
	int stop();
};

#endif
