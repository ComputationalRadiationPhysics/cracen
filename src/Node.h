#ifndef NODE_H
#define NUDE_H

#include <pthread.h>
#include "Types.h"

class Node {
private:
	int deviceIdentifier;
	bool finish;
	InputBuffer* iBuffer;
	pthread_t thread_tid;
	void run();
	static void* run_helper(void* This) { 
		static_cast<Node*>(This)->run();
		return NULL;
	};
	
public:
	Node(int deviceIdentifier, InputBuffer* input);
	int stop();
};

#endif
