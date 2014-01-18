#include <vector>
#include <iostream>

#include "Node.h"
#include "Constants.h"

int main(int argc, char* argv[]) {
	
	/* Get number of devices */
	int numberOfDevices;
	cudaError_t err;
	err = cudaGetDeviceCount(&numberOfDevices);
	
	/* Check the cuda runtime environment */
	if(err != cudaSuccess) {
		std::cerr << "Something went wrong during the creation the context, or no Cuda capable devices are installed on the system." << std::endl;
		std::cerr << "Exit." << std::endl;
		return 1;
	}
	
	/* Initialise input buffer */
	InputBuffer inputBuffer(CHUNK_BUFFER_COUNT);
	
	
	
	std::vector<Node> devices;
	for(int i = 0; i < numberOfDevices; i++) {
		/* Start threads to handle Nodes */
		devices.push_back(Node(i, &inputBuffer));
	}
	
	return 0;
}
