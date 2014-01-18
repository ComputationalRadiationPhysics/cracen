#include <vector>
#include <iostream>

#include "Node.h"
#include "OutputStream.h"
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
	
	std::string filename = "Results.txt";
	if(argc >= 1) {
		filename = argv[1];	
	}
	
	/* Initialise buffer */
	InputBuffer inputBuffer(CHUNK_BUFFER_COUNT);
	OutputStream os(filename);
	
	std::vector<Node> devices;
	for(int i = 0; i < numberOfDevices; i++) {
		/* Start threads to handle Nodes */
		devices.push_back(Node(i, &inputBuffer, os.getBuffer()));
	}
	
	return 0;
}
