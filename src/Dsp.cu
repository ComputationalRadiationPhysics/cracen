#include <vector>
#include <iostream>

#include "Node.h"
#include "OutputStream.h"
#include "Constants.h"
#include "DataReader.h"

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
	
	std::string input_filename = FILENAME_TESTFILE;
	std::string output_filename =  OUTPUT_FILENAME;
	
	if(argc > 1) {
		input_filename = argv[1];	
	}
	if(argc > 2) {
		output_filename = argv[2];
	}
	
	std::cout << "Args read (" << input_filename << ", " << output_filename << ")" << std::endl;
	/* Initialise buffer */
	InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	OutputStream os(output_filename, numberOfDevices);
	
	std::cout << "Buffer created." << std::endl;
	
	std::vector<Node*> devices;
	for(int i = 0; i < numberOfDevices; i++) {
		/* Start threads to handle Nodes */
		devices.push_back(new Node(i, &inputBuffer, os.getBuffer()));
	}
	
	std::cout << "Nodes created." << std::endl;
	
	DataReader reader(input_filename, &inputBuffer);	
	std::cout << "DataReader created" << std::endl;
	
	reader.readToBufferAsync();
	std::cout << "Data read." << std::endl;

	//Make sure all results are written back
	os.join();
	
	return 0;
}
