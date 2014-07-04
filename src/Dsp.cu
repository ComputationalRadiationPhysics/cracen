#include <vector>
#include <iostream>

#include "Node.h"
#include "OutputStream.h"
#include "Constants.h"
#include "DataReader.h"
#include "TimeInterval.h"

int main(int argc, char* argv[]) {
	
	/* Get number of devices */
	int numberOfDevices;
	cudaError_t err;
	err = cudaGetDeviceCount(&numberOfDevices);
	numberOfDevices = 1;
	
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

    int nSample = -1;
    int nSegments = -1;
    int nWaveforms = -1;

    DataReader::readHeader(input_filename, nSample, nSegments, nWaveforms);
	std::cout << "Header read. File compatible." << std::endl;

	/* Initialize input buffer (with dynamic elements) */
    Chunk dc(CHUNK_COUNT * nSample);
    std::fill(dc.begin(), dc.end(), 0);
	InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1, dc);
    /* Initialize output buffer (with static elements) */
	OutputStream os(output_filename, numberOfDevices);
	
	std::cout << "Buffer created." << std::endl;
	
    DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
    std::cout << "DataReader created." << std::endl;

	std::vector<Node*> devices;
	TimeIntervall ti;
	ti.toggleStart();
	for(int i = 0; i < numberOfDevices; i++) {
		/* Start threads to handle Nodes */
		devices.push_back(new Node(i, &inputBuffer, os.getBuffer()));
	}
	
	std::cout << "Nodes created." << std::endl;
		
	reader.readToBuffer();
	std::cout << "Data read." << std::endl;

	//Make sure all results are written back
	os.join();
	ti.toggleEnd();
	std::cout << "Time: " << ti.printInterval() << std::endl;
	std::cout << "Throuput: " << 382/(ti.getInterval()/1000) << "MiB/s."<< std::endl;
	return 0;
}
