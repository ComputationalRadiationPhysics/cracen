#include <vector>
#include <iostream>

#include "Node.hpp"
#include "OutputStream.hpp"
#include "Constants.hpp"
#include "DataReader.hpp"
#include "StopWatch.hpp"


//Taken from https://github.com/ComputationalRadiationPhysics/HaseOnGpu
//Author: Erik Zenker, Carlchristian Eckert
std::vector<unsigned> getFreeDevices(unsigned maxGpus){
	cudaDeviceProp prop;
	int minMajor = MIN_COMPUTE_CAPABILITY_MAJOR;
	int minMinor = MIN_COMPUTE_CAPABILITY_MINOR;
	int count;
	std::vector<unsigned> devices;

	// Get number of devices
	cudaGetDeviceCount(&count);

	// Check devices for compute capability and if device is busy
	unsigned devicesAllocated = 0;
	for(int i=0; i < count; ++i){
		cudaGetDeviceProperties(&prop, i);
		if( (prop.major > minMajor) || (prop.major == minMajor && prop.minor >= minMinor) ){
			cudaSetDevice(i);
			int* occupy; //TODO: occupy gets allocated, but never cudaFree'd -> small memory leak!
			if(cudaMalloc((void**) &occupy, sizeof(int)) == cudaSuccess){
				devices.push_back(i);
				devicesAllocated++;
				if(devicesAllocated == maxGpus)
					break;
			}
		}
	}
	// Exit if no device was found
	if(devices.size() == 0){
		std::cout << "None of the free CUDA-capable devices is sufficient!" << std::endl;
		exit(1);
	}

	// Print device information
	cudaSetDevice(devices.at(0));
	std::cout << "Found " << int(devices.size()) << " available CUDA devices with Compute Capability >= " << minMajor << "." << minMinor << "):" << std::endl;
	for(unsigned i=0; i<devices.size(); ++i){
		cudaGetDeviceProperties(&prop, devices[i]);
		std::cout << "[" << devices[i] << "] " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
	}

	return devices;

}
  
int main(int argc, char* argv[]) {
	
	/* Get number of devices */
	int numberOfDevices;
	cudaError_t err;
	err = cudaGetDeviceCount(&numberOfDevices);
	std::vector<unsigned> freeDevices = getFreeDevices(maxNumberOfDevices);
	
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
	OutputStream os(output_filename, freeDevices.size());
	
	std::cout << "Buffer created." << std::endl;
	
    DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
    std::cout << "DataReader created." << std::endl;

	std::vector<Node*> devices;
	StopWatch sw;
	sw.start();
	for(int i = 0; i < freeDevices.size(); i++) {
		/* Start threads to handle Nodes */
		devices.push_back(new Node(freeDevices[i], &inputBuffer, os.getBuffer()));
	}
	reader.readToBuffer();
	std::cout << "Data read." << std::endl;
	
	std::cout << "Nodes created." << std::endl;
		

	//Make sure all results are written back
	os.join();
	sw.stop();
	std::cout << "Time: " << sw << std::endl;
	//std::cout << "Throuput: " << 382/(sw.elapsedSeconds()) << "MiB/s."<< std::endl;
	return 0;
}
