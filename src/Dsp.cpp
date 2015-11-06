#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Input/DataReader.hpp"
#include "Input/ScopeReader.hpp"
#include "Utility/StopWatch.hpp"
#include "Device/CudaUtil.hpp"
  
int main(int argc, char* argv[]) {	
	/* Get number of devices */
	std::vector<unsigned> freeDevices = cuda::getFreeDevices(maxNumberOfDevices);

	std::string input_filename = FILENAME_TESTFILE;
	std::string scope_filename = SCOPE_PARAMETERFILE;
	std::string output_filename =  OUTPUT_FILENAME;

	if(argc > 1) {
		input_filename = argv[1];	
		scope_filename = argv[1];
	}
	if(argc > 2) {
		output_filename = argv[2];
	}
	
	std::cout << "Args read (" << input_filename << ", " << output_filename << ")" << std::endl;
	    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	#ifdef DATAREADER
		int nSample, nbrSegments, nWaveforms;
		DataReader::readHeader(input_filename, nSample, nbrSegments, nWaveforms);
		std::cout << "Header read. File compatible." << std::endl;
		DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
		std::cout << "DataReader created." << std::endl;
	#else
		/* Initialize input buffer (with dynamic elements) */
		ScopeReader::ScopeParameter parameter(scope_filename);
		//int nSegments = parameter.nbrSegments;
		//int nWaveforms = parameter.nbrWaveforms;
		ScopeReader reader(parameter, &inputBuffer, CHUNK_COUNT);
	#endif
	
	/* Initialize output buffer (with static elements) */
	OutputStream os(output_filename, freeDevices.size());
	
	std::cout << "Buffer created." << std::endl;

	std::vector<Node*> devices;
	StopWatch sw;
	sw.start();
	for(unsigned int i = 0; i < freeDevices.size(); i++) {
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
