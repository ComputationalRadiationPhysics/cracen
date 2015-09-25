#include <vector>
#include <iostream>

#include "../include/graybat/include/graybat.hpp"
#include "GrayBat/Pipeline.hpp"
#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Input/DataReader.hpp"
#include "Input/ScopeReader.hpp"
#include "Utility/StopWatch.hpp"
#include "Utility/CudaUtil.hpp"
  
int main(int argc, char* argv[]) {
	// CommunicationPolicy
    typedef graybat::communicationPolicy::BMPI CP;
    
    // GraphPolicy
    typedef graybat::graphPolicy::BGL<Cell>    GP;
    
    // Cage
    typedef graybat::Cage<CP, GP>   OuterCage;
    typedef typename MyCage::Event  Event;
    typedef typename MyCage::Vertex Vertex;
    typedef typename MyCage::Edge   Edge;

	std::vector<unsigned int> pipelineStage(3);
	pipelineStage[0] = 1;
	pipelineStage[1] = 1;
	pipelineStage[2] = 1;
	OuterCage outerCage(graybat::pattern::Pipeline(pipelineStage));
	
	/* Get number of devices */
	int numberOfDevices;
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
	    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1, NULL);
	
	#ifdef DATAREADER
		int nSample, nbrSegments, nWaveforms;
		DataReader::readHeader(input_filename, nSample, nbrSegments, nWaveforms);
		std::cout << "Header read. File compatible." << std::endl;
		DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
		std::cout << "DataReader created." << std::endl;
	#else
		/* Initialize input buffer (with dynamic elements) */
		ScopeReader::ScopeParameter::ScopeParameter parameter(scope_filename);
		//int nSegments = parameter.nbrSegments;
		//int nWaveforms = parameter.nbrWaveforms;
		int nSample = parameter.nbrSamples;
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
