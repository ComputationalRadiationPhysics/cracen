#include <iostream>
#include <thread>

#include "Config/NetworkGraph.hpp"
#include "Config/Constants.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/ScopeReader.hpp"

int main(int argc, char** argv) {
	cage.distribute(graybat::mapping::PeerGroupMapping(1));
	
	/* Get number of devices */
	int numberOfDevices;
	
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
	
	//std::cout << "Args read (" << input_filename << ", " << output_filename << ")" << std::endl;
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	/* Initialize input buffer (with dynamic elements) */
	ScopeReader::ScopeParameter parameter(scope_filename);
	//int nSegments = parameter.nbrSegments;
	//int nWaveforms = parameter.nbrWaveforms;
	int nSample = parameter.nbrSamples;
	ScopeReader reader(parameter, &inputBuffer, CHUNK_COUNT);
	GrayBatStream<Chunk, Cage> os(1, cage);
	
	std::thread sendingThread([&inputBuffer, &os](){
		while(inputBuffer.isFinished()) {
			inputBuffer.popTry([&os](Chunk& t){
				os.send(t);
			});
		}
		
		os.quit();
	});
	
	//std::cout << "Buffer created." << std::endl;

	reader.readToBuffer();
	
	//Make sure all results are written back
	sendingThread.join();
	return 0;	
}