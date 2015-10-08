#include <iostream>
#include <thread>

#include "Config/Constants.hpp"
#include "Output/OutputStream.hpp"
#include "Input/ScopeReader.hpp"

int main(int argc, char** argv) {
	
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
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1, NULL);
	
	/* Initialize input buffer (with dynamic elements) */
	ScopeReader::ScopeParameter parameter(scope_filename);
	//int nSegments = parameter.nbrSegments;
	//int nWaveforms = parameter.nbrWaveforms;
	int nSample = parameter.nbrSamples;
	ScopeReader reader(parameter, &inputBuffer, CHUNK_COUNT);
	GrayBatStream<Chunk> os(1,masterUri, fitterUri);
	
	std::thread sendingThread([inputBuffer, os](){
		while(inputBuffer.isFinished())
		Chunk* tail inputBuffer.reserveTailTry();
			if(tail != NULL) {
				os.send(*tail);
			}
		}
		os.quit();
	});
	
	std::cout << "Buffer created." << std::endl;

	reader.readToBuffer();
	
	//Make sure all results are written back
	sendingThread.join();
	os.join();
	return 0;	
}