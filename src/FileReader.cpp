#include <vector>
#include <iostream>

#include "Config/Constants.hpp"
#include "Config/NetworkGraph.hpp"
#include "Device/Node.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/DataReader.hpp"
  
int main(int argc, char* argv[]) {
	cage.distribute(graybat::mapping::PeerGroupMapping(0));
	
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
	
	int nSample, nbrSegments, nWaveforms;
	DataReader::readHeader(input_filename, nSample, nbrSegments, nWaveforms);
	std::cout << "Header read. File compatible." << std::endl;
	DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
	std::cout << "DataReader created." << std::endl;
	
	/* Initialize output buffer (with static elements) */
	GrayBatStream<Chunk, Cage> os(1, cage);
	std::cout << "GrayBatStream" << std::endl;
	
	
	std::thread cpyThread([&inputBuffer, &os](){
		while(!inputBuffer.isFinished()) {
			std::cout << "Filereader sending dataset." << std::endl;
			auto t = inputBuffer.pop();
			os.getBuffer().push(t);
		}
		os.getBuffer().producerQuit();
	//	os.join();
	});
	std::cout << "cpyThread created." << std::endl;
	
	reader.readToBuffer();
	std::cout << "Data read." << std::endl;
	cpyThread.join();
	//Make sure all results are written back

	return 0;
}