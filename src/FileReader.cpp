#include <vector>
#include <iostream>

#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Device/Node.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/DataReader.hpp"
  
int main(int argc, char* argv[]) {
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory cageFactory(vm);
	CageFactory::Cage cage(cageFactory.commPoly(), cageFactory.graphPoly());
	cageFactory.map(cage);
	
	std::string input_filename = vm["inputFile"].as<std::string>();
	
	std::cout << "Args read (" << input_filename << ")" << std::endl;
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	int nSample, nbrSegments, nWaveforms;
	DataReader::readHeader(input_filename, nSample, nbrSegments, nWaveforms);
	std::cout << "Header read. File compatible." << std::endl;
	DataReader reader(input_filename, &inputBuffer, CHUNK_COUNT);
	std::cout << "DataReader created." << std::endl;
	
	/* Initialize output buffer (with static elements) */
	GrayBatStream<Chunk, decltype(cage)> os(1, cage);
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