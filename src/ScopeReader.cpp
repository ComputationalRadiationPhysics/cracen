#include <iostream>
#include <thread>

#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Config/Constants.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/ScopeReader.hpp"

int main(int argc, char** argv) {
	std::cout << sizeof(ViSession) << std::endl;
	std::cout << sizeof(ViStatus) << std::endl;
	std::cout << sizeof(Chunk) << std::endl;

	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);
	
	/* Get number of devices */
	std::string scope_filename = vm["scopeFile"].as<std::string>();
	
	//std::cout << "Args read (" << input_filename << ", " << output_filename << ")" << std::endl;
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	/* Initialize input buffer (with dynamic elements) */
	ScopeReader::ScopeParameter parameter(scope_filename);
	ScopeReader reader(parameter, &inputBuffer, CHUNK_COUNT);
	
	GrayBatStream<Chunk, decltype(cage)> os(1, cage);
	
	std::thread sendingThread([&inputBuffer, &os](){
		while(!inputBuffer.isFinished()) {
			std::cout << "Filereader sending dataset." << std::endl;
			auto t = inputBuffer.pop();
			os.getBuffer().push(t);
		}
		os.getBuffer().producerQuit();
	});
	
	//std::cout << "Buffer created." << std::endl;

	reader.readToBuffer();
	
	//Make sure all results are written back
	sendingThread.join();
	return 0;	
}