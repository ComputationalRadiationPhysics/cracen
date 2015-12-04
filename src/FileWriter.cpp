#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Input/GrayBatReader.hpp"
#include "Input/ScopeReader.hpp"
#include "Device/CudaUtil.hpp"
  
int main(int argc, char* argv[]) {
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);	
	
	std::string output_filename =  vm["outputFile"].as<std::string>();
		
	GrayBatReader<Output, decltype(cage)> gbReader(cage);
	
	std::cout << "Buffer created." << std::endl;

	std::thread writerThread([&gbReader](){
		std::fstream out;
		out.open("results.txt");
		Ringbuffer<Output>* inputBuffer = gbReader.getBuffer();
		while(!inputBuffer->isFinished() || true) {
			auto elem = inputBuffer->pop();
			out << elem.status << " " << elem.woffset << " ";
			std::cout << "Write fit:" << elem.status << " " << elem.woffset << " ";
			for(auto p : elem.param) out << p << " ";
			out << std::endl;
			std::cout << std::endl;
		}
		out.close();
	});
	
	gbReader.readToBuffer();
	std::cout << "Data read." << std::endl;
	while(1);

	//Make sure all results are written back
	//writerThread.join();
	
	return 0;
}
