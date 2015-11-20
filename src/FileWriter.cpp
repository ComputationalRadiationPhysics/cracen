#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Config/NetworkGraph.hpp"
#include "Input/GrayBatReader.hpp"
#include "Input/ScopeReader.hpp"
#include "Device/CudaUtil.hpp"
  
int main(int argc, char* argv[]) {
	cage.distribute(graybat::mapping::PeerGroupMapping(2));
	
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
	
	GrayBatReader<Output, Cage> gbReader(cage);
	
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

	//Make sure all results are written back
	writerThread.join();
	
	return 0;
}
