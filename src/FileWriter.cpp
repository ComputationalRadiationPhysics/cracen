#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Input/GrayBatReader.hpp"
#include "Input/ScopeReader.hpp"
#include "Utility/StopWatch.hpp"
#include "Device/CudaUtil.hpp"
  
int main(int argc, char* argv[]) {
	
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
    OutputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	GrayBatReader<Output> gbReader(masterUri, onlineDspUri);
	
	std::cout << "Buffer created." << std::endl;

	std::thread writerThread([&inputBuffer](){
		std::fstream out;
		out.open("results.txt");
		while(!inputBuffer.isFinished()) {
			auto elem = inputBuffer.pop();
			out << elem.status << " " << elem.woffset << " ";
			for(auto p : elem.param) out << p << " ";
			out << std::endl;
		}
		out.close();
	});
	
	gbReader.readToBuffer();
	std::cout << "Data read." << std::endl;

	//Make sure all results are written back
	writerThread.join();
	
	return 0;
}
