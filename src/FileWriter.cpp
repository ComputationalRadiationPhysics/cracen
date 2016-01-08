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
  
using namespace std::chrono_literals;

int main(int argc, char* argv[]) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;

	Clock::time_point t0 = Clock::now();
			
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);	
	
	std::string output_filename =  vm["outputFile"].as<std::string>();
		
	GrayBatReader<Output, decltype(cage)> gbReader(cage);
	
	std::cout << "Buffer created." << std::endl;
	
	size_t fits = 0;
	std::thread writerThread([&gbReader, &fits, t0](){
		std::fstream out;
		out.open("results.txt");
		Ringbuffer<Output>* inputBuffer = gbReader.getBuffer();
		
		Clock::time_point t1 = Clock::now();
		Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
		
		while(!inputBuffer->isFinished() || true) {
			
			auto elem = inputBuffer->pop();
			fits++;
			out << elem.status << " " << elem.woffset << " ";
			std::cout << "Write fit:" << elem.status << " " 
<< elem.woffset << " " << elem.param[0] << " " << elem.param[1] << " " 
<< elem.param[2];
			for(auto p : elem.param) out << p << " ";
			out << std::endl;
			std::cout << std::endl;
		}
		out.close();
	});
	
	std::thread benchThread([&fits, t0](){
		while(1) {
			Clock::time_point t1 = Clock::now();
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			
			std::cout << fits*SAMPLE_COUNT*sizeof(DATATYPE) / s.count() / 1024 / 1024 << "MiB/s" << std::endl;
			std::this_thread::sleep_for(10s);
		};
	});
	
	gbReader.readToBuffer();
	std::cout << "Data read." << std::endl;
	while(1);

	//Make sure all results are written back
	//writerThread.join();
	
	return 0;
}
