#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Input/GrayBatReader.hpp"
#include "Device/CudaUtil.hpp"

#include "root/TFile.h"
#include "root/TNtuple.h"

  
using namespace std::chrono_literals;

int main(int argc, char* argv[]) {
	

	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;
	
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory cageFactory(vm);
	CageFactory::Cage cage(cageFactory.commPoly(), cageFactory.graphPoly());
	cageFactory.map(cage);	
	
	std::string output_filename =  vm["outputFile"].as<std::string>();
		
	GrayBatReader<Output, decltype(cage)> gbReader(cage);
	
	std::cout << "Buffer created." << std::endl;
	
	size_t fits = 0;
	std::thread writerThread([&gbReader, &fits, &output_filename](){
		
		Ringbuffer<Output>* inputBuffer = gbReader.getBuffer();
		
		fits = 0;
		
		TFile outFile("results.root", "RECREATE");
		TNtuple fitData("fitData", "FitTuple", "status:woffset:param0:param1:param2");
		
		while(!inputBuffer->isFinished() || true) {
			
			auto elemBuff = inputBuffer->pop();
			fits++;
			for(auto& elem : elemBuff) {
				
				//std::cout << "Write fit:" << elem.status << " " << elem.woffset << " " << elem.param[0] << " " << elem.param[1] << " " << elem.param[2];
				fitData.Fill(elem.status, elem.woffset, elem.param[0], elem.param[1], elem.param[2]);
				//std::cout << std::endl;
			}
			fitData.Write();
		}
		outFile.Close();
		
	});
	
	std::thread benchThread([&fits](){
		while(1) {
			fits = 0;
			Clock::time_point t0 = Clock::now();
			std::this_thread::sleep_for(3s);
			Clock::time_point t1 = Clock::now();
			
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			
			std::cout << static_cast<double>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / s.count() / 1024 / 1024 << "MiB/s" << std::endl;
		};
	});
	
	gbReader.readToBuffer();
	std::cout << "Data read." << std::endl;
	while(1);

	//Make sure all results are written back
	//writerThread.join();
	
	return 0;
}
