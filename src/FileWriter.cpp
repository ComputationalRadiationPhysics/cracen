#include <vector>
#include <iostream>

#include "Device/Node.hpp"
#include "Output/OutputStream.hpp"
#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Input/GrayBatReader.hpp"
#include "Device/CudaUtil.hpp"
#include "StaticTTY.h"

using namespace std::chrono_literals;

int main(int argc, char* argv[]) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;
	
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);	
	
	std::string output_filename =  vm["outputFile"].as<std::string>();
		
	GrayBatReader<Output, decltype(cage)> gbReader(cage);
	
	std::cout << "Buffer created." << std::endl;
	
	size_t fits = 0;
	Ringbuffer<Output>* inputBuffer = gbReader.getBuffer();
	
	std::thread writerThread([&gbReader, &fits, &inputBuffer, &output_filename](){
		std::ofstream out;
		out.open(output_filename, std::ofstream::out);

		
		fits = 0;
		
		while(!inputBuffer->isFinished() || true) {
			
			auto elemBuff = inputBuffer->pop();
			fits++;
			for(auto elem : elemBuff) {
				
				//out << elem.status << " " << elem.woffset << " ";
				//std::cout << "Write fit:" << elem.status << " " << elem.woffset << " " << elem.param[0] << " " << elem.param[1] << " " << elem.param[2];
				for(auto p : elem.param) out << p << " ";
				out << std::endl;
				//std::cout << std::endl;
			}
		}
		out.close();
	});
	
	float writeRateLogic = 0;
	float writeRatePhysical = 0;
	unsigned int outputBufferSize = 0;
	unsigned int inputBufferSize = 0;
	
	std::thread benchThread([&fits, &gbReader, &inputBuffer, &writeRateLogic, &writeRatePhysical, &outputBufferSize, &inputBufferSize](){
		Clock::time_point t0;
		while(1) {
			fits = 0;
			const unsigned int fps = 2;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000/fps));
			Clock::time_point t1 = Clock::now();
			
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

			outputBufferSize = inputBuffer->getSize();
			inputBufferSize = gbReader.getBuffer()->getSize();

			writeRateLogic = static_cast<float>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / ms.count() / 1024 / 1024 * 1000;
			const unsigned int CharsPerFloat = 10;
			const unsigned int CharsPerLine = FitFunction::numberOfParams * (CharsPerFloat + 1); //+1 for space and linebreak
			writeRatePhysical = static_cast<float>(fits)*CharsPerLine*CHUNK_COUNT / ms.count() / 1024 / 1024 * 1000;	

			fits = 0;
			t0 = Clock::now();
		};
	});
	
	StaticTTY tty;
	tty << HSpace('#') << " FileWriter " << HSpace('#') << "\n";
	tty << "\n";
	tty << "Input buffer usage:" << HSpace(' ') << ProgressBar<unsigned int>(30, 0, CHUNK_BUFFER_COUNT, inputBufferSize) << "   \n";
	tty << "Output buffer usage:" << HSpace(' ') << ProgressBar<unsigned int>(30, 0, CHUNK_BUFFER_COUNT, outputBufferSize) << "   \n";
	tty << "Write rate logical: " << writeRateLogic << " Mib/s\n";
	tty << "Write rate physical: " << writeRatePhysical << " Mib/s\n";
	
	gbReader.readToBuffer();
	std::cout << "Data read." << std::endl;
	while(1);
	tty.finish();
	//Make sure all results are written back
	//writerThread.join();
	
	return 0;
}
