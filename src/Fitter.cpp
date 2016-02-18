#include <chrono>
#include <iostream>
#include <memory>

#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Device/CudaUtil.hpp"
#include "Input/GrayBatReader.hpp"
#include "Output/GrayBatStream.hpp"
#include "Device/Node.hpp"
#include "StaticTTY.h"

using namespace std::chrono_literals;

int main(int argc, char** argv) {
	typedef std::chrono::high_resolution_clock Clock;
	
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);
	
	GrayBatReader<Chunk, decltype(cage)> reader(cage);
	std::cout << "GrayBatReader created." << std::endl;
	GrayBatStream<Output, decltype(cage)> stream(1,cage);
	std::cout << "GrayBatStream created." << std::endl;
	
	std::vector<Node*> devices;
	std::vector<unsigned int> freeDevices = cuda::getFreeDevices(4);
	//StopWatch sw;
	//sw.start();
	size_t fits = 0;
	
	for(unsigned int i = 0; i < freeDevices.size(); i++) {
		/* Start threads to handle Nodes */
		devices.push_back(new Node(freeDevices[i], reader.getBuffer(), &(stream.getBuffer()), &fits));
	}
	
	float computeRate = 0;
	unsigned int outputBufferSize = 0;
	unsigned int inputBufferSize = 0;
	
	std::thread benchThread([&fits, &reader, &stream, &computeRate, &outputBufferSize, &inputBufferSize](){
		Clock::time_point t0;
		while(1) {
			fits = 0;
			const unsigned int fps = 2;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000/fps));
			Clock::time_point t1 = Clock::now();
			
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

			outputBufferSize = stream.getBuffer().getSize();
			inputBufferSize = reader.getBuffer()->getSize();

			computeRate = static_cast<float>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / ms.count() / 1024 / 1024 * 1000;	
			fits = 0;
			t0 = Clock::now();
		};
	});
	
	StaticTTY tty;
	tty << HSpace('#') << " Fitter " << HSpace('#') << "\n";
	tty << "\n";
	tty << "Input buffer usage:" << HSpace(' ') << ProgressBar<unsigned int>(30, 0, CHUNK_BUFFER_COUNT, inputBufferSize) << "   \n";
	tty << "Output buffer usage:" << HSpace(' ') << ProgressBar<unsigned int>(30, 0, CHUNK_BUFFER_COUNT, outputBufferSize) << "   \n";
	tty << "Compute data rate: " << computeRate << " Mib/s\n";
	
	reader.readToBuffer();
	while(1);
	tty.finish();		
	
	return 0;
}
