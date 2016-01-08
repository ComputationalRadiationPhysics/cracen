#include <vector>
#include <iostream>
#include <chrono>

#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Device/Node.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/DataReader.hpp"
 

const float a = -0.01;
const float b = 10;
const float c = -2400;

const float dataRate = 10; // Datarate in MB/s
int main(int argc, char* argv[]) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;

	Clock::time_point t0 = Clock::now();
   
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);
		
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	int nSample, nbrSegments, nWaveforms;
	
	/* Initialize output buffer (with static elements) */
	GrayBatStream<Chunk, decltype(cage)> os(1, cage);
	std::cout << "GrayBatStream" << std::endl;	
	
	std::thread cpyThread([&inputBuffer, &os, t0](){
		Chunk chunk;
		
		size_t chunks = 0;
		
		auto fn = [a, b, c](int x) { 
			if(a*x*x + b*x+ c > 0) return a*x*x + b*x + c;
			else return 0.0f;						
		};
		for(int cc = 0; cc < CHUNK_COUNT; cc++) {
			for(int i = 0; i < SAMPLE_COUNT; i++) {
				chunk[cc*CHUNK_COUNT + i] = fn(i);
			}
		}
		while(!inputBuffer.isFinished()) {
			Clock::time_point t1 = Clock::now();
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			if(static_cast<double>(s.count())*dataRate > static_cast<double>(sizeof(DATATYPE))*CHUNK_COUNT*SAMPLE_COUNT/1000000*chunks) {
				chunks++;
				os.getBuffer().push(chunk);
			}
		}
		os.getBuffer().producerQuit();
	//	os.join();
	});
	std::cout << "cpyThread created." << std::endl;
	
	std::cout << "Data read." << std::endl;
	cpyThread.join();
	//Make sure all results are written back

	return 0;
}
