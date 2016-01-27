#include <vector>
#include <iostream>
#include <chrono>

#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Device/Node.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/DataReader.hpp"
 



using namespace std::chrono_literals;

const float dataRate = 76; // Datarate in MB/s
int main(int argc, char* argv[]) {
	
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;
	
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory::Cage cage(CageFactory::commPoly(vm), CageFactory::graphPoly(vm));
	CageFactory::map(cage, vm);
	
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
	
	int nSample, nbrSegments, nWaveforms;
	
	/* Initialize output buffer (with static elements) */
	GrayBatStream<Chunk, decltype(cage)> os(1, cage);
	std::cout << "GrayBatStream" << std::endl;	
		
	size_t fits = 0;
	Clock::time_point t0;
	std::thread cpyThread([&inputBuffer, &os, &t0, &fits](){
		Chunk chunk;
		
		size_t chunks = 0;
		
		auto fn = [](int x) { 
			const float a = -0.01;
			const float b = 11;
			const float c = -2400;

			if(a*x*x + b*x+ c > 0) return a*x*x + b*x + c;
			else return 0.0f;						
		};
		for(int cc = 0; cc < CHUNK_COUNT; cc++) {
			for(int i = 0; i < SAMPLE_COUNT; i++) {
				chunk[cc*SAMPLE_COUNT + i] = fn(i);
			}
		}
		while(!inputBuffer.isFinished()) {
			Clock::time_point t1 = Clock::now();
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			if(static_cast<double>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / s.count() / 1024 / 1024 < dataRate) {
				fits++;
				os.getBuffer().push(chunk);
			}
			std::this_thread::sleep_for(10ms);

		}
		os.getBuffer().producerQuit();
	//	os.join();
	});
	
	std::thread benchThread([&fits, &t0](){
		while(1) {
			fits = 0;
			t0 = Clock::now();
			std::this_thread::sleep_for(3s);
			Clock::time_point t1 = Clock::now();
			
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			
			std::cout << static_cast<double>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / s.count() / 1024 / 1024 << "MiB/s" << std::endl;
		};
	});
	
	std::cout << "cpyThread created." << std::endl;
	
	std::cout << "Data read." << std::endl;
	cpyThread.join();
	//Make sure all results are written back

	return 0;
}
