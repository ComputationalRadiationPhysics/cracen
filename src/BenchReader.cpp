#include <vector>
#include <iostream>
#include <chrono>
#include <random>

#include "Config/Constants.hpp"
#include "Config/CommandLineParser.hpp"
#include "graybat/CageFactory.hpp"
#include "Device/Node.hpp"
#include "Output/GrayBatStream.hpp"
#include "Input/DataReader.hpp"
#include "StaticTTY.h"



using namespace std::chrono_literals;

const float dataRate = 500000; // Datarate in MB/s
int main(int argc, char* argv[]) {
	std::random_device rd;
	std::mt19937 gen(rd());	
	std::normal_distribution<float> d(0,3);

	typedef std::chrono::high_resolution_clock Clock;
	
	auto vm = CommandLineParser::parse(argc, argv);
	CageFactory cageFactory(vm);
	CageFactory::Cage cage(cageFactory.commPoly(), cageFactory.graphPoly());
	cageFactory.map(cage);
	
    InputBuffer inputBuffer(CHUNK_BUFFER_COUNT, 1);
		
	/* Initialize output buffer (with static elements) */
	GrayBatStream<Chunk, decltype(cage)> os(1, cage);
	std::cout << "GrayBatStream" << std::endl;	
		
	size_t fits = 0;
	Clock::time_point t0;
	std::thread cpyThread([&inputBuffer, &os, &t0, &fits, &gen, &d](){
		Chunk chunk;
				
		auto fn = [&](int x) { 
			const float a = -0.01;
			const float b = 10;
			const float c = -2400;

			if(a*x*x + b*x+ c > 0) return a*x*x + b*x + c + d(gen);
			else return 0.0f;						
		};
		for(unsigned int cc = 0; cc < CHUNK_COUNT; cc++) {
			for(unsigned int i = 0; i < SAMPLE_COUNT; i++) {
				chunk[cc*SAMPLE_COUNT + i] = fn(i);
			}
		}
		while(!inputBuffer.isFinished()) {
			Clock::time_point t1 = Clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
			if(static_cast<double>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / ms.count() * 1000 / 1024 / 1024 < dataRate) {
				fits++;
				os.getBuffer().push(chunk);
			}
			std::this_thread::sleep_for(10ms);

		}
		os.getBuffer().producerQuit();
	//	os.join();
	});
	
	float dataRateOut = 0;
	unsigned int outputBufferSize = 0;
	
	std::thread benchThread([&t0, &fits, &os, &dataRateOut, &outputBufferSize](){
		while(1) {
			fits = 0;
			const unsigned int fps = 2;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000/fps));
			Clock::time_point t2 = Clock::now();
			
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0);

			outputBufferSize = os.getBuffer().getSize();
			dataRateOut = static_cast<float>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / ms.count() / 1024 / 1024 * 1000;	
			fits = 0;
			t0 = Clock::now();
		};
	});
	
	StaticTTY tty;
	tty << HSpace('#') << " BenchReader " << HSpace('#') << "\n";
	tty << "\n";
	tty << "Output buffer usage:" << HSpace(' ') << ProgressBar<unsigned int>(30, 0, CHUNK_BUFFER_COUNT, outputBufferSize) << "   \n";
	tty << "Output data rate: " << dataRateOut << " Mib/s\n";

	cpyThread.join();	
	tty.finish();	
	return 0;
}
