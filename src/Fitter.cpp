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

using namespace std::chrono_literals;

int main(int argc, char** argv) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::seconds Seconds;
	
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
	
	/*
	std::thread test([&](){
		for(int i = 0; true; i++) {
			ib->pop();
			std::cout << "Fitter sending package." << std::endl;
			Output o;
			o.status = i;
			stream.send(o);
		}
	});
	*/
	std::thread benchThread([&fits, &reader](){
		while(1) {
			fits = 0;
			Clock::time_point t0 = Clock::now();
			std::this_thread::sleep_for(3s);
			Clock::time_point t1 = Clock::now();
			Seconds s = std::chrono::duration_cast<Seconds>(t1 - t0);
			int elems = reader.getBuffer()->getSize();
			std::cout << static_cast<double>(fits)*SAMPLE_COUNT*CHUNK_COUNT*sizeof(DATATYPE) / s.count() / 1024 / 1024 << "MiB/s, " << elems << " elements in queue." << std::endl;
		};
	});
	
	std::cout << "Nodes created." << std::endl;
	
	reader.readToBuffer();
	std::cout << "Data read." << std::endl;
	//test.join();
	//Make sure all results are written back
	//stream.join();
	while(1);
	return 0;
}
