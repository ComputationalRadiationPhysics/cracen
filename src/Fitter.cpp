#include "Config/Constants.hpp"
#include "Config/NetworkGraph.hpp"
#include "Device/CudaUtil.hpp"
#include "Input/GrayBatReader.hpp"
#include "Output/GrayBatStream.hpp"
#include "Device/Node.hpp"

int main(int argc, char** argv) {
	cage.distribute(graybat::mapping::PeerGroupMapping(1));
	
	GrayBatReader<Chunk, Cage> reader(cage);
	std::cout << "GrayBatReader created." << std::endl;
	GrayBatStream<Output, Cage> stream(1,cage);
	std::cout << "GrayBatStream created." << std::endl;
	
	
	std::vector<Node*> devices;
	std::vector<unsigned int> freeDevices = cuda::getFreeDevices(4);
	//StopWatch sw;
	//sw.start();
	for(unsigned int i = 0; i < freeDevices.size(); i++) {
		/* Start threads to handle Nodes */
		//devices.push_back(new Node(freeDevices[i], reader.getBuffer(), &(stream.getBuffer())));
	}
	
	
	InputBuffer* ib = reader.getBuffer();
	std::thread test([&](){
		for(int i = 0; true; i++) {
			ib->pop();
			std::cout << "Fitter sending package." << std::endl;
			Output o;
			o.status = i;
			stream.send(o);
		}
	});
	
	std::cout << "Nodes created." << std::endl;
	
	reader.readToBuffer();
	std::cout << "Data read." << std::endl;
	test.join();
	//Make sure all results are written back
	//stream.join();
	return 0;
}