#include "../Cracen/Cracen.hpp"

#include <boost/mpi.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <array>
#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"
#include "Cracen/Util/Whoami.hpp"
constexpr unsigned int kilo = 1024;
constexpr unsigned int mega = kilo * kilo;
constexpr unsigned int chunkSize = 10*mega;
using Chunk = std::vector<char>;
Chunk DefaultChunk(chunkSize);

unsigned int actions;

struct BandwidthSource {
	Chunk chunk;

	BandwidthSource() :
		chunk(chunkSize)
	{}

	Chunk operator()() {
// 		std::cout << "Send" << std::endl;
		actions++;
		return chunk;
	}
};

struct BandwidthIntermediate {

	BandwidthIntermediate()
	{}

	Chunk operator()(Chunk chunk) {
// 		std::cout << "Forward" << std::endl;
		actions++;
		return chunk;
	}
};

struct BandwidthSink {
	BandwidthSink()
	{
	}

	void operator()(Chunk chunk) {
 	//	std::cout << "Receive" << std::endl;
		actions++;
	}
};

struct CageFactory {
private:
public:

	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<unsigned int>    GP;
	typedef graybat::Cage<CP, GP> Cage;
	typedef std::pair<std::string, unsigned int> Endpoint;

	const Endpoint signalingIp;
	const Endpoint localIp ;

	int worldSize;
	int rank;
	CageFactory(int worldSize, int rank, const Endpoint& signalingIp, const Endpoint& localIp) :
		signalingIp(signalingIp),
		localIp(localIp),
		worldSize(worldSize),
		rank(rank)
	{}

	CP::Config commPoly(std::string contextName) {

		const std::string masterUri = "tcp://"+signalingIp.first+":"+std::to_string(signalingIp.second);
		const std::string peerUri = "tcp://"+localIp.first+":"+std::to_string(localIp.second);
		//std::cout << "My URI =" << peerUri << std::endl;
		const unsigned int contextSize = worldSize;

		return CP::Config({masterUri, peerUri, contextSize, contextName}); //ZMQ Config
		//return CP::Config({}); //BMPI Config
	}
	auto graphPoly() {
		std::cout << "Stages:{" << 1  << ", " << worldSize - 2 << ", " << 1 << "}" << std::endl;
		return graybat::pattern::Pipeline<GP>(
			std::vector<unsigned int>({
				1,
				static_cast<unsigned>(worldSize) - 2,
				1
			})
		);
	}

	auto mapping() {
		unsigned int group = [&](){
			if(rank == 0) return 0;
			if(rank == 1) return 2;
			return 1;
		}();
		std::cout << "group:" << group  << " rank:" << rank << std::endl;
		return graybat::mapping::PeerGroupMapping(group);
	}
};

int main(int argc, char* argv[]) {

	if(argc < 2) {
		std::cerr << "No configuration file passed. Usage \"mpirun [...] ./Benchmark path/to/config.ini.\"" << std::endl;
		return 1;
	}

	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(argv[1], pt);

	if(pt.get<bool>("local.autoconfiguration")) {
		std::cout << "Ip autoconfiguration enabled." << std::endl;
		const std::string local = Whoami(pt.get<std::string>("signaling.ip"));
		std::cout << "Resolved " << local << std::endl;
		pt.put<std::string>("local.ip", local);
	}


	std::atomic<bool> done(false);
	boost::mpi::environment env;
	boost::mpi::communicator world;

    // Get the number of processes
    int worldSize = world.size();

    // Get the rank of the process
    int rank = world.rank();

	constexpr size_t byteSize = sizeof(Chunk::value_type) * chunkSize;

	std::thread printer;

	if(rank == 1) {
		printer = std::thread([&]() {
			std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
			while(!done) {
				float rate = actions * byteSize / std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - begin).count();
				std::cout << rank << " " << "Datarate = " << rate / mega << "MiB/s" << std::endl;
				actions = 0;
				begin = std::chrono::high_resolution_clock::now();
				std::this_thread::sleep_for(std::chrono::seconds(2));
			}
		});
	}

	CageFactory::Endpoint signaling = std::make_pair(
		pt.get<std::string>("signaling.ip"),
		std::stoul(pt.get<std::string>("signaling.port"), nullptr, 10)
	);
	CageFactory::Endpoint local = std::make_pair(
		pt.get<std::string>("local.ip"),
		std::stoul(pt.get<std::string>("local.port"), nullptr, 10)
	);
	CageFactory cf(worldSize, rank, signaling, local);

	if(rank == 0) {
		auto cracen = Cracen::make_cracen(BandwidthSource(), cf, Cracen::RoundRobinPolicy());
		std::this_thread::sleep_for(std::chrono::seconds(10));
		cracen->release();
		done = true;
	} else if(rank  > 1) {
		auto cracen = Cracen::make_cracen(BandwidthIntermediate(), cf, Cracen::RoundRobinPolicy());
		std::this_thread::sleep_for(std::chrono::seconds(12));
		cracen->release();
		done = true;
	} else {
		auto cracen = Cracen::make_cracen(BandwidthSink(), cf, Cracen::NoSend());
		std::this_thread::sleep_for(std::chrono::seconds(15));
		cracen->release();
		done = true;
	};

	if(rank == 1)
		printer.join();

	return 0;
};
