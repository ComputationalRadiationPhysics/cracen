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

enum TestCases {
	CracenConstructor,
	CracenDestructor,
	CracenSourceExecute,
	CracenSourceSend,
	CracenIntermediateSend,
	CracenIntermediateReceive,

	NumberOfTestcases
};

std::array<bool, static_cast<size_t>(NumberOfTestcases)> tests;
std::vector<std::string> messages;
constexpr unsigned int kilo = 1024;
constexpr unsigned int mega = kilo * kilo;
using Chunk = std::array<char, mega>;

unsigned int actions;

struct BandwidthSource {
	Chunk chunk;

	BandwidthSource() :
		chunk()
	{}

	Chunk operator()() {
// 		std::cout << "Send" << std::endl;
		actions++;
		return chunk;
	}
};

struct BandwidthIntermediate {
	Chunk chunk;

	BandwidthIntermediate() :
		chunk()
	{}

	Chunk operator()(Chunk chunk) {
// 		std::cout << "Forward" << std::endl;
		actions++;
		return chunk;
	}
};

struct BandwidthSink {
	BandwidthSink() :
		chunk()
	{}

	Chunk chunk;

	void operator()(Chunk) {
// 		std::cout << "Receive" << std::endl;
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
		unsigned int stageSize = static_cast<int>(worldSize) / 3;
		std::cout << "Stages:{" << stageSize + (worldSize - 3*stageSize) << ", " << stageSize << ", " << stageSize << "}" << std::endl;
		return graybat::pattern::Pipeline<GP>(
			std::vector<unsigned int>({
				stageSize + (worldSize - 3*stageSize),
				stageSize,
				stageSize
			})
		);
	}

	auto mapping() {
		const int stageSize = static_cast<int>(worldSize) / 3;

		unsigned int group = [&](){
				if(rank < stageSize + (worldSize - 3*stageSize)) return 0;
			else if(rank < 2*stageSize + (worldSize - 3*stageSize)) return 1;
			else return 2;
		}();
		std::cout << "group:" << group << std::endl;
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

	int stageSize = worldSize / 3;
	int overhang = worldSize - stageSize * 3;

	constexpr size_t chunkSize = sizeof(Chunk::value_type) * mega;

	unsigned int group = [&](){
			if(rank < stageSize + (worldSize - 3*stageSize)) return 0;
		else if(rank < 2*stageSize + (worldSize - 3*stageSize)) return 1;
		else return 2;
	}();

	std::thread printer;
	if(group == 2) {
		printer = std::thread([&]() {
			std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
			while(!done) {
				float rate = actions * chunkSize / std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - begin).count();
				std::cout << "Rank " << rank << ", Datarate = " << rate / mega << "MiB/s" << std::endl;
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

	if(rank < stageSize + overhang ) {
		auto cracen = Cracen::make_cracen(BandwidthSource(), cf, Cracen::BroadCastPolicy());
		std::this_thread::sleep_for(std::chrono::seconds(10));
		cracen->release();
		done = true;
	} else if(rank < stageSize * 2 + overhang) {
		auto cracen = Cracen::make_cracen(BandwidthIntermediate(), cf, Cracen::BroadCastPolicy());
		std::this_thread::sleep_for(std::chrono::seconds(12));
		cracen->release();
		done = true;
	} else {
		auto cracen = Cracen::make_cracen(BandwidthSink(), cf, Cracen::BroadCastPolicy());
		std::this_thread::sleep_for(std::chrono::seconds(15));
		cracen->release();
		done = true;
	};

	if(group == 2)
		printer.join();


	bool allPassed = true;
	for(bool passed : tests) {
		allPassed &= passed;
	}
	if(allPassed) {
		std::cout << "All tests passed." << std::endl;
		return 0;
	} else {
		std::cout << "Tests failed." << std::endl;
		return 1;
	}
};
