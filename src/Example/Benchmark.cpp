#include "../Cracen/Cracen.hpp"

#include <boost/mpi.hpp>
#include <array>
#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"


constexpr unsigned int kilo = 1024;
constexpr unsigned int mega = kilo * kilo;
using Chunk = std::vector<char>;
Chunk DefaultChunk(mega);

unsigned int actions;

struct BandwidthSource {
	Chunk chunk;

	BandwidthSource() :
		chunk(mega)
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
		chunk(mega)
	{}

	Chunk operator()(Chunk chunk) {
// 		std::cout << "Forward" << std::endl;
		actions++;
		return chunk;
	}
};

struct BandwidthSink {
	BandwidthSink() :
		chunk(mega)
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


	int worldSize;
	int rank;
	CageFactory(int worldSize, int rank) :
		worldSize(worldSize),
		rank(rank)
	{}

	CP::Config commPoly(std::string contextName) {
		const std::string signalingIp = "127.0.0.1";
		const std::string localIp = "127.0.0.1";
		const std::string masterUri = "localhost:"+std::to_string(5000);
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(5002);
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

	std::thread printer([&]() {
		std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
		while(!done) {
			float rate = actions * chunkSize / std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - begin).count();
			std::cout << "Datarate = " << rate / mega << "MiB/s" << std::endl;
			actions = 0;
			begin = std::chrono::high_resolution_clock::now();
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
	});

	CageFactory cf(worldSize, rank);

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

	printer.join();

	return 0;
};
