#include "../Cracen/Cracen.hpp"

#include <mpi.h>
#include <array>
#include <vector>
#include <cstdint>
#include <chrono>
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"


constexpr unsigned int kilo = 1024;
constexpr unsigned int mega = kilo * 1024;
using Chunk = std::array<std::uint8_t, mega>;

std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> actions;

struct BandwidthSource {
	Chunk chunk;

	Chunk operator()() {
		actions.push_back(std::chrono::high_resolution_clock::now());
		return chunk;
	}
};

struct BandwidthIntermediate {
	Chunk chunk;

	Chunk operator()(Chunk chunk) {
		actions.push_back(std::chrono::high_resolution_clock::now());
		return chunk;
	}
};

struct BandwidthSink {
	Chunk chunk;

	void operator()(Chunk) {
		actions.push_back(std::chrono::high_resolution_clock::now());
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
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(5000);
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

	MPI_Init(NULL, NULL);

    // Get the number of processes
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int stageSize = worldSize / 3;
	int overhang = worldSize - stageSize * 3;

	constexpr size_t chunkSize = sizeof(Chunk::value_type) * Chunk().size();

	std::thread printer([&]() {
		while(true) {
			if(actions.size() == 0) continue;
			float rate = actions.size() * chunkSize / std::chrono::duration<float>(actions.back() - actions.front()).count();
			std::cout << "Datarate = " << rate / mega << "MiB/s" << std::endl;
			actions.clear();
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
	});

	CageFactory cf(worldSize, rank);
	try {
	if(rank < stageSize + overhang ) {
		auto cracen = Cracen::make_cracen(BandwidthSource(), cf, Cracen::BroadCastPolicy());
	} else if(rank < stageSize * 2 + overhang) {
		auto cracen = Cracen::make_cracen(BandwidthIntermediate(), cf, Cracen::BroadCastPolicy());
	} else {
		auto cracen = Cracen::make_cracen(BandwidthSink(), cf, Cracen::BroadCastPolicy());

	};
	} catch(std::exception e) {
		std::cout << e.what() << std::endl;
	}
	MPI_Finalize();
	return 0;
};
