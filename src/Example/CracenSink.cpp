#include <string>
#include <typeinfo>
#include "../Cracen/Cracen.hpp"
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"

#include "SignalHandler.hpp"

class ReceiveFunctor {
public:
	void operator()(std::array<char,12> in) {
		for(char c : in) {
			std::cout << c;
		}
		std::cout << std::endl;
	}
};

struct CageFactory {
private:
public:

	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<unsigned int>    GP;
	typedef graybat::Cage<CP, GP> Cage;

	CageFactory()
	{}

	CP::Config commPoly(std::string contextName) {
		const std::string signalingIp = "127.0.0.1";
		const std::string localIp = "127.0.0.1";
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(5000);
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(5001);
		//std::cout << "My URI =" << peerUri << std::endl;
		const unsigned int contextSize = 2;

		return CP::Config({masterUri, peerUri, contextSize, contextName}); //ZMQ Config
		//return CP::Config({}); //BMPI Config
	}

	auto graphPoly() {
		return graybat::pattern::Pipeline<GP>(
			std::vector<unsigned int>({
				1,
				1
			})
		);
	}

	auto mapping() {
		return graybat::mapping::PeerGroupMapping(1);
	}
};

int main(int argc, char* argv[]) {

	//std::cout << "Input:" << typeid(CracenType::Input).name() << std::endl;
	//std::cout << "Output:" << typeid(CracenType::Output).name() << std::endl;

	CageFactory cf;

	auto receiveCracen = Cracen::make_cracen(ReceiveFunctor(), cf);

	waitForSignal(SIGINT);
	std::cout << "Received SIGINT. Shutting down." << std::endl;
	receiveCracen.release();

	return 0;
}
