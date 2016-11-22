#include <string>
#include "../Cracen.hpp"
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"

struct CageFactory {
private:
public:
	
	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<unsigned int>    GP;
	typedef graybat::Cage<CP, GP> Cage;
	
	CageFactory()
	{}
	
	CP::Config commPoly() {
		const std::string signalingIp = "127.0.0.1";
		const std::string localIp = "127.0.0.1";
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(5000);
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(5001);
		std::cout << "My URI =" << peerUri << std::endl;
		const unsigned int contextSize = 2;

		return CP::Config({masterUri, peerUri, contextSize}); //ZMQ Config
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

std::ostream& operator<<(std::ostream& lhs, const  std::vector<char>& rhs) {
	for(const char c : rhs) {
		lhs << c;
	}
	return lhs;
}

class ReceiveFunctor {
public:
	using InputType =  std::vector<char>;
	using OutputType = void;
	
	OutputType operator()(InputType in) {
		std::cout << in << std::endl;
		return OutputType();
	}
};

int main(int argc, char* argv[]) {
	CageFactory cf;

	
	Cracen::Cracen<
		ReceiveFunctor,
		CageFactory,
		Cracen::BroadCastPolicy
	>
	receiveCracen(cf);
	
	receiveCracen.release();
	
	return 0;
}
