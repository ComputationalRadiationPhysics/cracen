#include <string>
#include "../Cracen.hpp"
#include "../graybat/mapping/PeerGroupMapping.hpp"
#include "../graybat/pattern/Pipeline.hpp"

#include <array>

/*
class SendFunctor {
public:
	using InputType = void;
	using OutputType = int;
	
	int i;
	SendFunctor() :
		i(0)
	{};
	
	OutputType operator()() {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		return i++;
	}
};

class IntermediateFunctor {
public:
	using InputType = int;
	using OutputType = int;
	
	OutputType operator()(int i) {
		std::string received = std::to_string(i);
		int message = i + 2;
		std::cout << "Intermediate received:" << i << " adding two and sending:" << message << std::endl;
		return message;
	}
};

class ReceiveFunctor {
public:
	using InputType = int;
	using OutputType = void;
	
	OutputType operator()(int i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::cout << "End node received: " << i << std::endl;
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
	
	CP::Config commPoly() {
		const std::string signalingIp = "127.0.0.1";
		const std::string localIp = "127.0.0.1";
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(5000);
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(5002);
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
		return graybat::mapping::PeerGroupMapping(0);
	}
};

int main(int argc, char* argv[]) {
	CageFactory cf;

	Cracen::Cracen<
		SendFunctor,
		CageFactory,
		Cracen::BroadCastPolicy
	> sendCracen(cf);
	
	Cracen::Cracen<
		IntermediateFunctor,
		CageFactory,
		Cracen::BroadCastPolicy
	> intermediateCracen(cf);
	
	Cracen::Cracen<
		ReceiveFunctor,
		CageFactory,
		Cracen::BroadCastPolicy
	> endCracen(cf);
		
	sendCracen.release();
	intermediateCracen.release();
	endCracen.release();

	return 0;
}
*/
