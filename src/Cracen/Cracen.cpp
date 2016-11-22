#include "Cracen.h"
#include "Cracen.hpp"

#include <boost/any.hpp>
#include "graybat/pattern/Pipeline.hpp"
#include "graybat/mapping/PeerGroupMapping.hpp"

class CWrapperSender {
public:
	using InputType = void;
	using OutputType = std::vector<char>;

	std::function<void(char*, size_t size)> operation;
	size_t size;

	OutputType operator()() {
		OutputType data(size);
		operation(data.data(), data.size());
		return data;
	}
};

class CWrapperReceiver {
public:
	using InputType = std::vector<char>;
	using OutputType = void;

	std::function<void(char*, size_t size)> operation;

	OutputType operator()(InputType data) {
		operation(data.data(), data.size());
	}
};

class CWrapperIntermediate {
public:
	using InputType = std::vector<char>;
	using OutputType = std::vector<char>;

	std::function<void(char*, size_t size)> operation;

	OutputType operator()(InputType data) {
		operation(data.data(), data.size());
		return data;
	}
};

template <class Cage, class OutputType>
class DynamicSendPolicy {

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

using SourceCracen =
	Cracen::Cracen<
		CWrapperSender,
		CageFactory,
		DynamicSendPolicy
	>;

using IntermediateCracen =
	Cracen::Cracen<
		CWrapperIntermediate,
		CageFactory,
		DynamicSendPolicy
	>;

using SinkCracen =
	Cracen::Cracen<
		CWrapperReceiver,
		CageFactory,
		DynamicSendPolicy
	>;


void CracenInit(CracenHandle* cracenHandle, CracenEnum::Role role) {
	cracenHandle = new CracenHandle;
	/*
	*cracenHandle = cracen.size();
	CageFactory cf;
	switch(role) {
		case CracenEnum::Source:
			cracen.push_back(
 				SourceCracen(cf)
			);
		break;
		case CracenEnum::Intermediate:
			cracen.push_back(
 				IntermediateCracen(cf)
			);
		break;
		case CracenEnum::Sink:
			cracen.push_back(
 				SinkCracen(cf)
			);
		break;
	}
	*/
};
void CracenBind(CracenHandle* cracenHandle, void* fn, CracenEnum::Hook hook) {
	//Cracen& thisCracen = cracen[*cracenHandle];
	//thisCracen.getKernel().setFunction(fn);
}

void CracenRelease(CracenHandle* cracen) {

};
