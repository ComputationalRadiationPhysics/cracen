#pragma once

#include <type_traits>
#include <functional>
#include <list>
#include <tuple>
#include <vector>
#include <graybat/Cage.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>

#include "Ringbuffer.hpp"

#include "BufferTraits.hpp"
#include "SendPolicies.hpp"

namespace Cracen {

template <
    class KernelFunktor,
    class CageFactory,
    template<class, class> class SendPolicy
>
class Cracen : 
	public InputBufferEnable<typename KernelFunktor::InputType>,
	public OutputBufferEnable<typename KernelFunktor::OutputType>
{
	using Output = typename KernelFunktor::OutputType;
	using Input = typename KernelFunktor::InputType;
	using Cage = typename CageFactory::Cage;
	using Vertex = typename Cage::Vertex;
	using Edge = typename Cage::Edge;
	using Event = typename Cage::Event;
	
	Cage cage;
	
	std::thread receiveThread;
	std::thread sendThread;
	std::thread kernelThread;
	
	KernelFunktor kf;

	template <
		class ReceiveType = Input,
		typename std::enable_if<
			!std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void receive() {
		if(cage.hostedVertices.size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.hostedVertices.size() > 0);
		
		while(true) {
			ReceiveType data();
			cage.recv(data);
			this->inputBuffer.push(data);
		}
	}
	
	template <
		class ReceiveType = Input,
		typename std::enable_if<
			std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void receive() {
	}
	
	
	template <
		class SendType  = Output,
		typename std::enable_if<
			!std::is_same<SendType, void>::value
		>::type * = nullptr
	>
	void send() {
		if(cage.hostedVertices.size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.hostedVertices.size() > 0);
		SendPolicy<Cage, SendType> sendPolicy(cage);
		
		while(!this->outputBuffer.isFinished()) {
			//Send dataset away	
			//Vertex source = cage.hostedVertices.at(0);
			//std::vector<Edge> source_sink = cage.getOutEdges(source);
			
			const SendType out = this->outputBuffer.pop(); 
			sendPolicy(out);
		}
	}
	
	template <
		class SendType = Output,
		typename std::enable_if<
			std::is_same<SendType, void>::value
		>::type * = nullptr
	>
	void send() {}
	
	template <
		class ReceiveType = Input,
		class SendType = Output,
		typename std::enable_if<
			!std::is_same<ReceiveType, void>::value && 
			!std::is_same<SendType, void>::value
		>::type * = nullptr
	>
	void run() {
		while(true) {
			this->outputBuffer.push(kf(this->inputBuffer.pop()));
		}
	}
	
	template <
		class ReceiveType = Input,
		class SendType = Output,
		typename std::enable_if<
			!std::is_same<SendType, void>::value &&
			std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void run() {
		while(true) {
			this->outputBuffer.push(kf()); 
		}
	}
	
	template <
		class ReceiveType = Input,
		class SendType = Output,
		typename std::enable_if<
			!std::is_same<ReceiveType, void>::value &&
			std::is_same<SendType, void>::value
		>::type * = nullptr
	>
	void run() {
		while(true) {
			kf(this->inputBuffer.pop());
			
		}
	}
	
public:
	const static int inputBufferSize = 1000;
	const static int outputBufferSize = 1000;
	Cracen(CageFactory cf) :
		InputBufferEnable<Input>(inputBufferSize, 1),
		OutputBufferEnable<Output>(outputBufferSize, 1),
		cage(cf.commPoly(),
			 cf.graphPoly()
		)
	{
		//Graybat mapping
		cage.distribute(cf.mapping());;
	}
	
	void release() {
		//Fork threads
		try {
			std::thread receiveThread([=](){this->receive<>();});
			std::thread sendThread([=](){this->send<>();});
			this->run<>();
			//std::thread kernelThread([=](){this->run<>();});
		} catch(std::exception e) {
			std::cerr << "Exception thrown" << std::endl;
		};
		
		receiveThread.join();
		kernelThread.join();
		sendThread.join();
	}
	
	KernelFunktor& getKernel() {
		return kf;
	}

};

}
