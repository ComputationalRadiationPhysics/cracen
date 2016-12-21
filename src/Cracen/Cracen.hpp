#pragma once

#include <type_traits>
#include <functional>
#include <list>
#include <tuple>
#include <vector>
#include <future>
#include <graybat/Cage.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>

#include "Ringbuffer.hpp"

#include "BufferTraits.hpp"
#include "SendPolicies.hpp"

namespace Cracen {

template <class Type, class enable = void>
struct Identity;

template <class Type>
struct Identity<
	Type,
	typename std::enable_if<!std::is_same<Type,void>::value>::type
>
{
	Type operator()(Type value) {
		return value;
	}
};

template <class Type>
struct Identity<
	Type,
	typename std::enable_if<std::is_same<Type,void>::value>::type
>
{
	Type operator()() {
	}
};

template <
    class KernelFunktor,
    class CageFactory,
    template<class, class> class SendPolicy,
	class Incoming = Identity<typename KernelFunktor::InputType>,
	class Outgoing = Identity<typename KernelFunktor::OutputType>,
	std::launch IncomingPolicy = std::launch::deferred,
	std::launch OutgoingPolicy = std::launch::deferred
>
class Cracen :
	public InputBufferEnable<std::future<typename KernelFunktor::InputType>>,
	public OutputBufferEnable<std::future<typename KernelFunktor::OutputType>>
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

	Incoming incomingFunctor;
	KernelFunktor kf;
	Outgoing outgoingFunctor;

	template <
		class ReceiveType = Input,
		typename std::enable_if<
			!std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void receive() {
		if(cage.getHostedVertices().size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.getHostedVertices().size() > 0);

		while(true) {
			ReceiveType data;
			cage.recv(data);
			this->inputBuffer.push(
				std::async(
					IncomingPolicy,
					incomingFunctor,
					data
				)
			);
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
		if(cage.getHostedVertices().size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.getHostedVertices().size() > 0);
		SendPolicy<Cage, SendType> sendPolicy(cage);

		while(!this->outputBuffer.isFinished()) {
			//Send dataset away
			//Vertex source = cage.hostedVertices.at(0);
			//std::vector<Edge> source_sink = cage.getOutEdges(source);
			auto sendFuture = this->outputBuffer.pop();
			sendPolicy(sendFuture.get());
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
			this->outputBuffer.push(
				std::async(
					OutgoingPolicy,
					outgoingFunctor,
					kf(this->inputBuffer.pop())
				)
			);
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
			this->outputBuffer.push(
				std::async(
					OutgoingPolicy,
					outgoingFunctor,
					kf()
				)
			);
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
			kf(this->inputBuffer.pop().get());
		}
	}

public:
	const static int inputBufferSize = 1000;
	const static int outputBufferSize = 1000;
	Cracen(CageFactory cf) :
		InputBufferEnable<std::future<Input>>(inputBufferSize, 1),
		OutputBufferEnable<std::future<Output>>(outputBufferSize, 1),
		cage(
			cf.commPoly(),
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
