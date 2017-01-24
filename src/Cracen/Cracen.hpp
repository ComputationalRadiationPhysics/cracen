#pragma once

#include <type_traits>
#include <functional>
#include <vector>
#include <atomic>
#include <cstdint>
#include <future>
#include <chrono>
#include <tuple>
#include <graybat/Cage.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include "graybat/pattern/MirrorEdges.hpp"

#include "Ringbuffer.hpp"
#include "Util/OptionalAttribute.hpp"
#include "Cracen/Functor/Identity.hpp"
#include "Cracen/Meta/FunctionInfo.hpp"
#include "Cracen/Meta/ConditionalInvoke.hpp"

#include "BufferTraits.hpp"
#include "SendPolicies.hpp"

namespace Cracen {

template <
    class KernelFunktor,
    class CageFactory,
    template<class> class SendPolicy,
	template<class> class Incoming = Functor::Identity,
	template<class> class Outgoing = Functor::Identity,
	std::launch IncomingPolicy = std::launch::deferred,
	std::launch OutgoingPolicy = std::launch::deferred
>
class Cracen
{
	using Self = Cracen<
		KernelFunktor,
		CageFactory,
		SendPolicy,
		Incoming,
		Outgoing,
		IncomingPolicy,
		OutgoingPolicy
	>;
	using KernelInfo = Meta::FunctionInfo<KernelFunktor>;
	static_assert(std::tuple_size<typename KernelInfo::ParamList>::value <= 1, "Kernels for cracen can have at most 1 Argument.");

	const static int inputBufferSize = 1000;
	const static int outputBufferSize = 1000;

public: 	// Get in- and output from Functor
	using Input = typename std::tuple_element<
		0, // Get first element of parameter list
		typename std::conditional< // check if parameter is empty
			std::tuple_size<typename KernelInfo::ParamList>::value == 0,
			std::tuple<void>, // parameter list is empty => add a void type, so that Input is well defined
			typename KernelInfo::ParamList // parameter list is not empty => use parameter list
		>::type
	>::type;

	using Output = typename std::result_of<
		typename Meta::ConditionalInvoke<
			!std::is_same<Input, void>::value,
			KernelFunktor,
			Input
		>::type
	>::type;

	struct KeepAlive {
		unsigned int edgeWeight;
	};

	using Cage = typename CageFactory::Cage;

private:
	// Define graybat types

	using Vertex = typename Cage::Vertex;
	using Edge = typename Cage::Edge;
	using Event = typename Cage::Event;

	// Ringbuffers for in and output
	OptionalAttribute<Ringbuffer<std::future<Input>>, !std::is_same<Input, void>::value> inputBuffer;
	OptionalAttribute<Ringbuffer<std::future<Output>>, !std::is_same<Output, void>::value> outputBuffer;

	// cages for dataflow and meta-dataflow
	Cage dataCage;
	Cage metaCage;

	// SendPolicy instance
	OptionalAttribute<SendPolicy<Self>, !std::is_same<Output, void>::value> sendPolicy;

	std::map<
		typename Cage::Vertex::VertexID,
		std::atomic<std::uint64_t>
	> edgeWeights;
	std::thread receiveThread;
	std::thread sendThread;
	std::thread kernelThread;

	Incoming<Input> incomingFunctor;
	KernelFunktor kf;
	Outgoing<Output> outgoingFunctor;

	template <
		class ReceiveType = Input,
		typename std::enable_if<
			!std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void receive() {
		if(dataCage.getHostedVertices().size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(dataCage.getHostedVertices().size() > 0);
		while(true) {
			ReceiveType data;
			dataCage.recv(data);
			std::cout << "Message received." << std::endl;
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
		if(dataCage.getHostedVertices().size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(dataCage.getHostedVertices().size() > 0);

		while(!this->outputBuffer.isFinished()) {
			//Send dataset away
			//Vertex source = dataCage.hostedVertices.at(0);
			//std::vector<Edge> source_sink = dataCage.getOutEdges(source);
			auto sendFuture = this->outputBuffer.pop();
			auto  message = sendFuture.get();
			std::cout << "Message sent.("<< &message[0] <<")" << std::endl;
			sendPolicy(std::move(message));

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
	Cracen(CageFactory cf) :
		inputBuffer(inputBufferSize, 1),
		outputBuffer(outputBufferSize, 1),
		dataCage(
			cf.commPoly("DataContext"),
			cf.graphPoly()
		),
		metaCage(
			cf.commPoly("MetaContext"),
			mirrorEdges(cf.graphPoly())
		),
		sendPolicy( // distribute dataCage, for the sendPolicy to be initilised correctly
			[](auto& dataCage, auto& cf) -> auto {
				dataCage.distribute(cf.mapping());
				return std::ref(dataCage);
			}(dataCage, cf)
		)
	{
		//Graybat mapping
		metaCage.distribute(cf.mapping());
	}

	void release() {
		//Fork threads

		try {
			receiveThread = std::thread([=](){this->receive<>();});
			sendThread = std::thread([=](){this->send<>();});
			kernelThread = std::thread([=](){this->run<>();});
		} catch(std::exception e) {
			std::cerr << "Exception thrown" << std::endl;
		};

		auto nextWakeUp = std::chrono::steady_clock::now();

		using KeepAliveMessage = std::tuple<
			typename Cage::Edge,
			std::vector<KeepAlive>,
			std::vector<typename Cage::Event>
		>;

		std::vector<
			KeepAliveMessage
		> keepAliveMessages;

		for(const auto& vertex : metaCage.getHostedVertices()) {
			for(const auto& edge : metaCage.getInEdges(vertex)) {
				keepAliveMessages.push_back(KeepAliveMessage(edge, std::vector<KeepAlive>(1), {}));
				metaCage.recv(
					std::get<0>(keepAliveMessages.back()),
					std::get<1>(keepAliveMessages.back()),
					std::get<2>(keepAliveMessages.back())
				);
			}
		}

		while(true) {
			//Check for received KeepAlives
			for(auto& kam : keepAliveMessages) {
				while(std::get<2>(kam).at(0).ready()) {
					std::cout << "Received KeepAlive from " << std::get<0>(kam).source.id << std::endl;
					// Received KeepAlive, update record
					const typename Cage::Edge& edge = std::get<0>(kam);
					edgeWeights[edge.source.id] = std::get<1>(kam).at(0).edgeWeight;

					std::get<1>(kam).clear();
					std::get<1>(kam).push_back(KeepAlive());
					std::get<2>(kam).clear();
					// Try to receive next Message
					metaCage.recv(
						std::get<0>(kam),
						std::get<1>(kam),
						std::get<2>(kam)
					);
				}
			}

			// Send KeepAlive message to all sending nodes
			for(auto& vertex : metaCage.getHostedVertices()) {
				KeepAlive ka;
				using BufferType = typename decltype(inputBuffer)::type;
				ka.edgeWeight = inputBuffer.template optionalCall<decltype(&BufferType::getSize), &BufferType::getSize, int>();
				std::vector<KeepAlive> kam { ka };
				for(typename Cage::Edge& e : metaCage.getOutEdges(vertex)) {
					std::cout << "Send KeepAlive {" << ka.edgeWeight << "} to " << e.target.id << std::endl;
					metaCage.send( e, kam );
				}
			}

			/* Sleep until next KeepAlives */

			nextWakeUp += std::chrono::milliseconds(1000);
			std::this_thread::sleep_until(nextWakeUp);
		}


		receiveThread.join();
		kernelThread.join();
		sendThread.join();
	}

	KernelFunktor& getKernel() {
		return kf;
	}

};

}
