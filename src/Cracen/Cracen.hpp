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
//#include "Cracen/Util/TerminatableCall.hpp"

#include "BufferTraits.hpp"
#include "SendPolicies.hpp"

namespace Cracen {

template <
    class KernelFunktor,
    class CageFactory,
    class SendPolicy = NoSend,
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

	const int inputBufferSize = 100;
	const int outputBufferSize = 100;
	// Interval in which cracen is checking if it has to terminate in milliseconds
	// Don't make this interval too short, since it can cause little overhead and a second is acceptable for termination
	const int pollingInterval = 100;

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
	OptionalAttribute<SendPolicy, !std::is_same<Output, void>::value> sendPolicy;

	std::shared_ptr<std::atomic<bool>> running;
	std::thread receiveThread;
	std::thread sendThread;
	std::thread kernelThread;
	std::thread keepAliveThread;

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
		if(dataCage.getHostedVertices().size() != 1) {
			std::cerr
				<< "Error: Wrong number of hosted vertices! cage.hostedVericies().size() == "
				<<  dataCage.getHostedVertices().size()
				<< ", but it has to equal 1"
				<< std::endl;
			std::exit(EXIT_FAILURE);
		}
		while(*running) {
// 			std::cout << "Receive " << this->inputBuffer.getSize() << std::endl;
			ReceiveType data;
			dataCage.recv(data);
			//std::cout << "Message received." << std::endl;
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
		while(*running) {
			//Send dataset away
			//Vertex source = dataCage.hostedVertices.at(0);
			//std::vector<Edge> source_sink = dataCage.getOutEdges(source);
// 			std::cout << "SendBuffer " << this->outputBuffer.getSize() << std::endl;
			auto sendFuture = this->outputBuffer.pop();
			auto  message = sendFuture.get();
			//std::cout << "Message sent.("<< &message[0] <<")" << std::endl;
			sendPolicy.template operator()<Self>(dataCage, std::move(message));

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
		while(*running) {
			this->outputBuffer.push(
				std::async(
					OutgoingPolicy,
					outgoingFunctor,
					kf(this->inputBuffer.pop().get())
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
		while(*running) {
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
		while(*running) {
			kf(this->inputBuffer.pop().get());
		}
	}


	void keepAlive() {
		auto nextWakeUp = std::chrono::steady_clock::now();

		using KeepAliveMessage = std::tuple<
			typename Cage::Edge,
			std::vector<KeepAlive>,
			std::vector<typename Cage::Event>
		>;

		std::vector<
			KeepAliveMessage
		> keepAliveMessages;

		std::vector<typename Cage::Event> sendEvent{};

		for(const auto& vertex : metaCage.getHostedVertices()) {
			for(const auto& edge : metaCage.getInEdges(vertex)) {
				keepAliveMessages.push_back(KeepAliveMessage(edge, std::vector<KeepAlive>(1), {}));
				//TODO: Wrap in std::async to terminate on !running
				metaCage.recv(
					std::get<0>(keepAliveMessages.back()),
					std::get<1>(keepAliveMessages.back()),
					std::get<2>(keepAliveMessages.back())
				);
			}
		}

		while(*running) {
			//Check for received KeepAlives
			for(auto& kam : keepAliveMessages) {
				if(std::get<2>(kam).at(0).ready()) {
					//std::cout << "Received KeepAlive from " << std::get<0>(kam).source.id << std::endl;
					// Received KeepAlive, update record
					const typename Cage::Edge& edge = std::get<0>(kam);
					const KeepAlive& ka = std::get<1>(kam).at(0);
					sendPolicy.template optionalCall(&SendPolicy::template receiveKeepAlive<Self>, edge, ka);

					//edgeWeights[edge.source.id] = std::get<1>(kam).at(0).edgeWeight;
					//TODO: Call send policy with edge and weight
					std::get<1>(kam).clear();
					std::get<1>(kam).push_back(KeepAlive());
					std::get<2>(kam).clear();
					// Try to receive next Message
					//TODO: Wrap in std::async to terminate on !running
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
				ka.edgeWeight = inputBuffer.template optionalCall(&BufferType::getSize);
				std::vector<KeepAlive> kam { ka };
				for(typename Cage::Edge& e : metaCage.getOutEdges(vertex)) {
					//std::cout << "Send KeepAlive {" << ka.edgeWeight << "} to " << e.target.id << std::endl;
					metaCage.send( e, kam, sendEvent);
					/*
					Util::terminateAble(
						[&](){
							metaCage.send( e, kam, sendEvent);
						},
						running
					);
					*/
				}
			}

			/* Sleep until next KeepAlives */

			nextWakeUp += std::chrono::milliseconds(1000);
			std::this_thread::sleep_until(nextWakeUp);
		}
	}


public:
	Cracen(KernelFunktor kf, CageFactory cf, SendPolicy sendPolicy = NoSend()) :
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
		sendPolicy( sendPolicy ),
		running(
			new std::atomic<bool>(true)
		),
		kf(kf)
	{
		dataCage.distribute(cf.mapping());
		metaCage.distribute(cf.mapping());
		std::cout << "Cages distributed." << std::endl;
		receiveThread = std::thread(
			[=](){
				this->receive<>();
			}
		);
		sendThread = std::thread(
			[=](){
				this->send<>();
			}
		);
		kernelThread = std::thread(
			[=](){
				this->run<>();
			}
		);
		keepAliveThread = std::thread(
			[=](){
				this->keepAlive();
			}
		);
	}

	~Cracen() {
		receiveThread.join();
		sendThread.join();
		kernelThread.join();
		keepAliveThread.join();
	}

	Cracen(Cracen&& rhs) = delete;
	Self& operator=(Cracen&& rhs) = delete;

	Cracen(const Cracen& cracen) = delete;
	void operator=(const Cracen& cracen) = delete;

	void release() {
		*running = false;
	}

	KernelFunktor& getKernel() {
		return kf;
	}

};

template <class... Args>
std::unique_ptr<Cracen<Args...>> make_cracen(Args... args) {
	return std::unique_ptr<Cracen<Args...>>(new Cracen<Args...>(args...));
}

}
