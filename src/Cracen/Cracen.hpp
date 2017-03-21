/**
 *
 * @file
 *
 * @brief The main header of Cracen.
 *
 */

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
#include "Meta/OptionalAttribute.hpp"
#include "Cracen/Functor/Identity.hpp"
#include "Cracen/Meta/FunctionInfo.hpp"
#include "Cracen/Meta/ConditionalInvoke.hpp"
//#include "Cracen/Util/TerminatableCall.hpp"

#include "BufferTraits.hpp"
#include "SendPolicies.hpp"


namespace Cracen {

/**
 *
 * @class Cracen::Cracen
 *
 * @brief This class provides the interface and main functionality of the cracen toolkit.
 *
 * The this class owns the memory for the buffers, all the corresponing threads and graybat cage.
 * The threads will be spawned withing the constructor. The destructor will join all spawned threads.
 *
 *
 * @tparam KernelFunktor
 *     Type of the KernelFunctor, first argument of the constructor. To avoid writing all the
 *     template arguments explicitly, it is recomended to use the make_cracen() function instead,
 *     if possible. The KernelFunctor is called sequentially. The order of Kernel calls is the same
 *     as the order the data is received.
 *
 *
 * @tparam CageFactory
 *     Type of CageFactor, the second argument of the constructor. The minimum api this type must
 *     have:
 *
 *         struct CageFactory {
 *             typedef graybat::communicationPolicy::ZMQ CP;
 *             //typedef graybat::communicationPolicy::BMPI CP;
 *             typedef graybat::graphPolicy::BGL<unsigned int>    GP;
 *             typedef graybat::Cage<CP, GP> Cage;
 *
 *             CageFactory();
 *
 *             CP::Config commPoly(std::string contextName);
 *             GraphPolicy graphPoly();
 *             Mapping mapping();
 *         };
 *
 *
 *
 *
 * @tparam SendPolicy
 *     Type of the SendPolicy. The SendPolicy is a functor, that takes a cage and a value and sends
 *     the value over one or multiple out edges of the cage. Cracen comes with multiple send SendPolicies
 *     implemented and ready to use. For example broadcast, round robin, minimum workload and no send.
 *
 * @tparam Incoming
 *     Incoming is a Functor, that will be executed before the KernelFunctor. It can be executed asynchron,
 *     but is guaranteed to be finished before the KernelFunctor is called. This is mainly used for long
 *     running kernels, where it makes sense to run preprocessing or logging parallel to the kernel.
 *     In some cases it might be a good idea to put the main work from the kernel into the Incoming functor to enable
 *     parallel execution.
 *     Default value is Cracen::Functor::Identity
 *
 * @tparam Outgoing
 *    Same as the Incoming. The execution time is right before the data is put to the send buffer of the underlying
 *    communication policy (zeromq, mpi, ...)
 *
 * @tparam IncomingPolicy
 *    The launch policy for Incoming functor. It is of type std::launch and can be `std::launch::deferred`,
 *    `std::launch::async`, or  `std::launch::async | std::launch::deferred`. `std::launch::async` must be
 *    used with caution. It does spawn a new thread for every incoming package and can have a big impact on
 *    the overall performance. Another problem can arrive, if the maximum number of threads is exeeded. This
 *    will throw an exception, that is not catched by the Cracen framework (performance consideration). If
 *    this execption is not catched by the outside code. The programm may crash.
 *
 * @tparam OutgoingPolicy
 *     Same as IncomingPolicy
 *
 *
 */
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

	//TODO: make buffer sizes configurable
	const int inputBufferSize = 100;
	const int outputBufferSize = 100;
	// Interval in which cracen is checking if it has to terminate in milliseconds
	// Don't make this interval too short, since it can cause little overhead and a second is acceptable for termination
	//TODO: make polling interval configurable
	const int pollingInterval = 100;

public: 	// Get in- and output from Functor

	/**
	 * @typedef Input
	 *     The type of the Input, derived form the KernelFunctor. The Input type is the type of the
	 *     first argument of the functor. If it has no arguments, then it is `void`.
	 */
	using Input = typename std::tuple_element<
		0, // Get first element of parameter list
		typename std::conditional< // check if parameter is empty
			std::tuple_size<typename KernelInfo::ParamList>::value == 0,
			std::tuple<void>, // parameter list is empty => add a void type, so that Input is well defined
			typename KernelInfo::ParamList // parameter list is not empty => use parameter list
		>::type
	>::type;


	/**
	 * @typedef Output
	 *     The type of the Output, derived form the KernelFunctor. The Output type is the type of the
	 *     result of the functor. The result may be `void` if it is a data sink.
	 *
	 */
	using Output = typename std::result_of<
		typename Meta::ConditionalInvoke<
			!std::is_same<Input, void>::value,
			KernelFunktor,
			Input
		>::type
	>::type;

	/**
	 * @class Cracen::Cracen::KeepAlive
	 *
	 * @brief
	 *     This class is send as KeepAlive message on all edges in the backward direction (data sink
	 *     to data source). The interval of KeepAlive Messages can be configured (inf if not wanted).
	 *     The EdgeWeight data can be used by the SendPolicies. The MinimumWorkloadPolicy uses this
	 *     value to select the node with the least stress.
	 *
	 */
	struct KeepAlive {
		unsigned int edgeWeight;
	};


	/**
	 * @typedef Cage
	 *     The type of the cage, that the cracen owns.
	 */
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
	Cage dataCage; // cage for sending data
	Cage metaCage; // back channel for keep alive messages and maybe more in the future

	// SendPolicy instance
	OptionalAttribute<SendPolicy, !std::is_same<Output, void>::value> sendPolicy;

	std::shared_ptr<std::atomic<bool>> running; //flag is set to false, if the cracen is released.
	std::thread receiveThread;
	std::thread sendThread;
	std::thread kernelThread;
	std::thread keepAliveThread;

	Incoming<Input> incomingFunctor;
	KernelFunktor kf;
	Outgoing<Output> outgoingFunctor;


	template <
		class ReceiveType = Input,
		//SFINAE expression to check if the cracen is not a data source
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
		//SFINAE expression to check if the cracen is not a data source
		typename std::enable_if<
			std::is_same<ReceiveType, void>::value
		>::type * = nullptr
	>
	void receive() {
		// Implementation intentionally empty. Sources dont receive data.
	}


	template <
		class SendType  = Output,
		//SFINAE expression to check if the cracen is not a data sink
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
		//SFINAE expression to check if the cracen is a data sink
		typename std::enable_if<
			std::is_same<SendType, void>::value
		>::type * = nullptr
	>
	void send() {
		// Implementation intentionally empty. Sinks dont send data.
	}

	template <
		class ReceiveType = Input,
		class SendType = Output,
		// Check if cracen is not source and not sink (intermediate node)
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
		// Check if cracen is source
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
		// Check if cracen is sink
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
					// TODO: fix termination
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
	/**
	 * @fn Cracen(KernelFunktor kf, CageFactory cf, SendPolicy sendPolicy = NoSend())
	 *
	 * @brief
	 *     The constructor will spawn all threads and create the cages. It will also
	 *     block the execution until the cages are distributed and the communication context is
	 *     established for all participants.
	 *
	 * @param KernelFunctor
	 *     The executed functor object. See the corresponding template parameter for additional info.
	 *
	 * @param CageFactory
	 *     The cage factory object. See the corresponding template parameter for additional info.
	 *
	 * @param SendPolicy
	 *     The send policy object. See the corresponding template parameter for additional info.
	 *
	 */
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
		// distribute data cage and meta cage. These calls will block the execution until all
		// participants joind the communication
		dataCage.distribute(cf.mapping());
		metaCage.distribute(cf.mapping());
		std::cout << "Cages distributed." << std::endl;
		// Spawen the threads for sending, receiving and computing. This must be done after the
		// distribution of the cages, because some of the threads require to have a valid cage present
		// at all times.
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

	/**
	 * @fn ~Cracen()
	 *
	 * @brief
	 *     The destructor will join all threads. For the threads to be able to terminate `release()`
	 *     must be called on cracen. This will stop the main execution loop of the send and receive threads.
	 *     Once the buffer are emptied, the threads will terminate and the destructor return.
	 */

	~Cracen() {
		receiveThread.join();
		sendThread.join();
		kernelThread.join();
		keepAliveThread.join();
	}

	// Copy and move constructor is deleted, because cracen holds multiple threads, which have a pointer to
	// the original cracen. There is no effective way to copy or move the cracen and keep all threads
	// in a valid state.
	Cracen(Cracen&& rhs) = delete;
	Self& operator=(Cracen&& rhs) = delete;
	Cracen(const Cracen& cracen) = delete;
	void operator=(const Cracen& cracen) = delete;

	/**
	 * @fn void release()
	 *
	 * @brief
	 *     Request releasing the resources, the cracen holds. The main loop of the sending/receiving/computing
	 *     threads will stop execution. After the release and termination of all spawned threads, the destructor
	 *     of cracen will return.
	 *
	 *
	 */
	void release() {
		*running = false;
	}


	/**
	 * @fn KernelFunktor& getKernel()
	 *
	 * @brief
	 *     Simple getter for the kernel functor.
	 *
	 */
	KernelFunktor& getKernel() {
		return kf;
	}

};

/**
 * @fn std::unique_ptr<Cracen<Args...>> make_cracen(Args... args)
 *
 * @brief
 *     Helper function to create a cracen, similar to the STL functions like make_pair, make_shared
 *     and so on. The only difference is, that the function does not return a value (since cracen is
 *     neither copyable nor moveable), but return a unique_ptr.
 *     This may get obsolete with C++17 template parameter deduction.
 *
 */
template <class... Args>
std::unique_ptr<Cracen<Args...>> make_cracen(Args... args) {
	return std::unique_ptr<Cracen<Args...>>(new Cracen<Args...>(args...));
}

}
