#pragma once

#include <type_traits>
#include <graybat/Cage.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>

#include "../Utility/Ringbuffer.hpp"

namespace Cracen {

//Nulltype for input nodes without incoming dataflow
class NullType {}; 

template <class Cage, class OutputType>
class RoundRobinPolicy {
	int roundRobinCounter;
public:
	RoundRobinPolicy() :
		roundRobinCounter(0)
	{}
	
	void operator()(Cage& cage, const OutputType& out) {
		const auto outEdges = cage.getOutEdges(cage.hostedVertices());
		cage.send(outEdges.at(roundRobinCounter), out);
		roundRobinCounter = (roundRobinCounter+1) % outEdges.size();
	}
};

template <class Cage, class OutputType>
class BroadCastPolicy {
public:
	void operator()(Cage& cage, const OutputType& out) {
		for(auto edge : cage.getOutEdges(cage.hostedVertices())) {
			cage.send(edge, out);
		}
	}
};

template <class Input, class Ouput, class KernelFunktor, class CageFactory, template<class, class> class SendPolicy = RoundRobinPolicy>
class Cracen {
	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<unsigned int> GP;
	typedef graybat::Cage<CP, GP> Cage;
	typedef typename Cage::Vertex Vertex;
	typedef typename Cage::Edge Edge;
	typedef typename Cage::Event Event;
	
	Cage cage;
	
	Ringbuffer<Input> inputBuffer;
	Ringbuffer<Output> outputBuffer;
	
	std::thread receiveThread;
	std::thread sendThread;
	
	void receive() {
		if(cage.hostedVertices.size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.hostedVertices.size() > 0);
		Vertex sink = cage.hostedVertices.at(0);

		std::this_thread::sleep_for(1ms);
		Input r;
		std::vector<Event> events;
		std::vector<Input> buffer;
		auto inEdges = cage.getInEdges(sink);
		for(Edge e : inEdges) {
			buffer.push_back(Input());
			cage.recv(e, buffer.back(), events);
		}

		bool done = false;
		while(!done) {
			for(int i = 0; i < events.size(); i++) {
				if(events.at(i).ready()) {
					inputBuffer.push(buffer.at(i));
					buffer.erase(buffer.begin()+i);
					events.erase(buffer.begin()+i);
				}
			}
		}
	};
	void send() {
		if(cage.hostedVertices.size() == 0) std::cerr << "Error: No hostedVertices!" << std::endl;
		assert(cage.hostedVertices.size() > 0);
		SendPolicy<Cage, Output> sendPolicy;
		
		while(!outputBuffer.isFinished()) {
			//Send dataset away	
			Vertex source = cage.hostedVertices.at(0);
			std::vector<Edge> source_sink = cage.getOutEdges(source);
			
			sendPolicy(cage, outputBuffer.pop());
		}
	};
	
public:
	Cracen(CageFactory cf) :
		cage(cf.commPoly(), cf.grafPoly())
	{
		//Graybat mapping
		cage.distribute(cf.mapping());;
		
		//Fork threads
		std::thread receiveThread(&Cracen::receive);
		std::thread sendThread(&Cracen::send);
	};
	
	void release(){
		while(true) {
			 if(!std::is_same<Input, NullType>::value && !std::is_same<Output, NullType>::value) {
				//Intermediate or output node
				outputBuffer.push(KernelFunktor()(inputBuffer.pop()));
			 } else if(std::is_same<Input, NullType>::value) {
				//Input node
				outputBuffer.push(KernelFunktor()()); 
			 } else if(std::is_same<Output, NullType>::value) {
				//Output node
				 KernelFunktor()(inputBuffer.pop());
			 } else {
				std::cerr << "Invalid state of Cracen. Neither Input, nor Output is specified." << std::endl;
			 }
			
		}
	}

};

}