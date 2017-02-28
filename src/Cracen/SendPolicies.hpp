#pragma once

#include <limits>

namespace Cracen {

struct NoSend {
public:

	/*
	void operator()(const typename Cracen::Output& out) {
		std::cout << "The NoSend Policy is not meant to be used. For cracen with outgoing edges you have to specify a valid SendPolicy." << std::endl;
		assert(false);
	}
	*/

	template <class Cracen>
	int receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::KeepAlive ka) { return 0; };
};

struct RoundRobinPolicy {
	int vertexCounter;
	int edgeCounter;

public:
	RoundRobinPolicy() :
		vertexCounter(0),
		edgeCounter(0)
	{}

	template <class Cracen>
	void operator()(typename Cracen::Cage& cage, const typename Cracen::Output& out) {
		//TODO: Wrap in std::async to terminate on !running
		auto& vertices = cage.getHostedVertices();
		while(edgeCounter > vertices.at(vertexCounter).getOutEdges()) {
			vertexCounter = (vertexCounter + 1) % vertices.size();
			edgeCounter = 0;
		}
		cage.send(vertices.at(vertexCounter).getOutEdges().at(edgeCounter), out);
		edgeCounter++;
	}

	template <class Cracen>
	int receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::KeepAlive ka) { return 0; };

};

struct MinimumWorkloadPolicy {
	std::map<unsigned int, unsigned int> edgeWeights; // Map from edgeId to edgeWeight

	//unsigned int maxEdgeWeight = 200; // Maximum edge weight. if the edgeWeight extends that value, the edge is considered broken
	//unsigned int keepAliveInterval = 500; // Interval between two keepAlive messages in ms

public:

	template <class Cracen>
	void operator()(typename Cracen::Cage& cage, const typename Cracen::Output& out) {
		unsigned int minWeight = std::numeric_limits<unsigned int>::max();
		typename Cracen::Cage::Edge minEdge;

		for(typename Cracen::Cage::Vertex& v : cage.getHostedVertices()) {
			for(typename Cracen::Cage::Edge e : cage.getOutEdges(v)) {
				if(edgeWeights[e.id] < minWeight) {
					minWeight = edgeWeights[e.id];
					minEdge = e;
				}
			}
		}

		//TODO: Wrap in std::async to terminate on !running
		cage.send(minEdge, out);
		edgeWeights[minEdge.id]++;
	}

	template <class Cracen>
	int receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::Cage::KeepAlive ka) {
		edgeWeights[e.id] = ka.edgeWeight;
		return 0;
	}
};

struct BroadCastPolicy {

	template <class Cracen>
	void operator()(typename Cracen::Cage& cage, const typename Cracen::Output& out) {

		for(typename Cracen::Cage::Vertex& v : cage.getHostedVertices()) {
			for(typename Cracen::Cage::Edge e : cage.getOutEdges(v)) {
				//TODO: Wrap in std::async to terminate on !running
				cage.send(e, out);
			}
		}
	}

	template <class Cracen>
	int receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::KeepAlive ka) { return 0; };
};


}
