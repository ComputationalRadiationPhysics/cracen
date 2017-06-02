/**
 *
 * @file
 *
 * @brief This file contains all SendPolicies, that come with the cracen library.
 *
 */


#pragma once

#include <limits>

namespace Cracen {

/**
 *
 * @class Cracen::NoSend
 *
 * @brief The NoSend send policy.
 *
 * This policy is used for sinks, since they dont send any data an therefore dont need a send policy.
 * It is important to use this policy in these cases, because the sendPolicy may block on creation, if
 * there are no hosted verticies.
 *
 */
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

/**
 *
 * @class Cracen::RoundRobinPolicy
 *
 * @brief The RoundRobin send policy.
 *
 * This is a simple send policy, that implements the round robin algorith.
 *
 */
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
		while(static_cast<unsigned int>(edgeCounter) >= cage.getOutEdges(vertices.at(vertexCounter)).size()) {
			vertexCounter = (vertexCounter + 1) % vertices.size();
			edgeCounter = 0;
		}
		cage.send(cage.getOutEdges(vertices.at(vertexCounter)).at(edgeCounter), out);
		edgeCounter++;
	}

	template <class Cracen>
	int receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::KeepAlive ka) { return 0; };

};

/**
 *
 * @class Cracen::MinimumWorkloadPolicy
 *
 * @brief The minimum workload send policy.
 *
 * This policy takes incomeing keep alive messages into account and selects the edge, where the corresponding
 * cracen has the least amount of items in its working queue. Since this policy does send less and less
 * packages to "dead" nodes, it also provides a level of fault tolerance. In cases where it is not acceptable
 * to loose any package, the user has keep track of sending and receiving packages. For these purposes,
 * the additional hooks can be used.
 *
 */
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

/**
 *
 * @class Cracen::BroadCastPolicy
 *
 * @brief The BroadCast send policy.
 *
 * This is a simple send policy, that implements a broadcast of every package.
 *
 */
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
