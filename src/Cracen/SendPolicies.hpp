#pragma once

#include <limits>

namespace Cracen {

template <class Cracen>
class SendPolicyBase {
public:
	unsigned int keepAliveInterval = std::numeric_limits<unsigned int>::max(); // Interval between two keepAlive messages in ms

	void receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::KeepAlive ka) {};
};

template <class Cracen>
class RoundRobinPolicy :
	public SendPolicyBase<Cracen>
{
	int roundRobinCounter;
	std::vector<typename Cracen::Cage::Edge> outEdges;
	typename Cracen::Cage& cage;

public:
	RoundRobinPolicy(typename Cracen::Cage& cage) :
		roundRobinCounter(0),
		cage(cage)
	{
		for(typename Cracen::Cage::Vertex& v : cage.hostedVertices) {
			for(typename Cracen::Cage::Edge e : cage.getOutEdges(v)) {
				outEdges.push_back(e);
			}
		}
	}

	void operator()(const typename Cracen::Output& out) {
		cage.send(outEdges.at(roundRobinCounter), out);

		roundRobinCounter = (roundRobinCounter+1) % outEdges.size();
	}
};

template <class Cracen>
class MinimumWorkloadPolicy  :
	public SendPolicyBase<typename Cracen::Cage>
{
	std::map<typename Cracen::Cage::Edge, unsigned int> edgeWeights;

	typename Cracen::Cage& cage;
	unsigned int maxEdgeWeight = 200; // Maximum edge weight. if the edgeWeight extends that value, the edge is considered broken
	unsigned int keepAliveInterval = 500; // Interval between two keepAlive messages in ms

public:

	MinimumWorkloadPolicy(typename Cracen::Cage& cage) :
		cage(cage)
	{
		for(typename Cracen::Cage::Vertex& v : cage.hostedVertices) {
			for(typename Cracen::Cage::Edge e : cage.getOutEdges(v)) {
				edgeWeights[e] = 0;
			}
		}
	}

	void operator()(const typename Cracen::Output& out) {
		unsigned int minWeight = std::numeric_limits<unsigned int>::max();
		typename Cracen::Cage::Edge minEdge;
		for(const auto& edgePair : edgeWeights) {
			if(edgePair.second < minWeight) {
				minWeight = edgePair.second;
				minEdge = edgePair.first;
			}
		}
		cage.send(minEdge, out);
		edgeWeights[minEdge]++;
	}

	void receiveKeepAlive(typename Cracen::Cage::Edge e, typename Cracen::Cage::KeepAlive ka) {
		edgeWeights[e] = ka.edgeWeight;
	}
};

template <class Cracen>
class BroadCastPolicy  :
	public SendPolicyBase<Cracen>
{
private:
	using Cage = typename Cracen::Cage;
	using Edge = typename Cage::Edge;
	using OutputType = typename Cracen::Output;

	std::vector<Edge> outEdges;
	Cage& cage;

public:
	BroadCastPolicy(Cage& cage) :
		cage(cage)
	{
		for(typename Cage::Vertex& v : cage.getHostedVertices()) {
			for(typename Cage::Edge e : cage.getOutEdges(v)) {
				outEdges.push_back(e);
			}
		}
	}

	void operator()(const OutputType& out) {
		for(auto& edge : outEdges) {
			cage.send(edge, out);
		}
	}
};


}
