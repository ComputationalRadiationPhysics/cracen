#pragma once

namespace Cracen {

template <class Cage, class OutputType>
class RoundRobinPolicy {
	int roundRobinCounter;
	std::vector<typename Cage::Edge> outEdges;
	Cage& cage;

public:
	RoundRobinPolicy(Cage& cage) :
		roundRobinCounter(0),
		cage(cage)
	{
		for(typename Cage::Vertex& v : cage.hostedVertices) {
			for(typename Cage::Edge e : cage.getOutEdges(v)) {
				outEdges.push_back(e);
			}
		}
	}

	void operator()(const OutputType& out) {
		cage.send(outEdges.at(roundRobinCounter), out);

		roundRobinCounter = (roundRobinCounter+1) % outEdges.size();
	}
};

template <class Cage, class OutputType>
class MinimumWorkloadPolicy {
	int roundRobinCounter;
	std::vector<typename Cage::Edge> outEdges;
	Cage& cage;

public:
	MinimumWorkloadPolicy(Cage& cage) :
		roundRobinCounter(0),
		cage(cage)
	{
		for(typename Cage::Vertex& v : cage.hostedVertices) {
			for(typename Cage::Edge e : cage.getOutEdges(v)) {
				outEdges.push_back(e);
			}
		}
	}

	void operator()(const OutputType& out) {
		cage.send(outEdges.at(roundRobinCounter), out);

		roundRobinCounter = (roundRobinCounter+1) % outEdges.size();
	}
};

template <class Cage, class OutputType>
class BroadCastPolicy {
private:
	std::vector<typename Cage::Edge> outEdges;
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
