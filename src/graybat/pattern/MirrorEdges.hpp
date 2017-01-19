#pragma once

namespace graybat {

namespace pattern {

template <class GraphPolicy_T>
struct MirrorEdges {

	using GraphPolicy       = typename GraphPolicy_T::GraphPolicy;
	using VertexDescription = typename GraphPolicy_T::VertexDescription;
	using EdgeDescription   = typename GraphPolicy_T::EdgeDescription;
	using GraphDescription  = typename GraphPolicy_T::GraphDescription;
	using EdgeProperty      = typename GraphPolicy_T::EdgeProperty;
	using VertexProperty    = typename GraphPolicy_T::VertexProperty;

	GraphPolicy_T gp;

	MirrorEdges(GraphPolicy_T gp) :
		gp(gp)
	{}

	GraphDescription operator()() {
		GraphDescription graph = gp();
		for(EdgeDescription& e : graph.second) {
			std::swap(e.first.first, e.first.second);
		}

		return graph;
	}
};

template <class GraphPolicy>
MirrorEdges<GraphPolicy> mirrorEdges(GraphPolicy gd) {
	return MirrorEdges<GraphPolicy>(gd);
};

} // End of namespace pattern

} // End of namespace graybat
