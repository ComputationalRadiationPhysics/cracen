#ifndef PEERGROUPMAPPING_HPP
#define PEERGROUPMAPPING_HPP

#include <vector>    /* std::vector */

namespace graybat {

    namespace mapping {
	struct PeerGroupMapping {
	    
		unsigned int stage;
		
		PeerGroupMapping(unsigned int stage) :
			stage(stage)
		{}
		
	    template<typename T_Graph>
	    std::vector<typename T_Graph::Vertex> operator()(const unsigned processID, const unsigned processCount, T_Graph &graph){
			typedef typename T_Graph::Vertex Vertex;
			
			std::vector<Vertex> myVertices;
			assert(graph.getVertices().size() > 0);
			std::cout << graph.getVertices().size() << std::endl;
			myVertices.push_back(graph.getVertex(stage));
			
			return myVertices;
	    }
	};
    } /* mapping */
    
} /* graybat */

#endif