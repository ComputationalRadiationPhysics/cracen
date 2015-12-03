# pragma once

// STL
#include <utility> /* std::make_pair */

// GRAYBAT
#include <graybat/graphPolicy/Traits.hpp>

namespace graybat {

    namespace pattern {

        template<typename T_GraphPolicy>
        struct FullyConnected {

            using GraphPolicy       = T_GraphPolicy;
            using VertexDescription = graybat::graphPolicy::VertexDescription<GraphPolicy>;
            using EdgeDescription   = graybat::graphPolicy::EdgeDescription<GraphPolicy>;
            using GraphDescription  = graybat::graphPolicy::GraphDescription<GraphPolicy>;
            using EdgeProperty      = graybat::graphPolicy::EdgeProperty<GraphPolicy>;
            using VertexProperty    = graybat::graphPolicy::VertexProperty<GraphPolicy>;  

            const unsigned verticesCount;

            FullyConnected(const unsigned verticesCount) :
                verticesCount(verticesCount){

            }
      
            GraphDescription operator()(){
                std::vector<VertexDescription> vertices(verticesCount);

                assert(vertices.size() == verticesCount);

                std::vector<EdgeDescription> edges;

                for(unsigned i = 0; i < vertices.size(); ++i){
                    vertices.at(i) = std::make_pair(i, VertexProperty());
                }
                
                for(unsigned i = 0; i < vertices.size(); ++i){
                    for(unsigned j = 0; j < vertices.size(); ++j){
                        if(i == j){
                            continue;
                        } 
                        else {
                            edges.push_back(std::make_pair(std::make_pair(vertices[i].first, vertices[j].first), EdgeProperty()));
	      
                        }
	    
                    }

                }

                return std::make_pair(vertices,edges);
            }

        };
      
    } /* pattern */

} /* graybat */
