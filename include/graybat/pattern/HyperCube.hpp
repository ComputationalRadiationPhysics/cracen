# pragma once

// STL
#include <utility> /* std::make_pair */

// GRAYBAT
#include <graybat/graphPolicy/Traits.hpp>

namespace graybat {

    namespace pattern {

        template<typename T_GraphPolicy>    
        struct HyperCube {

            using GraphPolicy       = T_GraphPolicy;
            using VertexDescription = graybat::graphPolicy::VertexDescription<GraphPolicy>;
            using EdgeDescription   = graybat::graphPolicy::EdgeDescription<GraphPolicy>;
            using GraphDescription  = graybat::graphPolicy::GraphDescription<GraphPolicy>;
            using EdgeProperty      = graybat::graphPolicy::EdgeProperty<GraphPolicy>;
            using VertexProperty    = graybat::graphPolicy::VertexProperty<GraphPolicy>;  
            
            const unsigned dimension;

            HyperCube(const unsigned dimension) :
                dimension(dimension){

            }
      
            unsigned hammingDistance(unsigned a, unsigned b){
                unsigned abXor = a xor b;
                return (unsigned) __builtin_popcount(abXor);
            }

            GraphDescription operator()(){
                assert(dimension >= 1);
                std::vector<EdgeDescription> edges;

                unsigned verticesCount = pow(2, dimension);
                std::vector<VertexDescription> vertices(verticesCount);

                for(unsigned i = 0; i < vertices.size(); ++i){
                    vertices.at(i) = std::make_pair(i, VertexProperty());
                }
                
                for(unsigned i = 0; i < vertices.size(); ++i){
                    for(unsigned j = 0; j < vertices.size(); ++j){
                        if(hammingDistance(i, j) == 1){
                            edges.push_back(std::make_pair(std::make_pair(vertices[i].first, vertices[j].first), EdgeProperty()));
                        }

                    }
                }
    
                return std::make_pair(vertices,edges);
            }

        };

    } /* pattern */

} /* graybat */
