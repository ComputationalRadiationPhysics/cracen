#pragma once

// STL
#include <utility> /* std::make_pair */

// GRAYBAT
#include <graybat/graphPolicy/Traits.hpp>

namespace graybat {

    namespace pattern {



        template<typename T_GraphPolicy>
        struct Chain {

            using GraphPolicy       = T_GraphPolicy;
            using VertexDescription = graybat::graphPolicy::VertexDescription<GraphPolicy>;
            using EdgeDescription   = graybat::graphPolicy::EdgeDescription<GraphPolicy>;
            using GraphDescription  = graybat::graphPolicy::GraphDescription<GraphPolicy>;
            using EdgeProperty      = graybat::graphPolicy::EdgeProperty<GraphPolicy>;
            using VertexProperty    = graybat::graphPolicy::VertexProperty<GraphPolicy>;            
            
            const unsigned verticesCount;

            Chain(const unsigned verticesCount) :
                verticesCount(verticesCount) {

            }
    

            GraphDescription operator()(){
                std::vector<VertexDescription> vertices;
                
                for(size_t i = 0; i < verticesCount; ++i){
                    if(i == 0){
                        vertices.push_back(std::make_pair(i, VertexProperty()));
                        
                    }
                    else if(i == verticesCount - 1){
                        vertices.push_back(std::make_pair(i, VertexProperty()));
                        
                    }
                    else {
                        vertices.push_back(std::make_pair(i, VertexProperty()));
                        
                    }
                    
                }
                
                std::vector<EdgeDescription> edges;
                if(vertices.size() != 1) {
                    for(size_t i = 0; i < vertices.size() - 1; ++i){
                        edges.push_back(std::make_pair(std::make_pair(vertices[i].first, vertices[i + 1].first), EdgeProperty()));
                            
                    }
                    
                }

                return std::make_pair(vertices,edges);
            
            }

        };

    } /* pattern */

} /* graybat */
