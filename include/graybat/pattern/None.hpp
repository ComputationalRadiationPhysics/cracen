#pragma once

#include <graybat/graphPolicy/Traits.hpp>

namespace graybat {

    namespace pattern {


        template <typename T_GraphPolicy>
        struct None {
            using VertexDescription = graybat::graphPolicy::VertexDescription<T_GraphPolicy>;
            using EdgeDescription   = graybat::graphPolicy::EdgeDescription<T_GraphPolicy>;
            using GraphDescription  = graybat::graphPolicy::GraphDescription<T_GraphPolicy>;
            
            GraphDescription operator()(){
                std::vector<VertexDescription> vertices;
                std::vector<EdgeDescription> edges;

                return std::make_pair(vertices,edges);
            }

        };

    } /* pattern */

} /* graybat */
