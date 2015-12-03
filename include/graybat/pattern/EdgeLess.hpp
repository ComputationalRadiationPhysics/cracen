# pragma once

// STL
#include <utility> /* std::make_pair */

// GRAYBAT
#include <graybat/graphPolicy/Traits.hpp>

namespace graybat {

  namespace pattern {


      template<typename T_GraphPolicy>
      struct EdgeLess {

          using GraphPolicy       = T_GraphPolicy;
          using VertexDescription = graybat::graphPolicy::VertexDescription<GraphPolicy>;
          using EdgeDescription   = graybat::graphPolicy::EdgeDescription<GraphPolicy>;
          using GraphDescription  = graybat::graphPolicy::GraphDescription<GraphPolicy>;

          const unsigned verticesCount;

          EdgeLess(const unsigned verticesCount) :
              verticesCount(verticesCount){

          }
      
          GraphDescription operator()(){
              std::vector<VertexDescription> vertices(verticesCount);
              assert(vertices.size() == verticesCount);
              std::vector<EdgeDescription> edges;
              return std::make_pair(vertices,edges);
          }

      };
      
  } /* pattern */

} /* graybat */
