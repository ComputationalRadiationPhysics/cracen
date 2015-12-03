#pragma once

// STL
#include <utility> /* std::pair */
#include <vector>

namespace graybat {

    namespace graphPolicy {

        // namespace traits {
        // }

        template <typename T_GraphPolicy>
        using VertexProperty = typename T_GraphPolicy::VertexProperty;

        template <typename T_GraphPolicy>
        using EdgeProperty = typename T_GraphPolicy::EdgeProperty;

        using VertexID = size_t;
        
        template <typename T_GraphPolicy>        
        using VertexDescription = std::pair<VertexID, VertexProperty<T_GraphPolicy> >;

        template <typename T_GraphPolicy>        
        using EdgeDescription = std::pair< std::pair<
                                               VertexID
                                               ,VertexID >
                                           ,EdgeProperty<T_GraphPolicy> >;

        template <typename T_GraphPolicy>
        using GraphDescription = std::pair<
            std::vector<VertexDescription<T_GraphPolicy> >,
            std::vector<EdgeDescription<T_GraphPolicy> >
            >;

        
    } // namespace graphPolicy
    
} // namespace graybat
