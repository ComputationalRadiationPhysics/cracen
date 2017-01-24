#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <algorithm>
#include <graybat/graphPolicy/Traits.hpp>

#include <graybat/graphPolicy/BGL.hpp>

namespace graybat {
	namespace pattern {

		template <class T_GraphPolicy>
		struct Pipeline {
			/*
			 * T_GraphPolicy::VertexProperty
			 *	- operator+
			 *	- operator==
			 * 	- VertexProperty(unsigned)
			 *
			 * T_GraphPolicy::EdgeProperty
			 *	- EdgeProperty()
			 *
			 */

			using GraphPolicy       = T_GraphPolicy;
            using VertexDescription = graybat::graphPolicy::VertexDescription<GraphPolicy>;
            using EdgeDescription   = graybat::graphPolicy::EdgeDescription<GraphPolicy>;
            using GraphDescription  = graybat::graphPolicy::GraphDescription<GraphPolicy>;
            using EdgeProperty      = graybat::graphPolicy::EdgeProperty<GraphPolicy>;
            using VertexProperty    = graybat::graphPolicy::VertexProperty<GraphPolicy>;

			const std::vector<unsigned int> verticesCount;

			Pipeline(const std::vector<unsigned int>& verticesCount) :
				verticesCount(verticesCount)
			{}
			GraphDescription operator()(){
				std::vector<VertexDescription> vertices;
				unsigned int vertexId = 0;
				for(unsigned int stage = 0; stage < verticesCount.size(); stage++) {
					for(unsigned int i = 0; i < verticesCount[stage]; i++) {
						vertices.push_back(std::make_pair(vertexId++, VertexProperty(stage)));
						//std::cout << "vertex.push_back" << vertexId -1 << " " << stage << std::endl;
					}
				};
				std::vector<EdgeDescription> edges;

				for(const auto& v : vertices) {
					for(const auto& vNext : vertices) {
						//std::cout << "v.second " << v.second << " vNext.second " << vNext.second << std::endl;
						if((v.second + 1 == vNext.second)) {
							std::cout << v.first << "->" << vNext.first << std::endl;

							edges.push_back(
								std::make_pair(
									std::make_pair(
										v.first,
										vNext.first
									),
									EdgeProperty()
								)
							);
						}
					}
				}

				return std::make_pair(vertices,edges);
			}
		};
	}
}

#endif
