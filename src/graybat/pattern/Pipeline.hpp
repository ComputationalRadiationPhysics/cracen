#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <algorithm>

namespace graybat {
	namespace pattern {
	    typedef unsigned                                                        VertexID;
	    typedef std::pair<VertexID, VertexID>                                   EdgeDescription;
	    typedef std::pair<std::vector<VertexID>, std::vector<EdgeDescription> > GraphDescription;

		struct Pipeline {

			const std::vector<unsigned int> verticesCount;

			Pipeline(const std::vector<unsigned int>& verticesCount) :
				verticesCount(verticesCount) 
			{}
			struct IncGenerator {
			    int current_;
			    IncGenerator (int start) : current_(start) {}
			    int operator() () { return current_++; }
			};
			GraphDescription operator()(){
				const unsigned int totalVertexCount = std::accumulate(verticesCount.begin(), verticesCount.end(), 0);
				std::vector<VertexID> vertices(totalVertexCount);
				std::generate(vertices.begin(), vertices.end(), IncGenerator(0));
				std::vector<EdgeDescription> edges;
				
				unsigned int pos = 0;
				for(unsigned int stage = 1; stage < verticesCount.size() - 1; stage++) {
					for(unsigned int from = pos; from < pos+verticesCount[stage-1]; from++) {
						for(unsigned int to = pos+verticesCount[stage-1]; to < pos+verticesCount[stage-1]+verticesCount[stage]; to++) {
							edges.push_back(std::make_pair(from, to));
						}
					}
					pos += verticesCount[stage-1];
				}
				
				return std::make_pair(vertices,edges);
			}

		};
	}
}

#endif