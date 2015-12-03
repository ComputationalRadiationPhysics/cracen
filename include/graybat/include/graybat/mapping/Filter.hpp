#pragma once

#include <algorithm> /* std::sort*/
#include <vector>    /* std::vector */

namespace graybat {
    
    namespace mapping {
    
	struct Filter {

            const size_t vertexTag;
            
            Filter(size_t vertexTag):
                vertexTag(vertexTag){

            }
        
	    template<typename T_Cage>
	    std::vector<typename T_Cage::Vertex> operator()(const unsigned processID, const unsigned processCount, T_Cage &cage){

                using CommunicationPolicy = typename T_Cage::CommunicationPolicy;
                using Vertex              = typename T_Cage::Vertex;
                using Context             = typename CommunicationPolicy::Context;
                using VAddr               = typename CommunicationPolicy::VAddr;

                std::vector<VAddr> peersWithSameTag;
                
                CommunicationPolicy& comm = cage.comm;
                Context context = comm.getGlobalContext();
                

                // Get the information about who wants to
                // host vertices with the same tag
                std::array<size_t, 1> sendData{vertexTag};                
                for(VAddr vAddr = 0; vAddr < context.size(); vAddr++){
                    comm.asyncSend(vAddr, 0, context, sendData);
                }

                for(VAddr vAddr = 0; vAddr < context.size(); vAddr++){
                    std::array<size_t, 1> recvData{0};
                    comm.recv(vAddr, 0, context, recvData);
                    if(recvData[0] == vertexTag){
                        peersWithSameTag.push_back(vAddr);
                    }
                }

                // Distribute vertices to peers with same tag
                std::sort(peersWithSameTag.begin(), peersWithSameTag.end());

                const size_t nPeers = peersWithSameTag.size();
                size_t peer_i = 0;
                
                std::vector<Vertex> vertices = cage.getVertices();
                std::vector<Vertex> myVertices;
                
                for(size_t i = 0; i < vertices.size(); ++i){
                    if(vertices[i]().tag == vertexTag){
                        if(peersWithSameTag.at(peer_i) == context.getVAddr()){
                            myVertices.push_back(vertices[i]);
                            
                        }
                        peer_i = (peer_i + 1 % nPeers);
                        
                    }

                }
                
 		return myVertices;

	    }

	};

    } /* mapping */
    
} /* graybat */
