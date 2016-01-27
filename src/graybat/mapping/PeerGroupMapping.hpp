/*
 * (C) Erik Zenker (2015)
 * 	   Fabian Jung (2015)
 * 
 */

#ifndef PEERGROUPMAPPING_HPP
#define PEERGROUPMAPPING_HPP

#include <algorithm> /* std::sort*/
#include <vector>    /* std::vector */

#include <iostream>
namespace graybat {
    
    namespace mapping {
    
	struct PeerGroupMapping {

            const unsigned int stage;
            
            PeerGroupMapping(unsigned int stage):
                stage(stage){
				std::cout << "Stage : " << stage << std::endl;
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
                std::array<size_t, 1> sendData{stage};                
                for(VAddr vAddr = 0; vAddr < context.size(); vAddr++){
                    comm.asyncSend(vAddr, 0, context, sendData);
                }

                for(VAddr vAddr = 0; vAddr < context.size(); vAddr++){
                    std::array<size_t, 1> recvData{0};
                    comm.recv(vAddr, 0, context, recvData);
                    if(recvData[0] == stage){
                        peersWithSameTag.push_back(vAddr);
					}
                }
				std::cout << "Peers with same Tag " << peersWithSameTag.size() << std::endl;
                    
                // Distribute vertices to peers with same tag
                std::sort(peersWithSameTag.begin(), peersWithSameTag.end());

                const size_t nPeers = peersWithSameTag.size();
                size_t peer_i = 0;
                
                std::vector<Vertex> vertices = cage.getVertices();
                std::vector<Vertex> myVertices;
                
				std::cout << "vertices size" << vertices.size() << std::endl;
                for(size_t i = 0; i < vertices.size(); ++i){
                    if(vertices[i]() == stage){
                        if(peersWithSameTag.at(peer_i) == context.getVAddr()){
                            myVertices.push_back(vertices[i]);
                            
                        }
                        peer_i = (peer_i + 1 % nPeers);
                        
                    }

                }
                
	            std::cout << "myVertices size " << myVertices.size() << std::endl;
                
 		return myVertices;

	    }

	};

    } /* mapping */
    
} /* graybat */

#endif
