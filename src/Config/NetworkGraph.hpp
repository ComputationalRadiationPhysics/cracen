#ifndef NETWORKGRAPH_HPP
#define NETWORKGRAPH_HPP

#include <string>
// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
// GRAYBAT mappings
#include <graybat/mapping/PeerGroupMapping.hpp>
// GRAYBAT pattern
#include <graybat/pattern/Pipeline.hpp>

#include "../Utility/Whoami.hpp"

/***************************************************************************
 * graybat configuration
 ****************************************************************************/
// CommunicationPolicy
typedef graybat::communicationPolicy::ZMQ CP;
//typedef graybat::communicationPolicy::BMPI CP;
typedef graybat::graphPolicy::BGL<>    GP;
typedef graybat::Cage<CP, GP> Cage;
struct ZMQConfigDescriptor {
	const std::string masterIp = "127.0.0.1";
	const std::string localIp = Whoami(masterIp);
	
	const std::string masterUri = "tcp://"+masterIp+":5000";
	const std::string peerUri = "tcp://"+masterIp+":5001";
	unsigned int contextSize = 3;
};
ZMQConfigDescriptor zmqConfig;
CP cp(zmqConfig);

Cage cage(cp, graybat::pattern::Pipeline(std::vector<unsigned int>({1, 1, 1})));

#endif