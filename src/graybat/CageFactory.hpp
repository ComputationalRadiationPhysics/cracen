#ifndef CAGEFACTORY_HPP
#define CAGEFACTORY_HPP

#include <string>
#include <cstdlib>
#include <iostream>

#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/PeerGroupMapping.hpp>
#include <graybat/pattern/Pipeline.hpp>

#include <boost/program_options.hpp>

#include "../Utility/Whoami.hpp"

/***************************************************************************
 * graybat configuration
 ****************************************************************************/

struct CageFactory {
	
	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<>    GP;
	typedef graybat::Cage<CP, GP> Cage;
	
	struct ZMQConfigDescriptor {
		const std::string masterUri, peerUri;
		const unsigned int contextSize;
	};
		
	static Cage buildCage (boost::program_options::variables_map vm) {
		
		const std::string signalingIp = vm["signalingIp"].as<std::string>();
		const std::string localIp = Whoami(signalingIp);
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(vm["signalingPort"].as<unsigned int>());
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(vm["4communicationPort"].as<unsigned int>());
		const unsigned int contextSize = vm["sources"].as<unsigned int>() + vm["fitters"].as<unsigned int>() + vm["sinks"].as<unsigned int>();
			
		
		const std::string name = vm["programName"].as<std::string>();
		unsigned int peer;
		if(name == "FileReader" || name == "ScopeReader") {
			peer = 0;
		} else if (name == "Fitter") {
			peer = 1;
		} else if (name == "FileWriter") {
			peer = 2;
		} else {
			std::cerr << "Could not assign peer to executable. PeerGroupMapping not possible." << std::endl;
			std::cerr << "Program name has to be one of [FileReader, ScopeReader, Fitter, FileWriter], to assign the peer to the executable." << std::endl;
			std::cerr << "Program name given was: " << name << std::endl;
			std::exit(1);
		}
		
		ZMQConfigDescriptor zmqConfig{masterUri, peerUri, contextSize};
		CP cp(zmqConfig);
		Cage cage(cp, graybat::pattern::Pipeline(std::vector<unsigned int>({vm["sources"].as<unsigned int>(), vm["fitters"].as<unsigned int>(), vm["sinks"].as<unsigned int>()})));
		cage.distribute(graybat::mapping::PeerGroupMapping(peer));
		return cage;
	}
};
#endif