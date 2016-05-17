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

class CageFactory {
private:
	boost::program_options::variables_map vm;
public:
	
	typedef graybat::communicationPolicy::ZMQ CP;
	//typedef graybat::communicationPolicy::BMPI CP;
	typedef graybat::graphPolicy::BGL<unsigned int>    GP;
	typedef graybat::Cage<CP, GP> Cage;
	
	CageFactory(boost::program_options::variables_map vm) :
		vm(vm)
	{}
	
	CP::Config commPoly() {
		const std::string signalingIp = vm["signalingIp"].as<std::string>();
		const std::string localIp = Whoami(signalingIp);
		const std::string masterUri = "tcp://"+signalingIp+":"+std::to_string(vm["signalingPort"].as<unsigned int>());
		const std::string peerUri = "tcp://"+localIp+":"+std::to_string(vm["communicationPort"].as<unsigned int>());
		std::cout << "My URI =" << peerUri << std::endl;
		const unsigned int contextSize = vm["sources"].as<unsigned int>() + vm["fitters"].as<unsigned int>() + vm["sinks"].as<unsigned int>();

		return CP::Config({masterUri, peerUri, contextSize}); //ZMQ Config
		//return CP::Config({}); //BMPI Config
	}
	
	auto graphPoly() {
		return graybat::pattern::Pipeline<GP>(
			std::vector<unsigned int>({
				vm["sources"].as<unsigned int>(),
				vm["fitters"].as<unsigned int>(),
				vm["sinks"].as<unsigned int>()
				
			})
		);
	}
	
	void map(Cage& cage) {
		std::string name = vm["programName"].as<std::string>();
		name = name.substr(name.find_last_of("/") + 1); 
		unsigned int peer;
		if(name == "FileReader" || name == "ScopeReader" || name == "BenchReader") {
			peer = 0;
		} else if (name == "Fitter") {
			peer = 1;
		} else if (name == "FileWriter" || name == "RootWriter") {
			peer = 2;
		} else {
			std::cerr << "Could not assign peer to executable. PeerGroupMapping not possible." << std::endl;
			std::cerr << "Program name has to be one of [FileReader, ScopeReader, Fitter, FileWriter, RootWriter], to assign the peer to the executable." << std::endl;
			std::cerr << "Program name given was: " << name << std::endl;
			std::exit(1);
		}
		
		cage.distribute(graybat::mapping::PeerGroupMapping(peer));
	}
	
	auto mapping() {
		std::string name = vm["programName"].as<std::string>();
		name = name.substr(name.find_last_of("/") + 1); 
		unsigned int peer;
		if(name == "FileReader" || name == "ScopeReader" || name == "BenchReader") {
			peer = 0;
		} else if (name == "Fitter") {
			peer = 1;
		} else if (name == "FileWriter" || name == "RootWriter") {
			peer = 2;
		} else {
			std::cerr << "Could not assign peer to executable. PeerGroupMapping not possible." << std::endl;
			std::cerr << "Program name has to be one of [FileReader, ScopeReader, Fitter, FileWriter, RootWriter], to assign the peer to the executable." << std::endl;
			std::cerr << "Program name given was: " << name << std::endl;
			std::exit(1);
		}
		
		return graybat::mapping::PeerGroupMapping(peer);
	}
};
#endif
