#ifndef COMMANDLINEPARSER_HPP
#define COMMANDLINEPARSER_HPP

#include <string>
#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/any.hpp> /* boost::any_cast */

#include "Utility/Whoami.hpp"
#include "Config/Constants.hpp"

namespace CommandLineParser {
	
	namespace po = boost::program_options;
	
	po::variables_map parse(int argc, char** argv) {
		
		po::options_description networkOptions( "Network Options" );
		networkOptions.add_options()
		("signalingIp,sip", po::value<std::string>()->default_value("127.0.0.1"),
			"Required: The ip adress of the signaling server")
		("communicationIp,cip", po::value<std::string>(),
			 std::string(std::string() + "Optional: The ip adress of the client, by which it is reachable by other clients. This parameter is optional and " +
			 "should be used, only if the clients can not establish a connection amongst them. " +
			 "Note: This parameter only needed if network adress translation (NAT) is used in the network and the " +
			 "ip by which the client is reachable by the signaling server is different than the ip the client is " +
			 "reachable by other clients. If all clients use public ip adresses this parameter is not used.").c_str())
		("signalingPort,sp", po::value<unsigned int>()->default_value(5000),
			std::string(std::string() + "Optional: The port for the communication with the signaling server. " +
			"Must be different than the communication port. " +
			"Default port is 5000.").c_str())
		("communicationPort,cp", po::value<unsigned int>()->default_value(5001),
			std::string(std::string() + "Optional: The the port for communication between peers. " +
			"Must be different than the signaling port. " +
			"Default port is 5001").c_str())
		("sources,src", po::value<unsigned int>()->default_value(1),
			std::string(std::string() + "Optional: Number of data sources, both scopeReader and FileReader. The correct number is required to " +
			"establish the networking graph. If there are more than one sources, the parameter must be set accordingly." +
			"Default value is 1").c_str())
		("fitters,fit", po::value<unsigned int>()->default_value(1),
			std::string(std::string() + "Optional: Number of fitters. The correct number is required to " +
			"establish the networking graph. If there are more than one fitter, the parameter must be set accordingly." +
			"Default value is 1").c_str())
		("sinks,snk", po::value<unsigned int>()->default_value(1),
			std::string(std::string() + "Optional: Number of data sinks, both FileWriter and online monitors. The correct number is required to " +
			"establish the networking graph. If there are more than one sinks, the parameter must be set accordingly." +
			"Default value is 1").c_str());
		
		po::options_description inputOutputOptions( "Fitting Options" );
		inputOutputOptions.add_options()
		("inputFile, if", po::value<std::string>()->default_value(FILENAME_TESTFILE.c_str()),
			(std::string() + "Optional: Path to the input file of the FileReader. Only used by FileReader. " +
			"Default is \"" + FILENAME_TESTFILE + "\"").c_str())
		("outputFile, of", po::value<std::string>()->default_value(OUTPUT_FILENAME.c_str()),
			(std::string() + "Optional: Path to the output file of the FileReader. Only used by FileWriter. " +
			"Default is \"" + OUTPUT_FILENAME + "\"").c_str())
		("scopeFile, of", po::value<std::string>()->default_value(SCOPE_PARAMETERFILE.c_str()),
			(std::string() + "Optional: Path to the scope ini file of the ScopeReader. Only used by Scopereader " +
			"Default is \"" + SCOPE_PARAMETERFILE + "\"").c_str());
		
		//po::options_description fittingOptions( "Fitting Options" );
		
		po::options_description commandLineOptions( "Command line options" );
		commandLineOptions.add_options()
		("help, h",
			 "Print help message.");
		
		po::variables_map vm;
		po::options_description optionsDescritption;
		optionsDescritption
			.add(networkOptions)
			.add(inputOutputOptions)
			.add(commandLineOptions);
			
		po::store(po::parse_command_line( argc, argv, optionsDescritption ), vm);
		
		if(vm.count("help")){
			std::cout << "Usage: " << argv[0] << std::endl;
			std::cout << optionsDescritption << std::endl;
			std::exit(0);
		}
		
		if(!vm.count("communicationIp")){
			vm.insert(
				std::make_pair(
					std::string("communicationIp"), 
					po::variable_value(
						boost::any(Whoami(vm["signalingIp"].as<std::string>())),
						false
					)
				)
			);
		}
		
		std::string programName(argv[0]);
		vm.insert(
			std::make_pair(
				std::string("programName"),
				po::variable_value(
					boost::any(programName),
					false
				)
			)
		);
		
		vm.notify();
		return vm;
	}
}

#endif