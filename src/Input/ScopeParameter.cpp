#include "ScopeReader.hpp"

#include <map>
#include <cctype>
#include <boost/bind.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

ScopeReader::ScopeParameter::ScopeParameter() :
	nbrSamples(42000),
	nbrSegments(1),
	nbrSessions(1),
	nbrWaveforms(100),
	sampInterval(2.38),
	delayTime(-50000),
	coupling0(0),
	coupling1(0),
	bandwidth0(0),
	bandwidth1(0),
	fullScale0(2000),
	fullScale1(2000),
	offset0(0),
	offset1(0),
	trigType(1),
	trigCoupling_int(0),
	trigCoupling_ext(3),
	trigSlope(0),
	trigLevel(100),
	timeout(0),
	i_Simulation(0)
{}

ScopeReader::ScopeParameter::ScopeParameter(const std::string& filename) :
	nbrSamples(42000),
	nbrSegments(1),
	nbrSessions(1),
	nbrWaveforms(100),
	sampInterval(2.38),
	delayTime(-50000),
	coupling0(0),
	coupling1(0),
	bandwidth0(0),
	bandwidth1(0),
	fullScale0(2000),
	fullScale1(2000),
	offset0(0),
	offset1(0),
	trigType(1),
	trigCoupling_int(0),
	trigCoupling_ext(3),
	trigSlope(0),
	trigLevel(100),
	timeout(0),
	i_Simulation(0)
{
	std::map<std::string, ViInt32*> intParameterMap;
	std::map<std::string, ViReal64*> doubleParameterMap;
	/* int */
	intParameterMap["segments"] = &nbrSegments;
	intParameterMap["waveforms"] = &nbrWaveforms;
	intParameterMap["samples"] = &nbrSamples;
	intParameterMap["sessions"] = &nbrSessions;
	intParameterMap["coupling0"] = &coupling0;
	intParameterMap["coupling1"] = &coupling1; 
	intParameterMap["bandwidth0"] = &bandwidth0;
	intParameterMap["bandwidth0"] = &bandwidth1; 
	intParameterMap["trigger_type"] = &trigType;
	intParameterMap["trigger_coupling_int"] = &trigCoupling_int;
	intParameterMap["trigger_coupling_ext"] = &trigCoupling_ext;
	intParameterMap["trigger_slope"] = &trigSlope;
	intParameterMap["timeout"] = &timeout;
	intParameterMap["simulation"] = &i_Simulation;
	/* real */
	doubleParameterMap["trigger_level"] = &trigLevel;
	doubleParameterMap["sampling_interval"] = &sampInterval;
	doubleParameterMap["delay_time"] = &delayTime; 
	doubleParameterMap["full_scale0"] = &fullScale0;
	doubleParameterMap["full_scale1"] = &fullScale1;
	doubleParameterMap["offset0"] = &offset0;
	doubleParameterMap["offset1"] = &offset1;
	
	/* Open file */
	std::ifstream myfile(filename.c_str());
	std::string line;
	if(myfile.is_open()) {
		/* Read line by line */
		while(std::getline (myfile,line) ) {
			std::cout << line << std::endl;
		
			/* Strip Whitespaces */
			line.erase(std::remove_if(line.begin(), line.end(), boost::bind( std::isspace<char>, _1, std::locale::classic())));
			/* Strip Comments */
			line.erase(std::find(line.begin(), line.end(), ';'), line.end());
			/* Explode on '=' */
			boost::char_separator<char> sep("=");
		    boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
		    std::vector<std::string> tokenVector;
		    BOOST_FOREACH(std::string tok, tokens) {
		    	tokenVector.push_back(tok);
		    }
		    if(tokenVector.size() == 2) {
				try {
				if(intParameterMap.count(tokenVector[0]) > 0) {
					/* lookup in intParameterMap */
					*intParameterMap[tokenVector[0]] = boost::lexical_cast<ViInt32>(tokenVector[1]);
				} else if(doubleParameterMap.count(tokenVector[0]) > 0) {
					/* lookup in doubleParameterMap */
					*doubleParameterMap[tokenVector[0]] = boost::lexical_cast<ViReal64>(tokenVector[1]);
				} else {
					std::cerr << "Warning: Key in scope inifile is invalid." << std::endl;
					std::cerr << "	The key-value pair is ignored for settings." << std::endl;
					std::cerr << "	Key: " << tokenVector[0] << std::endl; 
				}
				} catch(boost::bad_lexical_cast e) {
					std::cerr << "Warning: Bad Value in scope inifile." << std::endl;
					std::cerr << "	The value could not be parsed to numeric value, which is required. The key-value pair is ignored." << std::endl;
					std::cerr << "	Key: " << tokenVector[0] << ", Value: " << tokenVector[1] << std::endl;
				}
			}
		}
		myfile.close();
	} else {
		std::cout << "Unable to open file" << std::endl;
	}	
}
