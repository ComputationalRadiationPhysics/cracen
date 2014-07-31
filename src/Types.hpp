#ifndef TYPES_HPP
#define TYPES_HPP

#include <ostream>
#include <vector>
#include "Constants.hpp"
#include "Ringbuffer.hpp"
#ifndef __CUDACC__
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#endif

/*!
 * \brief input data datatype for Levenberg Marquardt (if data texture is used, can not be changed to integer types)
*/
typedef float DATATYPE;
typedef short int MeasureType;

struct FitData {
	float param[FitFunction::numberOfParams];
	int status;
	FitData() {};
	#ifndef __CUDACC__
	void save(boost::property_tree::ptree& pt) {
		using boost::property_tree::ptree;
		ptree thisFit;
		ptree params;
		thisFit.put("status", status);
		BOOST_FOREACH(float name, param) {
			ptree value;
			value.put("", name);
			params.push_back(std::make_pair("", value));
		}
		thisFit.add_child("params", params);
		pt.push_back(std::make_pair("",thisFit));
	}
	#endif
};
typedef FitData Output;
//typedef std::vector<DATATYPE> Wform;
typedef std::vector<DATATYPE> Chunk;
typedef Ringbuffer<Chunk> InputBuffer;
typedef Ringbuffer<Output> OutputBuffer;


#endif
