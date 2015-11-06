#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>
#include <array>
#include <iostream>
#include "Constants.hpp"
#include "../Utility/Ringbuffer.hpp"
#ifndef __CUDACC__
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#endif

/*!
 * \brief input data datatype for Levenberg Marquardt (if data texture is used, can not be changed to integer types)
*/

typedef float DATATYPE;
typedef short int MeasureType;

struct FitData {
	float param[FitFunction::numberOfParams];
	int status;
	int woffset;
	
	FitData() {}
	~FitData() {}
	
	FitData(const FitData& cpy) :
		status(cpy.status),
		woffset(cpy.woffset) 
	{
		for(unsigned int i = 0; i < FitFunction::numberOfParams; i++) {
			param[i] = cpy.param[i];
		}
	}
	
	void swap(FitData& rhs) {
		std::swap(param, rhs.param);
		std::swap(status, rhs.status);
		std::swap(woffset, rhs.woffset);
	}
	
	FitData & operator= (const FitData & assign) {
		status = assign.status;
		woffset = assign.woffset;
		for(unsigned int i = 0; i < FitFunction::numberOfParams; i++) {
			param[i] = assign.param[i];
		}
		return *this;
	}
	
	#ifndef __CUDACC__
	void save(boost::property_tree::ptree& pt) {
		using boost::property_tree::ptree;
		ptree thisFit;
		ptree params;
		thisFit.put("status", status);
		thisFit.put("woffset", woffset);
		for(unsigned int i = 0; i < FitFunction::numberOfParams; i++) {
			ptree value;
			std::ostringstream out;
			out << std::setprecision(12) << param[i];
			value.put("", out.str());
			params.push_back(std::make_pair("", value));
		}
		thisFit.add_child("params", params);
		pt.push_back(std::make_pair("",thisFit));
	}
	
	friend class boost::serialization::access;
    template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::make_array<float>(param, FitFunction::numberOfParams);
		ar & status;
		ar & woffset;
	}
	
	#endif
	
	
	float* data() {
		return param;
	}
};
typedef FitData Output;
//typedef std::vector<DATATYPE> Wform;
typedef std::array<DATATYPE, CHUNK_COUNT*SAMPLE_COUNT> Chunk;
typedef Ringbuffer<Chunk> InputBuffer;
typedef Ringbuffer<Output> OutputBuffer;


#endif
