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
	
	#endif
	
	
	float* data() {
		return param;
	}
	
};

template <class type>
struct GrayBatAdapter {
	typedef type value_type;
	type d;
	
	GrayBatAdapter(type t) :
		d(t)
	{}
	
	void operator=(const GrayBatAdapter& rhs) { 
		d = type(rhs.d);
	}
	
	GrayBatAdapter() { }
	
	type* data() { return &d; }
	
	type const* data() const { return &d; }
	
	size_t size() const { return sizeof(type); };
	
	void swap(GrayBatAdapter rhs) { 
		d.swap(rhs.d);
	}
};

typedef FitData Output;
//typedef std::vector<DATATYPE> Wform;

template <class type, size_t size_v>
struct FixedSizeVector {
private:
	std::vector<type> values;
	//FixedSizeVector(const FixedSizeVector& rhs) {};
	//FixedSizeVector& operator=(const FixedSizeVector& rhs) {};
public:
	typedef type value_type;
	
	FixedSizeVector() :
		values(size_v)
	{}

	type& operator[](size_t k) {
		return values[k];
	}
	
	type& at(size_t k) {
		return values.at(k);
	}
	
	type& front() {
		return values.front();
	}
	
	size_t size() const {
		return size_v;
	};
	
	type* data() {
		return values.data();
	}
	
	type  const* data() const {
		return values.data();
	}
	
	void swap(FixedSizeVector& s) {
		values.swap(s.values);
	}
};


//typedef std::array<DATATYPE, CHUNK_COUNT*SAMPLE_COUNT> Chunk;
typedef FixedSizeVector<DATATYPE, CHUNK_COUNT*SAMPLE_COUNT> Chunk;
typedef Ringbuffer<Chunk> InputBuffer;
typedef Ringbuffer<GrayBatAdapter<Output>> OutputBuffer;


#endif
