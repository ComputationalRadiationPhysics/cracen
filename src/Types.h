#ifndef TYPES_H
#define TYPES_H

#include <ostream>
#include <vector>
#include "Constants.h"
#include "Ringbuffer.h"

/*!
 * \brief input data datatype for Levenberg Marquardt (if data texture is used, can not be changed to integer types)
*/
typedef float DATATYPE;
typedef short int MeasureType;

template<unsigned int numberOfParams>
struct FitData {
	float param[numberOfParams];
	int status;
	FitData() {}
};

//typedef std::vector<DATATYPE> Wform;
typedef std::vector<DATATYPE> Chunk;
typedef Ringbuffer<Chunk> InputBuffer;
typedef FitData<numberOfParams> Output;
typedef Ringbuffer<Output> OutputBuffer;

#endif
