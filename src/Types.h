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

//typedef std::vector<DATATYPE> Wform;
typedef std::vector<DATATYPE> Chunk;

struct fitData {
	float param[COUNTPARAM];
	float startValue;
	float endValue;
	float extremumPos;
	float extremumValue;
	float euclidNormResidues;
	float averageAbsResidues;
	int status;
};
typedef fitData Output;

typedef Ringbuffer<Chunk> InputBuffer;
typedef Ringbuffer<Output> OutputBuffer;

#endif
