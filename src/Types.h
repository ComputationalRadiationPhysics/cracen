#ifndef TYPES_H
#define TYPES_H

#include <ostream>
#include <vector>
#include "Constants.h"
#include "Ringbuffer.h"

typedef short int Precision;
typedef Precision Sample[SAMPLE_COUNT];
typedef Sample SampleChunk[CHUNK_COUNT];
struct fitData {
	float param[COUNTPARAM];
	float startValue;
	float endValue;
	float extremumPos;
	float extremumValue;
	int status;
};
typedef fitData Output;

// a waveform consisting of SAMPLE_COUNT samples is of type: wform_t
typedef std::vector<short int> wform_t;

typedef Ringbuffer<SampleChunk> InputBuffer;
typedef Ringbuffer<Output> OutputBuffer;

#endif
