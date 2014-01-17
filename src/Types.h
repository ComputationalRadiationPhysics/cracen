#ifndef TYPES_H
#define TYPES_H

#include <ostream>
#include "Constants.h"
#include "Ringbuffer.h"

typedef short int Precision;
typedef Precision Sample[SAMPLE_COUNT];
typedef Sample SampleChunk[CHUNK_COUNT];
typedef struct {
	Precision p1,p2,p3;
} Output;

/* defines how the output is written into the output file */
std::ostream& operator<<(std::ostream& lhs, const Output& rhs) {
	lhs << "{p1=" << rhs.p1 << ",p2=" << rhs.p2 << ",p3=" << rhs.p3 << "}";
	return lhs;
}

typedef Ringbuffer<SampleChunk, CHUNK_BUFFER_COUNT> InputBuffer;
typedef Ringbuffer<Output, CHUNK_BUFFER_COUNT> OutputBuffer;

#endif
