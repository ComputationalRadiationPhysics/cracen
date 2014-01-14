#ifndef TYPES_H
#define TYPES_H

#include "Constants.h"
#include "Ringbuffer.h"

typedef short int Precision;
typedef Precision Sample[SAMPLE_COUNT];
typedef Sample SampleChunk[CHUNK_COUNT];
typedef struct {
	Precision p1,p2,p3;
} Output;

typedef Ringbuffer<SampleChunk, CHUNK_BUFFER_COUNT> InputBuffer;
typedef Ringbuffer<Output, CHUNK_BUFFER_COUNT> OutputBuffer;

#endif
