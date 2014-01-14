#ifndef CONSTANTS_H
#define CONSTANTS_H

/* Number of samples per event */
const unsigned int SAMPLE_COUNT = 16000;
/* Number of events copied to the GPU in one step */
const unsigned int CHUNK_COUNT = 100;
/* Number of chunks in the input buffer */
const unsigned int CHUNK_BUFFER_COUNT = 1024;

#endif
