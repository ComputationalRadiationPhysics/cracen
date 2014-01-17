#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/* Number of samples per event */
const unsigned int SAMPLE_COUNT = 1000;
/* Number of events copied to the GPU in one step */
const unsigned int CHUNK_COUNT = 100;
/* Number of chunks in the input buffer */
const unsigned int CHUNK_BUFFER_COUNT = 1024;
/* Filtermode for Interpolation */
const cudaTextureFilterMode FILTER_MODE = cudaFilterModeLinear;
/* Number of samples per waveform (event) in testfile
 * Al_25keV-259.cdb
 */
const std::string FILENAME_TESTFILE = "../data/Al_25keV-259.cdb";
const unsigned int SAMPLE_COUNT_TESTFILE = 1000;
const unsigned int SEGMENT_COUNT_TESTFILE = 1;
const unsigned int WAVEFORM_COUNT_TESTFILE = 100000;
#endif
