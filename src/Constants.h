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



#define CUDA //defined: runs on GPU, otherwise on CPU (useful for debugging)

//#define MAXCOUNTDATA 2450 //for compute capability 2.0 or higher - currently ca. 2450 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 48 kB (= max. shared memory)
#define MAXCOUNTDATA 800 //for compute capability 1.x - currently ca. 800 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 16 kB (= max. shared memory)

#define DATATYPE float //if data texture is used, can not be changed to integer types
#define MAXCALL 100
#define COUNTPARAM 3
#define PARAMSTARTVALUE { 1, 1, 1 } //any value, but not { 0, 0, 0 } (count = COUNTPARAM)

#define FITVALUETHRESHOLD 0.0 //0.5 //threshold between min (0.0) and max (1.0) value to define the data using interval to calculate the fit function
#define STARTENDPROPORTION 0.01 //proportion of countData for calculating the average of start/end value (e. g. 0.1 means average of the first 10% of data for start value and the last 10% for end value)


#endif
